"""
v16 MINER - Anti-Convergence Edition

Based on v12 with three key additions to break local optima:

1. DIVERSITY TAX: After iteration 5, applies a penalty (0.005 per occurrence)
   to molecules using overrepresented components. Aggressively favors
   exploration of underused building blocks.

2. PRE-RESET EXPLOIT: At iterations 5, 11, 17, 23... (one before each reset),
   runs fingerprint exploitation to squeeze maximum value from current optimum
   before exploring new space.

3. SCHEDULED REGION RESETS: At iterations 6, 12, 18, 24..., forces anti-dominant
   exploration using region-based sampling. Periodically "pulses" the search
   to explore underrepresented chemical space.

Key parameters:
- REGION_RESET_INTERVAL = 6 (explore every 6 iters)
- REGION_RESET_START = 6 (start resets at iter 6)
- Diversity tax = 0.005 per occurrence in top_pool
- Pre-reset exploit → Region reset → Exploit → Amplification (priority order)
"""
import os
from traceback import print_exc
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys
import json
import time
import argparse
import bittensor as bt
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import pandas as pd
from pathlib import Path
import nova_ph2
from itertools import combinations

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PARENT_DIR)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")

from nova_ph2.PSICHIC.wrapper import PsichicWrapper
from nova_ph2.PSICHIC.psichic_utils.data_utils import virtual_screening
from molecules import (
    generate_valid_random_molecules_batch,
    select_diverse_elites,
    build_component_weights,
    compute_tanimoto_similarity_to_pool,
    sample_random_valid_molecules,
    compute_maccs_entropy,
    SynthonLibrary,
    generate_molecules_from_synthon_library,
    validate_molecules,
    generate_inchikey,
    run_exploit,
    get_top_n_unexploited,
    compute_diverse_components,
    generate_diverse_molecules,
    run_amplification,
    # v12: Progressive amplification (full famy.py approach)
    ProgressiveAmplificationState,
    run_progressive_amplification,
)

# v12: Region-based sampling (anti-dominant strategy)
from regions import RegionBasedSampler

DB_PATH = str(Path(nova_ph2.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")

# Exploit settings
TOP_N_EXPLOIT = 5                # Max molecules to consider per exploit call
TARGET_REACTANTS = 15            # Target number of NEW (role,reactant) pairs per exploit (5 mols × 3 components)
LIMIT_PER_REACTANT = 600         # Candidates per reactant (ranked by fingerprint similarity)

target_models = []
antitarget_models = []


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Nova Blueprint Miner")
    parser.add_argument('--rxn', type=int, help='Override allowed_reaction (e.g., --rxn 5)')
    parser.add_argument('--input', type=str, default=os.path.join(BASE_DIR, "input.json"),
                        help='Path to input.json config file')
    return parser.parse_args()


def get_config(input_file: str = os.path.join(BASE_DIR, "input.json"), rxn_override: int = None):
    with open(input_file, "r") as f:
        d = json.load(f)
    config = {**d.get("config", {}), **d.get("challenge", {})}

    # Override allowed_reaction if --rxn argument provided
    if rxn_override is not None:
        config["allowed_reaction"] = f"rxn:{rxn_override}"
        bt.logging.warning(f"[Miner] Overriding allowed_reaction to rxn:{rxn_override}")

    return config


def initialize_models(config: dict):
    """Initialize separate model instances for each target and antitarget sequence."""
    global target_models, antitarget_models
    target_models = []
    antitarget_models = []

    for seq in config["target_sequences"]:
        wrapper = PsichicWrapper()
        wrapper.initialize_model(seq)
        target_models.append(wrapper)

    for seq in config["antitarget_sequences"]:
        wrapper = PsichicWrapper()
        wrapper.initialize_model(seq)
        antitarget_models.append(wrapper)


# ---------- scoring helpers (reuse pre-initialized models) ----------
def target_score_from_data(data: pd.Series):
    """Score molecules against all target models."""
    global target_models, antitarget_models
    try:
        target_scores = []
        smiles_list = data.tolist()
        for target_model in target_models:
            scores = target_model.score_molecules(smiles_list)
            for antitarget_model in antitarget_models:
                antitarget_model.smiles_list = smiles_list
                antitarget_model.smiles_dict = target_model.smiles_dict

            scores.rename(columns={'predicted_binding_affinity': "target"}, inplace=True)
            target_scores.append(scores["target"])
        # Average across all targets
        target_series = pd.DataFrame(target_scores).mean(axis=0)
        return target_series
    except Exception as e:
        bt.logging.error(f"Target scoring error: {e}")
        return pd.Series(dtype=float)


def antitarget_scores():
    """Score molecules against all antitarget models."""

    global antitarget_models
    try:
        antitarget_scores = []
        for i, antitarget_model in enumerate(antitarget_models):
            antitarget_model.create_screen_loader(antitarget_model.protein_dict, antitarget_model.smiles_dict)
            antitarget_model.screen_df = virtual_screening(antitarget_model.screen_df,
                                            antitarget_model.model,
                                            antitarget_model.screen_loader,
                                            os.getcwd(),
                                            save_interpret=False,
                                            ligand_dict=antitarget_model.smiles_dict,
                                            device=antitarget_model.device,
                                            save_cluster=False,
                                            )
            scores = antitarget_model.screen_df[['predicted_binding_affinity']]
            scores.rename(columns={'predicted_binding_affinity': f"anti_{i}"}, inplace=True)
            antitarget_scores.append(scores[f"anti_{i}"])

        if not antitarget_scores:
            return pd.Series(dtype=float)

        # average across antitargets
        anti_series = pd.DataFrame(antitarget_scores).mean(axis=0)
        return anti_series
    except Exception as e:
        bt.logging.error(f"Antitarget scoring error: {e}")
        return pd.Series(dtype=float)


def _cpu_random_candidates_with_similarity(
    iteration: int,
    n_samples: int,
    subnet_config: dict,
    top_pool_df: pd.DataFrame,
    avoid_inchikeys: set[str] | None = None,
    thresh: float = 0.8
) -> pd.DataFrame:
    """
    CPU-side helper:
    - draws a random batch of valid molecules (independent of the GPU batch),
    - computes Tanimoto similarity vs. current top_pool,
    - returns a DataFrame with name, smiles, InChIKey, tanimoto_similarity.
    """
    try:
        random_df = sample_random_valid_molecules(
            n_samples=n_samples,
            subnet_config=subnet_config,
            avoid_inchikeys=avoid_inchikeys,
            focus_neighborhood_of=top_pool_df
        )
        if random_df.empty or top_pool_df.empty:
            return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

        sims = compute_tanimoto_similarity_to_pool(
            candidate_smiles=random_df["smiles"],
            pool_smiles=top_pool_df["smiles"],
        )
        random_df = random_df.copy()
        random_df["tanimoto_similarity"] = sims.reindex(random_df.index).fillna(0.0)
        random_df = random_df.sort_values(by="tanimoto_similarity", ascending=False)
        random_df_filtered = random_df[random_df["tanimoto_similarity"] >= thresh]

        if random_df_filtered.empty:
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "tanimoto_similarity"])

        random_df_filtered = random_df_filtered.reset_index(drop=True)
        return random_df_filtered[["name", "smiles", "InChIKey"]]
    except Exception as e:
        bt.logging.warning(f"[Miner] _cpu_random_candidates_with_similarity failed: {e}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

def select_diverse_subset(pool, top_95_smiles, subset_size=5, entropy_threshold=0.1):
    smiles_list = pool["smiles"].tolist()
    for combination in combinations(smiles_list, subset_size):
        test_subset = top_95_smiles + list(combination)
        entropy = compute_maccs_entropy(test_subset)
        if entropy >= entropy_threshold:
            bt.logging.info(f"Entropy Threshold Met: {entropy:.4f}")
            return pool[pool["smiles"].isin(combination)]

    bt.logging.warning("No combination exceeded the given entropy threshold.")
    return pd.DataFrame()


def main(config: dict):
    # WINNING MODEL: Combines best of richard1220v3 (balanced) + patarcom1 (exploration) + smart adaptations
    # Target: Beat richard1220v3 (0.0124) by 0.05+ to reach 0.0174+
    print(f"[Miner] main() started, rxn={config['allowed_reaction']}", flush=True)
    base_n_samples = 800  # Optimized: proven pattern for high scores (more iterations, faster learning)
    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score", "Target", "Anti"])
    rxn_id = int(config["allowed_reaction"].split(":")[-1])
    print(f"[Miner] rxn_id={rxn_id}", flush=True)
    iteration = 0

    mutation_prob = 0.25
    elite_frac = 0.75

    seen_inchikeys = set()
    seed_df = pd.DataFrame(columns=["name", "smiles", "InChIKey", "tanimoto_similarity"])
    start = time.time()
    prev_avg_score = None
    current_avg_score = None
    score_improvement_rate = 0.0
    no_improvement_counter = 0

    synthon_lib = None
    use_synthon_search = False

    # Track best molecules and score history for adaptive strategies
    best_molecules_history = []
    max_score_history = []

    # Enhanced first iteration - good exploration
    n_samples_first_iteration = base_n_samples * 4 if config["allowed_reaction"] != "rxn:5" else base_n_samples * 2

    # Mode tracking - iterations 1-17: GA/Synthon, 18-21: Exploit, 22+: Progressive Amplification
    exploit_start_iteration = 18
    exploit_end_iteration = 21
    amplification_start_iteration = 22
    amplification_end_iteration = 999  # runs until end
    use_amplification_mode = False
    use_exploit_mode = False
    seen_names = set()  # Track all molecule names for exploit/amplification
    exploited_reactants = set()  # Track which reactants have been exploited
    amplification_rounds = 0  # Track amplification iteration count

    # v12: Progressive amplification state (persists across rounds, like famy.py)
    progressive_amp_state = None

    # v16: Cached region sampler for scheduled exploration windows
    cached_region_sampler = None

    # v16: Scheduled exploration window settings
    REGION_RESET_INTERVAL = 6  # Run anti-dominant exploration every N iterations
    REGION_RESET_START = 6     # Start scheduled resets after this iteration

    print(f"[Miner] Entering main loop (v16: pre-reset exploit at 5,11,17..., region reset at 6,12,18..., diversity tax=0.005)...", flush=True)

    # Use single CPU worker - proven pattern reduces overhead, improves stability
    with ProcessPoolExecutor(max_workers=2) as cpu_executor:
        while time.time() - start < 2700:  # 45 minutes
            iteration += 1
            iter_start_time = time.time()
            print(f"[Miner] === Starting iteration {iteration} ===", flush=True)

            # Switch to exploit mode at iteration 15-20
            if iteration >= exploit_start_iteration and iteration <= exploit_end_iteration and not use_exploit_mode:
                use_exploit_mode = True
                use_amplification_mode = False
                seen_names = set(top_pool["name"].tolist())
                bt.logging.info(f"[Miner] Switching to EXPLOIT MODE at iteration {iteration}")
                print(f"[Miner] Switching to EXPLOIT MODE at iteration {iteration}", flush=True)

            # Switch to progressive amplification mode at iteration 21+
            if iteration >= amplification_start_iteration and not use_amplification_mode:
                use_amplification_mode = True
                use_exploit_mode = False  # Turn off exploit
                seen_names = set(top_pool["name"].tolist())

                # v12: Initialize ProgressiveAmplificationState (famy.py approach)
                try:
                    progressive_amp_state = ProgressiveAmplificationState(DB_PATH, rxn_id)
                    bt.logging.info(f"[Miner] Initialized ProgressiveAmplificationState for rxn:{rxn_id}")
                except Exception as e:
                    bt.logging.error(f"[Miner] Failed to initialize ProgressiveAmplificationState: {e}")
                    progressive_amp_state = None

                bt.logging.info(f"[Miner] Switching to PROGRESSIVE AMPLIFICATION MODE at iteration {iteration}")
                print(f"[Miner] Switching to PROGRESSIVE AMPLIFICATION MODE at iteration {iteration}", flush=True)

            # Adaptive n_samples: maintain good throughput
            remaining_time = 2700 - (time.time() - start)
            if remaining_time > 1500:
                n_samples = base_n_samples
            elif remaining_time > 900:
                n_samples = int(base_n_samples * 0.95)
            elif remaining_time > 600:
                n_samples = int(base_n_samples * 0.90)
            elif remaining_time > 300:
                n_samples = int(base_n_samples * 0.85)
            else:
                n_samples = int(base_n_samples * 0.80)

            # Build synthon library after first iteration
            if iteration == 2 and not top_pool.empty and synthon_lib is None:
                try:
                    bt.logging.info("[Miner] Building synthon library from top molecules...")
                    synthon_lib_start = time.time()
                    synthon_lib = SynthonLibrary(DB_PATH, rxn_id)
                    use_synthon_search = True
                    bt.logging.info(f"[Miner] Synthon library ready! Built in {time.time() - synthon_lib_start:.2f}s")
                except Exception as e:
                    bt.logging.warning(f"[Miner] Could not build synthon library: {e}")
                    use_synthon_search = False

            component_weights = build_component_weights(top_pool, rxn_id) if not top_pool.empty else None
            # Enhanced elite pool
            elite_df = select_diverse_elites(top_pool, min(150, len(top_pool))) if not top_pool.empty else pd.DataFrame()
            elite_names = elite_df["name"].tolist() if not elite_df.empty else None

            # v16: CHECK FOR SCHEDULED REGION RESET FIRST (takes priority over exploit/amplification)
            is_region_reset_iteration = (
                iteration >= REGION_RESET_START and
                iteration % REGION_RESET_INTERVAL == 0 and
                not top_pool.empty
            )

            # v16: PRE-RESET EXPLOIT - maximize current optimum right before resetting
            # Runs at iterations 5, 11, 17, 23, 29... (one before each reset)
            is_pre_reset_exploit_iteration = (
                iteration >= REGION_RESET_START - 1 and
                (iteration + 1) % REGION_RESET_INTERVAL == 0 and
                not top_pool.empty
            )

            # v16: PRE-RESET EXPLOIT (squeeze out best molecules before reset)
            if is_pre_reset_exploit_iteration:
                print(f"[Miner] Iteration {iteration}: PRE-RESET EXPLOIT (maximizing before reset)...", flush=True)
                bt.logging.info(f"[Miner] Iteration {iteration}: v16 pre-reset exploit to maximize current optimum")

                # Get molecules with unexploited reactants
                all_pool_mols = top_pool.to_dict("records")
                top_mols = get_top_n_unexploited(all_pool_mols, exploited_reactants, n=TOP_N_EXPLOIT, target_reactants=TARGET_REACTANTS)

                if not top_mols:
                    bt.logging.warning(f"[Miner] All reactants exhausted for pre-reset exploit, using GA")
                    data = generate_valid_random_molecules_batch(
                        rxn_id,
                        n_samples=n_samples,
                        db_path=DB_PATH,
                        subnet_config=config,
                        batch_size=400,
                        elite_names=elite_names,
                        elite_frac=elite_frac,
                        mutation_prob=mutation_prob,
                        avoid_inchikeys=seen_inchikeys,
                        component_weights=component_weights,
                    )
                else:
                    print(f"[Miner] Pre-reset exploit: {len(top_mols)} molecules selected (targeting {TARGET_REACTANTS} new reactants, {len(exploited_reactants)} already done)...", flush=True)

                    exploit_start = time.time()
                    try:
                        exploit_results, exploit_summary = run_exploit(
                            top_molecules=top_mols,
                            db_path=DB_PATH,
                            rxn_id=rxn_id,
                            top_n=TOP_N_EXPLOIT,
                            limit_per_reactant=LIMIT_PER_REACTANT,
                            avoid_names=seen_names,
                            exploited_reactants=exploited_reactants,
                            min_heavy_atoms=config.get('min_heavy_atoms', 20),
                            min_rotatable=config.get('min_rotatable_bonds', 1),
                            max_rotatable=config.get('max_rotatable_bonds', 10)
                        )

                        if exploit_summary and 'exploited_reactant_ids' in exploit_summary:
                            exploited_reactants.update(exploit_summary['exploited_reactant_ids'])

                        exploit_time = time.time() - exploit_start
                        bt.logging.info(f"[Miner] Pre-reset exploit returned {len(exploit_results)} candidates in {exploit_time:.1f}s")

                        if exploit_results:
                            data = pd.DataFrame(exploit_results)
                            seen_names.update(data["name"].tolist())
                        else:
                            bt.logging.warning(f"[Miner] Pre-reset exploit returned no results, falling back to GA")
                            data = generate_valid_random_molecules_batch(
                                rxn_id,
                                n_samples=n_samples,
                                db_path=DB_PATH,
                                subnet_config=config,
                                batch_size=400,
                                elite_names=elite_names,
                                elite_frac=elite_frac,
                                mutation_prob=mutation_prob,
                                avoid_inchikeys=seen_inchikeys,
                                component_weights=component_weights,
                            )
                    except Exception as e:
                        bt.logging.error(f"[Miner] Pre-reset exploit failed: {e}")
                        print_exc()
                        data = generate_valid_random_molecules_batch(
                            rxn_id,
                            n_samples=n_samples,
                            db_path=DB_PATH,
                            subnet_config=config,
                            batch_size=400,
                            elite_names=elite_names,
                            elite_frac=elite_frac,
                            mutation_prob=mutation_prob,
                            avoid_inchikeys=seen_inchikeys,
                            component_weights=component_weights,
                        )

            # v16: SCHEDULED REGION EXPLORATION WINDOWS
            # Every N iterations, force anti-dominant exploration to break local optima
            elif is_region_reset_iteration:
                print(f"[Miner] Iteration {iteration}: SCHEDULED REGION RESET (anti-dominant pulse)...", flush=True)
                bt.logging.info(f"[Miner] Iteration {iteration}: v16 scheduled region exploration window")

                try:
                    region_start = time.time()

                    # Build or reuse region sampler
                    if cached_region_sampler is None:
                        cached_region_sampler = RegionBasedSampler(DB_PATH, rxn_id)

                    # Analyze current top_pool dominance
                    def get_last_comp(name):
                        parts = name.split(':')
                        return parts[-1] if len(parts) >= 3 else ''

                    pool_comp_counts = top_pool['name'].apply(get_last_comp).value_counts()
                    dominant_comps = set(pool_comp_counts.head(5).index.tolist())
                    bt.logging.info(f"[Miner] Top 5 dominant components to AVOID: {dominant_comps}")

                    # Generate molecules explicitly avoiding dominant components
                    # Heavy focus on rare/uncommon regions: 50% rare, 35% uncommon, 15% common
                    region_molecules = cached_region_sampler.generate_region_stratified_molecules(
                        n_samples=n_samples,
                        rare_pct=0.50,
                        uncommon_pct=0.35,
                        common_pct=0.15,
                        avoid_inchikeys=seen_inchikeys,
                        subnet_config=config
                    )

                    region_time = time.time() - region_start
                    bt.logging.info(f"[Miner] Scheduled region reset generated {len(region_molecules)} molecules in {region_time:.1f}s")

                    if region_molecules:
                        data = pd.DataFrame(region_molecules)

                        # Extra filter: remove molecules with dominant components
                        data['_last_comp'] = data['name'].apply(get_last_comp)
                        pre_filter = len(data)
                        data = data[~data['_last_comp'].isin(dominant_comps)]
                        data = data.drop(columns=['_last_comp'])
                        bt.logging.info(f"[Miner] Filtered out {pre_filter - len(data)} molecules with dominant components")
                    else:
                        data = pd.DataFrame(columns=["name", "smiles", "InChIKey"])

                    # Supplement with pure random GA if needed
                    if len(data) < n_samples // 2:
                        n_supplement = n_samples - len(data)
                        bt.logging.info(f"[Miner] Supplementing with {n_supplement} pure random exploration molecules")

                        supplement_df = generate_valid_random_molecules_batch(
                            rxn_id,
                            n_samples=n_supplement,
                            db_path=DB_PATH,
                            subnet_config=config,
                            batch_size=400,
                            elite_names=None,  # No elite bias during reset
                            elite_frac=0.0,
                            mutation_prob=0.9,  # Very high mutation for diversity
                            avoid_inchikeys=seen_inchikeys.union(set(data["InChIKey"].tolist()) if not data.empty else set()),
                            component_weights=None,  # Pure random - no bias
                        )
                        data = pd.concat([data, supplement_df], ignore_index=True)
                        data = data.drop_duplicates(subset=["InChIKey"], keep="first")

                except Exception as e:
                    bt.logging.error(f"[Miner] Scheduled region reset failed: {e}, falling back to GA")
                    print_exc()
                    data = generate_valid_random_molecules_batch(
                        rxn_id,
                        n_samples=n_samples,
                        db_path=DB_PATH,
                        subnet_config=config,
                        batch_size=400,
                        elite_names=None,
                        elite_frac=0.0,
                        mutation_prob=0.9,
                        avoid_inchikeys=seen_inchikeys,
                        component_weights=None,
                    )

            # EXPLOIT MODE: After iteration threshold, use fingerprint exploitation
            elif use_exploit_mode and not top_pool.empty:
                bt.logging.info(f"[Miner] Iteration {iteration}: EXPLOIT MODE - fingerprint exploitation")

                # Get molecules with unexploited reactants (v4 approach)
                all_pool_mols = top_pool.to_dict("records")
                top_mols = get_top_n_unexploited(all_pool_mols, exploited_reactants, n=TOP_N_EXPLOIT, target_reactants=TARGET_REACTANTS)

                if not top_mols:
                    bt.logging.warning(f"[Miner] All reactants exhausted, falling back to GA")
                    data = generate_valid_random_molecules_batch(
                        rxn_id,
                        n_samples=n_samples,
                        db_path=DB_PATH,
                        subnet_config=config,
                        batch_size=400,
                        elite_names=elite_names,
                        elite_frac=elite_frac,
                        mutation_prob=mutation_prob,
                        avoid_inchikeys=seen_inchikeys,
                        component_weights=component_weights,
                    )
                else:
                    print(f"[Miner] Iteration {iteration}: EXPLOIT mode - {len(top_mols)} molecules selected (targeting {TARGET_REACTANTS} new reactants, {len(exploited_reactants)} already done)...", flush=True)

                    exploit_start = time.time()
                    try:
                        exploit_results, exploit_summary = run_exploit(
                            top_molecules=top_mols,
                            db_path=DB_PATH,
                            rxn_id=rxn_id,
                            top_n=TOP_N_EXPLOIT,
                            limit_per_reactant=LIMIT_PER_REACTANT,
                            avoid_names=seen_names,
                            exploited_reactants=exploited_reactants,
                            min_heavy_atoms=config.get('min_heavy_atoms', 20),
                            min_rotatable=config.get('min_rotatable_bonds', 1),
                            max_rotatable=config.get('max_rotatable_bonds', 10)
                        )

                        # ALWAYS update exploited reactants (fix tracking bug)
                        if exploit_summary and 'exploited_reactant_ids' in exploit_summary:
                            exploited_reactants.update(exploit_summary['exploited_reactant_ids'])

                        exploit_time = time.time() - exploit_start
                        bt.logging.info(f"[Miner] Exploit returned {len(exploit_results)} candidates in {exploit_time:.1f}s (exploited_reactants={len(exploited_reactants)})")

                        if exploit_results:
                            data = pd.DataFrame(exploit_results)
                            seen_names.update(data["name"].tolist())
                        else:
                            bt.logging.warning(f"[Miner] Exploit returned no results, falling back to GA")
                            data = generate_valid_random_molecules_batch(
                                rxn_id,
                                n_samples=n_samples,
                                db_path=DB_PATH,
                                subnet_config=config,
                                batch_size=400,
                                elite_names=elite_names,
                                elite_frac=elite_frac,
                                mutation_prob=mutation_prob,
                                avoid_inchikeys=seen_inchikeys,
                                component_weights=component_weights,
                            )
                    except Exception as e:
                        bt.logging.error(f"[Miner] Exploit failed: {e}")
                        print_exc()
                        data = generate_valid_random_molecules_batch(
                            rxn_id,
                            n_samples=n_samples,
                            db_path=DB_PATH,
                            subnet_config=config,
                            batch_size=400,
                            elite_names=elite_names,
                            elite_frac=elite_frac,
                            mutation_prob=mutation_prob,
                            avoid_inchikeys=seen_inchikeys,
                            component_weights=component_weights,
                        )

            # AMPLIFICATION MODE: Use PROGRESSIVE elite winner pattern amplification (famy.py approach)
            elif use_amplification_mode and not top_pool.empty:
                amplification_rounds += 1

                # Different sample counts based on reaction type
                # 3-component reactions (rxn 3, 5): 4500 samples, 100 elites
                # 2-component reactions (rxn 1, 2, 4): 1500 samples, 100 elites
                is_3_component = rxn_id in [3, 5]
                amp_n_samples = 4500 if is_3_component else 1500
                amp_elite_count = 100

                bt.logging.info(f"[Miner] Iteration {iteration}: PROGRESSIVE AMPLIFICATION (round {amplification_rounds}, n_samples={amp_n_samples}, elites={amp_elite_count})")
                print(f"[Miner] Iteration {iteration}: Running PROGRESSIVE AMPLIFICATION (round {amplification_rounds}, {'3-comp' if is_3_component else '2-comp'}, n={amp_n_samples}, elites={amp_elite_count})...", flush=True)

                # Convert top_pool to list of dicts for amplification
                top_mols = top_pool.head(amp_elite_count).to_dict("records")

                amp_start = time.time()
                try:
                    # v12: Use progressive amplification (famy.py approach with memory)
                    if progressive_amp_state is not None:
                        amp_results, amp_summary = run_progressive_amplification(
                            amp_state=progressive_amp_state,
                            top_pool=top_mols,
                            n_samples=amp_n_samples,
                            avoid_names=seen_names,
                            avoid_inchikeys=seen_inchikeys,
                            similarity_threshold=0.7,
                            min_heavy_atoms=config.get('min_heavy_atoms', 20),
                            min_rotatable=config.get('min_rotatable_bonds', 1),
                            max_rotatable=config.get('max_rotatable_bonds', 10),
                            verbose=True
                        )
                    else:
                        # Fallback to old amplification if state not initialized
                        bt.logging.warning("[Miner] ProgressiveAmplificationState not available, using fallback")
                        amp_results, amp_summary = run_amplification(
                            top_molecules=top_mols,
                            db_path=DB_PATH,
                            rxn_id=rxn_id,
                            n_samples=amp_n_samples,
                            avoid_names=seen_names,
                            avoid_inchikeys=seen_inchikeys,
                            min_heavy_atoms=config.get('min_heavy_atoms', 20),
                            min_rotatable=config.get('min_rotatable_bonds', 1),
                            max_rotatable=config.get('max_rotatable_bonds', 10),
                            verbose=True
                        )

                    amp_time = time.time() - amp_start
                    bt.logging.info(f"[Miner] Progressive Amplification returned {len(amp_results)} candidates in {amp_time:.1f}s")

                    if amp_results:
                        data = pd.DataFrame(amp_results)
                        seen_names.update(data["name"].tolist())

                        # Log strategy breakdown
                        for strategy, stats in amp_summary.get('strategies', {}).items():
                            bt.logging.info(f"[Miner]   {strategy}: {stats.get('generated', 0)} generated")
                    else:
                        bt.logging.warning(f"[Miner] Progressive Amplification returned no results, falling back to GA")
                        data = generate_valid_random_molecules_batch(
                            rxn_id,
                            n_samples=n_samples,
                            db_path=DB_PATH,
                            subnet_config=config,
                            batch_size=400,
                            elite_names=elite_names,
                            elite_frac=elite_frac,
                            mutation_prob=mutation_prob,
                            avoid_inchikeys=seen_inchikeys,
                            component_weights=component_weights,
                        )

                except Exception as e:
                    bt.logging.error(f"[Miner] Progressive Amplification failed: {e}")
                    print_exc()
                    data = generate_valid_random_molecules_batch(
                        rxn_id,
                        n_samples=n_samples,
                        db_path=DB_PATH,
                        subnet_config=config,
                        batch_size=400,
                        elite_names=elite_names,
                        elite_frac=elite_frac,
                        mutation_prob=mutation_prob,
                        avoid_inchikeys=seen_inchikeys,
                        component_weights=component_weights,
                    )

            # v12: REGION-BASED SAMPLING (anti-dominant strategy)
            # Instead of Tanimoto diversity, sample from underexplored feature regions
            elif iteration == 1:
                print(f"[Miner] Iteration 1: REGION-BASED SAMPLING for rxn:{rxn_id}...", flush=True)
                bt.logging.info(f"[Miner] Iteration {iteration}: Region-based anti-dominant exploration")

                try:
                    region_start = time.time()

                    # Build region sampler (maps components to feature regions)
                    # v16: Cache it for later scheduled resets
                    cached_region_sampler = RegionBasedSampler(DB_PATH, rxn_id)
                    region_sampler = cached_region_sampler

                    # Log region statistics
                    stats = region_sampler.get_region_stats()
                    for pool_name, pool_stats in stats.get('pools', {}).items():
                        bt.logging.info(
                            f"[Miner] Pool {pool_name}: {pool_stats['total']} mols, "
                            f"rare={pool_stats['n_rare']}, uncommon={pool_stats['n_uncommon']}, "
                            f"common={pool_stats['n_common']}, dominant={pool_stats['n_dominant']}"
                        )
                        # Log dominant regions (what we're avoiding)
                        for dom_region, dom_count in pool_stats.get('dominant_regions', []):
                            bt.logging.info(f"[Miner]   AVOIDING dominant: {dom_region} ({dom_count} mols)")

                    # Generate molecules using stratified region sampling
                    # Distribution: 40% rare, 35% uncommon, 25% common, 0% dominant
                    region_molecules = region_sampler.generate_region_stratified_molecules(
                        n_samples=n_samples_first_iteration,
                        rare_pct=0.40,
                        uncommon_pct=0.35,
                        common_pct=0.25,
                        avoid_inchikeys=seen_inchikeys,
                        subnet_config=config
                    )

                    region_time = time.time() - region_start
                    bt.logging.info(f"[Miner] Region sampling generated {len(region_molecules)} molecules in {region_time:.1f}s")

                    if region_molecules:
                        data = pd.DataFrame(region_molecules)
                    else:
                        data = pd.DataFrame(columns=["name", "smiles", "InChIKey"])

                    # If not enough, supplement with anti-dominant sampling
                    if len(data) < n_samples_first_iteration // 2:
                        bt.logging.warning(f"[Miner] Only {len(data)} region molecules, adding anti-dominant...")
                        n_supplement = n_samples_first_iteration - len(data)

                        supplement_molecules = region_sampler.generate_anti_dominant_molecules(
                            n_samples=n_supplement,
                            avoid_inchikeys=seen_inchikeys.union(set(data["InChIKey"].tolist())),
                            subnet_config=config
                        )

                        if supplement_molecules:
                            supplement_df = pd.DataFrame(supplement_molecules)
                            data = pd.concat([data, supplement_df], ignore_index=True)
                            data = data.drop_duplicates(subset=["InChIKey"], keep="first")

                except Exception as e:
                    bt.logging.error(f"[Miner] Region sampling failed: {e}, falling back to random")
                    print_exc()
                    data = generate_valid_random_molecules_batch(
                        rxn_id,
                        n_samples=n_samples_first_iteration,
                        db_path=DB_PATH,
                        subnet_config=config,
                        batch_size=400,
                        elite_names=None,
                        elite_frac=0.0,
                        mutation_prob=1.0,
                        avoid_inchikeys=seen_inchikeys,
                        component_weights=None,
                    )

            # v12: Iteration 2 - Second round of region exploration with different focus
            elif iteration == 2:
                print(f"[Miner] Iteration 2: REGION-BASED + GA HYBRID for rxn:{rxn_id}...", flush=True)
                bt.logging.info(f"[Miner] Iteration {iteration}: Region-based exploration (uncommon focus)")

                try:
                    # v16: Reuse cached region sampler
                    if cached_region_sampler is None:
                        cached_region_sampler = RegionBasedSampler(DB_PATH, rxn_id)
                    region_sampler = cached_region_sampler

                    # Second round: More focus on uncommon regions (adjacent to successful rare)
                    # Distribution: 25% rare, 45% uncommon, 30% common, 0% dominant
                    region_molecules = region_sampler.generate_region_stratified_molecules(
                        n_samples=n_samples,
                        rare_pct=0.25,
                        uncommon_pct=0.45,
                        common_pct=0.30,
                        avoid_inchikeys=seen_inchikeys,
                        subnet_config=config
                    )

                    bt.logging.info(f"[Miner] Region sampling (iter 2) generated {len(region_molecules)} molecules")

                    if region_molecules:
                        region_df = pd.DataFrame(region_molecules)
                    else:
                        region_df = pd.DataFrame(columns=["name", "smiles", "InChIKey"])

                    # Supplement with GA if needed
                    if len(region_df) < n_samples // 2:
                        n_supplement = n_samples - len(region_df)
                        bt.logging.info(f"[Miner] Supplementing with {n_supplement} GA molecules")
                        ga_df = generate_valid_random_molecules_batch(
                            rxn_id,
                            n_samples=n_supplement,
                            db_path=DB_PATH,
                            subnet_config=config,
                            batch_size=400,
                            elite_names=elite_names,
                            elite_frac=elite_frac,
                            mutation_prob=mutation_prob,
                            avoid_inchikeys=seen_inchikeys.union(set(region_df["InChIKey"].tolist())),
                            component_weights=component_weights,
                        )
                        data = pd.concat([region_df, ga_df], ignore_index=True)
                        data = data.drop_duplicates(subset=["InChIKey"], keep="first")
                    else:
                        data = region_df

                except Exception as e:
                    bt.logging.error(f"[Miner] Region sampling (iter 2) failed: {e}, using GA")
                    print_exc()
                    data = generate_valid_random_molecules_batch(
                        rxn_id,
                        n_samples=n_samples,
                        db_path=DB_PATH,
                        subnet_config=config,
                        batch_size=400,
                        elite_names=elite_names,
                        elite_frac=elite_frac,
                        mutation_prob=mutation_prob,
                        avoid_inchikeys=seen_inchikeys,
                        component_weights=component_weights,
                    )

            elif use_synthon_search and iteration > 2 and not top_pool.empty:
                bt.logging.info(f"[Miner] Iteration {iteration}: Smart synthon similarity search")

                # Get current max score for adaptive strategy
                current_max_score = top_pool['score'].max() if not top_pool.empty else None
                current_avg_score = top_pool['score'].mean() if not top_pool.empty else None
                max_score_history.append(current_max_score)
                if len(max_score_history) > 5:
                    max_score_history.pop(0)

                # SMART: Adaptive strategy based on improvement rate AND absolute score
                has_high_score = current_max_score is not None and current_max_score > 0.01
                has_very_high_score = current_max_score is not None and current_max_score > 0.015

                # Time-based strategy
                time_elapsed = time.time() - start
                is_late_stage = time_elapsed > 1200
                is_very_late_stage = time_elapsed > 1500

                if score_improvement_rate > 0.05:
                    # High improvement: tight exploration
                    sim_threshold = 0.75
                    n_per_base = 15
                    n_seeds = 20
                    synthon_ratio = 0.75
                    bt.logging.info(f"[Miner] High improvement ({score_improvement_rate:.4f}), tight similarity (0.75)")

                elif score_improvement_rate > 0.02:
                    # Good improvement: medium-tight exploration
                    sim_threshold = 0.70
                    n_per_base = 18
                    n_seeds = 25
                    synthon_ratio = 0.75
                    bt.logging.info(f"[Miner] Good improvement ({score_improvement_rate:.4f}), medium-tight similarity (0.70)")

                elif score_improvement_rate > 0.005:
                    # Moderate improvement: balanced exploration
                    sim_threshold = 0.65
                    n_per_base = 20
                    n_seeds = 30
                    synthon_ratio = 0.70
                    bt.logging.info(f"[Miner] Moderate improvement ({score_improvement_rate:.4f}), medium similarity (0.65)")

                else:
                    # Low/no improvement - PROVEN MULTI-RANGE STRATEGY (like richard1220v3)
                    bt.logging.info(f"[Miner] Low improvement ({score_improvement_rate:.4f}), using PROVEN MULTI-RANGE strategy")

                    # SMART: Adjust strategy based on absolute score and time
                    if has_very_high_score or is_very_late_stage:
                        # When we have very high scores, add focused exploitation on TOP 1
                        # Part 1: Ultra-tight on TOP 1 molecule (30% of synthon budget) - OPTIMIZED for high scores
                        n_synthon_top1 = int(n_samples * 0.25)  # Increased from 0.21 to 0.30 for maximum exploitation
                        synthon_top1_df = generate_molecules_from_synthon_library(
                            synthon_lib,
                            top_pool.head(1),  # TOP 1 ONLY
                            n_synthon_top1,
                            min_similarity=0.91,  # Increased from 0.85 to 0.90 for ultra-tight similarity
                            n_per_base=50  # Increased from 50 to 80 for maximum variations
                        )
                        bt.logging.info(f"[Miner] Generated {len(synthon_top1_df)} TOP-1 synthon candidates (sim=0.90, n_per_base=80)")

                        # Part 2: Ultra-tight on top 5 molecules (10% of synthon budget)
                        n_synthon_tight = int(n_samples * 0.07)  # 10% of 70%
                        synthon_tight_df = generate_molecules_from_synthon_library(
                            synthon_lib,
                            top_pool.head(5),
                            n_synthon_tight,
                            min_similarity=0.80,  # Tight
                            n_per_base=30
                        )
                        bt.logging.info(f"[Miner] Generated {len(synthon_tight_df)} TIGHT synthon candidates (sim=0.80)")

                        # Part 3: Medium on molecules 10-40 (30% of synthon budget)
                        n_synthon_medium = int(n_samples * 0.21)  # 30% of 70%
                        seed_medium = top_pool.iloc[10:40] if len(top_pool) > 40 else top_pool.iloc[5:]
                        synthon_medium_df = generate_molecules_from_synthon_library(
                            synthon_lib,
                            seed_medium,
                            n_synthon_medium,
                            min_similarity=0.55,  # Medium - like richard1220v3
                            n_per_base=15
                        )
                        bt.logging.info(f"[Miner] Generated {len(synthon_medium_df)} MEDIUM synthon candidates (sim=0.55)")

                        # Part 4: Broad on top 50 (30% of synthon budget)
                        n_synthon_broad = int(n_samples * 0.21)  # 30% of 70%
                        synthon_broad_df = generate_molecules_from_synthon_library(
                            synthon_lib,
                            top_pool.head(50),
                            n_synthon_broad,
                            min_similarity=0.40,  # Broad - like richard1220v3
                            n_per_base=20
                        )
                        bt.logging.info(f"[Miner] Generated {len(synthon_broad_df)} BROAD synthon candidates (sim=0.40)")

                        # Combine all synthon approaches
                        synthon_df = pd.concat([synthon_top1_df, synthon_tight_df, synthon_medium_df, synthon_broad_df], ignore_index=True)
                    else:
                        # Standard: PROVEN multi-range strategy from richard1220v3
                        # Part 1: Ultra-tight on top 5 molecules (40% of synthon budget)
                        n_synthon_tight = int(n_samples * 0.28)  # 40% of 70%
                        synthon_tight_df = generate_molecules_from_synthon_library(
                            synthon_lib,
                            top_pool.head(5),
                            n_synthon_tight,
                            min_similarity=0.80,  # Very tight!
                            n_per_base=30
                        )
                        bt.logging.info(f"[Miner] Generated {len(synthon_tight_df)} TIGHT synthon candidates (sim=0.80)")

                        # Part 2: Medium on molecules 10-40 (30% of synthon budget)
                        n_synthon_medium = int(n_samples * 0.21)  # 30% of 70%
                        seed_medium = top_pool.iloc[10:40] if len(top_pool) > 40 else top_pool.iloc[5:]
                        synthon_medium_df = generate_molecules_from_synthon_library(
                            synthon_lib,
                            seed_medium,
                            n_synthon_medium,
                            min_similarity=0.55,  # Medium - like richard1220v3
                            n_per_base=15
                        )
                        bt.logging.info(f"[Miner] Generated {len(synthon_medium_df)} MEDIUM synthon candidates (sim=0.55)")

                        # Part 3: Broad on top 50 (30% of synthon budget)
                        n_synthon_broad = int(n_samples * 0.21)  # 30% of 70%
                        synthon_broad_df = generate_molecules_from_synthon_library(
                            synthon_lib,
                            top_pool.head(50),
                            n_synthon_broad,
                            min_similarity=0.40,  # Broad - like richard1220v3
                            n_per_base=20
                        )
                        bt.logging.info(f"[Miner] Generated {len(synthon_broad_df)} BROAD synthon candidates (sim=0.40)")

                        # Combine all synthon approaches
                        synthon_df = pd.concat([synthon_tight_df, synthon_medium_df, synthon_broad_df], ignore_index=True)

                    synthon_df = synthon_df.drop_duplicates(subset=["name"], keep="first")

                    if not synthon_df.empty:
                        synthon_df = validate_molecules(synthon_df, config)
                        bt.logging.info(f"[Miner] {len(synthon_df)} multi-range synthon candidates passed validation")

                    # Generate remaining from GA with component weighting
                    n_traditional = n_samples - len(synthon_df)
                    if n_traditional > 0:
                        traditional_df = generate_valid_random_molecules_batch(
                            rxn_id,
                            n_samples=n_traditional,
                            db_path=DB_PATH,
                            subnet_config=config,
                            batch_size=400,
                            elite_names=elite_names,
                            elite_frac=elite_frac,
                            mutation_prob=mutation_prob,
                            avoid_inchikeys=seen_inchikeys,
                            component_weights=component_weights,
                        )
                    else:
                        traditional_df = pd.DataFrame(columns=["name", "smiles", "InChIKey"])

                    data = pd.concat([synthon_df, traditional_df], ignore_index=True)
                    data = data.drop_duplicates(subset=["name"], keep="first")
                    bt.logging.info(f"[Miner] Combined: {len(data)} total ({len(synthon_df)} multi-range synthon + {len(traditional_df)} GA)")

                    # Skip the standard synthon generation below
                    synthon_df = None

                # Standard single-range synthon generation (for high/medium improvement)
                if score_improvement_rate > 0.005:  # Only if not using multi-range
                    n_synthon = int(n_samples * synthon_ratio)
                    synthon_gen_start = time.time()
                    synthon_df = generate_molecules_from_synthon_library(
                        synthon_lib,
                        top_pool.head(n_seeds),
                        n_synthon,
                        min_similarity=sim_threshold,
                        n_per_base=n_per_base
                    )
                    bt.logging.info(f"[Miner] Generated {len(synthon_df)} synthon candidates in {time.time() - synthon_gen_start:.2f}s")

                    # Generate remaining from traditional method
                    n_traditional = n_samples - len(synthon_df)
                    if n_traditional > 0:
                        traditional_df = generate_valid_random_molecules_batch(
                            rxn_id,
                            n_samples=n_traditional,
                            db_path=DB_PATH,
                            subnet_config=config,
                            batch_size=300,
                            elite_names=elite_names,
                            elite_frac=elite_frac,
                            mutation_prob=mutation_prob,
                            avoid_inchikeys=seen_inchikeys,
                            component_weights=component_weights,
                        )
                    else:
                        traditional_df = pd.DataFrame(columns=["name", "smiles", "InChIKey"])

                    # Validate and combine
                    if not synthon_df.empty:
                        synthon_df = validate_molecules(synthon_df, config)
                        bt.logging.info(f"[Miner] {len(synthon_df)} synthon candidates passed validation")

                    data = pd.concat([synthon_df, traditional_df], ignore_index=True)
                    data = data.drop_duplicates(subset=["name"], keep="first")
                    bt.logging.info(f"[Miner] Combined: {len(data)} total ({len(synthon_df)} synthon + {len(traditional_df)} GA)")

            elif no_improvement_counter < 3:
                bt.logging.info(f"[Miner] Iteration {iteration}: Standard genetic algorithm")
                data = generate_valid_random_molecules_batch(
                    rxn_id,
                    n_samples=n_samples,
                    db_path=DB_PATH,
                    subnet_config=config,
                    batch_size=400,
                    elite_names=elite_names,
                    elite_frac=elite_frac,
                    mutation_prob=mutation_prob,
                    avoid_inchikeys=seen_inchikeys,
                    component_weights=component_weights,
                )

            elif no_improvement_counter < 6:
                bt.logging.info(f"[Miner] Iteration {iteration}: Exploring similar space (no_improvement={no_improvement_counter})")
                data = _cpu_random_candidates_with_similarity(
                    iteration,
                    30,
                    config,
                    top_pool.head(50)[["name", "smiles", "InChIKey"]],
                    seen_inchikeys,
                    0.65
                )
                seed_df = pd.DataFrame(columns=["name", "smiles", "InChIKey", "tanimoto_similarity"])

            else:
                bt.logging.info(f"[Miner] Iteration {iteration}: Broad exploration reset (no_improvement={no_improvement_counter})")
                data = _cpu_random_candidates_with_similarity(
                    iteration,
                    40,
                    config,
                    top_pool.head(100)[["name", "smiles", "InChIKey"]],
                    seen_inchikeys,
                    0.0
                )
                seed_df = pd.DataFrame(columns=["name", "smiles", "InChIKey", "tanimoto_similarity"])
                no_improvement_counter = 0

            gen_time = time.time() - iter_start_time
            print(f"[Miner] Iteration {iteration}: {len(data)} molecules generated in {gen_time:.1f}s", flush=True)
            bt.logging.info(f"[Miner] Iteration {iteration}: {len(data)} Samples Generated in ~{gen_time:.2f}s (pre-score)")

            if data.empty:
                print(f"[Miner] Iteration {iteration}: NO MOLECULES GENERATED - check rxn:{rxn_id} data", flush=True)
                bt.logging.warning(f"[Miner] Iteration {iteration}: No valid molecules produced; continuing")
                continue

            if not seed_df.empty:
                data = pd.concat([data, seed_df])
                data = data.drop_duplicates(subset=["InChIKey"], keep="first")
                seed_df = pd.DataFrame(columns=["name", "smiles", "InChIKey", "tanimoto_similarity"])

            try:
                filterd_data = data[~data["InChIKey"].isin(seen_inchikeys)]
                if len(filterd_data) < len(data):
                    bt.logging.warning(
                        f"[Miner] Iteration {iteration}: {len(data) - len(filterd_data)} molecules were previously seen"
                    )

                dup_ratio = (len(data) - len(filterd_data)) / max(1, len(data))

                if dup_ratio > 0.7:
                    mutation_prob = min(0.9, mutation_prob * 1.5)
                    elite_frac = max(0.15, elite_frac * 0.7)
                    bt.logging.warning(f"[Miner] SEVERE duplication ({dup_ratio:.2%})! mut={mutation_prob:.2f}, elite={elite_frac:.2f}")
                elif dup_ratio > 0.5:
                    mutation_prob = min(0.7, mutation_prob * 1.3)
                    elite_frac = max(0.2, elite_frac * 0.8)
                    bt.logging.warning(f"[Miner] High duplication ({dup_ratio:.2%}), mut={mutation_prob:.2f}, elite={elite_frac:.2f}")
                elif dup_ratio < 0.15 and not top_pool.empty and iteration > 10:
                    mutation_prob = max(0.05, mutation_prob * 0.95)
                    elite_frac = min(0.85, elite_frac * 1.05)

                data = filterd_data

            except Exception as e:
                bt.logging.warning(f"[Miner] Pre-score deduplication failed: {e}")

            if data.empty:
                bt.logging.error(f"[Miner] Iteration {iteration}: ALL molecules were duplicates! Skipping scoring and continuing...")
                # Force more diversity for next iteration
                mutation_prob = min(0.95, mutation_prob * 2.0)
                elite_frac = max(0.1, elite_frac * 0.5)
                bt.logging.warning(f"[Miner] Emergency diversity boost: mut={mutation_prob:.2f}, elite={elite_frac:.2f}")
                continue  # Skip to next iteration

            data = data.reset_index(drop=True)

            # Enhanced CPU similarity search - multiple parallel searches
            cpu_futures = []
            if not top_pool.empty and iteration > 3:
                # Multiple parallel CPU searches with different strategies
                if score_improvement_rate < 0.01:
                    # Strategy 1: Tight on top 5
                    cpu_futures.append((
                        cpu_executor.submit(
                            _cpu_random_candidates_with_similarity,
                            iteration,
                            40,
                            config,
                            top_pool.head(5)[["name", "smiles", "InChIKey"]],
                            seen_inchikeys,
                            0.80
                        ),
                        "tight-top5"
                    ))

                    # Strategy 2: Medium on top 20
                    cpu_futures.append((
                        cpu_executor.submit(
                            _cpu_random_candidates_with_similarity,
                            iteration,
                            30,
                            config,
                            top_pool.head(20)[["name", "smiles", "InChIKey"]],
                            seen_inchikeys,
                            0.65
                        ),
                        "medium-top20"
                    ))

            gpu_start_time = time.time()

            if len(data) == 0:
                bt.logging.error(f"[Miner] Iteration {iteration}: No molecules to score! Continuing...")
                continue

            print(f"[Miner] Iteration {iteration}: Scoring {len(data)} molecules with PSICHIC...", flush=True)
            data["Target"] = target_score_from_data(data["smiles"])
            data["Anti"] = antitarget_scores()
            data["score"] = data["Target"] - (config["antitarget_weight"] * data["Anti"])

            # v16: DIVERSITY TAX - penalize overused components to break local optima
            if not top_pool.empty and iteration > 5:
                try:
                    # Count comp2 (or comp3 for 3-component rxns) frequency in current top_pool
                    def get_last_comp(name):
                        parts = name.split(':')
                        return parts[-1] if len(parts) >= 3 else ''

                    pool_comp_counts = top_pool['name'].apply(get_last_comp).value_counts().to_dict()
                    data['_last_comp'] = data['name'].apply(get_last_comp)
                    data['_comp_count'] = data['_last_comp'].map(pool_comp_counts).fillna(0)

                    # Apply diversity tax: 0.005 penalty per occurrence in top_pool
                    # This aggressively favors molecules with underrepresented components
                    diversity_tax = 0.005 * data['_comp_count']
                    data['score'] = data['score'] - diversity_tax

                    taxed_count = (diversity_tax > 0).sum()
                    if taxed_count > 0:
                        bt.logging.info(f"[Miner] Diversity tax applied to {taxed_count} molecules (max tax: {diversity_tax.max():.4f})")

                    # Cleanup temp columns
                    data = data.drop(columns=['_last_comp', '_comp_count'])
                except Exception as e:
                    bt.logging.warning(f"[Miner] Diversity tax failed: {e}")

            if data["score"].isna().all():
                bt.logging.error(f"[Miner] Iteration {iteration}: Scoring failed (all NaN)! Continuing...")
                continue

            gpu_time = time.time() - gpu_start_time
            print(f"[Miner] Iteration {iteration}: PSICHIC scoring done in {gpu_time:.1f}s", flush=True)
            bt.logging.info(f"[Miner] Iteration {iteration}: GPU scoring time ~{gpu_time:.2f}s")

            # Collect all CPU results
            if cpu_futures:
                for cpu_future, strategy_name in cpu_futures:
                    try:
                        cpu_df = cpu_future.result(timeout=0)
                        if not cpu_df.empty:
                            if seed_df.empty:
                                seed_df = cpu_df.copy()
                            else:
                                seed_df = pd.concat([seed_df, cpu_df], ignore_index=True)
                            bt.logging.info(f"[Miner] CPU similarity ({strategy_name}) found {len(cpu_df)} candidates")
                    except TimeoutError:
                        pass
                    except Exception as e:
                        bt.logging.warning(f"[Miner] CPU similarity ({strategy_name}) failed: {e}")

                if not seed_df.empty:
                    seed_df = seed_df.drop_duplicates(subset=["InChIKey"], keep="first")

            seen_inchikeys.update([k for k in data["InChIKey"].tolist() if k])
            total_data = data[["name", "smiles", "InChIKey", "score", "Target", "Anti"]]
            prev_avg_score = top_pool['score'].mean() if not top_pool.empty else None

            # Safe concatenation
            if not total_data.empty:
                top_pool = pd.concat([top_pool, total_data], ignore_index=True)
                top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
                top_pool = top_pool.sort_values(by="score", ascending=False)

                # v12: Update progressive amplification memory with scored results
                if use_amplification_mode and progressive_amp_state is not None:
                    try:
                        scored_mols = total_data.to_dict("records")
                        progressive_amp_state.update_learning_memory_with_results(scored_mols)
                    except Exception as e:
                        bt.logging.warning(f"[Miner] Failed to update progressive amp memory: {e}")
            else:
                bt.logging.warning(f"[Miner] Iteration {iteration}: No valid scored data to add to pool")

            # Track best molecules for later intensive exploration
            if not top_pool.empty and iteration % 5 == 0:
                best_molecules_history.append({
                    'iteration': iteration,
                    'molecules': top_pool.head(10)[["name", "smiles", "InChIKey", "score"]].copy()
                })
                if len(best_molecules_history) > 6:
                    best_molecules_history.pop(0)

            remaining_time = 2700 - (time.time() - start)
            if remaining_time <= 60:
                # FINAL PHASE: Ensure entropy requirements met
                bt.logging.info("[Miner] === FINAL PHASE ===")
                entropy = compute_maccs_entropy(top_pool.iloc[:config["num_molecules"]]['smiles'].to_list())
                if entropy > config['entropy_min_threshold']:
                    top_pool = top_pool.head(config["num_molecules"])
                    bt.logging.info(f"[Miner] Iteration {iteration}: Sufficient Entropy = {entropy:.4f}")
                else:
                    try:
                        top_95 = top_pool.iloc[:95]
                        remaining_pool = top_pool.iloc[95:]
                        additional_5 = select_diverse_subset(remaining_pool, top_95["smiles"].tolist(),
                                                            subset_size=5, entropy_threshold=config['entropy_min_threshold'])
                        if not additional_5.empty:
                            top_pool = pd.concat([top_95, additional_5]).reset_index(drop=True)
                            entropy = compute_maccs_entropy(top_pool['smiles'].to_list())
                            bt.logging.info(f"[Miner] Iteration {iteration}: Adjusted Entropy = {entropy:.4f}")
                        else:
                            top_pool = top_pool.head(config["num_molecules"])
                    except Exception as e:
                        bt.logging.warning(f"[Miner] Entropy handling failed: {e}")
            else:
                top_pool = top_pool.head(config["num_molecules"])

            current_avg_score = top_pool['score'].mean() if not top_pool.empty else None

            if current_avg_score is not None:
                if prev_avg_score is not None:
                    score_improvement_rate = (current_avg_score - prev_avg_score) / max(abs(prev_avg_score), 1e-6)
                prev_avg_score = current_avg_score

            if score_improvement_rate == 0.0:
                no_improvement_counter += 1
            else:
                no_improvement_counter = 0

            iter_total_time = time.time() - iter_start_time
            total_time = time.time() - start

            # Pool stats (unified pool)
            pool_avg = top_pool['score'].mean()
            pool_max = top_pool['score'].max()
            pool_min = top_pool['score'].min() if len(top_pool) >= 100 else top_pool['score'].min()
            try:
                pool_entropy = compute_maccs_entropy(top_pool.head(100)['smiles'].tolist())
            except:
                pool_entropy = 0.0

            # Write best molecules to result.json
            top_entries = {"molecules": top_pool["name"].tolist()}

            mode_str = "EXPLOIT" if use_exploit_mode else ("AMPLIFY" if use_amplification_mode else "SYNTHON")
            bt.logging.warning(
                f"Iter {iteration} | {iter_total_time:.1f}s | Total: {total_time:.0f}s | "
                f"Mode: {mode_str} | "
                f"Pool: avg={pool_avg:.4f} max={pool_max:.4f} ent={pool_entropy:.3f}"
            )

            with open(os.path.join(OUTPUT_DIR, "result.json"), "w") as f:
                json.dump(top_entries, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    args = parse_args()
    config = get_config(input_file=args.input, rxn_override=args.rxn)
    start_time_1 = time.time()
    initialize_models(config)
    bt.logging.info(f"{time.time() - start_time_1} seconds for model initialization")
    main(config)
