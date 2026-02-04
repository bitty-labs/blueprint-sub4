from rdkit import Chem, DataStructs
import bittensor as bt
from rdkit.Chem import Descriptors, MACCSkeys, AllChem
from rdkit.Chem import rdFingerprintGenerator
from dotenv import load_dotenv
import pandas as pd
import warnings
import sqlite3
import random
import os
from functools import lru_cache
from typing import List, Tuple, Dict
load_dotenv(override=True)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
from nova_ph2.combinatorial_db.reactions import get_smiles_from_reaction, get_reaction_info
from nova_ph2.utils.molecules import get_heavy_atom_count
from collections import defaultdict
from itertools import chain
import numpy as np
import math

# Try to import synthon search
try:
    from rdkit.Chem import rdSynthonSpaceSearch
    SYNTHON_SEARCH_AVAILABLE = True
except ImportError:
    SYNTHON_SEARCH_AVAILABLE = False
    bt.logging.warning("RDKit synthon search not available, using fingerprint similarity")

# Create global Morgan fingerprint generator to avoid deprecation warnings
MORGAN_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# =============================================================================
# DIVERSITY SAMPLING - Fast matrix-based diverse component selection
# =============================================================================

def _compute_fingerprint_matrix(molecules: List[Tuple[int, str, int]]) -> Tuple[np.ndarray, List[int]]:
    """
    Compute fingerprint bit matrix for a list of molecules.
    Returns (fp_matrix, valid_mol_ids) - matrix shape (n_valid, 2048)
    """
    fps = []
    valid_ids = []
    for mol_id, smiles, _ in molecules:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = MORGAN_FP_GENERATOR.GetFingerprint(mol)
                # Convert to numpy array
                arr = np.zeros(2048, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps.append(arr)
                valid_ids.append(mol_id)
        except:
            continue

    if not fps:
        return np.array([]), []

    return np.stack(fps), valid_ids


def _compute_tanimoto_matrix(fp_matrix: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Tanimoto similarity matrix using matrix operations.
    Input: (n, 2048) fingerprint matrix
    Output: (n, n) Tanimoto similarity matrix
    """
    if len(fp_matrix) == 0:
        return np.array([])

    # Dot products: intersection counts
    dots = fp_matrix @ fp_matrix.T

    # Bit counts per molecule
    counts = fp_matrix.sum(axis=1)

    # Tanimoto: intersection / union
    # union = counts[i] + counts[j] - intersection
    union = counts[:, None] + counts[None, :] - dots

    # Avoid division by zero
    tanimoto = np.divide(dots, union, out=np.zeros_like(dots), where=union != 0)

    return tanimoto


def _select_diverse_subset_greedy(tanimoto_matrix: np.ndarray, mol_ids: List[int],
                                   n_select: int, max_similarity: float = 0.5) -> List[int]:
    """
    Greedy max-min diversity selection.
    Picks molecules that are maximally dissimilar to already selected ones.
    """
    if len(mol_ids) == 0:
        return []

    n = len(mol_ids)
    n_select = min(n_select, n)

    if n_select <= 1:
        return [mol_ids[0]] if mol_ids else []

    # Start with random molecule
    selected_indices = [random.randint(0, n - 1)]
    selected_set = set(selected_indices)

    for _ in range(n_select - 1):
        # For each unselected molecule, find its max similarity to any selected
        max_sims = np.full(n, -1.0)
        for i in range(n):
            if i not in selected_set:
                max_sims[i] = tanimoto_matrix[i, selected_indices].max()

        # Pick the one with LOWEST max similarity (most diverse)
        # Only consider those below threshold if possible
        candidates = np.where((max_sims >= 0) & (max_sims <= max_similarity))[0]

        if len(candidates) > 0:
            # Pick randomly among good candidates for some variance
            next_idx = random.choice(candidates)
        else:
            # Fallback: pick the one with lowest similarity even if above threshold
            valid_mask = max_sims >= 0
            if not valid_mask.any():
                break
            next_idx = np.argmin(np.where(valid_mask, max_sims, np.inf))

        selected_indices.append(next_idx)
        selected_set.add(next_idx)

    return [mol_ids[i] for i in selected_indices]


def compute_diverse_components(db_path: str, rxn_id: int,
                                n_per_pool: int = 50,
                                max_similarity: float = 0.5) -> Dict[str, List[int]]:
    """
    Compute diverse component subsets for each role in a reaction.
    Uses fast matrix operations for Tanimoto computation.

    Returns: {'A': [mol_ids], 'B': [mol_ids], 'C': [mol_ids]}
    """
    from tqdm import tqdm

    reaction_info = get_reaction_info(rxn_id, db_path)
    if not reaction_info:
        bt.logging.error(f"Could not get reaction info for rxn:{rxn_id}")
        return {}

    smarts, roleA, roleB, roleC = reaction_info
    is_three_component = roleC is not None and roleC != 0

    diverse_components = {}

    # Process each role
    roles = [('A', roleA), ('B', roleB)]
    if is_three_component:
        roles.append(('C', roleC))

    for role_name, role_mask in roles:
        bt.logging.info(f"[Diversity] Computing diverse {role_name} components...")

        # Load molecules for this role
        molecules = get_molecules_by_role(role_mask, db_path)

        if not molecules:
            diverse_components[role_name] = []
            continue

        # Compute fingerprint matrix
        fp_matrix, valid_ids = _compute_fingerprint_matrix(molecules)

        if len(fp_matrix) == 0:
            diverse_components[role_name] = []
            continue

        bt.logging.info(f"[Diversity] {role_name}: {len(valid_ids)} molecules, computing Tanimoto matrix...")

        # Compute Tanimoto matrix
        tanimoto = _compute_tanimoto_matrix(fp_matrix)

        # Select diverse subset
        diverse_ids = _select_diverse_subset_greedy(
            tanimoto, valid_ids, n_per_pool, max_similarity
        )

        diverse_components[role_name] = diverse_ids
        bt.logging.info(f"[Diversity] {role_name}: Selected {len(diverse_ids)} diverse components")

    return diverse_components


def generate_diverse_molecules(rxn_id: int, diverse_components: Dict[str, List[int]],
                                n_samples: int, subnet_config: dict,
                                avoid_inchikeys: set = None) -> pd.DataFrame:
    """
    Generate molecules by combining diverse components.
    Creates all combinations of diverse A × B (× C) and samples from them.
    """
    from tqdm import tqdm

    if avoid_inchikeys is None:
        avoid_inchikeys = set()

    A_ids = diverse_components.get('A', [])
    B_ids = diverse_components.get('B', [])
    C_ids = diverse_components.get('C', [])

    is_three_component = len(C_ids) > 0

    if not A_ids or not B_ids:
        bt.logging.warning("[Diversity] Empty component pools, cannot generate")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

    # Generate all combinations
    all_names = []
    if is_three_component:
        for a in A_ids:
            for b in B_ids:
                for c in C_ids:
                    all_names.append(f"rxn:{rxn_id}:{a}:{b}:{c}")
        bt.logging.info(f"[Diversity] {len(A_ids)}×{len(B_ids)}×{len(C_ids)} = {len(all_names)} diverse combinations")
    else:
        for a in A_ids:
            for b in B_ids:
                all_names.append(f"rxn:{rxn_id}:{a}:{b}")
        bt.logging.info(f"[Diversity] {len(A_ids)}×{len(B_ids)} = {len(all_names)} diverse combinations")

    # Shuffle and sample if we have more than needed
    random.shuffle(all_names)
    if len(all_names) > n_samples * 2:
        all_names = all_names[:n_samples * 2]

    # Validate molecules
    valid_molecules = []
    seen_keys = set()

    for name in tqdm(all_names, desc="[Diversity] Validating", leave=False):
        smiles = _get_smiles_from_reaction_cached(name)
        if not smiles:
            continue

        # Check properties
        heavy = get_heavy_atom_count(smiles)
        if heavy < subnet_config.get('min_heavy_atoms', 20):
            continue

        rotatable = num_rotatable_bonds(smiles)
        if rotatable < subnet_config.get('min_rotatable_bonds', 1):
            continue
        if rotatable > subnet_config.get('max_rotatable_bonds', 10):
            continue

        inchi = generate_inchikey(smiles)
        if not inchi:
            continue
        if inchi in seen_keys or inchi in avoid_inchikeys:
            continue

        seen_keys.add(inchi)
        valid_molecules.append({
            'name': name,
            'smiles': smiles,
            'InChIKey': inchi
        })

        if len(valid_molecules) >= n_samples:
            break

    bt.logging.info(f"[Diversity] Generated {len(valid_molecules)} valid diverse molecules")

    return pd.DataFrame(valid_molecules) if valid_molecules else pd.DataFrame(columns=["name", "smiles", "InChIKey"])


@lru_cache(maxsize=1000_000)
def _get_smiles_from_reaction_cached(name: str):
    """Cache SMILES retrieval to avoid repeated database queries."""
    try:
        return get_smiles_from_reaction(name)
    except Exception:
        return None

@lru_cache(maxsize=1000_000)
def _mol_from_smiles_cached(smiles: str):
    """Cache molecule parsing to avoid repeated SMILES parsing."""
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


@lru_cache(maxsize=1000_000)
def _maccs_fp_from_smiles_cached(smiles: str):
    """Cache MACCS fingerprints for SMILES strings for fast Tanimoto similarity."""
    if not smiles:
        return None
    try:
        mol = _mol_from_smiles_cached(smiles)
        if mol is None:
            return None
        return MACCSkeys.GenMACCSKeys(mol)
    except Exception:
        return None

@lru_cache(maxsize=1000_000)
def _inchikey_from_name_cached(name: str) -> str:
    """Cache InChIKey generation from molecule name to avoid repeated computation."""
    try:
        s = _get_smiles_from_reaction_cached(name)
        if not s:
            return ""
        return generate_inchikey(s)
    except Exception:
        return ""

def compute_maccs_entropy(smiles_list: list[str]) -> float:
    n_bits = 167  # RDKit uses 167 bits (index 0 is always 0)
    bit_counts = np.zeros(n_bits)
    valid_mols = 0

    for smi in smiles_list:
        fp = _maccs_fp_from_smiles_cached(smi)
        if fp is not None:
            arr = np.array(fp)
            bit_counts += arr
            valid_mols += 1

    if valid_mols == 0:
        raise ValueError("No valid molecules found.")

    probs = bit_counts / valid_mols
    entropy_per_bit = np.array([
        -p * math.log2(p) - (1 - p) * math.log2(1 - p) if 0 < p < 1 else 0
        for p in probs
    ])

    avg_entropy = np.mean(entropy_per_bit)

    return avg_entropy

def num_rotatable_bonds(smiles: str) -> int:
    """Get number of rotatable bonds from SMILES string."""
    if not smiles:
        return 0
    try:
        mol = _mol_from_smiles_cached(smiles)
        if mol is None:
            return 0
        return Descriptors.NumRotatableBonds(mol)
    except Exception:
        return 0

@lru_cache(maxsize=1000_000)
def generate_inchikey(smiles: str) -> str:
    """Generate InChIKey from SMILES string."""
    if not smiles:
        return ""
    try:
        mol = _mol_from_smiles_cached(smiles)
        if mol is None:
            return ""
        return Chem.MolToInchiKey(mol)
    except Exception as e:
        bt.logging.error(f"Error generating InChIKey for SMILES {smiles}: {e}")
        return ""


def compute_tanimoto_similarity_to_pool(
    candidate_smiles: pd.Series,
    pool_smiles: pd.Series,
) -> pd.Series:
    """
    Compute, for each candidate SMILES, the maximum MACCS Tanimoto similarity
    to any molecule in the reference pool.

    Returns a Series indexed like candidate_smiles.
    """
    if candidate_smiles.empty or pool_smiles.empty:
        # Return zeros with matching index
        return pd.Series(0.0, index=candidate_smiles.index, dtype=float)

    # Pre-compute fingerprints for pool molecules
    pool_fps = []
    for smi in pool_smiles.dropna().unique():
        fp = _maccs_fp_from_smiles_cached(smi)
        if fp is not None:
            pool_fps.append(fp)

    if not pool_fps:
        return pd.Series(0.0, index=candidate_smiles.index, dtype=float)

    similarities = {}
    for idx, smi in candidate_smiles.items():
        fp_cand = _maccs_fp_from_smiles_cached(smi)
        if fp_cand is None:
            similarities[idx] = 0.0
            continue
        max_sim = 0.0
        for fp_ref in pool_fps:
            try:
                sim = DataStructs.TanimotoSimilarity(fp_cand, fp_ref)
            except Exception:
                sim = 0.0
            if sim > max_sim:
                max_sim = sim
        similarities[idx] = float(max_sim)

    return pd.Series(similarities, dtype=float)

seen_cache = {}

def sample_random_valid_molecules(
    n_samples: int,
    subnet_config: dict,
    avoid_inchikeys: set[str] | None = None,
    focus_neighborhood_of: pd.DataFrame | None = None,
) -> pd.DataFrame:
    global seen_cache

    names = []
    for name in focus_neighborhood_of["name"]:
        try:
            parts = name.split(":")
            if len(parts) == 4:
                rxn_prefix, rxn_type, comp1_id, comp2_id = parts
                comp1_id = int(comp1_id)
                comp2_id = int(comp2_id)

                # Check if this molecule has been seen before, and adjust range accordingly
                seen_count = seen_cache.get(name, 0) + 1
                seen_cache[name] = seen_count

                comp1_range = chain(range(max(1, comp1_id - seen_count * n_samples), comp1_id - (seen_count-1) * n_samples), range(max(1, comp1_id + (seen_count - 1) * n_samples), comp1_id + seen_count * n_samples + 1))
                for new_comp1 in comp1_range:
                    new_name = f"{rxn_prefix}:{rxn_type}:{new_comp1}:{comp2_id}"
                    if avoid_inchikeys and new_name in avoid_inchikeys:
                        continue  # Skip if this molecule has already been seen
                    names.append(new_name)

                # Generate neighborhood around comp2_id (keep comp1_id fixed)
                comp2_range = chain(range(max(1, comp2_id - seen_count * n_samples), comp2_id - (seen_count-1) * n_samples), range(max(1, comp2_id + (seen_count - 1) * n_samples), comp2_id + seen_count * n_samples + 1))
                for new_comp2 in comp2_range:
                    new_name = f"{rxn_prefix}:{rxn_type}:{comp1_id}:{new_comp2}"
                    if avoid_inchikeys and new_name in avoid_inchikeys:
                        continue  # Skip if this molecule has already been seen
                    names.append(new_name)

            if len(parts) == 5:
                rxn_prefix, rxn_type, comp1_id, comp2_id, comp3_id = parts
                comp1_id = int(comp1_id)
                comp2_id = int(comp2_id)
                comp3_id = int(comp3_id)

                # Check if this molecule has been seen before, and adjust range accordingly
                seen_count = seen_cache.get(name, 0) + 1
                seen_cache[name] = seen_count
                # Generate neighborhood around comp1_id (keep comp2_id and comp3_id fixed)
                comp1_range = chain(range(max(1, comp1_id - seen_count * n_samples), comp1_id - (seen_count-1) * n_samples), range(max(1, comp1_id + (seen_count - 1) * n_samples), comp1_id + seen_count * n_samples + 1))
                for new_comp1 in comp1_range:
                    new_name = f"{rxn_prefix}:{rxn_type}:{new_comp1}:{comp2_id}:{comp3_id}"
                    if avoid_inchikeys and new_name in avoid_inchikeys:
                        continue  # Skip if this molecule has already been seen
                    names.append(new_name)

                # Generate neighborhood around comp2_id (keep comp1_id and comp3_id fixed)
                comp2_range = chain(range(max(1, comp2_id - seen_count * n_samples), comp2_id - (seen_count-1) * n_samples), range(max(1, comp2_id + (seen_count - 1) * n_samples), comp2_id + seen_count * n_samples + 1))
                for new_comp2 in comp2_range:
                    new_name = f"{rxn_prefix}:{rxn_type}:{comp1_id}:{new_comp2}:{comp3_id}"
                    if avoid_inchikeys and new_name in avoid_inchikeys:
                        continue  # Skip if this molecule has already been seen
                    names.append(new_name)

                # Generate neighborhood around comp3_id (keep comp1_id and comp2_id fixed)
                comp3_range = chain(range(max(1, comp3_id - seen_count * n_samples), comp3_id - (seen_count-1) * n_samples), range(max(1, comp3_id + (seen_count - 1) * n_samples), comp3_id + seen_count * n_samples + 1))
                for new_comp3 in comp3_range:
                    new_name = f"{rxn_prefix}:{rxn_type}:{comp1_id}:{comp2_id}:{new_comp3}"
                    if avoid_inchikeys and new_name in avoid_inchikeys:
                        continue  # Skip if this molecule has already been seen
                    names.append(new_name)

        except (ValueError, IndexError) as e:
            bt.logging.warning(f"Could not parse name '{name}': {e}")
            continue

    if not names:
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

    df = pd.DataFrame({"name": names})

    df = df[df["name"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

    df = validate_molecules(df, subnet_config)
    if df.empty:
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

    df = df.drop_duplicates(subset=["InChIKey"], keep="first")

    if avoid_inchikeys:
        df = df[~df["InChIKey"].isin(avoid_inchikeys)]

    return df[["name", "smiles", "InChIKey"]].copy()



def validate_molecules(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Validate molecules by checking heavy atom count and rotatable bonds.
    Returns DataFrame with validated molecules and their descriptors.
    Defer InChIKey generation until after validation to avoid waste.
    """
    if data.empty:
        return data

    data = data.copy()
    data['smiles'] = data["name"].apply(_get_smiles_from_reaction_cached)

    data = data[data['smiles'].notna()]
    if data.empty:
        return data

    data['heavy_atoms'] = data["smiles"].apply(get_heavy_atom_count)
    data['bonds'] = data["smiles"].apply(num_rotatable_bonds)

    mask = (
        (data['heavy_atoms'] >= config['min_heavy_atoms']) &
        (data['bonds'] >= config['min_rotatable_bonds']) &
        (data['bonds'] <= config['max_rotatable_bonds'])
    )
    data = data[mask]

    if not data.empty:
        data['InChIKey'] = data["smiles"].apply(generate_inchikey)

    return data


@lru_cache(maxsize=None)
def get_molecules_by_role(role_mask: int, db_path: str) -> List[Tuple[int, str, int]]:
    try:
        abs_db_path = os.path.abspath(db_path)
        with sqlite3.connect(f"file:{abs_db_path}?mode=ro&immutable=1", uri=True) as conn:
            conn.execute("PRAGMA query_only = ON")
            cursor = conn.cursor()
            cursor.execute(
                "SELECT mol_id, smiles, role_mask FROM molecules WHERE (role_mask & ?) = ?",
                (role_mask, role_mask)
            )
            results = cursor.fetchall()
        return results
    except Exception as e:
        bt.logging.error(f"Error getting molecules by role {role_mask}: {e}")
        return []


class SynthonLibrary:
    """Manages synthon-based similarity search for component selection using Morgan fingerprints."""

    def __init__(self, db_path: str, rxn_id: int):
        self.db_path = db_path
        self.rxn_id = rxn_id
        self.reaction_info = get_reaction_info(rxn_id, db_path)

        if not self.reaction_info:
            raise ValueError(f"Could not load reaction {rxn_id}")

        self.smarts, self.roleA, self.roleB, self.roleC = self.reaction_info
        self.is_three_component = self.roleC is not None and self.roleC != 0

        # Load all components
        self.molecules_A = get_molecules_by_role(self.roleA, db_path)
        self.molecules_B = get_molecules_by_role(self.roleB, db_path)
        self.molecules_C = get_molecules_by_role(self.roleC, db_path) if self.is_three_component else []

        # Build fingerprint indices
        self.fps_A = self._build_fingerprint_index(self.molecules_A)
        self.fps_B = self._build_fingerprint_index(self.molecules_B)
        self.fps_C = self._build_fingerprint_index(self.molecules_C) if self.is_three_component else {}

        bt.logging.info(f"SynthonLibrary initialized: {len(self.fps_A)} A components, "
                       f"{len(self.fps_B)} B components" +
                       (f", {len(self.fps_C)} C components" if self.is_three_component else ""))

    def _build_fingerprint_index(self, molecules: List[Tuple[int, str, int]]) -> Dict[int, object]:
        """Build fingerprint index for fast similarity search."""
        fps = {}
        for mol_id, smiles, _ in molecules:
            mol = _mol_from_smiles_cached(smiles)
            if mol:
                # Use MorganGenerator instead of deprecated method
                fp = MORGAN_FP_GENERATOR.GetFingerprint(mol)
                fps[mol_id] = fp
        return fps

    def find_similar_components(
        self,
        target_smiles: str,
        role: str = 'A',
        top_k: int = 80,  # BALANCED: Find good number of similar components
        min_similarity: float = 0.5
    ) -> List[Tuple[int, float]]:
        """
        Find components similar to target molecule.

        Args:
            target_smiles: SMILES string of target molecule
            role: 'A', 'B', or 'C' - which component pool to search
            top_k: Number of similar components to return
            min_similarity: Minimum Tanimoto similarity threshold

        Returns:
            List of (component_id, similarity_score) tuples
        """
        target_mol = _mol_from_smiles_cached(target_smiles)
        if not target_mol:
            return []

        # Use MorganGenerator instead of deprecated method
        target_fp = MORGAN_FP_GENERATOR.GetFingerprint(target_mol)

        # Select appropriate fingerprint index
        if role == 'A':
            fps_dict = self.fps_A
        elif role == 'B':
            fps_dict = self.fps_B
        elif role == 'C' and self.is_three_component:
            fps_dict = self.fps_C
        else:
            return []

        # Calculate similarities - MORE AGGRESSIVE: check all components
        similarities = []
        for mol_id, fp in fps_dict.items():
            try:
                sim = DataStructs.TanimotoSimilarity(target_fp, fp)
                if sim >= min_similarity:
                    similarities.append((mol_id, sim))
            except Exception:
                continue

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def find_similar_to_molecule_name(
        self,
        molecule_name: str,
        vary_component: str = 'both',
        top_k_per_component: int = 10,
        min_similarity: float = 0.6
    ) -> Dict[str, List[int]]:
        """
        Given a high-scoring molecule name, find similar components.

        Args:
            molecule_name: e.g., "rxn:1:123:456" or "rxn:3:123:456:789"
            vary_component: 'A', 'B', 'C', 'both', or 'all'
            top_k_per_component: How many similar components to find per role
            min_similarity: Minimum similarity threshold

        Returns:
            Dict with keys 'A', 'B', 'C' containing lists of similar component IDs
        """
        # Parse molecule name
        parts = molecule_name.split(":")
        if len(parts) < 4:
            return {}

        try:
            if len(parts) == 4:
                _, rxn, A_id, B_id = parts
                A_id, B_id = int(A_id), int(B_id)
                C_id = None
            else:
                _, rxn, A_id, B_id, C_id = parts
                A_id, B_id, C_id = int(A_id), int(B_id), int(C_id)
        except (ValueError, IndexError):
            return {}

        # Get SMILES for each component
        result = {}

        if vary_component in ['A', 'both', 'all']:
            A_smiles = self._get_component_smiles(A_id, 'A')
            if A_smiles:
                similar_As = self.find_similar_components(
                    A_smiles, 'A', top_k_per_component, min_similarity
                )
                result['A'] = [mol_id for mol_id, _ in similar_As if mol_id != A_id]

        if vary_component in ['B', 'both', 'all']:
            B_smiles = self._get_component_smiles(B_id, 'B')
            if B_smiles:
                similar_Bs = self.find_similar_components(
                    B_smiles, 'B', top_k_per_component, min_similarity
                )
                result['B'] = [mol_id for mol_id, _ in similar_Bs if mol_id != B_id]

        if self.is_three_component and C_id and vary_component in ['C', 'all']:
            C_smiles = self._get_component_smiles(C_id, 'C')
            if C_smiles:
                similar_Cs = self.find_similar_components(
                    C_smiles, 'C', top_k_per_component, min_similarity
                )
                result['C'] = [mol_id for mol_id, _ in similar_Cs if mol_id != C_id]

        return result

    def _get_component_smiles(self, mol_id: int, role: str) -> str:
        """Get SMILES for a component by ID and role."""
        if role == 'A':
            molecules = self.molecules_A
        elif role == 'B':
            molecules = self.molecules_B
        elif role == 'C':
            molecules = self.molecules_C
        else:
            return None

        for mid, smiles, _ in molecules:
            if mid == mol_id:
                return smiles
        return None

    def generate_similar_molecules(
        self,
        base_molecule_names: List[str],
        n_per_base: int = 5,
        min_similarity: float = 0.6
    ) -> List[str]:
        """
        Generate new molecule names by finding similar components to base molecules.
        ULTIMATE PERFECTED: Maximum variations for top molecules.

        Args:
            base_molecule_names: List of high-scoring molecule names
            n_per_base: How many variations to generate per base molecule
            min_similarity: Minimum component similarity threshold

        Returns:
            List of new molecule names to try
        """
        new_molecules = []

        # SUPER-AGGRESSIVE: When only one base molecule, MAXIMUM variations
        is_single_molecule = len(base_molecule_names) == 1
        # For single molecule with high n_per_base, use as-is; otherwise boost significantly
        if is_single_molecule:
            if n_per_base >= 80:
                effective_n_per_base = n_per_base  # Already maximum
            else:
                effective_n_per_base = n_per_base * 3  # 3x multiplier for single top molecule - BALANCED
        else:
            effective_n_per_base = n_per_base

        for base_name in base_molecule_names:
            parts = base_name.split(":")
            if len(parts) < 4:
                continue

            try:
                if len(parts) == 4:
                    _, rxn, A_id, B_id = parts
                    A_id, B_id = int(A_id), int(B_id)

                    # Find similar components - PERFECTED: find more for single molecule
                    similar_comps = self.find_similar_to_molecule_name(
                        base_name, 'both', effective_n_per_base, min_similarity
                    )

                    # Generate variations by replacing A
                    for new_A in similar_comps.get('A', [])[:effective_n_per_base]:
                        new_molecules.append(f"rxn:{rxn}:{new_A}:{B_id}")

                    # Generate variations by replacing B
                    for new_B in similar_comps.get('B', [])[:effective_n_per_base]:
                        new_molecules.append(f"rxn:{rxn}:{A_id}:{new_B}")

                else:  # 3-component
                    _, rxn, A_id, B_id, C_id = parts
                    A_id, B_id, C_id = int(A_id), int(B_id), int(C_id)

                    similar_comps = self.find_similar_to_molecule_name(
                        base_name, 'all', effective_n_per_base, min_similarity
                    )

                    # Generate variations
                    for new_A in similar_comps.get('A', [])[:effective_n_per_base]:
                        new_molecules.append(f"rxn:{rxn}:{new_A}:{B_id}:{C_id}")

                    for new_B in similar_comps.get('B', [])[:effective_n_per_base]:
                        new_molecules.append(f"rxn:{rxn}:{A_id}:{new_B}:{C_id}")

                    for new_C in similar_comps.get('C', [])[:effective_n_per_base]:
                        new_molecules.append(f"rxn:{rxn}:{A_id}:{B_id}:{new_C}")

            except (ValueError, IndexError) as e:
                bt.logging.warning(f"Could not parse molecule name {base_name}: {e}")
                continue

        # Remove duplicates while preserving order
        return list(dict.fromkeys(new_molecules))


def generate_molecules_from_synthon_library(
    synthon_lib: SynthonLibrary,
    top_molecules: pd.DataFrame,
    n_samples: int,
    min_similarity: float = 0.6,
    n_per_base: int = 10
) -> pd.DataFrame:
    """
    Generate new molecules using synthon similarity search.
    ULTIMATE PERFECTED: Maximum exploitation of top molecules.

    Args:
        synthon_lib: Initialized SynthonLibrary
        top_molecules: DataFrame with top-scoring molecules
        n_samples: Target number of molecules to generate
        min_similarity: Minimum component similarity
        n_per_base: Variations per base molecule

    Returns:
        DataFrame with new molecule names
    """
    if top_molecules.empty:
        return pd.DataFrame(columns=["name"])

    # SUPER-AGGRESSIVE: When only 1 molecule, MAXIMUM exploitation
    if len(top_molecules) == 1:
        # Single molecule: generate MAXIMUM variations
        seed_names = top_molecules["name"].tolist()
        # SUPER-AGGRESSIVE: For single top molecule, use 4x variations if n_per_base is high
        if n_per_base >= 80:
            effective_n_per_base = n_per_base  # Already high, use as-is
        else:
            effective_n_per_base = n_per_base * 4  # 4x multiplier for single molecule - SUPER-AGGRESSIVE
    else:
        # Multiple molecules: use appropriate number
        n_seeds = min(30, len(top_molecules))  # More seeds like model1
        seed_names = top_molecules.head(n_seeds)["name"].tolist()
        effective_n_per_base = n_per_base

    # Generate similar molecules
    new_names = synthon_lib.generate_similar_molecules(
        seed_names,
        n_per_base=effective_n_per_base,
        min_similarity=min_similarity
    )

    if not new_names:
        return pd.DataFrame(columns=["name"])

    # SUPER-AGGRESSIVE: Keep all high-quality variations, only sample if excessive
    if len(new_names) > n_samples * 3.0:  # Allow more overflow
        new_names = random.sample(new_names, int(n_samples * 2.0))  # Keep more

    return pd.DataFrame({"name": new_names})


def generate_valid_random_molecules_batch(
    rxn_id: int,
    n_samples: int,
    db_path: str,
    subnet_config: dict,
    batch_size: int = 200,
    seed: int = None,
    elite_names: list[str] | None = None,
    elite_frac: float = 0.5,
    mutation_prob: float = 0.1,
    avoid_inchikeys: set[str] | None = None,
    component_weights: dict | None = None,
) -> pd.DataFrame:
    from tqdm import tqdm

    reaction_info = get_reaction_info(rxn_id, db_path)
    if not reaction_info:
        print(f"[MolGen] ERROR: Could not get reaction info for rxn_id {rxn_id}", flush=True)
        bt.logging.error(f"Could not get reaction info for rxn_id {rxn_id}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

    smarts, roleA, roleB, roleC = reaction_info
    is_three_component = roleC is not None and roleC != 0
    print(f"[MolGen] rxn:{rxn_id} roleA={roleA}, roleB={roleB}, roleC={roleC}, 3-comp={is_three_component}", flush=True)

    molecules_A = get_molecules_by_role(roleA, db_path)
    molecules_B = get_molecules_by_role(roleB, db_path)
    molecules_C = get_molecules_by_role(roleC, db_path) if is_three_component else []
    print(f"[MolGen] Loaded: A={len(molecules_A)}, B={len(molecules_B)}, C={len(molecules_C)}", flush=True)

    if not molecules_A or not molecules_B or (is_three_component and not molecules_C):
        print(f"[MolGen] ERROR: No molecules found for roles A={roleA}, B={roleB}, C={roleC}", flush=True)
        bt.logging.error(f"No molecules found for roles A={roleA}, B={roleB}, C={roleC}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

    elite_As, elite_Bs, elite_Cs = set(), set(), set()
    if elite_names:
        for name in elite_names:
            A, B, C = _parse_components(name)
            if A is not None:
                elite_As.add(A)
            if B is not None:
                elite_Bs.add(B)
            if C is not None and is_three_component:
                elite_Cs.add(C)

    pool_A_ids = _ids_from_pool(molecules_A)
    pool_B_ids = _ids_from_pool(molecules_B)
    pool_C_ids = _ids_from_pool(molecules_C) if is_three_component else []
    valid_dfs = []
    seen_keys = set()
    total_valid = 0

    pbar = tqdm(total=n_samples, desc="[MolGen] Generating", unit="mol")
    while total_valid < n_samples:
        needed = n_samples - total_valid
        batch_size_actual = min(max(batch_size, 300), needed * 2)

        emitted_names = set()
        if elite_names:
            n_elite = max(0, min(batch_size_actual, int(batch_size_actual * elite_frac)))
            n_rand = batch_size_actual - n_elite

            elite_batch = generate_offspring_from_elites(
                rxn_id=rxn_id,
                n=n_elite,
                pool_A_ids=pool_A_ids,
                pool_B_ids=pool_B_ids,
                pool_C_ids=pool_C_ids,
                is_three_component=is_three_component,
                mutation_prob=mutation_prob,
                seed=seed,
                avoid_names=emitted_names,
                avoid_inchikeys=avoid_inchikeys,
                max_tries=10,
                elite_As=elite_As,
                elite_Bs=elite_Bs,
                elite_Cs=elite_Cs,
            )
            emitted_names.update(elite_batch)

            rand_batch = generate_molecules_from_pools(
                rxn_id, n_rand, molecules_A, molecules_B, molecules_C, is_three_component, seed, component_weights
            )
            rand_batch = [n for n in rand_batch if n and (n not in emitted_names)]
            batch_molecules = elite_batch + rand_batch

        else:
            batch_molecules = generate_molecules_from_pools(
                rxn_id, batch_size_actual, molecules_A, molecules_B, molecules_C, is_three_component, seed, component_weights
            )


        if not batch_molecules:
            continue

        batch_df = pd.DataFrame({"name": batch_molecules})
        batch_df = batch_df[batch_df["name"].notna()]  # Remove None values
        if batch_df.empty:
            continue

        batch_df = validate_molecules(batch_df, subnet_config)

        if batch_df.empty:
            continue

        batch_df = batch_df.drop_duplicates(subset=["InChIKey"], keep="first")

        mask = ~batch_df["InChIKey"].isin(seen_keys)
        if avoid_inchikeys:
            mask = mask & ~batch_df["InChIKey"].isin(avoid_inchikeys)
        batch_df = batch_df[mask]

        if batch_df.empty:
            continue

        seen_keys.update(batch_df["InChIKey"].values)
        valid_dfs.append(batch_df[["name", "smiles", "InChIKey"]].copy())
        pbar.update(len(batch_df))
        total_valid += len(batch_df)

        if total_valid >= n_samples:
            break

    pbar.close()

    if not valid_dfs:
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

    # Concatenate all DataFrames at once
    result_df = pd.concat(valid_dfs, ignore_index=True)
    return result_df.head(n_samples).copy()


def generate_molecules_from_pools(rxn_id: int, n: int, molecules_A: List[Tuple], molecules_B: List[Tuple],
                                molecules_C: List[Tuple], is_three_component: bool, seed: int = None,
                                component_weights: dict = None) -> List[str]:

    rng = random.Random(seed) if seed is not None else random

    A_ids = [a[0] for a in molecules_A]
    B_ids = [b[0] for b in molecules_B]
    C_ids = [c[0] for c in molecules_C] if is_three_component else None

    # Use weighted sampling if component weights are provided
    if component_weights:
        # Build weights for each component pool
        weights_A = [component_weights.get('A', {}).get(aid, 1.0) for aid in A_ids]
        weights_B = [component_weights.get('B', {}).get(bid, 1.0) for bid in B_ids]
        weights_C = [component_weights.get('C', {}).get(cid, 1.0) for cid in C_ids] if is_three_component else None

        # Normalize weights
        if weights_A:
            sum_w = sum(weights_A)
            weights_A = [w / sum_w if sum_w > 0 else 1.0/len(weights_A) for w in weights_A]
        if weights_B:
            sum_w = sum(weights_B)
            weights_B = [w / sum_w if sum_w > 0 else 1.0/len(weights_B) for w in weights_B]
        if weights_C:
            sum_w = sum(weights_C)
            weights_C = [w / sum_w if sum_w > 0 else 1.0/len(weights_C) for w in weights_C]

        picks_A = rng.choices(A_ids, weights=weights_A, k=n) if weights_A else rng.choices(A_ids, k=n)
        picks_B = rng.choices(B_ids, weights=weights_B, k=n) if weights_B else rng.choices(B_ids, k=n)
        if is_three_component:
            picks_C = rng.choices(C_ids, weights=weights_C, k=n) if weights_C else rng.choices(C_ids, k=n)
            names = [f"rxn:{rxn_id}:{a}:{b}:{c}" for a, b, c in zip(picks_A, picks_B, picks_C)]
        else:
            names = [f"rxn:{rxn_id}:{a}:{b}" for a, b in zip(picks_A, picks_B)]
    else:
        # Uniform random sampling
        picks_A = rng.choices(A_ids, k=n)
        picks_B = rng.choices(B_ids, k=n)
        if is_three_component:
            picks_C = rng.choices(C_ids, k=n)
            names = [f"rxn:{rxn_id}:{a}:{b}:{c}" for a, b, c in zip(picks_A, picks_B, picks_C)]
        else:
            names = [f"rxn:{rxn_id}:{a}:{b}" for a, b in zip(picks_A, picks_B)]

    # Remove duplicates while preserving order
    names = list(dict.fromkeys(names))
    return names

def _parse_components(name: str) -> tuple[int, int, int | None]:
    # name format: "rxn:{rxn_id}:{A}:{B}" or "rxn:{rxn_id}:{A}:{B}:{C}"
    parts = name.split(":")
    if len(parts) < 4:
        return None, None, None
    A = int(parts[2]); B = int(parts[3])
    C = int(parts[4]) if len(parts) > 4 else None
    return A, B, C

def _ids_from_pool(pool):
    return [x[0] for x in pool]

def generate_offspring_from_elites(rxn_id: int, n: int,
                                   is_three_component: bool,
                                   pool_A_ids:list,
                                   pool_B_ids:list,
                                   pool_C_ids:list,
                                   mutation_prob: float = 0.1, seed: int | None = None,
                                   avoid_names: set[str] = None,
                                   avoid_inchikeys: set[str] = None,
                                   max_tries: int = 10,
                                   elite_As: set[int] = None,
                                   elite_Bs: set[int] = None,
                                   elite_Cs: set[int] = None) -> list[str]:

    rng = random.Random(seed) if seed is not None else random

    elite_As_list = list(elite_As) if elite_As else []
    elite_Bs_list = list(elite_Bs) if elite_Bs else []
    elite_Cs_list = list(elite_Cs) if elite_Cs else []

    out = []
    local_names = set()
    check_inchikeys = avoid_inchikeys is not None and len(avoid_inchikeys) > 0

    for _ in range(n):
        cand = None
        name = None
        for _try in range(max_tries):
            use_mutA = (not elite_As) or (rng.random() < mutation_prob)
            use_mutB = (not elite_Bs) or (rng.random() < mutation_prob)
            use_mutC = (not elite_Cs) or (rng.random() < mutation_prob)

            A = rng.choice(pool_A_ids) if use_mutA else rng.choice(elite_As_list)
            B = rng.choice(pool_B_ids) if use_mutB else rng.choice(elite_Bs_list)
            if is_three_component:
                C = rng.choice(pool_C_ids) if use_mutC else rng.choice(elite_Cs_list)
                name = f"rxn:{rxn_id}:{A}:{B}:{C}"
            else:
                name = f"rxn:{rxn_id}:{A}:{B}"

            # Fast checks first (set membership is O(1))
            if avoid_names and name in avoid_names:
                continue
            if name in local_names:
                continue

            if check_inchikeys:
                try:
                    key = _inchikey_from_name_cached(name)
                    if key and key in avoid_inchikeys:
                        continue
                except Exception:
                    pass

            cand = name
            break

        if cand is None:
            if name is None:
                A = rng.choice(pool_A_ids)
                B = rng.choice(pool_B_ids)
                if is_three_component:
                    C = rng.choice(pool_C_ids) if pool_C_ids else 0
                    name = f"rxn:{rxn_id}:{A}:{B}:{C}"
                else:
                    name = f"rxn:{rxn_id}:{A}:{B}"
            cand = name
        out.append(cand)
        local_names.add(cand)
        if avoid_names is not None:
            avoid_names.add(cand)
    return out

def select_diverse_elites(top_pool: pd.DataFrame, n_elites: int, min_score_ratio: float = 0.65) -> pd.DataFrame:
    """
    Select diverse elite molecules: top by score, but ensure diversity in component space.
    ENHANCED: Lower threshold to include more candidates, better diversity.
    """
    if top_pool.empty or n_elites <= 0:
        return pd.DataFrame()

    # Take MORE top candidates for better diversity selection
    top_candidates = top_pool.head(min(len(top_pool), n_elites * 4))  # Increased from 3
    if len(top_candidates) <= n_elites:
        return top_candidates

    # Score threshold: LOWER threshold to include more candidates
    max_score = top_candidates['score'].max()
    threshold = max_score * min_score_ratio
    candidates = top_candidates[top_candidates['score'] >= threshold]

    # Select diverse set: prefer molecules with different components
    selected = []
    used_components = {'A': set(), 'B': set(), 'C': set()}

    # First, add top scorer
    if not candidates.empty:
        top_idx = candidates.index[0]
        top_row = candidates.iloc[0]
        selected.append(top_idx)
        parts = top_row['name'].split(":")
        if len(parts) >= 4:
            try:
                used_components['A'].add(int(parts[2]))
                used_components['B'].add(int(parts[3]))
                if len(parts) > 4:
                    used_components['C'].add(int(parts[4]))
            except (ValueError, IndexError):
                pass

    # Then add diverse molecules - MORE AGGRESSIVE diversity selection
    for idx, row in candidates.iterrows():
        if len(selected) >= n_elites:
            break
        if idx in selected:
            continue

        parts = row['name'].split(":")
        if len(parts) >= 4:
            try:
                A_id = int(parts[2])
                B_id = int(parts[3])
                C_id = int(parts[4]) if len(parts) > 4 else None

                # Prefer molecules with new components - MORE AGGRESSIVE
                is_diverse = (A_id not in used_components['A'] or
                             B_id not in used_components['B'] or
                             (C_id is not None and C_id not in used_components['C']))

                # Lower threshold for diversity - include more diverse molecules
                if is_diverse or len(selected) < n_elites * 0.6:  # Increased from 0.5
                    selected.append(idx)
                    used_components['A'].add(A_id)
                    used_components['B'].add(B_id)
                    if C_id is not None:
                        used_components['C'].add(C_id)
            except (ValueError, IndexError):
                if len(selected) < n_elites:
                    selected.append(idx)

    # Fill remaining slots
    for idx, row in candidates.iterrows():
        if len(selected) >= n_elites:
            break
        if idx not in selected:
            selected.append(idx)

    return candidates.loc[selected[:n_elites]] if selected else candidates.head(n_elites)


def build_component_weights(top_pool: pd.DataFrame, rxn_id: int) -> Dict[str, Dict[int, float]]:
    """
    Build component weights based on scores of molecules containing them.
    ENHANCED: Use exponential weighting for top molecules to emphasize best components.
    Returns dict with 'A', 'B', 'C' keys mapping to {component_id: weight}
    """
    weights = {'A': defaultdict(float), 'B': defaultdict(float), 'C': defaultdict(float)}
    counts = {'A': defaultdict(int), 'B': defaultdict(int), 'C': defaultdict(int)}

    if top_pool.empty:
        return weights

    # Get max score for normalization
    max_score = top_pool['score'].max() if not top_pool.empty else 1.0

    # Extract component IDs and scores with EXPONENTIAL weighting for top molecules
    for idx, row in top_pool.iterrows():
        name = row['name']
        score = row['score']

        # BALANCED exponential weighting: top molecules contribute more but not excessively
        # Rank-based exponential: rank 1 gets weight 2.5, rank 10 gets weight 1.2, etc.
        rank = idx + 1
        rank_weight = 2.5 * math.exp(-rank / 18.0)  # Balanced exponential decay
        weighted_score = max(0, score) * rank_weight

        parts = name.split(":")
        if len(parts) >= 4:
            try:
                A_id = int(parts[2])
                B_id = int(parts[3])
                weights['A'][A_id] += weighted_score
                weights['B'][B_id] += weighted_score
                counts['A'][A_id] += 1
                counts['B'][B_id] += 1

                if len(parts) > 4:
                    C_id = int(parts[4])
                    weights['C'][C_id] += weighted_score
                    counts['C'][C_id] += 1
            except (ValueError, IndexError):
                continue

    # Normalize by count and add smoothing - but preserve exponential weighting
    for role in ['A', 'B', 'C']:
        for comp_id in weights[role]:
            if counts[role][comp_id] > 0:
                # Average with exponential weighting preserved
                avg_weight = weights[role][comp_id] / counts[role][comp_id]
                # Add smoothing but keep the exponential boost
                weights[role][comp_id] = avg_weight + 0.15  # Balanced smoothing

    return weights


# =============================================================================
# EXPLOIT FUNCTIONALITY - Fingerprint-based exploitation
# Uses the same DB patterns and caching as the rest of molecules.py
# =============================================================================

from dataclasses import dataclass, field

# FCFP generator for exploit (feature-based Morgan)
FCFP_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
)

# Stage 1 thresholds
STAGE1_THRESHOLDS = {'tanimoto': 0.40, 'fcfp': 0.40}

# Stage 2 weights (3-metric composite)
STAGE2_WEIGHTS = {'tanimoto': 0.40, 'fcfp': 0.35, 'tversky': 0.25}


@dataclass
class WinnerRef:
    """Reference molecule for similarity comparison."""
    name: str
    smiles: str
    mol: object = None
    morgan_fp: object = None
    fcfp_fp: object = None

    def __post_init__(self):
        if self.mol is None and self.smiles:
            self.mol = Chem.MolFromSmiles(self.smiles)
        if self.mol and self.morgan_fp is None:
            self.morgan_fp = MORGAN_FP_GENERATOR.GetFingerprint(self.mol)
        if self.mol and self.fcfp_fp is None:
            self.fcfp_fp = FCFP_GEN.GetFingerprint(self.mol)


@dataclass
class ExploitCandidate:
    """A candidate molecule from exploitation."""
    name: str
    smiles: str
    mol: object = None
    morgan_fp: object = None
    fcfp_fp: object = None
    metrics: Dict[str, float] = field(default_factory=dict)
    composite_score: float = 0.0
    winner_ref: str = ""


@lru_cache(maxsize=100000)
def _get_mol_and_fps_cached(smiles: str):
    """Cache molecule parsing and fingerprint computation for exploit."""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        morgan_fp = MORGAN_FP_GENERATOR.GetFingerprint(mol)
        fcfp_fp = FCFP_GEN.GetFingerprint(mol)
        heavy_atoms = get_heavy_atom_count(smiles)  # Use same function as GA path
        rotatable = Descriptors.NumRotatableBonds(mol)
        return (mol, morgan_fp, fcfp_fp, heavy_atoms, rotatable)
    except:
        return None


def _calc_composite_score(metrics: Dict[str, float]) -> float:
    """Calculate Stage 2 composite score."""
    return sum(STAGE2_WEIGHTS.get(k, 0) * v for k, v in metrics.items())


def _get_reactant_type(mol_id: int, roleA: int, roleB: int, db_path: str) -> str:
    """Determine if a molecule is typeA or typeB based on role mask."""
    try:
        abs_db_path = os.path.abspath(db_path)
        with sqlite3.connect(f"file:{abs_db_path}?mode=ro&immutable=1", uri=True) as conn:
            conn.execute("PRAGMA query_only = ON")
            cursor = conn.cursor()
            cursor.execute("SELECT role_mask FROM molecules WHERE mol_id = ?", (mol_id,))
            row = cursor.fetchone()

        if not row:
            return None

        role_mask = row[0]
        is_A = (role_mask & roleA) == roleA
        is_B = (role_mask & roleB) == roleB

        if is_A and not is_B:
            return 'typeA'
        elif is_B and not is_A:
            return 'typeB'
        elif is_A and is_B:
            return 'typeA'
        return None
    except Exception:
        return None


def _exploit_single_reactant(
    reactant_id: int,
    reactant_type: str,
    winner: WinnerRef,
    rxn_id: int,
    db_path: str,
    roleA: int,
    roleB: int,
    limit: int = 50,
    avoid_names: set = None,
    min_heavy_atoms: int = 20,
    min_rotatable: int = 1,
    max_rotatable: int = 10
) -> Tuple[List[ExploitCandidate], Dict[str, int]]:
    """Exploit a single reactant using BULK fingerprint similarity."""
    from tqdm import tqdm

    if avoid_names is None:
        avoid_names = set()

    stats = {
        'total': 0, 'skipped_tested': 0, 'errors': 0,
        'low_atoms': 0, 'bad_rotatable': 0, 'stage1_fail': 0, 'stage1_pass': 0
    }

    # Load partners using existing function
    if reactant_type == 'typeA':
        partners = get_molecules_by_role(roleB, db_path)
    else:
        partners = get_molecules_by_role(roleA, db_path)

    stats['total'] = len(partners)

    # Build reaction names
    candidates_info = []
    for partner_id, partner_smiles, _ in partners:
        if reactant_type == 'typeA':
            rxn_name = f"rxn:{rxn_id}:{reactant_id}:{partner_id}"
        else:
            rxn_name = f"rxn:{rxn_id}:{partner_id}:{reactant_id}"

        if rxn_name in avoid_names:
            stats['skipped_tested'] += 1
            continue
        candidates_info.append((partner_id, rxn_name))

    # Generate products using existing cached function
    valid_candidates = []
    for partner_id, rxn_name in tqdm(candidates_info, desc="      Products", leave=False):
        product_smiles = _get_smiles_from_reaction_cached(rxn_name)
        if not product_smiles:
            stats['errors'] += 1
            continue

        mol_data = _get_mol_and_fps_cached(product_smiles)
        if mol_data is None:
            stats['errors'] += 1
            continue

        mol, morgan_fp, fcfp_fp, heavy_atoms, rotatable = mol_data

        if heavy_atoms < min_heavy_atoms:
            stats['low_atoms'] += 1
            continue
        if rotatable < min_rotatable or rotatable > max_rotatable:
            stats['bad_rotatable'] += 1
            continue

        valid_candidates.append((rxn_name, product_smiles, mol, morgan_fp, fcfp_fp))

    if not valid_candidates:
        return [], stats

    # Bulk similarity
    morgan_fps = [c[3] for c in valid_candidates]
    fcfp_fps = [c[4] for c in valid_candidates]
    morgan_sims = DataStructs.BulkTanimotoSimilarity(winner.morgan_fp, morgan_fps)
    fcfp_sims = DataStructs.BulkTanimotoSimilarity(winner.fcfp_fp, fcfp_fps)

    # Stage 1 filter
    stage1_passed = []
    for i, (rxn_name, product_smiles, mol, morgan_fp, fcfp_fp) in enumerate(valid_candidates):
        tanimoto = morgan_sims[i]
        fcfp = fcfp_sims[i]

        if tanimoto >= STAGE1_THRESHOLDS['tanimoto'] or fcfp >= STAGE1_THRESHOLDS['fcfp']:
            tversky = DataStructs.TverskySimilarity(morgan_fp, winner.morgan_fp, 0.7, 0.3)
            metrics = {'tanimoto': tanimoto, 'fcfp': fcfp, 'tversky': tversky}
            cand = ExploitCandidate(
                name=rxn_name, smiles=product_smiles, mol=mol,
                morgan_fp=morgan_fp, fcfp_fp=fcfp_fp,
                metrics=metrics, composite_score=_calc_composite_score(metrics),
                winner_ref=winner.name
            )
            stage1_passed.append(cand)
            stats['stage1_pass'] += 1
        else:
            stats['stage1_fail'] += 1

    stage1_passed.sort(key=lambda x: x.composite_score, reverse=True)
    return stage1_passed[:limit], stats


def _exploit_single_variation_3comp(
    fixed_ids: Tuple[int, int],
    vary_role: str,
    winner: WinnerRef,
    rxn_id: int,
    db_path: str,
    roleA: int,
    roleB: int,
    roleC: int,
    limit: int = 50,
    avoid_names: set = None,
    min_heavy_atoms: int = 20,
    min_rotatable: int = 1,
    max_rotatable: int = 10
) -> Tuple[List[ExploitCandidate], Dict[str, int]]:
    """Exploit 3-component reaction by varying one component."""
    from tqdm import tqdm

    if avoid_names is None:
        avoid_names = set()

    stats = {
        'total': 0, 'skipped_tested': 0, 'errors': 0,
        'low_atoms': 0, 'bad_rotatable': 0, 'stage1_fail': 0, 'stage1_pass': 0
    }

    # Load partners using existing function
    if vary_role == 'A':
        partners = get_molecules_by_role(roleA, db_path)
    elif vary_role == 'B':
        partners = get_molecules_by_role(roleB, db_path)
    else:
        partners = get_molecules_by_role(roleC, db_path)

    stats['total'] = len(partners)

    # Build reaction names
    candidates_info = []
    for partner_id, partner_smiles, _ in partners:
        if vary_role == 'A':
            rxn_name = f"rxn:{rxn_id}:{partner_id}:{fixed_ids[0]}:{fixed_ids[1]}"
        elif vary_role == 'B':
            rxn_name = f"rxn:{rxn_id}:{fixed_ids[0]}:{partner_id}:{fixed_ids[1]}"
        else:
            rxn_name = f"rxn:{rxn_id}:{fixed_ids[0]}:{fixed_ids[1]}:{partner_id}"

        if rxn_name in avoid_names:
            stats['skipped_tested'] += 1
            continue
        candidates_info.append((partner_id, rxn_name))

    # Generate products
    valid_candidates = []
    for partner_id, rxn_name in tqdm(candidates_info, desc=f"      Vary {vary_role}", leave=False):
        product_smiles = _get_smiles_from_reaction_cached(rxn_name)
        if not product_smiles:
            stats['errors'] += 1
            continue

        mol_data = _get_mol_and_fps_cached(product_smiles)
        if mol_data is None:
            stats['errors'] += 1
            continue

        mol, morgan_fp, fcfp_fp, heavy_atoms, rotatable = mol_data

        if heavy_atoms < min_heavy_atoms:
            stats['low_atoms'] += 1
            continue
        if rotatable < min_rotatable or rotatable > max_rotatable:
            stats['bad_rotatable'] += 1
            continue

        valid_candidates.append((rxn_name, product_smiles, mol, morgan_fp, fcfp_fp))

    if not valid_candidates:
        return [], stats

    # Bulk similarity
    morgan_fps = [c[3] for c in valid_candidates]
    fcfp_fps = [c[4] for c in valid_candidates]
    morgan_sims = DataStructs.BulkTanimotoSimilarity(winner.morgan_fp, morgan_fps)
    fcfp_sims = DataStructs.BulkTanimotoSimilarity(winner.fcfp_fp, fcfp_fps)

    # Stage 1 filter
    stage1_passed = []
    for i, (rxn_name, product_smiles, mol, morgan_fp, fcfp_fp) in enumerate(valid_candidates):
        tanimoto = morgan_sims[i]
        fcfp = fcfp_sims[i]

        if tanimoto >= STAGE1_THRESHOLDS['tanimoto'] or fcfp >= STAGE1_THRESHOLDS['fcfp']:
            tversky = DataStructs.TverskySimilarity(morgan_fp, winner.morgan_fp, 0.7, 0.3)
            metrics = {'tanimoto': tanimoto, 'fcfp': fcfp, 'tversky': tversky}
            cand = ExploitCandidate(
                name=rxn_name, smiles=product_smiles, mol=mol,
                morgan_fp=morgan_fp, fcfp_fp=fcfp_fp,
                metrics=metrics, composite_score=_calc_composite_score(metrics),
                winner_ref=winner.name
            )
            stage1_passed.append(cand)
            stats['stage1_pass'] += 1
        else:
            stats['stage1_fail'] += 1

    stage1_passed.sort(key=lambda x: x.composite_score, reverse=True)
    return stage1_passed[:limit], stats


def get_top_n_unexploited(top_molecules: list, exploited_reactants: set, n: int = 5) -> list:
    """
    Get top N molecules that have at least one unexploited reactant.
    Searches through the entire list to find N molecules with unexploited reactants.
    Handles both 2-component (4 parts) and 3-component (5 parts) reactions.
    """
    selected = []
    for mol in top_molecules:
        if len(selected) >= n:
            break
        name = mol.get("name", "")
        parts = name.split(":")
        if len(parts) >= 4:
            try:
                rxn_id = int(parts[1])
                id_A = int(parts[2])
                id_B = int(parts[3])
                # For 3-component reactions
                id_C = int(parts[4]) if len(parts) == 5 else None

                # Include if any reactant hasn't been exploited
                # rxn:5 uses tuple tracking (role, id) since roleB == roleC
                if id_C is not None and rxn_id == 5:
                    if ('A', id_A) not in exploited_reactants or ('B', id_B) not in exploited_reactants or ('C', id_C) not in exploited_reactants:
                        selected.append(mol)
                else:
                    if id_A not in exploited_reactants or id_B not in exploited_reactants:
                        selected.append(mol)
                    elif id_C is not None and id_C not in exploited_reactants:
                        selected.append(mol)
            except (ValueError, IndexError):
                pass
    return selected


def run_exploit(
    top_molecules: List[Dict],
    db_path: str,
    rxn_id: int,
    top_n: int = 1,
    limit_per_reactant: int = 2000,
    avoid_names: set = None,
    exploited_reactants: set = None,
    min_heavy_atoms: int = 20,
    min_rotatable: int = 1,
    max_rotatable: int = 10,
    verbose: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    Main exploit entry point - exploits UNIQUE reactants only.
    Uses same DB patterns as rest of molecules.py.
    """
    if avoid_names is None:
        avoid_names = set()
    if exploited_reactants is None:
        exploited_reactants = set()

    local_avoid = set(avoid_names)

    # Get reaction info using existing function
    reaction_info = get_reaction_info(rxn_id, db_path)
    if not reaction_info:
        if verbose:
            print(f"[Exploit] ERROR: Could not get reaction info for rxn:{rxn_id}", flush=True)
        return [], {}

    smarts, roleA, roleB, roleC = reaction_info
    is_three_component = roleC is not None and roleC != 0

    if verbose:
        print(f"[Exploit] rxn:{rxn_id} 3-comp={is_three_component}", flush=True)

    # Handle 3-component reactions
    if is_three_component:
        return _exploit_3comp(
            top_molecules, db_path, rxn_id, roleA, roleB, roleC,
            top_n, limit_per_reactant, local_avoid, exploited_reactants,
            min_heavy_atoms, min_rotatable, max_rotatable, verbose
        )

    # 2-component reaction exploitation
    unique_reactants = {}

    for mol_info in top_molecules[:top_n]:
        name = mol_info.get("name", "")
        smiles = mol_info.get("smiles", "")

        if not name or not smiles:
            continue

        parts = name.split(":")
        if len(parts) < 4:
            continue

        try:
            id_A = int(parts[2])
            id_B = int(parts[3])
        except (ValueError, IndexError):
            continue

        winner = WinnerRef(name=name, smiles=smiles)
        if winner.mol is None:
            continue

        type_A = _get_reactant_type(id_A, roleA, roleB, db_path)
        type_B = _get_reactant_type(id_B, roleA, roleB, db_path)

        if type_A and id_A not in unique_reactants and id_A not in exploited_reactants:
            unique_reactants[id_A] = {'type': type_A, 'winner': winner}
        if type_B and id_B not in unique_reactants and id_B not in exploited_reactants:
            unique_reactants[id_B] = {'type': type_B, 'winner': winner}

    if verbose:
        print(f"[Exploit] Found {len(unique_reactants)} unique reactants", flush=True)

    all_candidates = []
    summary = {
        'unique_reactants': len(unique_reactants),
        'reactants_exploited': 0,
        'total_candidates': 0,
        'exploited_reactant_ids': set()
    }

    for reactant_id, info in unique_reactants.items():
        if verbose:
            print(f"[Exploit] Exploiting {info['type']} {reactant_id}", flush=True)

        candidates, stats = _exploit_single_reactant(
            reactant_id=reactant_id,
            reactant_type=info['type'],
            winner=info['winner'],
            rxn_id=rxn_id,
            db_path=db_path,
            roleA=roleA,
            roleB=roleB,
            limit=limit_per_reactant,
            avoid_names=local_avoid,
            min_heavy_atoms=min_heavy_atoms,
            min_rotatable=min_rotatable,
            max_rotatable=max_rotatable
        )

        all_candidates.extend(candidates)
        summary['reactants_exploited'] += 1
        summary['exploited_reactant_ids'].add(reactant_id)

        if verbose:
            print(f"[Exploit]   {stats['total']:,} partners -> {stats['stage1_pass']} pass -> {len(candidates)} kept", flush=True)

        for c in candidates:
            local_avoid.add(c.name)

    summary['total_candidates'] = len(all_candidates)

    # Convert to dicts
    results = [{"name": c.name, "smiles": c.smiles, "InChIKey": generate_inchikey(c.smiles)} for c in all_candidates]

    if verbose:
        print(f"[Exploit] Total: {len(results)} candidates", flush=True)

    return results, summary


def _exploit_3comp(
    top_molecules: List[Dict],
    db_path: str,
    rxn_id: int,
    roleA: int,
    roleB: int,
    roleC: int,
    top_n: int,
    limit_per_reactant: int,
    local_avoid: set,
    exploited_reactants: set,
    min_heavy_atoms: int,
    min_rotatable: int,
    max_rotatable: int,
    verbose: bool
) -> Tuple[List[Dict], Dict]:
    """Handle 3-component reaction exploitation."""

    exploited_tuples = set()
    all_candidates = []
    summary = {
        'molecules_processed': 0,
        'variations_exploited': 0,
        'total_candidates': 0,
        'exploited_reactant_ids': set()
    }

    for mol_info in top_molecules[:top_n]:
        name = mol_info.get("name", "")
        smiles = mol_info.get("smiles", "")

        if not name or not smiles:
            continue

        parts = name.split(":")
        if len(parts) != 5:
            continue

        try:
            id_A = int(parts[2])
            id_B = int(parts[3])
            id_C = int(parts[4])
        except (ValueError, IndexError):
            continue

        winner = WinnerRef(name=name, smiles=smiles)
        if winner.mol is None:
            continue

        summary['molecules_processed'] += 1

        if verbose:
            print(f"[Exploit] 3-comp: {name}", flush=True)

        # rxn:5 uses tuple tracking (role, id) since roleB == roleC
        use_tuple_tracking = (rxn_id == 5)

        # Vary A
        tuple_key = ('A', id_B, id_C)
        check_key_A = ('A', id_A) if use_tuple_tracking else id_A
        if tuple_key not in exploited_tuples and check_key_A not in exploited_reactants:
            candidates, stats = _exploit_single_variation_3comp(
                fixed_ids=(id_B, id_C), vary_role='A', winner=winner,
                rxn_id=rxn_id, db_path=db_path,
                roleA=roleA, roleB=roleB, roleC=roleC,
                limit=limit_per_reactant, avoid_names=local_avoid,
                min_heavy_atoms=min_heavy_atoms,
                min_rotatable=min_rotatable, max_rotatable=max_rotatable
            )
            all_candidates.extend(candidates)
            summary['variations_exploited'] += 1
            summary['exploited_reactant_ids'].add(('A', id_A) if use_tuple_tracking else id_A)
            exploited_tuples.add(tuple_key)
            for c in candidates:
                local_avoid.add(c.name)
            if verbose:
                print(f"  [Exploit] Vary A -> {len(candidates)} candidates", flush=True)

        # Vary B
        tuple_key = ('B', id_A, id_C)
        check_key_B = ('B', id_B) if use_tuple_tracking else id_B
        if tuple_key not in exploited_tuples and check_key_B not in exploited_reactants:
            candidates, stats = _exploit_single_variation_3comp(
                fixed_ids=(id_A, id_C), vary_role='B', winner=winner,
                rxn_id=rxn_id, db_path=db_path,
                roleA=roleA, roleB=roleB, roleC=roleC,
                limit=limit_per_reactant, avoid_names=local_avoid,
                min_heavy_atoms=min_heavy_atoms,
                min_rotatable=min_rotatable, max_rotatable=max_rotatable
            )
            all_candidates.extend(candidates)
            summary['variations_exploited'] += 1
            summary['exploited_reactant_ids'].add(('B', id_B) if use_tuple_tracking else id_B)
            exploited_tuples.add(tuple_key)
            for c in candidates:
                local_avoid.add(c.name)
            if verbose:
                print(f"  [Exploit] Vary B -> {len(candidates)} candidates", flush=True)

        # Vary C
        tuple_key = ('C', id_A, id_B)
        check_key_C = ('C', id_C) if use_tuple_tracking else id_C
        if tuple_key not in exploited_tuples and check_key_C not in exploited_reactants:
            candidates, stats = _exploit_single_variation_3comp(
                fixed_ids=(id_A, id_B), vary_role='C', winner=winner,
                rxn_id=rxn_id, db_path=db_path,
                roleA=roleA, roleB=roleB, roleC=roleC,
                limit=limit_per_reactant, avoid_names=local_avoid,
                min_heavy_atoms=min_heavy_atoms,
                min_rotatable=min_rotatable, max_rotatable=max_rotatable
            )
            all_candidates.extend(candidates)
            summary['variations_exploited'] += 1
            summary['exploited_reactant_ids'].add(('C', id_C) if use_tuple_tracking else id_C)
            exploited_tuples.add(tuple_key)
            for c in candidates:
                local_avoid.add(c.name)
            if verbose:
                print(f"  [Exploit] Vary C -> {len(candidates)} candidates", flush=True)

    summary['total_candidates'] = len(all_candidates)

    # Convert to dicts
    results = [{"name": c.name, "smiles": c.smiles, "InChIKey": generate_inchikey(c.smiles)} for c in all_candidates]

    if verbose:
        print(f"[Exploit] 3-comp Total: {len(results)} candidates", flush=True)

    return results, summary


# =============================================================================
# AMPLIFICATION MODE - Elite Winner Pattern Amplification
# Ported from famy.py for integration into miner workflow
# =============================================================================

# Strategy weights for amplification (from famy.py)
AMPLIFICATION_STRATEGY_WEIGHTS = {
    'elite_winner_amplification': 0.50,
    'elite_similar_combinations': 0.30,
    'successful_reaction_focus': 0.15,
    'pattern_guided_exploration': 0.05
}

# Chemical patterns for pattern-guided exploration
CHEMICAL_PATTERNS = [
    ('sulfonamide', 'S(=O)(=O)N'),
    ('sulfonyl', 'S(=O)(=O)'),
    ('boron', 'B('),
    ('silicon', 'Si'),
    ('alkyne', 'C#C'),
    ('phosphate', 'P(=O)'),
    ('trifluoromethyl', 'C(F)(F)F'),
    ('cyclopropyl', 'C1CC1'),
    ('morpholine', 'N1CCOCC1'),
    ('piperazine', 'N1CCN'),
    ('pyridine', 'ccncc'),
    ('pyrimidine', 'ncncn'),
    ('benzene', 'c1ccccc1'),
    ('imidazole', 'n1ccnc1'),
    ('triazole', 'n1nccn1'),
    ('thiophene', 'cccs'),
]

# Pattern bonuses for similarity scoring
PATTERN_BONUSES = {
    'boron': 0.3, 'silicon': 0.3, 'alkyne': 0.25, 'phosphate': 0.25,
    'trifluoromethyl': 0.2, 'sulfonamide': 0.2, 'sulfonyl': 0.15,
    'pyridine': 0.15, 'morpholine': 0.15, 'piperazine': 0.15,
    'cyclopropyl': 0.15, 'imidazole': 0.1, 'triazole': 0.1,
    'thiophene': 0.1, 'benzene': 0.05, 'pyrimidine': 0.05,
}


class AmplificationLibrary:
    """
    Manages amplification-based molecule generation using elite winner patterns.
    Similar to SynthonLibrary but focuses on amplifying successful patterns.
    """

    def __init__(self, db_path: str, rxn_id: int):
        self.db_path = db_path
        self.rxn_id = rxn_id
        self.reaction_info = get_reaction_info(rxn_id, db_path)

        if not self.reaction_info:
            raise ValueError(f"Could not load reaction {rxn_id}")

        self.smarts, self.roleA, self.roleB, self.roleC = self.reaction_info
        self.is_three_component = self.roleC is not None and self.roleC != 0

        # Load all components
        self.molecules_A = get_molecules_by_role(self.roleA, db_path)
        self.molecules_B = get_molecules_by_role(self.roleB, db_path)
        self.molecules_C = get_molecules_by_role(self.roleC, db_path) if self.is_three_component else []

        # Build fingerprint indices (same as SynthonLibrary)
        self.fps_A = self._build_fingerprint_index(self.molecules_A)
        self.fps_B = self._build_fingerprint_index(self.molecules_B)
        self.fps_C = self._build_fingerprint_index(self.molecules_C) if self.is_three_component else {}

        # Build mol_id -> smiles lookup
        self.id_to_smiles_A = {m[0]: m[1] for m in self.molecules_A}
        self.id_to_smiles_B = {m[0]: m[1] for m in self.molecules_B}
        self.id_to_smiles_C = {m[0]: m[1] for m in self.molecules_C} if self.is_three_component else {}

        # Pattern cache for reactants
        self.pattern_cache = {}
        self._precompute_patterns()

        # Memory for learning (tracks successful reactants/reactions)
        self.successful_reactants = defaultdict(lambda: {'count': 0, 'avg_score': 0.0})
        self.successful_reactions = defaultdict(lambda: {'count': 0, 'avg_score': 0.0})

        bt.logging.info(f"AmplificationLibrary initialized: {len(self.fps_A)} A, "
                       f"{len(self.fps_B)} B" +
                       (f", {len(self.fps_C)} C" if self.is_three_component else ""))

    def _build_fingerprint_index(self, molecules: List[Tuple[int, str, int]]) -> Dict[int, object]:
        """Build fingerprint index for fast similarity search."""
        fps = {}
        for mol_id, smiles, _ in molecules:
            mol = _mol_from_smiles_cached(smiles)
            if mol:
                fp = MORGAN_FP_GENERATOR.GetFingerprint(mol)
                fps[mol_id] = fp
        return fps

    def _precompute_patterns(self):
        """Precompute chemical patterns for all reactants."""
        all_molecules = self.molecules_A + self.molecules_B + self.molecules_C
        for mol_id, smiles, _ in all_molecules:
            patterns_found = []
            for pattern_name, pattern_str in CHEMICAL_PATTERNS:
                if pattern_str in smiles:
                    patterns_found.append(pattern_name)
            self.pattern_cache[mol_id] = patterns_found

    def update_memory(self, top_molecules: List[Dict]):
        """Update memory with successful molecules."""
        for mol_info in top_molecules:
            name = mol_info.get("name", "")
            score = mol_info.get("score", 0.0)

            parts = name.split(":")
            if len(parts) < 4:
                continue

            try:
                rxn_id = int(parts[1])
                id_A = int(parts[2])
                id_B = int(parts[3])
                id_C = int(parts[4]) if len(parts) == 5 else None

                # Update successful reactants
                for comp_id in [id_A, id_B, id_C]:
                    if comp_id is not None:
                        mem = self.successful_reactants[comp_id]
                        old_avg = mem['avg_score']
                        old_count = mem['count']
                        new_count = old_count + 1
                        mem['avg_score'] = (old_avg * old_count + score) / new_count
                        mem['count'] = new_count

                # Update successful reactions
                rxn_mem = self.successful_reactions[rxn_id]
                old_avg = rxn_mem['avg_score']
                old_count = rxn_mem['count']
                new_count = old_count + 1
                rxn_mem['avg_score'] = (old_avg * old_count + score) / new_count
                rxn_mem['count'] = new_count

            except (ValueError, IndexError):
                continue

    def find_similar_reactants(
        self,
        target_smiles: str,
        role: str = 'A',
        top_k: int = 50,
        min_similarity: float = 0.5
    ) -> List[Tuple[int, float]]:
        """Find reactants similar to target molecule."""
        target_mol = _mol_from_smiles_cached(target_smiles)
        if not target_mol:
            return []

        target_fp = MORGAN_FP_GENERATOR.GetFingerprint(target_mol)

        if role == 'A':
            fps_dict = self.fps_A
        elif role == 'B':
            fps_dict = self.fps_B
        elif role == 'C' and self.is_three_component:
            fps_dict = self.fps_C
        else:
            return []

        similarities = []
        for mol_id, fp in fps_dict.items():
            try:
                sim = DataStructs.TanimotoSimilarity(target_fp, fp)
                # Add pattern bonus
                pattern_bonus = 0.0
                for pattern in self.pattern_cache.get(mol_id, []):
                    pattern_bonus += PATTERN_BONUSES.get(pattern, 0.0)
                # Add memory bonus
                memory_bonus = min(self.successful_reactants[mol_id]['count'] * 0.05, 0.2)

                combined_score = sim + min(pattern_bonus, 0.3) + memory_bonus
                if combined_score >= min_similarity:
                    similarities.append((mol_id, combined_score))
            except Exception:
                continue

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def extract_elite_components(self, top_molecules: List[Dict]) -> Dict[str, set]:
        """Extract component IDs from elite molecules."""
        elite_components = {'A': set(), 'B': set(), 'C': set()}

        for mol_info in top_molecules:
            name = mol_info.get("name", "")
            parts = name.split(":")
            if len(parts) >= 4:
                try:
                    elite_components['A'].add(int(parts[2]))
                    elite_components['B'].add(int(parts[3]))
                    if len(parts) == 5:
                        elite_components['C'].add(int(parts[4]))
                except (ValueError, IndexError):
                    continue

        return elite_components


def run_amplification(
    top_molecules: List[Dict],
    db_path: str,
    rxn_id: int,
    n_samples: int = 5000,
    avoid_names: set = None,
    avoid_inchikeys: set = None,
    min_heavy_atoms: int = 20,
    min_rotatable: int = 1,
    max_rotatable: int = 10,
    verbose: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    Amplification mode: generate molecules by amplifying elite winner patterns.

    Uses 4 strategies (weights from famy.py):
    - elite_winner_amplification (50%): Variations using elite reactants
    - elite_similar_combinations (30%): Combine elite-like reactants
    - successful_reaction_focus (15%): Focus on successful reaction types
    - pattern_guided_exploration (5%): Pattern-based discovery

    Args:
        top_molecules: List of top-scoring molecules with name, smiles, score
        db_path: Path to molecule database
        rxn_id: Reaction ID
        n_samples: Target number of molecules to generate
        avoid_names: Set of molecule names to avoid
        avoid_inchikeys: Set of InChIKeys to avoid
        min_heavy_atoms: Minimum heavy atom count
        min_rotatable: Minimum rotatable bonds
        max_rotatable: Maximum rotatable bonds
        verbose: Print progress

    Returns:
        (list of molecule dicts, summary dict)
    """
    from tqdm import tqdm

    if avoid_names is None:
        avoid_names = set()
    if avoid_inchikeys is None:
        avoid_inchikeys = set()

    local_avoid = set(avoid_names)
    local_avoid_keys = set(avoid_inchikeys)

    # Initialize library
    try:
        amp_lib = AmplificationLibrary(db_path, rxn_id)
    except Exception as e:
        if verbose:
            print(f"[Amplification] ERROR initializing library: {e}", flush=True)
        return [], {'error': str(e)}

    # Update memory with current top molecules
    amp_lib.update_memory(top_molecules)

    # Extract elite components
    elite_components = amp_lib.extract_elite_components(top_molecules)

    if verbose:
        print(f"[Amplification] rxn:{rxn_id} 3-comp={amp_lib.is_three_component}", flush=True)
        print(f"[Amplification] Elite components: A={len(elite_components['A'])}, "
              f"B={len(elite_components['B'])}, C={len(elite_components['C'])}", flush=True)

    summary = {
        'strategies': {},
        'total_generated': 0,
        'total_valid': 0,
        'elite_components': {k: len(v) for k, v in elite_components.items()}
    }

    all_candidates = []
    used_combinations = set()

    # Calculate samples per strategy
    strategy_samples = {}
    for strategy, weight in AMPLIFICATION_STRATEGY_WEIGHTS.items():
        strategy_samples[strategy] = int(n_samples * weight)
        summary['strategies'][strategy] = {'target': strategy_samples[strategy], 'generated': 0}

    # Strategy 1: Elite Winner Amplification (50%)
    if verbose:
        print(f"[Amplification] Strategy 1: Elite Winner Amplification ({strategy_samples['elite_winner_amplification']} target)", flush=True)

    elite_candidates = _generate_elite_winner_amplification(
        amp_lib, elite_components, strategy_samples['elite_winner_amplification'],
        used_combinations, local_avoid, verbose
    )
    summary['strategies']['elite_winner_amplification']['generated'] = len(elite_candidates)
    all_candidates.extend(elite_candidates)

    # Strategy 2: Elite Similar Combinations (30%)
    if verbose:
        print(f"[Amplification] Strategy 2: Elite Similar Combinations ({strategy_samples['elite_similar_combinations']} target)", flush=True)

    similar_candidates = _generate_elite_similar_combinations(
        amp_lib, top_molecules, strategy_samples['elite_similar_combinations'],
        used_combinations, local_avoid, verbose
    )
    summary['strategies']['elite_similar_combinations']['generated'] = len(similar_candidates)
    all_candidates.extend(similar_candidates)

    # Strategy 3: Successful Reaction Focus (15%)
    if verbose:
        print(f"[Amplification] Strategy 3: Successful Reaction Focus ({strategy_samples['successful_reaction_focus']} target)", flush=True)

    reaction_candidates = _generate_successful_reaction_focus(
        amp_lib, elite_components, strategy_samples['successful_reaction_focus'],
        used_combinations, local_avoid, verbose
    )
    summary['strategies']['successful_reaction_focus']['generated'] = len(reaction_candidates)
    all_candidates.extend(reaction_candidates)

    # Strategy 4: Pattern Guided Exploration (5%)
    if verbose:
        print(f"[Amplification] Strategy 4: Pattern Guided Exploration ({strategy_samples['pattern_guided_exploration']} target)", flush=True)

    pattern_candidates = _generate_pattern_guided_exploration(
        amp_lib, strategy_samples['pattern_guided_exploration'],
        used_combinations, local_avoid, verbose
    )
    summary['strategies']['pattern_guided_exploration']['generated'] = len(pattern_candidates)
    all_candidates.extend(pattern_candidates)

    summary['total_generated'] = len(all_candidates)

    if verbose:
        print(f"[Amplification] Total generated: {len(all_candidates)} candidates", flush=True)

    # Validate molecules
    if not all_candidates:
        return [], summary

    df = pd.DataFrame({"name": all_candidates})
    df = df.drop_duplicates(subset=["name"])

    # Get SMILES and validate
    df['smiles'] = df['name'].apply(_get_smiles_from_reaction_cached)
    df = df[df['smiles'].notna()]

    # Filter by properties
    df['heavy_atoms'] = df['smiles'].apply(get_heavy_atom_count)
    df['rotatable'] = df['smiles'].apply(num_rotatable_bonds)

    mask = (
        (df['heavy_atoms'] >= min_heavy_atoms) &
        (df['rotatable'] >= min_rotatable) &
        (df['rotatable'] <= max_rotatable)
    )
    df = df[mask]

    # Generate InChIKeys and filter duplicates
    if not df.empty:
        df['InChIKey'] = df['smiles'].apply(generate_inchikey)
        df = df[df['InChIKey'].notna()]
        df = df[~df['InChIKey'].isin(local_avoid_keys)]
        df = df.drop_duplicates(subset=['InChIKey'])

    # Convert to result format
    results = []
    for _, row in df.iterrows():
        results.append({
            "name": row['name'],
            "smiles": row['smiles'],
            "InChIKey": row['InChIKey']
        })

    summary['total_valid'] = len(results)

    if verbose:
        print(f"[Amplification] Valid molecules: {len(results)}", flush=True)

    return results, summary


def _generate_elite_winner_amplification(
    amp_lib: AmplificationLibrary,
    elite_components: Dict[str, set],
    count: int,
    used_combinations: set,
    avoid_names: set,
    verbose: bool
) -> List[str]:
    """Generate variations using reactants from elite winners."""
    candidates = []
    rxn_id = amp_lib.rxn_id

    elite_A = list(elite_components['A'])
    elite_B = list(elite_components['B'])
    elite_C = list(elite_components['C']) if amp_lib.is_three_component else []

    if not elite_A or not elite_B:
        return []

    all_A = list(amp_lib.id_to_smiles_A.keys())
    all_B = list(amp_lib.id_to_smiles_B.keys())
    all_C = list(amp_lib.id_to_smiles_C.keys()) if amp_lib.is_three_component else []

    attempts = 0
    max_attempts = count * 3

    while len(candidates) < count and attempts < max_attempts:
        attempts += 1

        # 80% elite, 20% random mutation
        if random.random() < 0.8 and elite_A:
            A_id = random.choice(elite_A)
        else:
            A_id = random.choice(all_A)

        if random.random() < 0.8 and elite_B:
            B_id = random.choice(elite_B)
        else:
            B_id = random.choice(all_B)

        if amp_lib.is_three_component:
            if random.random() < 0.8 and elite_C:
                C_id = random.choice(elite_C)
            elif all_C:
                C_id = random.choice(all_C)
            else:
                continue
            combo_key = (rxn_id, A_id, B_id, C_id)
            name = f"rxn:{rxn_id}:{A_id}:{B_id}:{C_id}"
        else:
            combo_key = (rxn_id, A_id, B_id)
            name = f"rxn:{rxn_id}:{A_id}:{B_id}"

        if combo_key not in used_combinations and name not in avoid_names:
            used_combinations.add(combo_key)
            candidates.append(name)

    return candidates


def _generate_elite_similar_combinations(
    amp_lib: AmplificationLibrary,
    top_molecules: List[Dict],
    count: int,
    used_combinations: set,
    avoid_names: set,
    verbose: bool
) -> List[str]:
    """Generate combinations using elite-winner-like reactants."""
    candidates = []
    rxn_id = amp_lib.rxn_id

    # Find similar reactants for each elite molecule
    similar_A = set()
    similar_B = set()
    similar_C = set()

    for mol_info in top_molecules[:10]:  # Top 10 elites
        smiles = mol_info.get("smiles", "")
        if not smiles:
            continue

        # Find similar reactants
        for mol_id, _ in amp_lib.find_similar_reactants(smiles, 'A', top_k=30, min_similarity=0.5):
            similar_A.add(mol_id)
        for mol_id, _ in amp_lib.find_similar_reactants(smiles, 'B', top_k=30, min_similarity=0.5):
            similar_B.add(mol_id)
        if amp_lib.is_three_component:
            for mol_id, _ in amp_lib.find_similar_reactants(smiles, 'C', top_k=30, min_similarity=0.5):
                similar_C.add(mol_id)

    similar_A = list(similar_A)
    similar_B = list(similar_B)
    similar_C = list(similar_C)

    if not similar_A or not similar_B:
        return []

    attempts = 0
    max_attempts = count * 3

    while len(candidates) < count and attempts < max_attempts:
        attempts += 1

        A_id = random.choice(similar_A)
        B_id = random.choice(similar_B)

        if amp_lib.is_three_component:
            if similar_C:
                C_id = random.choice(similar_C)
            else:
                C_id = random.choice(list(amp_lib.id_to_smiles_C.keys())) if amp_lib.id_to_smiles_C else None
                if C_id is None:
                    continue
            combo_key = (rxn_id, A_id, B_id, C_id)
            name = f"rxn:{rxn_id}:{A_id}:{B_id}:{C_id}"
        else:
            combo_key = (rxn_id, A_id, B_id)
            name = f"rxn:{rxn_id}:{A_id}:{B_id}"

        if combo_key not in used_combinations and name not in avoid_names:
            used_combinations.add(combo_key)
            candidates.append(name)

    return candidates


def _generate_successful_reaction_focus(
    amp_lib: AmplificationLibrary,
    elite_components: Dict[str, set],
    count: int,
    used_combinations: set,
    avoid_names: set,
    verbose: bool
) -> List[str]:
    """Focus on the successful reaction type with memory-guided selection."""
    candidates = []
    rxn_id = amp_lib.rxn_id

    # Combine elite and memory-successful reactants
    elite_A = list(elite_components['A'])
    elite_B = list(elite_components['B'])
    elite_C = list(elite_components['C']) if amp_lib.is_three_component else []

    # Get top successful reactants from memory
    memory_A = sorted(
        [(k, v['avg_score']) for k, v in amp_lib.successful_reactants.items() if k in amp_lib.id_to_smiles_A],
        key=lambda x: x[1], reverse=True
    )[:20]
    memory_B = sorted(
        [(k, v['avg_score']) for k, v in amp_lib.successful_reactants.items() if k in amp_lib.id_to_smiles_B],
        key=lambda x: x[1], reverse=True
    )[:20]

    # Combine elite and memory (elite has priority)
    combined_A = list(set(elite_A + [m[0] for m in memory_A]))
    combined_B = list(set(elite_B + [m[0] for m in memory_B]))

    if not combined_A or not combined_B:
        combined_A = list(amp_lib.id_to_smiles_A.keys())[:100]
        combined_B = list(amp_lib.id_to_smiles_B.keys())[:100]

    all_C = list(amp_lib.id_to_smiles_C.keys()) if amp_lib.is_three_component else []

    attempts = 0
    max_attempts = count * 3

    while len(candidates) < count and attempts < max_attempts:
        attempts += 1

        # 85% from combined pool
        if random.random() < 0.85:
            A_id = random.choice(combined_A[:min(10, len(combined_A))])
            B_id = random.choice(combined_B[:min(10, len(combined_B))])
        else:
            A_id = random.choice(combined_A)
            B_id = random.choice(combined_B)

        if amp_lib.is_three_component:
            if elite_C and random.random() < 0.85:
                C_id = random.choice(elite_C)
            elif all_C:
                C_id = random.choice(all_C)
            else:
                continue
            combo_key = (rxn_id, A_id, B_id, C_id)
            name = f"rxn:{rxn_id}:{A_id}:{B_id}:{C_id}"
        else:
            combo_key = (rxn_id, A_id, B_id)
            name = f"rxn:{rxn_id}:{A_id}:{B_id}"

        if combo_key not in used_combinations and name not in avoid_names:
            used_combinations.add(combo_key)
            candidates.append(name)

    return candidates


def _generate_pattern_guided_exploration(
    amp_lib: AmplificationLibrary,
    count: int,
    used_combinations: set,
    avoid_names: set,
    verbose: bool
) -> List[str]:
    """Use chemical patterns to guide exploration."""
    candidates = []
    rxn_id = amp_lib.rxn_id

    # Find reactants with high-value patterns
    pattern_reactants_A = []
    pattern_reactants_B = []
    pattern_reactants_C = []

    for mol_id, patterns in amp_lib.pattern_cache.items():
        if not patterns:
            continue
        pattern_score = sum(PATTERN_BONUSES.get(p, 0.0) for p in patterns)
        if pattern_score >= 0.1:
            if mol_id in amp_lib.id_to_smiles_A:
                pattern_reactants_A.append((mol_id, pattern_score))
            if mol_id in amp_lib.id_to_smiles_B:
                pattern_reactants_B.append((mol_id, pattern_score))
            if amp_lib.is_three_component and mol_id in amp_lib.id_to_smiles_C:
                pattern_reactants_C.append((mol_id, pattern_score))

    # Sort by pattern score
    pattern_reactants_A.sort(key=lambda x: x[1], reverse=True)
    pattern_reactants_B.sort(key=lambda x: x[1], reverse=True)
    pattern_reactants_C.sort(key=lambda x: x[1], reverse=True)

    # Take top pattern reactants
    top_A = [m[0] for m in pattern_reactants_A[:50]]
    top_B = [m[0] for m in pattern_reactants_B[:50]]
    top_C = [m[0] for m in pattern_reactants_C[:50]] if amp_lib.is_three_component else []

    if not top_A:
        top_A = list(amp_lib.id_to_smiles_A.keys())[:50]
    if not top_B:
        top_B = list(amp_lib.id_to_smiles_B.keys())[:50]
    if amp_lib.is_three_component and not top_C:
        top_C = list(amp_lib.id_to_smiles_C.keys())[:50]

    attempts = 0
    max_attempts = count * 3

    while len(candidates) < count and attempts < max_attempts:
        attempts += 1

        A_id = random.choice(top_A)
        B_id = random.choice(top_B)

        if amp_lib.is_three_component:
            if top_C:
                C_id = random.choice(top_C)
            else:
                continue
            combo_key = (rxn_id, A_id, B_id, C_id)
            name = f"rxn:{rxn_id}:{A_id}:{B_id}:{C_id}"
        else:
            combo_key = (rxn_id, A_id, B_id)
            name = f"rxn:{rxn_id}:{A_id}:{B_id}"

        if combo_key not in used_combinations and name not in avoid_names:
            used_combinations.add(combo_key)
            candidates.append(name)

    return candidates
