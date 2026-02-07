"""
Region-Based Component Sampling for v12

Instead of Tanimoto fingerprint diversity (which everyone uses),
this module implements feature-region based sampling to explore
underexplored parts of the chemical space.

Key insight: Dominant regions ({}, {small}) contain 30-50% of components
but everyone samples from them proportionally. By explicitly excluding
dominant regions and focusing on rare feature combinations, we can
explore different parts of the scoring landscape.
"""

import sqlite3
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple, FrozenSet
from functools import lru_cache
import bittensor as bt

# Try to import RDKit for true SMARTS matching
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    bt.logging.warning("[Regions] RDKit not available, using simple pattern matching")


# SMARTS patterns for true substructure matching (used when RDKit available)
SMARTS_PATTERNS = {
    # Highest value scaffolds
    'boron': '[#5]',                    # Any boron
    'silicon': '[#14]',                 # Any silicon
    'alkyne': 'C#C',                    # Terminal or internal alkyne
    'phosphate': 'P(=O)([O,N])([O,N])', # Phosphate/phosphonate

    # High value functional groups
    'sulfonamide': 'S(=O)(=O)N',        # Sulfonamide
    'sulfone': 'S(=O)(=O)[C,c]',        # Sulfone
    'cyclopropyl': 'C1CC1',             # Cyclopropane
    'azetidine': 'C1CNC1',              # Azetidine (4-membered N-ring)

    # Heterocycles (privileged scaffolds)
    'morpholine': 'C1COCCN1',           # Morpholine
    'piperazine': 'C1CNCCN1',           # Piperazine
    'piperidine': 'C1CCNCC1',           # Piperidine
    'pyrrolidine': 'C1CCNC1',           # Pyrrolidine
    'imidazole': 'c1cnc[nH]1',          # Imidazole
    'pyrazole': 'c1cc[nH]n1',           # Pyrazole
    'oxazole': 'c1cocn1',               # Oxazole
    'thiazole': 'c1cscn1',              # Thiazole
    'indole': 'c1ccc2[nH]ccc2c1',       # Indole
    'benzimidazole': 'c1ccc2nc[nH]c2c1', # Benzimidazole

    # Functional groups
    'nitrile': 'C#N',                   # Nitrile
    'nitro': '[N+](=O)[O-]',            # Nitro
    'azide': '[N-]=[N+]=[N-]',          # Azide
    'trifluoromethyl': 'C(F)(F)F',      # CF3
    'difluoromethyl': 'C(F)F',          # CHF2
    'amide': 'C(=O)N',                  # Amide
    'ester': 'C(=O)O[C,c]',             # Ester
    'carbamate': 'NC(=O)O',             # Carbamate
    'urea': 'NC(=O)N',                  # Urea
}

# Compiled SMARTS patterns (cached)
_COMPILED_SMARTS = {}

def _get_compiled_smarts():
    """Compile SMARTS patterns once."""
    global _COMPILED_SMARTS
    if not _COMPILED_SMARTS and HAS_RDKIT:
        for name, smarts in SMARTS_PATTERNS.items():
            mol = Chem.MolFromSmarts(smarts)
            if mol:
                _COMPILED_SMARTS[name] = mol
            else:
                bt.logging.warning(f"[Regions] Failed to compile SMARTS: {name} = {smarts}")
    return _COMPILED_SMARTS


# Simple string patterns fallback (when RDKit not available)
SIMPLE_PATTERNS = {
    'boron': 'B(',
    'silicon': 'Si',
    'alkyne': 'C#C',
    'phosphate': 'P(=O)',
    'sulfonamide': 'S(=O)(=O)N',
    'cyclopropyl': 'C1CC1',
    'morpholine': 'COCCN',
    'piperazine': 'N1CCN',
    'piperidine': 'N1CCCCC1',
    'pyrrolidine': 'N1CCCC1',
}


# Feature extraction from SMILES
def extract_features(smiles: str) -> FrozenSet[str]:
    """
    Extract structural features from SMILES string.
    Returns a frozenset of feature tags.

    Features include:
    - Basic: Halogens (F, Cl, Br, I), Heteroatoms (S, N4+, O4+)
    - Size: tiny, small, large
    - Ring systems: aromatic count, ring count
    - SMARTS patterns: True substructure matching for medchem scaffolds
    """
    features = set()

    # Try RDKit-based extraction first (more accurate)
    if HAS_RDKIT:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return _extract_features_rdkit(mol, smiles)
        except:
            pass  # Fall back to simple extraction

    # Simple string-based extraction (fallback)
    return _extract_features_simple(smiles)


def _extract_features_rdkit(mol, smiles: str) -> FrozenSet[str]:
    """Extract features using RDKit (accurate SMARTS matching)."""
    features = set()

    # Halogens (atom count)
    atom_counts = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        atom_counts[sym] = atom_counts.get(sym, 0) + 1

    if atom_counts.get('F', 0) > 0:
        features.add('F')
        if atom_counts.get('F', 0) >= 3:
            features.add('F3+')  # Polyfluorinated
    if atom_counts.get('Cl', 0) > 0:
        features.add('Cl')
    if atom_counts.get('Br', 0) > 0:
        features.add('Br')
    if atom_counts.get('I', 0) > 0:
        features.add('I')
    if atom_counts.get('S', 0) > 0:
        features.add('S')
    if atom_counts.get('B', 0) > 0:
        features.add('boron')
    if atom_counts.get('Si', 0) > 0:
        features.add('silicon')
    if atom_counts.get('P', 0) > 0:
        features.add('phosphorus')

    # Nitrogen count
    n_count = atom_counts.get('N', 0)
    if n_count >= 4:
        features.add('N4+')
    if n_count >= 6:
        features.add('N6+')

    # Oxygen count
    o_count = atom_counts.get('O', 0)
    if o_count >= 4:
        features.add('O4+')

    # Heavy atom count (size)
    heavy = mol.GetNumHeavyAtoms()
    if heavy < 15:
        features.add('tiny')
    elif heavy < 25:
        features.add('small')
    elif heavy >= 40:
        features.add('large')

    # Ring analysis
    ring_info = mol.GetRingInfo()
    n_rings = ring_info.NumRings()
    if n_rings >= 4:
        features.add('polyring')
    n_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
    if n_aromatic >= 3:
        features.add('polyaromatic')

    # Rotatable bonds (flexibility)
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    if n_rot >= 8:
        features.add('flexible')
    elif n_rot <= 2:
        features.add('rigid')

    # SMARTS pattern matching (true substructure search)
    compiled = _get_compiled_smarts()
    for pattern_name, pattern_mol in compiled.items():
        if mol.HasSubstructMatch(pattern_mol):
            features.add(pattern_name)

    return frozenset(features)


def _extract_features_simple(smiles: str) -> FrozenSet[str]:
    """Extract features using simple string matching (fallback)."""
    features = set()

    # Halogens
    if 'F' in smiles:
        features.add('F')
    if 'Cl' in smiles:
        features.add('Cl')
    if 'Br' in smiles:
        features.add('Br')
    if 'I' in smiles and 'In' not in smiles:
        features.add('I')

    # Heteroatoms
    if 'S' in smiles or 's' in smiles:
        features.add('S')

    # Nitrogen count
    n_count = smiles.count('N') + smiles.count('n')
    if n_count >= 4:
        features.add('N4+')
    if n_count >= 6:
        features.add('N6+')

    # Oxygen count
    o_count = smiles.count('O') + smiles.count('o')
    if o_count >= 4:
        features.add('O4+')

    # Size categories
    slen = len(smiles)
    if slen < 20:
        features.add('tiny')
    elif slen < 30:
        features.add('small')
    elif slen >= 45:
        features.add('large')

    # Special functional groups
    if 'C(F)(F)F' in smiles or 'CF3' in smiles:
        features.add('trifluoromethyl')
    if 'S(=O)(=O)' in smiles:
        features.add('sulfone')
    if 'C#N' in smiles or 'N#C' in smiles:
        features.add('nitrile')
    if 'N(=O)=O' in smiles or '[N+](=O)[O-]' in smiles:
        features.add('nitro')

    # Simple pattern matching
    for pattern_name, pattern_str in SIMPLE_PATTERNS.items():
        if pattern_str in smiles:
            features.add(pattern_name)

    return frozenset(features)


class RegionMap:
    """
    Maps components to feature regions for a given role.
    Enables stratified sampling by region rarity.
    """

    def __init__(self, db_path: str, role_mask: int, role_name: str = ""):
        self.db_path = db_path
        self.role_mask = role_mask
        self.role_name = role_name

        # Region -> list of mol_ids
        self.region_to_mols: Dict[FrozenSet[str], List[int]] = defaultdict(list)
        # mol_id -> region
        self.mol_to_region: Dict[int, FrozenSet[str]] = {}
        # mol_id -> smiles
        self.mol_to_smiles: Dict[int, str] = {}

        # Region categories
        self.rare_regions: List[FrozenSet[str]] = []      # <1%
        self.uncommon_regions: List[FrozenSet[str]] = []  # 1-5%
        self.common_regions: List[FrozenSet[str]] = []    # 5-20%
        self.dominant_regions: List[FrozenSet[str]] = []  # >20%

        self.total_molecules = 0
        self._build_map()

    def _build_map(self):
        """Build the region map from database."""
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro&immutable=1", uri=True)
        cur = conn.cursor()

        cur.execute(
            f"SELECT mol_id, smiles FROM molecules WHERE role_mask & {self.role_mask} = {self.role_mask}"
        )

        for mol_id, smiles in cur.fetchall():
            region = extract_features(smiles)
            self.region_to_mols[region].append(mol_id)
            self.mol_to_region[mol_id] = region
            self.mol_to_smiles[mol_id] = smiles

        conn.close()

        self.total_molecules = len(self.mol_to_region)

        # Categorize regions by population
        for region, mol_ids in self.region_to_mols.items():
            pct = 100 * len(mol_ids) / self.total_molecules
            if pct < 1:
                self.rare_regions.append(region)
            elif pct < 5:
                self.uncommon_regions.append(region)
            elif pct < 20:
                self.common_regions.append(region)
            else:
                self.dominant_regions.append(region)

        bt.logging.info(
            f"[RegionMap] {self.role_name} (mask {self.role_mask}): "
            f"{self.total_molecules} mols, {len(self.region_to_mols)} regions "
            f"(rare={len(self.rare_regions)}, uncommon={len(self.uncommon_regions)}, "
            f"common={len(self.common_regions)}, dominant={len(self.dominant_regions)})"
        )

    def get_molecules_from_regions(
        self,
        region_list: List[FrozenSet[str]],
        n_samples: int,
        exclude_mol_ids: Set[int] = None
    ) -> List[Tuple[int, str]]:
        """
        Sample molecules from specified regions.
        Returns list of (mol_id, smiles) tuples.
        """
        if exclude_mol_ids is None:
            exclude_mol_ids = set()

        # Collect all eligible molecules from these regions
        eligible = []
        for region in region_list:
            for mol_id in self.region_to_mols.get(region, []):
                if mol_id not in exclude_mol_ids:
                    eligible.append((mol_id, self.mol_to_smiles[mol_id]))

        # Sample
        if len(eligible) <= n_samples:
            return eligible
        return random.sample(eligible, n_samples)

    def get_stratified_sample(
        self,
        n_samples: int,
        rare_pct: float = 0.40,
        uncommon_pct: float = 0.35,
        common_pct: float = 0.25,
        dominant_pct: float = 0.0,
        exclude_mol_ids: Set[int] = None
    ) -> List[Tuple[int, str]]:
        """
        Get stratified sample across region categories.

        Default distribution:
        - 40% from rare regions (<1% of pool)
        - 35% from uncommon regions (1-5%)
        - 25% from common regions (5-20%)
        - 0% from dominant regions (>20%) - EXCLUDED
        """
        if exclude_mol_ids is None:
            exclude_mol_ids = set()

        results = []

        # Sample from each category
        n_rare = int(n_samples * rare_pct)
        n_uncommon = int(n_samples * uncommon_pct)
        n_common = int(n_samples * common_pct)
        n_dominant = int(n_samples * dominant_pct)

        # Rare regions
        if n_rare > 0 and self.rare_regions:
            rare_mols = self.get_molecules_from_regions(
                self.rare_regions, n_rare, exclude_mol_ids
            )
            results.extend(rare_mols)
            bt.logging.debug(f"[RegionMap] Sampled {len(rare_mols)} from rare regions")

        # Uncommon regions
        if n_uncommon > 0 and self.uncommon_regions:
            uncommon_mols = self.get_molecules_from_regions(
                self.uncommon_regions, n_uncommon, exclude_mol_ids
            )
            results.extend(uncommon_mols)
            bt.logging.debug(f"[RegionMap] Sampled {len(uncommon_mols)} from uncommon regions")

        # Common regions
        if n_common > 0 and self.common_regions:
            common_mols = self.get_molecules_from_regions(
                self.common_regions, n_common, exclude_mol_ids
            )
            results.extend(common_mols)
            bt.logging.debug(f"[RegionMap] Sampled {len(common_mols)} from common regions")

        # Dominant regions (usually 0%)
        if n_dominant > 0 and self.dominant_regions:
            dominant_mols = self.get_molecules_from_regions(
                self.dominant_regions, n_dominant, exclude_mol_ids
            )
            results.extend(dominant_mols)

        return results

    def get_anti_dominant_sample(
        self,
        n_samples: int,
        exclude_mol_ids: Set[int] = None
    ) -> List[Tuple[int, str]]:
        """
        Sample from ALL non-dominant regions uniformly.
        Simpler alternative to stratified sampling.
        """
        non_dominant = self.rare_regions + self.uncommon_regions + self.common_regions
        return self.get_molecules_from_regions(non_dominant, n_samples, exclude_mol_ids)

    def get_medchem_prioritized_sample(
        self,
        n_samples: int,
        exclude_mol_ids: Set[int] = None
    ) -> List[Tuple[int, str]]:
        """
        Sample prioritizing regions with high-value medicinal chemistry patterns.
        Focuses on: boron, silicon, alkyne, phosphate, sulfonamide, cyclopropyl
        """
        if exclude_mol_ids is None:
            exclude_mol_ids = set()

        # High-value pattern names (from MEDCHEM_PATTERNS)
        high_value_patterns = {'boron', 'silicon', 'alkyne', 'phosphate', 'sulfonamide', 'cyclopropyl'}
        medium_value_patterns = {'morpholine', 'piperazine', 'piperidine', 'pyrrolidine'}

        # Find regions containing high-value patterns
        high_value_regions = []
        medium_value_regions = []
        other_non_dominant = []

        all_non_dominant = self.rare_regions + self.uncommon_regions + self.common_regions

        for region in all_non_dominant:
            region_patterns = set(region)
            if region_patterns & high_value_patterns:
                high_value_regions.append(region)
            elif region_patterns & medium_value_patterns:
                medium_value_regions.append(region)
            else:
                other_non_dominant.append(region)

        bt.logging.info(
            f"[RegionMap] MedChem regions: high_value={len(high_value_regions)}, "
            f"medium_value={len(medium_value_regions)}, other={len(other_non_dominant)}"
        )

        results = []

        # 50% from high-value pattern regions
        n_high = int(n_samples * 0.50)
        if n_high > 0 and high_value_regions:
            high_mols = self.get_molecules_from_regions(high_value_regions, n_high, exclude_mol_ids)
            results.extend(high_mols)
            bt.logging.info(f"[RegionMap] Sampled {len(high_mols)} from high-value medchem regions")

        # 30% from medium-value pattern regions
        n_medium = int(n_samples * 0.30)
        if n_medium > 0 and medium_value_regions:
            exclude_so_far = exclude_mol_ids.union({m[0] for m in results})
            medium_mols = self.get_molecules_from_regions(medium_value_regions, n_medium, exclude_so_far)
            results.extend(medium_mols)
            bt.logging.info(f"[RegionMap] Sampled {len(medium_mols)} from medium-value medchem regions")

        # 20% from other non-dominant regions
        n_other = n_samples - len(results)
        if n_other > 0 and other_non_dominant:
            exclude_so_far = exclude_mol_ids.union({m[0] for m in results})
            other_mols = self.get_molecules_from_regions(other_non_dominant, n_other, exclude_so_far)
            results.extend(other_mols)

        return results


class RegionBasedSampler:
    """
    Orchestrates region-based sampling for a reaction.
    Handles both 2-component and 3-component reactions.
    """

    def __init__(self, db_path: str, rxn_id: int):
        self.db_path = db_path
        self.rxn_id = rxn_id

        # Get reaction info
        from nova_ph2.combinatorial_db.reactions import get_reaction_info
        reaction_info = get_reaction_info(rxn_id, db_path)
        if not reaction_info:
            raise ValueError(f"Could not get reaction info for rxn:{rxn_id}")

        self.smarts, self.roleA, self.roleB, self.roleC = reaction_info
        self.is_three_component = self.roleC is not None and self.roleC != 0

        # Build region maps for each role
        bt.logging.info(f"[RegionSampler] Building region maps for rxn:{rxn_id}...")
        self.map_A = RegionMap(db_path, self.roleA, "Role A")
        self.map_B = RegionMap(db_path, self.roleB, "Role B")
        self.map_C = None
        if self.is_three_component:
            self.map_C = RegionMap(db_path, self.roleC, "Role C")

        bt.logging.info(f"[RegionSampler] Region maps ready for rxn:{rxn_id}")

    def generate_region_stratified_molecules(
        self,
        n_samples: int,
        rare_pct: float = 0.40,
        uncommon_pct: float = 0.35,
        common_pct: float = 0.25,
        avoid_inchikeys: Set[str] = None,
        subnet_config: dict = None
    ) -> List[Dict]:
        """
        Generate molecules using region-stratified component selection.

        Strategy:
        1. Sample components from each pool using stratified sampling
        2. Combine A × B (× C) to create molecules
        3. Validate and return
        """
        from nova_ph2.combinatorial_db.reactions import get_smiles_from_reaction
        from nova_ph2.utils.molecules import get_heavy_atom_count
        from molecules import generate_inchikey, num_rotatable_bonds

        if avoid_inchikeys is None:
            avoid_inchikeys = set()
        if subnet_config is None:
            subnet_config = {}

        min_heavy = subnet_config.get('min_heavy_atoms', 20)
        min_rot = subnet_config.get('min_rotatable_bonds', 1)
        max_rot = subnet_config.get('max_rotatable_bonds', 10)

        # Calculate how many components to sample per pool
        # Need enough to generate n_samples combinations after validation
        # For 2-comp: sqrt(n_samples * 4) to account for validation losses
        # For 3-comp: cbrt(n_samples * 6) to account for validation losses
        if self.is_three_component:
            n_per_pool = int((n_samples * 6) ** 0.34) + 20
        else:
            n_per_pool = int((n_samples * 4) ** 0.52) + 30

        bt.logging.info(
            f"[RegionSampler] Sampling {n_per_pool} components per pool "
            f"(rare={rare_pct:.0%}, uncommon={uncommon_pct:.0%}, common={common_pct:.0%})"
        )

        # Sample from each pool using stratified sampling
        A_samples = self.map_A.get_stratified_sample(
            n_per_pool, rare_pct, uncommon_pct, common_pct, 0.0
        )
        B_samples = self.map_B.get_stratified_sample(
            n_per_pool, rare_pct, uncommon_pct, common_pct, 0.0
        )

        C_samples = []
        if self.is_three_component:
            C_samples = self.map_C.get_stratified_sample(
                n_per_pool, rare_pct, uncommon_pct, common_pct, 0.0
            )

        bt.logging.info(
            f"[RegionSampler] Sampled A={len(A_samples)}, B={len(B_samples)}, "
            f"C={len(C_samples) if C_samples else 'N/A'}"
        )

        # Generate all combinations
        all_names = []
        if self.is_three_component:
            for a_id, _ in A_samples:
                for b_id, _ in B_samples:
                    for c_id, _ in C_samples:
                        all_names.append(f"rxn:{self.rxn_id}:{a_id}:{b_id}:{c_id}")
        else:
            for a_id, _ in A_samples:
                for b_id, _ in B_samples:
                    all_names.append(f"rxn:{self.rxn_id}:{a_id}:{b_id}")

        bt.logging.info(f"[RegionSampler] Generated {len(all_names)} combinations")

        # Shuffle and limit
        random.shuffle(all_names)
        if len(all_names) > n_samples * 3:
            all_names = all_names[:n_samples * 3]

        # Validate molecules
        valid_molecules = []
        seen_keys = set()

        for name in all_names:
            try:
                smiles = get_smiles_from_reaction(name)
                if not smiles:
                    continue

                # Property checks
                heavy = get_heavy_atom_count(smiles)
                if heavy < min_heavy:
                    continue

                rotatable = num_rotatable_bonds(smiles)
                if rotatable < min_rot or rotatable > max_rot:
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

            except Exception:
                continue

        bt.logging.info(
            f"[RegionSampler] Validated {len(valid_molecules)} molecules from region-stratified sampling"
        )

        return valid_molecules

    def generate_rare_focused_molecules(
        self,
        n_samples: int,
        avoid_inchikeys: Set[str] = None,
        subnet_config: dict = None
    ) -> List[Dict]:
        """
        Generate molecules focusing heavily on rare regions.
        Distribution: 60% rare, 30% uncommon, 10% common, 0% dominant
        """
        return self.generate_region_stratified_molecules(
            n_samples=n_samples,
            rare_pct=0.60,
            uncommon_pct=0.30,
            common_pct=0.10,
            avoid_inchikeys=avoid_inchikeys,
            subnet_config=subnet_config
        )

    def generate_anti_dominant_molecules(
        self,
        n_samples: int,
        avoid_inchikeys: Set[str] = None,
        subnet_config: dict = None
    ) -> List[Dict]:
        """
        Generate molecules excluding dominant regions entirely.
        Uses uniform sampling from non-dominant regions.
        """
        from nova_ph2.combinatorial_db.reactions import get_smiles_from_reaction
        from nova_ph2.utils.molecules import get_heavy_atom_count
        from molecules import generate_inchikey, num_rotatable_bonds

        if avoid_inchikeys is None:
            avoid_inchikeys = set()
        if subnet_config is None:
            subnet_config = {}

        min_heavy = subnet_config.get('min_heavy_atoms', 20)
        min_rot = subnet_config.get('min_rotatable_bonds', 1)
        max_rot = subnet_config.get('max_rotatable_bonds', 10)

        # Sample from non-dominant regions
        # Need more components to ensure enough combinations
        if self.is_three_component:
            n_per_pool = int((n_samples * 6) ** 0.34) + 25
        else:
            n_per_pool = int((n_samples * 4) ** 0.52) + 40

        A_samples = self.map_A.get_anti_dominant_sample(n_per_pool)
        B_samples = self.map_B.get_anti_dominant_sample(n_per_pool)
        C_samples = []
        if self.is_three_component:
            C_samples = self.map_C.get_anti_dominant_sample(n_per_pool)

        bt.logging.info(
            f"[RegionSampler] Anti-dominant: A={len(A_samples)}, B={len(B_samples)}, "
            f"C={len(C_samples) if C_samples else 'N/A'}"
        )

        # Generate combinations
        all_names = []
        if self.is_three_component:
            for a_id, _ in A_samples:
                for b_id, _ in B_samples:
                    for c_id, _ in C_samples:
                        all_names.append(f"rxn:{self.rxn_id}:{a_id}:{b_id}:{c_id}")
        else:
            for a_id, _ in A_samples:
                for b_id, _ in B_samples:
                    all_names.append(f"rxn:{self.rxn_id}:{a_id}:{b_id}")

        random.shuffle(all_names)
        if len(all_names) > n_samples * 3:
            all_names = all_names[:n_samples * 3]

        # Validate
        valid_molecules = []
        seen_keys = set()

        for name in all_names:
            try:
                smiles = get_smiles_from_reaction(name)
                if not smiles:
                    continue

                heavy = get_heavy_atom_count(smiles)
                if heavy < min_heavy:
                    continue

                rotatable = num_rotatable_bonds(smiles)
                if rotatable < min_rot or rotatable > max_rot:
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

            except Exception:
                continue

        bt.logging.info(
            f"[RegionSampler] Generated {len(valid_molecules)} anti-dominant molecules"
        )

        return valid_molecules

    def get_region_stats(self) -> Dict:
        """Get statistics about region distribution."""
        stats = {
            'rxn_id': self.rxn_id,
            'is_three_component': self.is_three_component,
            'pools': {}
        }

        for name, region_map in [('A', self.map_A), ('B', self.map_B), ('C', self.map_C)]:
            if region_map is None:
                continue
            stats['pools'][name] = {
                'total': region_map.total_molecules,
                'n_regions': len(region_map.region_to_mols),
                'n_rare': len(region_map.rare_regions),
                'n_uncommon': len(region_map.uncommon_regions),
                'n_common': len(region_map.common_regions),
                'n_dominant': len(region_map.dominant_regions),
                'dominant_regions': [
                    (set(r), len(region_map.region_to_mols[r]))
                    for r in region_map.dominant_regions
                ]
            }

        return stats
