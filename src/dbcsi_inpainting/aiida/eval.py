from concurrent.futures import ProcessPoolExecutor
from pymatgen.analysis.structure_matcher import StructureMatcher
import pandas as pd
from tqdm import tqdm
from dbcsi_inpainting.aiida.data import BatchedStructuresData, BatchedStructures
from pymatgen.core.structure import Structure

def get_structure_keys(
    structures: dict | BatchedStructuresData | BatchedStructures,
) -> set[str]:
    """
    Get the unique keys of the structures with out sample indices.
    
    :param structures: BatchedStructuresData object containing the inpainted structures.
    :return: Set of unique keys representing the inpainted structures.
    """
    keys = structures.keys if isinstance(structures, (BatchedStructuresData, BatchedStructures)) else structures.keys()
    structure_keys = []
    for key in keys:
        key = key.split('_sample_')[0]  # Extract the base key before any sample index
        structure_keys.append(key)
    return structure_keys

def worker_init(ref_structures, inp_structures_grp):
    """Initialize worker."""
    global matcher, reference_structures, inpainted_structures_grouped
    matcher = StructureMatcher()
    reference_structures = ref_structures
    inpainted_structures_grouped = inp_structures_grp

def _compare_key(key: str) -> bool:
    """
    For a given base key, compare all its inpainted samples
    against the corresponding reference and return True if any match.
    """
    ref = reference_structures[key]
    return [matcher.fit(sample, ref) for sample in inpainted_structures_grouped[key]]

def _parallel_structure_comparison(
    inpainted_structures: dict[str, Structure],
    reference_structures: dict[str, Structure],
):
    structure_keys = get_structure_keys(inpainted_structures)
    inpainted_structures_grouped = {}
    for strct_key, inpainted_structures in zip(
        structure_keys, inpainted_structures.values()
    ):
        inpainted_structures_grouped.setdefault(strct_key, []).append(inpainted_structures)
    

    with ProcessPoolExecutor(
        max_workers=6,
        initializer=worker_init,
        initargs=(reference_structures, inpainted_structures_grouped),
        ) as executor:
        # preserve the initial key order
        match_flags = {}
        pbar = tqdm(executor.map(_compare_key, structure_keys, chunksize=50), total=len(structure_keys))
        for key, match_flag in zip(structure_keys, pbar):
            match_flags[key] = match_flag

    return match_flags


def evaluate_inpainting(
    inpainted_structures: BatchedStructuresData,
    reference_structures: dict[str, Structure],
):
    if set(get_structure_keys(inpainted_structures)) != set(reference_structures.keys()):
        raise ValueError(
            "The keys of inpainted structures do not match the keys of reference structures."
        )

    compared = _parallel_structure_comparison(
        inpainted_structures.get_structures(strct_type='pymatgen'),
        reference_structures,
    )
    
    return compared