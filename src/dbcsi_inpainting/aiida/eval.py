from concurrent.futures import ProcessPoolExecutor
from pymatgen.analysis.structure_matcher import StructureMatcher
import pandas as pd
import numpy as np
from tqdm import tqdm
from dbcsi_inpainting.aiida.data import (
    BatchedStructuresData,
    BatchedStructures,
)
from pymatgen.core.structure import Structure
from mattergen.evaluation.utils.utils import compute_rmsd_angstrom
from functools import partial

def _check_for_nan(structure: Structure) -> bool:
    """Check if a pymatgen Structure has NaN values in its atomic positions."""
    positions = structure.cart_coords
    return np.isnan(positions).any()

def _rmsd(strct1, strct2) -> float:
    """Compute RMSD between two structures."""
    return compute_rmsd_angstrom(strct1, strct2)

def _match(strct1, strct2) -> bool:
    """Check if two structures match using StructureMatcher."""
    return bool(matcher.fit(strct1, strct2))

COMPARISON_METHODS = {
    "match": _match,
    "rmsd": _rmsd,
}

def _comparison_per_key(
    key: str,
    metric: str,
) -> bool:
    """
    For a given base key, compare all its inpainted samples
    against the corresponding reference and return True if any match.
    """
    ref = reference_structures[key]

    comparisons = []
    comp_func = COMPARISON_METHODS[metric]
    for sample_idx, sample in inpainted_structures_grouped[key]:
        if _check_for_nan(sample):
            comparison = None
        else:
            comparison = comp_func(sample, ref)
        comparisons.append((sample_idx, comparison))

    return comparisons


def get_structure_keys(
    structures: dict | BatchedStructuresData | BatchedStructures,
) -> set[str]:
    """
    Get the unique keys of the structures with out sample indices.

    :param structures: BatchedStructuresData object containing the inpainted structures.
    :return: Set of unique keys representing the inpainted structures.
    """
    keys = structures.keys()
    structure_keys = []
    sample_indices = []
    for key in keys:
        if "_sample_" in key:
            key, sample_idx = key.split("_sample_")
        else:
            sample_idx = None
        structure_keys.append(key)
        sample_indices.append(sample_idx)

    return structure_keys, sample_indices


def worker_init(ref_structures, inp_structures_grp):
    """Initialize worker."""
    global matcher, reference_structures, inpainted_structures_grouped
    matcher = StructureMatcher()
    reference_structures = ref_structures
    inpainted_structures_grouped = inp_structures_grp


def _parallel_structure_comparison(
    inpainted_structures: dict[str, Structure],
    reference_structures: dict[str, Structure],
    metric: str,
    max_workers: int = 6,
    chunksize: int = 50,
):
    structure_keys, sample_indices = get_structure_keys(inpainted_structures)
    inpainted_structures_grouped = {}
    for (strct_key, sample_idx, inpainted_structures) in zip(
        structure_keys, sample_indices, inpainted_structures.values()
    ):
        inpainted_structures_grouped.setdefault(strct_key, []).append(
            (sample_idx, inpainted_structures)
        )
    #ToDo: Calculate both metrics at once? or at least in one job to avoid uploads
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=worker_init,
        initargs=(reference_structures, inpainted_structures_grouped),
    ) as executor:
        # preserve the initial key order
        metric_agg = {}
        metric_individual = {}

        pbar = tqdm(
            executor.map(partial(_comparison_per_key, metric=metric), structure_keys, chunksize=chunksize),
            total=len(structure_keys),
        )

        for key, metric_value in zip(structure_keys, pbar):
            metric_agg[key] = [m[1] for m in metric_value]

            for sample_idx, match in metric_value:
                if sample_idx is not None:
                    metric_individual[f"{key}_sample_{sample_idx}"] = match
                else:
                    metric_individual[key] = match

    return metric_agg, metric_individual


def evaluate_inpainting(
    inpainted_structures: (
        BatchedStructuresData | BatchedStructures | dict[str, Structure]
    ),
    reference_structures: (
        BatchedStructuresData | BatchedStructures | dict[str, Structure]
    ),
    *,
    metric: str = "match",
    max_workers: int = 6,
    chunksize: int = 50,
) -> dict[str, bool]:
    if set(get_structure_keys(inpainted_structures)[0]) != set(
        reference_structures.keys()
    ):
        raise ValueError(
            "The keys of inpainted structures do not match the keys of reference structures."
        )

    if isinstance(
        inpainted_structures,
        (BatchedStructuresData, BatchedStructures),
    ):
        inpainted_structures = inpainted_structures.get_structures(strct_type="pymatgen")
    if isinstance(
        reference_structures,
        (BatchedStructuresData, BatchedStructures),
    ):
        reference_structures = reference_structures.get_structures(strct_type="pymatgen")

    metric_agg, metric_individual = _parallel_structure_comparison(
        inpainted_structures=inpainted_structures,
        reference_structures=reference_structures,
        metric=metric,
        max_workers=max_workers,
        chunksize=chunksize,
    )

    return metric_agg, metric_individual
