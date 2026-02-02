"""Module to evaluate inpainted structures against reference structures."""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, TypeAlias, Union

import numpy as np
import pandas as pd
from mattergen.evaluation.utils.utils import compute_rmsd_angstrom
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from tqdm import tqdm

from xtalpaint.data import BatchedStructures
from xtalpaint.utils import _is_batched_structure

if TYPE_CHECKING:
    from xtalpaint.aiida.data import BatchedStructuresData

StructureInput: TypeAlias = Union[
    "BatchedStructuresData", BatchedStructures, dict[str, Structure]
]


def _check_for_nan(structure: Structure) -> bool:
    """Check if a pymatgen Structure has NaN values in its atomic positions."""
    positions = structure.cart_coords
    return np.isnan(positions).any()


def _rmsd(strct1, strct2, normalization_element: str | None = None) -> float:
    """Compute RMSD between two structures.

    If normalization_element is provided, normalize by the number of
    atoms of that element, instead of total atoms.
    """
    rmsd = compute_rmsd_angstrom(strct1, strct2)
    if normalization_element:
        total_atoms = len(strct1)
        n_elem_atoms = strct1.composition[normalization_element]
        if n_elem_atoms == 0:
            return float("inf")
        rmsd *= (total_atoms / n_elem_atoms) ** 0.5

    return rmsd


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
    **kwargs,
) -> bool:
    """For a given base key, compare all its inpainted samples."""
    ref = reference_structures[key]

    comparisons = []
    comp_func = COMPARISON_METHODS[metric]
    for sample_idx, sample in inpainted_structures_grouped[key]:
        if _check_for_nan(sample):
            comparison = None
        else:
            comparison = comp_func(sample, ref, **kwargs)
        comparisons.append((sample_idx, comparison))

    return comparisons


def get_structure_keys(
    structures: StructureInput,
) -> tuple[list[str], list[str | None]]:
    """Get the unique keys of the structures with out sample indices.

    This is used to group structures that are samples of the same
    base structure.

    Args:
        structures (dict | BatchedStructuresData | BatchedStructures):
            The structures to get the keys from.

    Returns:
        set[str]: The unique structure keys.
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
    **comparison_kwargs,
):
    structure_keys, sample_indices = get_structure_keys(inpainted_structures)

    inpainted_structures_grouped = {}
    for strct_key, sample_idx, inpainted_structures in zip(
        structure_keys, sample_indices, inpainted_structures.values()
    ):
        inpainted_structures_grouped.setdefault(strct_key, []).append(
            (sample_idx, inpainted_structures)
        )

    grouped_keys = list(inpainted_structures_grouped.keys())
    # ToDo: Calculate both metrics at once? or at least in one job to avoid
    # uploads
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=worker_init,
        initargs=(reference_structures, inpainted_structures_grouped),
    ) as executor:
        metric_individual = {}

        pbar = tqdm(
            executor.map(
                partial(
                    _comparison_per_key, metric=metric, **comparison_kwargs
                ),
                grouped_keys,
                chunksize=chunksize,
            ),
            total=len(grouped_keys),
        )

        for key, metric_value in zip(grouped_keys, pbar):
            for sample_idx, match in metric_value:
                if sample_idx is not None:
                    metric_individual[f"{key}_sample_{sample_idx}"] = match
                else:
                    metric_individual[key] = match

    return metric_individual


def evaluate_inpainting(
    inpainted_structures: StructureInput,
    reference_structures: StructureInput,
    *,
    metric: str = "match",
    max_workers: int = 6,
    chunksize: int = 50,
    **comparison_kwargs,
) -> pd.DataFrame:
    """Evaluate inpainting by comparing inpainted structures with references.

    Args:
        inpainted_structures:
            The inpainted structures to evaluate.
        reference_structures:
            The reference structures to compare against.
        metric (str, optional): The metric to use for comparison.
            Defaults to "match". Either "match" or "rmsd".
        max_workers (int, optional): The maximum number of worker processes to
            use. Defaults to 6.
        chunksize (int, optional): The chunk size for processing.
            Defaults to 50.
        **comparison_kwargs: Additional keyword arguments for the comparison
            function.

    Raises:
        ValueError: If the keys of inpainted structures do not match the keys
            of reference structures.

    Returns:
        dict[str, bool]: A dictionary with the aggregated and individual
            metric results.
    """
    if set(get_structure_keys(inpainted_structures)[0]) != set(
        reference_structures.keys()
    ):
        raise ValueError(
            "The keys of inpainted structures do not match the keys of "
            "reference structures."
        )

    if _is_batched_structure(inpainted_structures):
        inpainted_structures = inpainted_structures.get_structures(
            strct_type="pymatgen"
        )
    if _is_batched_structure(reference_structures):
        reference_structures = reference_structures.get_structures(
            strct_type="pymatgen"
        )

    metric_results = _parallel_structure_comparison(
        inpainted_structures=inpainted_structures,
        reference_structures=reference_structures,
        metric=metric,
        max_workers=max_workers,
        chunksize=chunksize,
        **comparison_kwargs,
    )

    return pd.DataFrame(
        metric_results.items(), columns=["keys", metric]
    ).set_index("keys")
