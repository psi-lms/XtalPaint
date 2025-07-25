"""This module provides functions to generate inpainting candidates for crystal structures"""

from typing import Dict, Iterable, List, Tuple, Union
import numpy as np
from pymatgen.core import Structure
from aiida.orm import StructureData
from aiida_workgraph import task
from dbcsi_inpainting.aiida.data import BatchedStructures


def _add_inpainting_sites(structure, n_sites, element):
    """Add n_sites sites with the element to the structure."""
    structure = structure.copy()
    for _ in range(n_sites):
        structure.append(element, np.full(3, fill_value=np.nan))
    return structure


def _structures_to_pymatgen(
    structures: Union[
        List[Union[Structure, StructureData]],
        Dict[str, Union[Structure, StructureData]],
    ],
) -> List[Structure]:
    """
    Convert a list or dictionary of structures to a list of pymatgen Structure objects.

    Args:
        structures: List or dictionary of pymatgen Structure objects or AiiDA StructureData objects.

    Returns:
        List of pymatgen Structure objects.
    """
    if isinstance(structures, list):
        structures = {f"{i}": s for i, s in enumerate(structures)}

    if isinstance(structures, dict):
        if isinstance(list(structures.values())[0], Structure):
            return structures
        elif isinstance(list(structures.values())[0], StructureData):
            structures_pmt = {}
            for key, s in structures.items():
                s_pmt = s.get_pymatgen_structure()
                s_pmt.properties["uuid"] = s.uuid
                structures_pmt[key] = s_pmt
            return structures_pmt
    else:
        raise TypeError("Input must be a list or dictionary of pymatgen Structure objects.")

def _prepare_inpainting_inputs(
    structures: Union[Structure, Iterable[Structure], Dict[str, Structure]],
    n_inp: Union[int, Tuple[int, int], List[int], List[Tuple[int, int]]],
    element: Union[str, List[str]],
):
    if not isinstance(structures, (list, dict)):
        structures = {'0': structures}

    structures = _structures_to_pymatgen(structures)

    if isinstance(element, str):
        element = {key: element for key in structures.keys()}
    if isinstance(n_inp, int):
        n_inp = {key: n_inp for key in structures.keys()}

    if not all(
        [len(n) == 2 if isinstance(n, list) else isinstance(n, int) for n in n_inp.values()]
    ):
        raise ValueError("n_inp must be an int or a list of two ints (start, end)")

    return structures, n_inp, element

def structure_to_inpainting_candidates(
    structure: Structure, strct_key: str, num_inpaint_sites: Union[int, Tuple], element: str, num_samples: int = 1
) -> List[Structure]:

    if isinstance(num_inpaint_sites, tuple) and not len(num_inpaint_sites) == 2:
        raise ValueError(
            "num_inpaint_sites must be an int or a tuple of two ints (start, end)"
        )

    structures_sites_removed = {}
    if isinstance(num_inpaint_sites, int):
        num_inpaint_sites = (num_inpaint_sites, num_inpaint_sites)

    for i_sample in range(num_samples):
        for j in range(num_inpaint_sites[0], num_inpaint_sites[1] + 1):
            s_removed = structure.copy()
            s_removed.remove_species(element)
            s_removed = _add_inpainting_sites(s_removed, j, element)

            label = strct_key
            if num_inpaint_sites[0] != num_inpaint_sites[1]:
                label += f"_n_inp_{j}"
                
            if num_samples > 1:
                label += f"_sample_{i_sample}"

            s_removed.properties["material_id"] = label
            
            structures_sites_removed[label] = s_removed

    return structures_sites_removed


def generate_inpainting_candidates(
    structures: Union[Structure, Iterable[Structure]],
    n_inp: Union[int, Tuple[int, int], List[int], List[Tuple[int, int]]],
    element: Union[str, List[str]],
    num_samples: int = 1,
) -> List[Structure]:
    """
    Generate inpainting candidates for a list of structures by removing the specified element and adding
    a variable number of inpainting sites.

    Args:
        structures: Iterable of pymatgen Structure objects.
        n_inp: Number of inpainting sites to add, can be an int, a tuple (start, end), or a list of ints or tuples.
        element: Element to be removed and replaced with inpainting sites.

    Returns:
        List of pymatgen Structure objects with inpainting candidates.
    """
    structures, n_inp, element = _prepare_inpainting_inputs(
        structures=structures, n_inp=n_inp, element=element
    )

    candidates = {}
    for key in structures:
        candidates.update(
            structure_to_inpainting_candidates(
                structures[key], key, n_inp[key], element[key], num_samples
                )
            )

    return candidates

@task(
    inputs=[
        {
            "name": "structures",
            "identifier": "workgraph.namespace",
            "metadata": {"dynamic": True},
        }
    ],
    outputs=[
        {
            "name": "candidates",
            # "identifier": "workgraph.namespace",
            # "metadata": {"dynamic": True},
        }
    ],
)
def _aiida_generate_inpainting_candidates(
    structures: Union[Structure, Iterable[Structure]],
    n_inp: Union[int, Tuple[int, int], List[int], List[Tuple[int, int]]],
    element: Union[str, List[str]],
    num_samples: int = 1,
) -> BatchedStructures:

    candidates = generate_inpainting_candidates(
        structures=structures, n_inp=n_inp, element=element, num_samples=num_samples
    )
    return {
        "candidates": BatchedStructures(candidates)
    }