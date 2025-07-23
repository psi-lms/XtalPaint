import numpy as np
from typing import Callable, Sequence
from pymatgen.core import Structure, Composition
import torch
import json


from mattergen.diffusion.data.batched_data import BatchedData
# from mattergen.evaluation.utils.relaxation import relax_structures
from dbcsi_inpainting.utils.relaxation_utils import relax_structures
from mattergen.common.utils.globals import get_device
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.utils.globals import get_device


def _collate_fn_w_mask(
    batch: Sequence[ChemGraph],
    collate_fn: Callable[[Sequence[ChemGraph]], BatchedData],
    fix_cell: bool = True
) -> tuple[BatchedData, None]:
    """Collate a batch of ChemGraphs and add a mask for missing positions."""
    batch = collate_fn(batch)
    nan_pos = torch.isnan(batch.pos).any(dim=1)

    mask = torch.ones_like(batch.pos, dtype=torch.float)
    mask[nan_pos] = 0
    batch['pos'] = torch.nan_to_num(batch['pos'])
    
    mask_dict = {
        'pos': mask
    }
    if fix_cell:
        mask_dict['cell'] =  torch.ones_like(batch.cell, dtype=torch.float)
    

    return batch, mask_dict


def relax_structure(
    structures: Structure,
    device: str = str(get_device()),
    load_path: str | None = None,
    **kwargs
) -> tuple[Structure, float]:
    """
    Relax a single pymatgen Structure using mattersim and return the relaxed
    structure and its total energy.

    Args:
        structure (Structure): The structure to relax.
        device (str): The device to run the relaxation on.
        load_path (str, optional): Path to potential weights if needed.
        **kwargs: Additional arguments for relaxation.

    Returns:
        tuple[Structure, float]: The relaxed structure and its total energy.
    """
    # relax_structures expects a list of Structure objects.
    relaxed_structures, energies = relax_structures(
        structures=structures, device=device, load_path=load_path, **kwargs
    )
    return relaxed_structures, energies


def _add_n_sites_to_be_found(structure, n_sites, element):
    """Add n_sites sites with the element to the structure."""
    structure = structure.copy()
    for _ in range(n_sites):
        structure.append(element, np.full(3, fill_value=np.nan))
    return structure

def load_mc3d_with_H():
    from aiida import orm, load_profile

    load_profile()

    query_structures_w_H = orm.QueryBuilder().append(
        orm.Group, filters={'label': 'mc3d-structures-with-H'}, tag='group'
    ).append(
        orm.StructureData, with_group='group'
    )
    
    return query_structures_w_H.all(flat=True)

def get_mattergen_unknown_formulas(path):
    """Get the formulas that are not in the training or validation set of MatterGen."""
    with open(f'{path}/formula_not_in_train.json', 'r') as f:
        formula_not_in_train = json.load(f)

    with open(f'{path}/formula_not_in_val.json', 'r') as f:
        formula_not_in_val = json.load(f)

    formulas_to_choose = set(formula_not_in_train).union(set(formula_not_in_val))
    
    return formulas_to_choose
