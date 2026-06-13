"""Utility functions for data handling in XtalPaint."""

from functools import partial
from typing import Callable, Sequence

import torch
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from mattergen.common.data.dataset import CrystalDataset
from mattergen.diffusion.data.batched_data import BatchedData
from pymatgen.core.structure import Structure
from torch.utils.data import DataLoader

from xtalpaint.data import BatchedStructures


def get_structure_keys(
    structures: BatchedStructures | dict[str, Structure],
) -> tuple[list[str], list[str | None]]:
    """Get the unique keys of the structures with out sample indices.

    This is used to group structures that are samples of the same
    base structure. Example keys are ``mp-1234_sample_0``,
    ``mp-1234_sample_1``, etc. This function will return ``mp-1234`` as the
    unique key for both of these, and the sample indices as ``0`` and ``1``
    respectively. If a key does not have a ``_sample_`` suffix, it is returned
    as-is with a sample index of ``None``.

    Args:
        structures (dict | BatchedStructures):
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


def create_dataloader(
    dataset: CrystalDataset, batch_size: int, fix_cell: bool = True
) -> DataLoader:
    """Create a dataloader that repeats each sample."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(
            _collate_fn_w_mask, collate_fn=collate, fix_cell=fix_cell
        ),
        shuffle=False,
    )


def _collate_fn_w_mask(
    batch: Sequence[ChemGraph],
    collate_fn: Callable[[Sequence[ChemGraph]], BatchedData],
    fix_cell: bool = True,
) -> tuple[BatchedData, None]:
    """Collate a batch of ChemGraphs and add a mask for missing positions."""
    batch = collate_fn(batch)
    nan_pos = torch.isnan(batch.pos).any(dim=1)

    mask = torch.ones_like(batch.pos, dtype=torch.float)
    mask[nan_pos] = 0
    batch["pos"] = torch.nan_to_num(batch["pos"])

    mask_dict = {"pos": mask}
    if fix_cell:
        mask_dict["cell"] = torch.ones_like(batch.cell, dtype=torch.float)

    return batch, mask_dict
