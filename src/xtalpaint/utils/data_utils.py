"""Utility functions for data handling in XtalPaint."""

from functools import partial
from typing import Callable, Sequence

import torch
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from mattergen.common.data.dataset import CrystalDataset
from mattergen.diffusion.data.batched_data import BatchedData
from torch.utils.data import DataLoader


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
