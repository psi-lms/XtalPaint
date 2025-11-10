from pathlib import Path
from typing import Iterable, List
import bz2
import json
import numpy as np
from pymatgen.core import Structure
from mattergen.common.data.dataset import CrystalDataset, structures_to_numpy
from mattergen.common.data.transform import (
    symmetrize_lattice,
    set_chemical_system_string,
)
from mattergen.common.utils.eval_utils import save_structures
from mattergen.common.data.collate import collate
from torch.utils.data import DataLoader
from functools import partial

from dbcsi_inpainting.utils.misc import _collate_fn_w_mask
from dbcsi_inpainting.utils.misc import _add_n_sites_to_be_found


def load_structures(
    formulas_to_choose: Iterable[str],
    N_structures: int,
    max_num_atoms: int = 30,
    save_prefix: str = "",
) -> List[Structure]:
    """Load and filter structures with hydrogen."""
    input_file = Path(
        "/data/user/reents_t/projects/mlip/git/diffusion-based-crystal-structure-inpainting/mc3d_structures_with_H.bz2"
    )
    with input_file.open("rb") as f:
        decompressed_data = bz2.decompress(f.read())
    structures_json = json.loads(decompressed_data.decode("utf-8"))
    structures_w_H: List[Structure] = [
        Structure.from_dict(s) for s in structures_json
    ]

    if formulas_to_choose:
        structures_w_H = [
            s
            for s in structures_w_H
            if (
                s.num_sites < max_num_atoms
                and s.composition.reduced_formula in formulas_to_choose
            )
        ]
    if N_structures == -1:
        indices = list(range(len(structures_w_H)))
    else:
        indices = np.random.choice(
            range(len(structures_w_H)), N_structures, replace=False
        )
    subset: List[Structure] = [structures_w_H[i] for i in indices]
    print(
        f"Number of unique formulas: {len({s.composition.formula for s in subset})}"
    )
    print(f"Number of structures: {len(subset)}")

    save_dir = Path(f"./{save_prefix}_initial_structures/")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_structures(save_dir, subset)
    return subset


def prepare_dataset(structures_subset: List[Structure]) -> CrystalDataset:
    """Remove H, add missing sites and create a CrystalDataset."""
    structures_H_removed: List[Structure] = []
    for i, s in enumerate(structures_subset):
        n_H: int = int(s.composition["H"])
        s_wo_H: Structure = s.copy()
        s_wo_H.remove_species("H")
        s_wo_H = _add_n_sites_to_be_found(s_wo_H, n_H, "H")
        s_wo_H.properties["material_id"] = i  # s.composition.formula
        structures_H_removed.append(s_wo_H)

    structures_numpy, properties = structures_to_numpy(structures_H_removed)
    properties["structure_id"] = np.arange(len(structures_subset))
    dataset = CrystalDataset(
        **structures_numpy,
        properties=properties,
        transforms=[symmetrize_lattice, set_chemical_system_string],
    )
    return dataset


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
