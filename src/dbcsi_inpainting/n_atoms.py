import numpy as np
import pandas as pd

from pymatgen.core import Composition
import os
from pathlib import Path
from mattergen.common.data.dataset import CrystalDataset, structures_to_numpy
from torch.utils.data import DataLoader
from mattergen.common.data.collate import collate
from functools import partial
from pymatgen.analysis.structure_matcher import StructureMatcher
from prettytable import PrettyTable
from mattergen.common.utils.eval_utils import save_structures
from dbcsi_inpainting.utils import (
    _collate_fn_w_mask, 
    _add_n_sites_to_be_found,
    relax_structure,
    get_mattergen_unknown_formulas,
    load_mc3d_with_H
)

from dbcsi_inpainting.generate_inpainting import generate_reconstructed_structures
import mlflow



np.random.seed(1234)

N_structures = 20
N_steps = 20
N_samples_per_structure = 1
batch_size = 12


mc3d_structures_with_H = load_mc3d_with_H()
formulas_to_choose = get_mattergen_unknown_formulas()

structures_w_H = [
    s for s in mc3d_structures_with_H if (
        len(s.sites) < 30 and Composition(s.get_formula()).reduced_formula in formulas_to_choose
        ) #if 20 < len(s.sites) <= 35
]

structures_w_H_subset =[s.get_pymatgen_structure() for s in np.random.choice(structures_w_H, N_structures)]
print(f'Number of unique formulas: {len(set([s.composition.reduced_formula for s in structures_w_H_subset]))}')
print(f'Number of structures: {len(structures_w_H_subset)}')



structures_H_removed = []

for s in structures_w_H_subset:
    n_H = int(s.composition['H'])
    for i_n_H in range(1, n_H+5):
        s_wo_H = s.copy()
        s_wo_H.remove_species('H')
        
        s_wo_H = _add_n_sites_to_be_found(s_wo_H, i_n_H, 'H')
        s_wo_H.properties['material_id'] = s.composition.formula
        
        structures_H_removed.append(s_wo_H)
    
structures_H_removed_numpy = structures_to_numpy(structures_H_removed)
structures_H_removed_dataset = CrystalDataset(
    **structures_H_removed_numpy[0]
)
structures_H_removed_dl = DataLoader(
    structures_H_removed_dataset.repeat(N_samples_per_structure),
    batch_size=batch_size, 
    collate_fn=partial(_collate_fn_w_mask, collate_fn=collate),
    shuffle=False,
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("n-missing-sites")

mlflow.autolog()

n_corrector_steps = 1
with mlflow.start_run() as run:
    params = {
        'N_structures': N_structures,
        'N_steps': N_steps,
        'n_corrector_steps': n_corrector_steps
        
    }
    
    mlflow.log_params(params)
    
    structures_wo_H_regenerated = generate_reconstructed_structures(
        structures_to_reconstruct=structures_H_removed_dl,
        sampling_config_overrides=[
            f'sampler_partial.N={N_steps}',
            f'sampler_partial.n_steps_corrector={n_corrector_steps}',
            f'~sampler_partial.predictor_partials.cell',
            f'~sampler_partial.corrector_partials.cell',
            ],
        config_overrides=[
            'lightning_module.diffusion_module.corruption.discrete_corruptions'
            f'.atomic_numbers.d3pm.schedule.num_steps={N_steps}',
            
            ]
    )



    # relaxed_wo_H_regen = relax_structure(structures_wo_H_regenerated)
    # relaxed_wo_H_regen = [r[0] for r in zip(*relaxed_wo_H_regen)]

    # if not os.path.exists('./structures_relaxed/'):
    #     os.makedirs('./structures_relaxed/')
    # save_structures(Path('./structures_relaxed/'), relaxed_wo_H_regen)

