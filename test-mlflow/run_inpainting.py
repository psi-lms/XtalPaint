import numpy as np
import pandas as pd

import os
from pathlib import Path
from mattergen.common.data.dataset import CrystalDataset, structures_to_numpy
from torch.utils.data import DataLoader
from mattergen.common.data.collate import collate
from functools import partial
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure
import bz2
import json
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

N_structures = 40
N_steps = 200
N_samples_per_structure = 1
batch_size = 14
coordinates_snr = 0.2
n_corrector_steps = 10
n_resample_steps = 3


# mc3d_structures_with_H = load_mc3d_with_H()
formulas_to_choose = get_mattergen_unknown_formulas()

# structures_w_H = [
#     s for s in mc3d_structures_with_H if (
#         len(s.sites) < 30 and Composition(s.get_formula()).reduced_formula in formulas_to_choose
#         ) #if 20 < len(s.sites) <= 35
# ]

with open('mc3d_structures_with_H.bz2', 'rb') as f:
    decompressed_data = bz2.decompress(f.read())
structures_json = json.loads(decompressed_data.decode('utf-8'))
structures_w_H = [Structure.from_dict(s) for s in structures_json]

structures_w_H_subset =[s.get_pymatgen_structure() for s in np.random.choice(structures_w_H, N_structures)]
print(f'Number of unique formulas: {len(set([s.composition.reduced_formula for s in structures_w_H_subset]))}')
print(f'Number of structures: {len(structures_w_H_subset)}')

structures_H_removed = []

for s in structures_w_H_subset:
    n_H = int(s.composition['H'])
    s_wo_H = s.copy()
    s_wo_H.remove_species('H')
    
    s_wo_H = _add_n_sites_to_be_found(s_wo_H, n_H, 'H')
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



mlflow.set_tracking_uri("sqlite:////home/reents_t/project/mlip/git/diffusion-based-crystal-structure-inpainting/test-mlflow/mlflow.db")
mlflow.set_experiment("first-experiment")

mlflow.autolog()

with mlflow.start_run(run_name='next-try4') as run:
    params = {
        'N_structures': N_structures,
        'N_steps': N_steps,
        'n_corrector_steps': n_corrector_steps,
        'coordinates_snr': coordinates_snr,
        'batch_size': batch_size,
        'N_samples_per_structure': N_samples_per_structure,
        'n_resample_steps': n_resample_steps,
    }
    
    mlflow.log_params(params)
    
    structures_wo_H_regenerated = generate_reconstructed_structures(
        structures_to_reconstruct=structures_H_removed_dl,
        sampling_config_overrides=[
            f'sampler_partial.N={N_steps}',
            f'sampler_partial.n_steps_corrector={n_corrector_steps}',
            f'~sampler_partial.predictor_partials.cell',
            f'~sampler_partial.corrector_partials.cell',
            f'sampler_partial._target_=dbcsi_inpainting.custom_predictor_corrector.CustomGuidedPredictorCorrectorRePaint.from_pl_module',
            f'sampler_partial.corrector_partials.pos.snr={coordinates_snr}',
            f'+sampler_partial.n_resample_steps={n_resample_steps}'
            ],
        config_overrides=[
            'lightning_module.diffusion_module.corruption.discrete_corruptions'
            f'.atomic_numbers.d3pm.schedule.num_steps={N_steps}',
            '~lightning_module.diffusion_module.corruption.sdes.cell'
            
            ]
    )


    # relaxed_wo_H_regen = [
    #     relax_structure(s)[0][0] for s in structures_wo_H_regenerated
    # ]
    relaxed_wo_H_regen = relax_structure(structures_wo_H_regenerated)
    relaxed_wo_H_regen = [r[0] for r in zip(*relaxed_wo_H_regen)]

    if not os.path.exists('./structures_relaxed/'):
        os.makedirs('./structures_relaxed/')
    save_structures(Path('./structures_relaxed/'), relaxed_wo_H_regen)


    # %%
    matcher = StructureMatcher()
    results = []

    structures_w_H_mapping = {
        s.composition.alphabetical_formula: s for s in structures_w_H_subset
    }


    for i in range(len(relaxed_wo_H_regen)):
        formula = structures_wo_H_regenerated[i].composition.alphabetical_formula
        # assert structures_w_H_subset[i].composition == structures_wo_H_regenerated[i].composition
        # assert structures_w_H_subset[i].composition == relaxed_wo_H_regen[i].composition
        
        ref_strct = structures_w_H_mapping[formula]
        
        matches = matcher.fit(structures_wo_H_regenerated[i], ref_strct)
        matches_relaxed = matcher.fit(relaxed_wo_H_regen[i], ref_strct)
        
        results.append(
        (
            structures_wo_H_regenerated[i].composition.iupac_formula,
            # matcher.fit(structures_H_removed[i], structures_wo_H_regenerated[i]),
            matches,
            matches_relaxed
            # matcher.fit(structures_H_removed[i], relaxed_wo_H_regen[i][0])
        )
    )

    df_results = pd.DataFrame(
        results, columns=['composition', 'Matches', 'Matches after relaxation']
    )

    df_results_sum = df_results.groupby('composition').sum()

    table = PrettyTable(["Formula", "Matches", "Matches after relaxation"])

    for i, row in df_results_sum.iterrows():
        table.add_row([
            i,
            row['Matches'],
            row['Matches after relaxation']
        ]
        )
        
    print(table)
    
    correctly_matched = df_results_sum['Matches'].gt(0).mean() * 100
    correctly_matched_relaxed = df_results_sum['Matches after relaxation'].gt(0).mean() * 100
    
    metrics = {
        'correctly_matched_perc': correctly_matched,
        'correctly_matched_relaxed_perc': correctly_matched_relaxed
    }
    
    mlflow.log_metrics(metrics)
        
