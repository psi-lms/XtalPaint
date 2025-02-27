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
)
from mattergen.common.data.transform import symmetrize_lattice, set_chemical_system_string

from dbcsi_inpainting.generate_inpainting import generate_reconstructed_structures
import mlflow

np.random.seed(1234)

N_structures = 100
batch_size = 128
N_samples_per_structure = 1

# N_steps = 200
# coordinates_snr = 0.2
# n_corrector_steps = 5
# n_resample_steps = 3


# mc3d_structures_with_H = load_mc3d_with_H()
formulas_to_choose = get_mattergen_unknown_formulas('/data/user/reents_t/projects/mlip/git/diffusion-based-crystal-structure-inpainting/')

with open('/data/user/reents_t/projects/mlip/git/diffusion-based-crystal-structure-inpainting/mc3d_structures_with_H.bz2', 'rb') as f:
    decompressed_data = bz2.decompress(f.read())
structures_json = json.loads(decompressed_data.decode('utf-8'))
structures_w_H = [Structure.from_dict(s) for s in structures_json]

structures_w_H = [
    s for s in structures_w_H if (
        s.num_sites < 30 and s.composition.reduced_formula in formulas_to_choose
        ) #if 20 < len(s.sites) <= 35
]

structures_w_H_subset =[structures_w_H[i] for i in np.random.choice(range(len(structures_w_H)), N_structures, replace=False)]
print(f'Number of unique formulas: {len(set([s.composition.formula for s in structures_w_H_subset]))}')
print(f'Number of structures: {len(structures_w_H_subset)}')

save_structures(
    Path('./initial_structures/'), structures_w_H_subset
)

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
    **structures_H_removed_numpy[0],
    transforms=[symmetrize_lattice, set_chemical_system_string]
)
structures_H_removed_dl = DataLoader(
    structures_H_removed_dataset.repeat(N_samples_per_structure),
    batch_size=batch_size, 
    collate_fn=partial(_collate_fn_w_mask, collate_fn=collate),
    shuffle=False,
)


mlflow.set_tracking_uri("sqlite:////data/user/reents_t/projects/mlip/git/diffusion-based-crystal-structure-inpainting/mlflow.db")
mlflow.set_experiment("parameter-screening")

mlflow.autolog()

# N_steps = 200
# coordinates_snr = 0.2
# n_corrector_steps = 5
# n_resample_steps = 3

for N_steps in [
    # 20, 50, 100, 200, 500#, 
    1000
    ]:
    for coordinates_snr in [0.2, 0.4, 0.6]:
        for n_corrector_steps in [1, 2, 5, 10]:
            for n_resample_steps in [1, 3, 5]:
                if N_steps * n_corrector_steps * n_resample_steps > 5500:
                    continue
                    
                params = {
                    'N_structures': N_structures,
                    'N_steps': N_steps,
                    'n_corrector_steps': n_corrector_steps,
                    'coordinates_snr': coordinates_snr,
                    'batch_size': batch_size,
                    'N_samples_per_structure': N_samples_per_structure,
                    'n_resample_steps': n_resample_steps,
                }
                # Create a unique directory path based on the parameters.
                param_str = '__'.join([f"{key}-{value}" for key, value in params.items()])
                
                results_path = Path(param_str)
                
                if results_path.exists():
                    print(f'...Path {param_str} already exists...')
                    continue
                
                results_path.mkdir(parents=True, exist_ok=True)

                with mlflow.start_run(run_name=param_str) as run:
                    
                    mlflow.log_params(params)
                    
                    structures_wo_H_regenerated = generate_reconstructed_structures(
                        structures_to_reconstruct=structures_H_removed_dl,
                        output_path=results_path,
                        sampling_config_overrides=[
                            f'sampler_partial.N={N_steps}',
                            f'sampler_partial.n_steps_corrector={n_corrector_steps}',
                            f'~sampler_partial.predictor_partials.cell',
                            f'~sampler_partial.corrector_partials.cell',
                            f'~sampler_partial.predictor_partials.atomic_numbers',
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

                    relaxed_wo_H_regen = relax_structure(structures_wo_H_regenerated)
                    relaxed_wo_H_regen = [r[0] for r in zip(*relaxed_wo_H_regen)]

                    if not os.path.exists(results_path / 'structures_relaxed/'):
                        os.makedirs(results_path / 'structures_relaxed/')
                    save_structures(results_path / Path('structures_relaxed/'), relaxed_wo_H_regen)

                    matcher = StructureMatcher()
                    results = []

                    structures_w_H_mapping = {
                        f'{s.composition.alphabetical_formula} {s.volume:.0f}': s for s in structures_w_H_subset
                    }
                    print(f'Len {len(structures_w_H_mapping)}')

                    for i in range(len(structures_wo_H_regenerated)):
                        formula = structures_wo_H_regenerated[i].composition.alphabetical_formula
                        # nsites = structures_wo_H_regenerated[i].num_sites
                        volume = structures_wo_H_regenerated[i].volume
                        # assert structures_w_H_subset[i].composition == structures_wo_H_regenerated[i].composition
                        # assert structures_w_H_subset[i].composition == relaxed_wo_H_regen[i].composition
                        ref_key = f'{formula} {volume:.0f}'
                        ref_strct = structures_w_H_mapping[ref_key]
                        
                        matches = matcher.fit(structures_wo_H_regenerated[i], ref_strct)
                        matches_relaxed = matcher.fit(relaxed_wo_H_regen[i], ref_strct)
                        
                        results.append(
                        (
                            ref_key,
                            structures_wo_H_regenerated[i].composition.iupac_formula,
                            # matcher.fit(structures_H_removed[i], structures_wo_H_regenerated[i]),
                            matches,
                            matches_relaxed
                            # matcher.fit(structures_H_removed[i], relaxed_wo_H_regen[i][0])
                        )
                    )

                    df_results = pd.DataFrame(
                        results, columns=['ref_key', 'composition', 'Matches', 'Matches after relaxation']
                    )

                    df_results.to_csv(Path(param_str) / 'df_results.csv')
                    
                    df_results_sum = df_results.groupby('ref_key').sum()

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
                    
                    print(correctly_matched, 'after relaxation: ', correctly_matched_relaxed)
                    
                    metrics = {
                        'correctly_matched_perc': correctly_matched,
                        'correctly_matched_relaxed_perc': correctly_matched_relaxed
                    }
                    
                    mlflow.log_metrics(metrics)
                        
