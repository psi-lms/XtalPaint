import numpy as np
from typing import Any, Dict
from dbcsi_inpainting.data_utils import load_structures, prepare_dataset, create_dataloader
from dbcsi_inpainting.sampling import run_all_experiments  # run_all_experiments can be defined in sampling.py or a separate module
from dbcsi_inpainting.utils import get_mattergen_unknown_formulas
from pymatgen.core import Structure

def main(param_grid: Dict[str, Any], predictor_corrector: str, fix_cell: bool = True, relax_kwargs: Dict[str, Any] | None = None, max_num_atoms: int = 30) -> None:
    np.random.seed(1234)
    formulas_to_choose = get_mattergen_unknown_formulas(
        '/data/user/reents_t/projects/mlip/git/diffusion-based-crystal-structure-inpainting/'
    )
    N_structures = param_grid['N_structures']
    structures_subset: list[Structure] = load_structures(formulas_to_choose, N_structures, max_num_atoms)
    dataset = prepare_dataset(structures_subset)
    N_samples_per_structure = param_grid['N_samples_per_structure']
    batch_size = param_grid['batch_size']
    structures_dl = create_dataloader(dataset, N_samples_per_structure, batch_size)
    # run_all_experiments would iterate over combinations and call run_experiment for each.
    run_all_experiments(predictor_corrector, structures_dl, structures_subset, param_grid, fix_cell, relaxation_kwargs=relax_kwargs)

if __name__ == '__main__':
    param_grid_repaint: Dict[str, Any] = {
        'N_structures': 250,
        'N_steps': [200],
        'coordinates_snr': [0.2],
        'n_corrector_steps': [5],
        'n_resample_steps': [3],
        'jump_length': [10],
        'batch_size': 512,
        'N_samples_per_structure': 1,
    }
    predictor_corrector = 'repaint-v2'
    
    print('\n\nRun repaint-v2 up to 20 atoms\n\n')
    main(param_grid=param_grid_repaint, predictor_corrector=predictor_corrector, max_num_atoms=20)
    print('\n\nRun repaint-v2 up to 100 atoms\n\n')
    main(param_grid=param_grid_repaint, predictor_corrector=predictor_corrector, max_num_atoms=100)
    
    predictor_corrector = 'baseline'
    param_grid_baseline: Dict[str, Any] = {
        'N_structures': 250,
        'N_steps': [200],
        'coordinates_snr': [0.2],
        'n_corrector_steps': [5],
        # 'n_resample_steps': [3],
        # 'jump_length': [10],
        'batch_size': 512,
        'N_samples_per_structure': 1,
    }
    
    print('\n\nRun baseline up to 20 atoms\n\n')
    main(param_grid=param_grid_baseline, predictor_corrector=predictor_corrector, max_num_atoms=20)
    print('\n\nRun baseline up to 100 atoms\n\n')
    main(param_grid=param_grid_baseline, predictor_corrector=predictor_corrector, max_num_atoms=100)