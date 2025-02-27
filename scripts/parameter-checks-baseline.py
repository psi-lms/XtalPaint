import numpy as np
from typing import Any, Dict
from dbcsi_inpainting.data_utils import load_structures, prepare_dataset, create_dataloader
from dbcsi_inpainting.sampling import run_all_experiments  # run_all_experiments can be defined in sampling.py or a separate module
from dbcsi_inpainting.utils import get_mattergen_unknown_formulas
from pymatgen.core import Structure

def main(param_grid: Dict[str, Any], predictor_corrector: str) -> None:
    np.random.seed(1234)
    formulas_to_choose = get_mattergen_unknown_formulas(
        '/data/user/reents_t/projects/mlip/git/diffusion-based-crystal-structure-inpainting/'
    )
    N_structures = param_grid['N_structures']
    structures_subset: list[Structure] = load_structures(formulas_to_choose, N_structures)
    dataset = prepare_dataset(structures_subset)
    N_samples_per_structure = param_grid['N_samples_per_structure']
    batch_size = param_grid['batch_size']
    structures_dl = create_dataloader(dataset, N_samples_per_structure, batch_size)
    # run_all_experiments would iterate over combinations and call run_experiment for each.
    run_all_experiments(predictor_corrector, structures_dl, structures_subset, param_grid)

if __name__ == '__main__':
    param_grid: Dict[str, Any] = {
        'N_structures': 100,
        'N_steps': [20, 50, 100, 200, 500, 1000],
        'coordinates_snr': [0.2, 0.4, 0.6],
        'n_corrector_steps': [1, 2, 5, 10],
        # 'n_resample_steps': [1, 3, 5],
        'batch_size': 128,
        'N_samples_per_structure': 1,
    }
    predictor_corrector = 'baseline'
    predictor_corrector = 'baseline-with-noise'
    main(param_grid=param_grid, predictor_corrector=predictor_corrector)