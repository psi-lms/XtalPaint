import os
import bz2
import json
from pathlib import Path
from functools import partial
from typing import Any, Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd
import mlflow

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from torch.utils.data import DataLoader
from mattergen.common.data.dataset import CrystalDataset, structures_to_numpy
from mattergen.common.data.collate import collate
from mattergen.common.data.transform import symmetrize_lattice, set_chemical_system_string
from mattergen.common.utils.eval_utils import save_structures
from dbcsi_inpainting.utils import (
    _collate_fn_w_mask, 
    _add_n_sites_to_be_found,
    relax_structure,
    get_mattergen_unknown_formulas,
)
from dbcsi_inpainting.generate_inpainting import generate_reconstructed_structures
from prettytable import PrettyTable
import itertools

GUIDED_PREDICTOR_CORRECTOR_MAPPING: Dict[str, str] = {
    'baseline': 'mattergen.diffusion.sampling.classifier_free_guidance.GuidedPredictorCorrector.from_pl_module',
    'baseline-with-noise': 'dbcsi_inpainting.custom_predictor_corrector.CustomGuidedPredictorCorrector.from_pl_module',
    'repaint-v1': 'dbcsi_inpainting.custom_predictor_corrector.CustomGuidedPredictorCorrectorRePaint.from_pl_module'
}


def load_structures(formulas_to_choose: Iterable[str], N_structures: int) -> List[Structure]:
    """Load and filter structures with hydrogen."""
    input_file = Path(
        "/data/user/reents_t/projects/mlip/git/diffusion-based-crystal-structure-inpainting/mc3d_structures_with_H.bz2"
    )
    with input_file.open('rb') as f:
        decompressed_data = bz2.decompress(f.read())
    structures_json = json.loads(decompressed_data.decode('utf-8'))
    structures_w_H: List[Structure] = [Structure.from_dict(s) for s in structures_json]

    structures_w_H = [
        s for s in structures_w_H if (
            s.num_sites < 30 and s.composition.reduced_formula in formulas_to_choose
        )
    ]
    indices = np.random.choice(range(len(structures_w_H)), N_structures, replace=False)
    subset: List[Structure] = [structures_w_H[i] for i in indices]
    print(f'Number of unique formulas: {len({s.composition.formula for s in subset})}')
    print(f'Number of structures: {len(subset)}')

    # Save initial structures for record keeping
    save_structures(Path('./initial_structures/'), subset)
    return subset


def prepare_dataset(structures_subset: List[Structure]) -> CrystalDataset:
    """Remove H, add missing sites and create a CrystalDataset."""
    structures_H_removed: List[Structure] = []
    for s in structures_subset:
        n_H: int = int(s.composition['H'])
        s_wo_H: Structure = s.copy()
        s_wo_H.remove_species('H')
        s_wo_H = _add_n_sites_to_be_found(s_wo_H, n_H, 'H')
        s_wo_H.properties['material_id'] = s.composition.formula
        structures_H_removed.append(s_wo_H)

    structures_numpy = structures_to_numpy(structures_H_removed)
    dataset = CrystalDataset(
        **structures_numpy[0],
        transforms=[symmetrize_lattice, set_chemical_system_string]
    )
    return dataset


def create_dataloader(dataset: CrystalDataset, N_samples_per_structure: int, batch_size: int) -> DataLoader:
    """Create a dataloader that repeats each sample."""
    return DataLoader(
        dataset.repeat(N_samples_per_structure),
        batch_size=batch_size,
        collate_fn=partial(_collate_fn_w_mask, collate_fn=collate),
        shuffle=False,
    )


def evaluate_results(
    structures_regen: List[Structure],
    relaxed_structures: List[Structure],
    structures_subset: List[Structure],
    output_dir: Path,
    param_str: str
) -> Tuple[float, float]:
    """Evaluate regenerated structures against the originals."""
    matcher = StructureMatcher()
    results: List[Tuple[str, str, Any, Any]] = []
    mapping: Dict[str, Structure] = {
        f'{s.composition.alphabetical_formula} {s.volume:.0f}': s for s in structures_subset
    }
    print(f'Mapping length: {len(mapping)}')

    for i in range(len(structures_regen)):
        formula = structures_regen[i].composition.alphabetical_formula
        volume = structures_regen[i].volume
        ref_key = f'{formula} {volume:.0f}'
        ref_struct = mapping[ref_key]
        matches = matcher.fit(structures_regen[i], ref_struct)
        matches_relaxed = matcher.fit(relaxed_structures[i], ref_struct)
        results.append((ref_key, structures_regen[i].composition.iupac_formula, matches, matches_relaxed))

    df = pd.DataFrame(results, columns=['ref_key', 'composition', 'Matches', 'Matches after relaxation'])
    df.to_csv(output_dir / f'{param_str}_df_results.csv')

    df_sum = df.groupby('ref_key').sum()
    table = PrettyTable(["Formula", "Matches", "Matches after relaxation"])
    for key, row in df_sum.iterrows():
        table.add_row([key, row['Matches'], row['Matches after relaxation']])
    print(table)

    correctly_matched = df_sum['Matches'].gt(0).mean() * 100
    correctly_matched_relaxed = df_sum['Matches after relaxation'].gt(0).mean() * 100
    print(f"{correctly_matched}% correctly matched, after relaxation: {correctly_matched_relaxed}%")
    return correctly_matched, correctly_matched_relaxed


def run_sampling(
    predictor_corrector: str,
    structures_dl: DataLoader,
    params: Dict[str, Any],
    results_path: Path
) -> List[Structure]:
    """
    Run the structure reconstruction sampling using the given parameters.
    This function logs parameters and returns the reconstructed structures.
    """
    mlflow.log_params(params)
    
    sampling_config_overrides = [
            f'sampler_partial.N={params["N_steps"]}',
            f'sampler_partial.n_steps_corrector={params["n_corrector_steps"]}',
            f'~sampler_partial.predictor_partials.cell',
            f'~sampler_partial.corrector_partials.cell',
            f'~sampler_partial.predictor_partials.atomic_numbers',
            f'sampler_partial.corrector_partials.pos.snr={params["coordinates_snr"]}',
            f'sampler_partial._target_={GUIDED_PREDICTOR_CORRECTOR_MAPPING[predictor_corrector]}',
        ]
    if 'n_resample_steps' in params:
        sampling_config_overrides.append(f'sampler_partial.n_resample_steps={params["n_resample_steps"]}')
    
    structures_wo_H_regenerated: List[Structure] = generate_reconstructed_structures(
        structures_to_reconstruct=structures_dl,
        output_path=results_path,
        sampling_config_overrides=sampling_config_overrides,
        config_overrides=[
            'lightning_module.diffusion_module.corruption.discrete_corruptions.atomic_numbers.d3pm.schedule.num_steps=' + str(params["N_steps"]),
            '~lightning_module.diffusion_module.corruption.sdes.cell'
        ]
    )
    return structures_wo_H_regenerated


def evaluate_sampling(
    structures_regen: List[Structure],
    structures_subset: List[Structure],
    results_path: Path,
    param_str: str
) -> float:
    """
    Evaluate the reconstructed (and relaxed) structures against originals.
    Logs result metrics and saves a CSV summary.
    """
    matcher = StructureMatcher()
    results: List[Tuple[str, str, Any]] = []
    # Create a mapping key from the original structures.
    structures_w_H_mapping: Dict[str, Structure] = {
        f'{s.composition.alphabetical_formula} {s.volume:.0f}': s 
        for s in structures_subset
    }
    for i in range(len(structures_regen)):
        structure = structures_regen[i]
        formula = structure.composition.alphabetical_formula
        volume = structure.volume
        ref_key = f'{formula} {volume:.0f}'
        ref_strct = structures_w_H_mapping[ref_key]
        matches = matcher.fit(structures_regen[i], ref_strct)
        results.append((ref_key, structure.composition.iupac_formula, matches))
    
    df_results = pd.DataFrame(results, columns=['ref_key', 'composition', 'Matches'])
    df_results.to_csv(results_path / f'{param_str}_df_results.csv', index=False)
    
    df_results_sum = df_results.groupby('ref_key').sum()
    table = PrettyTable(["Formula", "Matches"])
    for i, row in df_results_sum.iterrows():
        table.add_row([i, row['Matches']])
    print(table)
    
    correctly_matched: float = df_results_sum['Matches'].gt(0).mean() * 100
    print(f"Correctly Matched: {correctly_matched}%")
    return correctly_matched


def run_experiment(
    predictor_corrector: str,
    structures_dl: DataLoader,
    structures_subset: List[Structure],
    params: Dict[str, Any]
) -> None:
    """
    Run a single experiment:
      - Set up a dedicated results directory.
      - Run sampling (with MLflow logging in place).
      - Apply relaxation if needed.
      - Evaluate the regenerated structures.
    """
    param_str: str = '__'.join([f"{key}-{value}" for key, value in params.items()])
    results_path: Path = Path(param_str)
    results_path.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=param_str):
        structures_wo_H_regenerated = run_sampling(
            predictor_corrector, structures_dl, params, results_path
            )
        
        # Optionally, perform relaxation here if necessary:
        relaxed = relax_structure(structures_wo_H_regenerated)
        # Flatten the relaxed structure tuples.
        relaxed_structures: List[Structure] = [r[0] for r in zip(*relaxed)]
        relaxed_dir: Path = results_path / 'structures_relaxed'
        if not os.path.exists(relaxed_dir):
            os.makedirs(relaxed_dir)
        save_structures(relaxed_dir, relaxed_structures)

        metric: float = evaluate_sampling(structures_wo_H_regenerated, structures_subset, results_path, param_str)
        mlflow.log_metrics({'correctly_matched_perc': metric})


def run_all_experiments(
    predictor_corrector: str,
    structures_dl: DataLoader,
    structures_subset: List[Structure],
    param_grid: Dict[str, Any]
) -> None:
    """Loop over parameter grid and run each experiment."""
    # Convert each parameter to a list if it isn't one already.
    grid_params = {
        key: (value if isinstance(value, list) else [value])
        for key, value in param_grid.items()
    }

    # Generate all combinations.
    for combo in itertools.product(*grid_params.values()):
        params = dict(zip(grid_params.keys(), combo))
        
        # Example constraint: if the parameters exist then skip over combinations violating the constraint.
        if all(key in params for key in ['N_steps', 'n_corrector_steps', 'n_resample_steps']):
            if params['N_steps'] * params['n_corrector_steps'] * params['n_resample_steps'] > 5500:
                continue

        run_experiment(predictor_corrector, structures_dl, structures_subset, params)


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
        'n_resample_steps': [1, 3, 5],
        'batch_size': 128,
        'N_samples_per_structure': 1,
    }
    predictor_corrector = 'baseline'
    main(param_grid=param_grid, predictor_corrector=predictor_corrector)

