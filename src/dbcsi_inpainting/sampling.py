import os
from pathlib import Path
import itertools
from typing import Any, Dict, List
import mlflow
from torch.utils.data import DataLoader
from pymatgen.core import Structure
from dbcsi_inpainting.generate_inpainting import (
    generate_reconstructed_structures,
)
from dbcsi_inpainting.utils.misc import relax_structure
from mattergen.common.utils.eval_utils import save_structures
from dbcsi_inpainting.evaluation import (
    evaluate_results,
)  # Assuming you created an evaluation.py module

GUIDED_PREDICTOR_CORRECTOR_MAPPING: Dict[str, str] = {
    "baseline": "mattergen.diffusion.sampling.classifier_free_guidance.GuidedPredictorCorrector.from_pl_module",
    "baseline-reverted-order": "dbcsi_inpainting.custom_predictor_corrector.GuidedPredictorCorrectorRevertedOrder.from_pl_module",
    "baseline-with-noise": "dbcsi_inpainting.custom_predictor_corrector.CustomGuidedPredictorCorrector.from_pl_module",
    "repaint-v1": "dbcsi_inpainting.custom_predictor_corrector.CustomGuidedPredictorCorrectorRePaint.from_pl_module",
    "repaint-v2": "dbcsi_inpainting.custom_predictor_corrector.CustomGuidedPredictorCorrectorRePaintV2.from_pl_module",
}


def run_sampling(
    predictor_corrector: str,
    structures_dl: DataLoader,
    params: Dict[str, Any],
    results_path: Path,
    fix_cell: bool = True,
) -> List[Structure]:
    """
    Run the structure reconstruction sampling using the given parameters.
    Logs parameters with MLflow and returns the reconstructed structures.
    """
    mlflow.log_params(params)
    sampling_config_overrides = [
        f"sampler_partial.N={params['N_steps']}",
        f"sampler_partial.n_steps_corrector={params['n_corrector_steps']}",
        "~sampler_partial.predictor_partials.atomic_numbers",
        f"sampler_partial.corrector_partials.pos.snr={params['coordinates_snr']}",
        f"sampler_partial._target_={GUIDED_PREDICTOR_CORRECTOR_MAPPING[predictor_corrector]}",
    ]
    config_overrides = [
        "lightning_module.diffusion_module.corruption.discrete_corruptions.atomic_numbers.d3pm.schedule.num_steps="
        + str(params["N_steps"]),
    ]

    if fix_cell:
        sampling_config_overrides.extend(
            [
                "~sampler_partial.predictor_partials.cell",
                "~sampler_partial.corrector_partials.cell",
            ]
        )
        config_overrides.append(
            "~lightning_module.diffusion_module.corruption.sdes.cell"
        )

    if "n_resample_steps" in params:
        sampling_config_overrides.append(
            f"+sampler_partial.n_resample_steps={params['n_resample_steps']}"
        )
    if "jump_length" in params:
        sampling_config_overrides.append(
            f"+sampler_partial.jump_length={params['jump_length']}"
        )

    structures_wo_H_regenerated: List[Structure] = (
        generate_reconstructed_structures(
            structures_to_reconstruct=structures_dl,
            output_path=results_path,
            sampling_config_overrides=sampling_config_overrides,
            config_overrides=config_overrides,
        )
    )
    return structures_wo_H_regenerated


def run_experiment(
    predictor_corrector: str,
    structures_dl: DataLoader,
    structures_subset: List["Structure"],
    params: Dict[str, Any],
    fix_cell: bool = True,
    relaxation_kwargs: Dict[str, Any] = None,
    save_prefix: str = "",
) -> None:
    """
    Run a single experiment:
      - Create a results directory.
      - Run sampling, perform optional relaxation.
      - Evaluate the regenerated structures.
    """
    param_str: str = (
        f"{save_prefix}_"
        + "__".join([f"{key}-{value}" for key, value in params.items()])
        + f"__{predictor_corrector}"
    )
    results_path: Path = Path(param_str)
    results_path.mkdir(parents=True, exist_ok=True)

    if relaxation_kwargs is None:
        relaxation_kwargs = {}

    with mlflow.start_run(run_name=param_str):
        structures_wo_H_regenerated = run_sampling(
            predictor_corrector, structures_dl, params, results_path, fix_cell
        )
        relaxed = relax_structure(
            structures_wo_H_regenerated, **relaxation_kwargs
        )
        relaxed_structures: List[Structure] = [r[0] for r in zip(*relaxed)]
        relaxed_dir: Path = results_path / "structures_relaxed"
        if not os.path.exists(relaxed_dir):
            os.makedirs(relaxed_dir)
        save_structures(relaxed_dir, relaxed_structures)
        correctly_matched, correctly_matched_relaxed = evaluate_results(
            structures_wo_H_regenerated,
            relaxed_structures,
            structures_subset,
            results_path,
            param_str,
        )

        metrics = {
            "correctly_matched_perc": correctly_matched,
            "correctly_matched_relaxed_perc": correctly_matched_relaxed,
        }

        mlflow.log_metrics(metrics)


def run_all_experiments(
    predictor_corrector: str,
    structures_dl: DataLoader,
    structures_subset: List[Structure],
    param_grid: Dict[str, Any],
    fix_cell: bool = True,
    relaxation_kwargs: Dict[str, Any] = None,
    save_prefix: str = "",
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
        # if all(key in params for key in ['N_steps', 'n_corrector_steps', 'n_resample_steps']):
        # if params['N_steps'] * params['n_corrector_steps'] * params['n_resample_steps'] > 5500:
        #     continue

        run_experiment(
            predictor_corrector,
            structures_dl,
            structures_subset,
            params,
            fix_cell,
            relaxation_kwargs,
            save_prefix,
        )
