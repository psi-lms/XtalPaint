import os
from pathlib import Path
import itertools
from typing import Any, Dict, List, Tuple
from torch.utils.data import DataLoader
from pymatgen.core import Structure
from dbcsi_inpainting.generate_inpainting import (
    generate_reconstructed_structures,
)
from mattergen.common.utils.eval_utils import save_structures
from dbcsi_inpainting.evaluation import (
    evaluate_results,
)
from dbcsi_inpainting.aiida.config_schema import InpaintingPipelineParams
import numpy as np
from pymatgen.core import Structure
from dbcsi_inpainting.utils.data_utils import create_dataloader

from dbcsi_inpainting.aiida.data import BatchedStructures, InpaintingStructure
from mattergen.common.data.dataset import CrystalDataset, structures_to_numpy
from mattergen.common.data.transform import (
    symmetrize_lattice,
    set_chemical_system_string,
)
from dbcsi_inpainting.aiida.config_schema import InpaintingPipelineParams
from ase import Atoms


DBSCI_INPAINTING_BASE = "dbcsi_inpainting.custom_predictor_corrector"
GUIDED_PREDICTOR_CORRECTOR_MAPPING = {
    "baseline": "mattergen.diffusion.sampling.classifier_free_guidance.GuidedPredictorCorrector.from_pl_module",
    "baseline-reverted-order": f"{DBSCI_INPAINTING_BASE}.GuidedPredictorCorrectorRevertedOrder.from_pl_module",
    "baseline-with-noise": f"{DBSCI_INPAINTING_BASE}.CustomGuidedPredictorCorrector.from_pl_module",
    "repaint-v1": f"{DBSCI_INPAINTING_BASE}.CustomGuidedPredictorCorrectorRePaint.from_pl_module",
    "repaint-v2": f"{DBSCI_INPAINTING_BASE}.CustomGuidedPredictorCorrectorRePaintV2.from_pl_module",
    "TD": f"{DBSCI_INPAINTING_BASE}.CustomGuidedPredictorCorrectorNewTimesteps.from_pl_module",
}


def _get_overrides(
    inpainting_model_params: Dict[str, Any],
    predictor_corrector: str,
    fix_cell: bool = True,
    pretrained_name=None,
) -> Tuple[List[str], List[str]]:
    sampling_config_overrides = [
        f'sampler_partial.N={inpainting_model_params["N_steps"]}',
        f'sampler_partial.n_steps_corrector={inpainting_model_params["n_corrector_steps"]}',
        f"~sampler_partial.predictor_partials.atomic_numbers",
        f'sampler_partial.corrector_partials.pos.snr={inpainting_model_params["coordinates_snr"]}',
        f"sampler_partial._target_={GUIDED_PREDICTOR_CORRECTOR_MAPPING[predictor_corrector]}",
    ]
    if predictor_corrector == "TD":
        sampling_config_overrides.append(
            "sampler_partial.corrector_partials.pos._target_=dbcsi_inpainting.time_dependent.corrector.TDWrappedLangevinCorrector"
        )
    config_overrides = []
    if pretrained_name is not None:
        config_overrides = [
            "lightning_module.diffusion_module.corruption.discrete_corruptions.atomic_numbers.d3pm.schedule.num_steps="
            + f'{inpainting_model_params["N_steps"]}',
        ]

    if fix_cell:
        sampling_config_overrides.extend(
            [
                "~sampler_partial.predictor_partials.cell",
                "~sampler_partial.corrector_partials.cell",
            ]
        )
        if pretrained_name is not None:
            config_overrides.append(
                "~lightning_module.diffusion_module.corruption.sdes.cell"
            )

    if "n_resample_steps" in inpainting_model_params:
        sampling_config_overrides.append(
            f'+sampler_partial.n_resample_steps={inpainting_model_params["n_resample_steps"]}'
        )
    if "jump_length" in inpainting_model_params:
        sampling_config_overrides.append(
            f'+sampler_partial.jump_length={inpainting_model_params["jump_length"]}'
        )

    return sampling_config_overrides, config_overrides


def _run_inpainting(
    predictor_corrector: str,
    structures_dl: DataLoader,
    inpainting_model_params: Dict[str, Any],
    fix_cell: bool = True,
    record_trajectories: bool = False,
    pretrained_name=None,
    model_path=None,
) -> list[Structure]:
    """Run the inpainting process using MatterGen."""
    sampling_config_overrides, config_overrides = _get_overrides(
        inpainting_model_params, predictor_corrector, fix_cell, pretrained_name
    )

    reconstructed_structures = generate_reconstructed_structures(
        structures_to_reconstruct=structures_dl,
        sampling_config_overrides=sampling_config_overrides,
        config_overrides=config_overrides,
        model_path=model_path,
        pretrained_name=pretrained_name,
        record_trajectories=record_trajectories,
        fix_cell=fix_cell,
    )

    return reconstructed_structures


def run_inpainting_pipeline(
    structures: dict[str, Structure | InpaintingStructure],
    config: InpaintingPipelineParams | dict[str, Any],
):
    """Run the inpainting experiment using MatterGen.

    Args:
        structures: Input structures for inpainting.
        config: Configuration for the inpainting process.
        record_trajectories: Whether to record trajectories during the inpainting process.

    Returns:
        A dictionary containing inpainted structures, relaxed structures (if applicable), and trajectories.
    """
    if isinstance(structures, BatchedStructures):
        structures = structures.get_structures("pymatgen")
    labels, structures = map(list, zip(*structures.items()))

    print(
        "This is the batch size:",
        config["inpainting_model_params"].get("batch_size", 64),
    )
    prepared_structures = __prepare_structures(
        structures,
        batch_size=config["inpainting_model_params"].get("batch_size", 64),
    )

    inpainted_structures, trajectories = _run_inpainting(
        structures_dl=prepared_structures, **config
    )

    inpainted_structures = {
        labels[i]: s for i, s in enumerate(inpainted_structures)
    }
    trajectories = {labels[i]: t for i, t in enumerate(trajectories)}

    outputs = {
        "structures": BatchedStructures(structures=inpainted_structures),
    }

    if config["record_trajectories"]:
        outputs.update(
            {
                "trajectories": trajectories,
            }
        )

    return outputs


def __prepare_structures(
    structures: dict[str, Structure | InpaintingStructure],
    batch_size: int = 64,
) -> DataLoader:
    """Prepare structures for inpainting by converting them to a DataLoader."""
    np.random.seed(1234)

    structures_numpy, properties = structures_to_numpy(structures)
    dataset = CrystalDataset(
        **structures_numpy,
        properties=properties,
        transforms=[symmetrize_lattice, set_chemical_system_string],
    )

    return create_dataloader(dataset, batch_size)
