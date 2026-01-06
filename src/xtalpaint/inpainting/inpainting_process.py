"""Inpainting pipeline for crystal structures using MatterGen."""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mattergen.common.data.dataset import CrystalDataset, structures_to_numpy
from mattergen.common.data.transform import (
    set_chemical_system_string,
    symmetrize_lattice,
)
from pymatgen.core import Structure
from torch.utils.data import DataLoader

from xtalpaint.aiida.config_schema import InpaintingPipelineParams
from xtalpaint.aiida.data import BatchedStructures
from xtalpaint.generate_inpainting import (
    generate_reconstructed_structures,
)
from xtalpaint.utils.data_utils import create_dataloader

XTALPAINT_BASE = "xtalpaint.predictor_corrector"

GUIDED_PREDICTOR_CORRECTOR_MAPPING = {
    "baseline": (
        "mattergen.diffusion.sampling.classifier_free_guidance."
        "GuidedPredictorCorrector.from_pl_module"
    ),
    "baseline-with-noise": (
        f"{XTALPAINT_BASE}.CustomGuidedPredictorCorrector.from_pl_module"
    ),
    "baseline-store-scores": (
        f"{XTALPAINT_BASE}.AdditionalDataPredictorCorrector.from_pl_module"
    ),
    "repaint-v1": (
        f"{XTALPAINT_BASE}.RePaintLegacyGuidedPredictorCorrector.from_pl_module"
    ),
    "repaint-v2": (
        f"{XTALPAINT_BASE}.RePaintV2GuidedPredictorCorrector.from_pl_module"
    ),
    "TD": f"{XTALPAINT_BASE}.TDPaintGuidedPredictorCorrector.from_pl_module",
}


def _get_sampling_config_overrides(
    inpainting_model_params: dict[str, Any],
    predictor_corrector: str,
    fix_cell: bool = True,
) -> list[str]:
    """Get sampling configuration overrides.

    Args:
        inpainting_model_params: Parameters for the inpainting model.
        predictor_corrector: Type of predictor-corrector to use.
        fix_cell: Whether to fix the cell during sampling.

    Returns:
        List of sampling configuration override strings.
    """
    overrides = [
        f"sampler_partial.N={inpainting_model_params['N_steps']}",
        (
            f"sampler_partial.n_steps_corrector="
            f"{inpainting_model_params['n_corrector_steps']}"
        ),
        "~sampler_partial.predictor_partials.atomic_numbers",
        (
            f"sampler_partial.corrector_partials.pos.snr="
            f"{inpainting_model_params['coordinates_snr']}"
        ),
        (
            f"sampler_partial._target_="
            f"{GUIDED_PREDICTOR_CORRECTOR_MAPPING[predictor_corrector]}"
        ),
    ]

    # Add TD-specific override
    if predictor_corrector == "TD":
        overrides.append(
            "sampler_partial.corrector_partials.pos._target_="
            "xtalpaint.time_dependent.corrector.TDWrappedLangevinCorrector"
        )

    # Add cell fixing overrides
    if fix_cell:
        overrides.extend(
            [
                "~sampler_partial.predictor_partials.cell",
                "~sampler_partial.corrector_partials.cell",
            ]
        )

    # Add optional parameters
    if "n_resample_steps" in inpainting_model_params:
        overrides.append(
            f"+sampler_partial.n_resample_steps="
            f"{inpainting_model_params['n_resample_steps']}"
        )

    if "jump_length" in inpainting_model_params:
        overrides.append(
            f"+sampler_partial.jump_length="
            f"{inpainting_model_params['jump_length']}"
        )

    return overrides


def _get_model_config_overrides(
    inpainting_model_params: dict[str, Any],
    fix_cell: bool = True,
    pretrained_name: str | None = None,
) -> list[str]:
    """Get model configuration overrides.

    Args:
        inpainting_model_params: Parameters for the inpainting model.
        fix_cell: Whether to fix the cell during sampling.
        pretrained_name: Name of pretrained model, if any.

    Returns:
        List of model configuration override strings.
    """
    if pretrained_name is None:
        return []

    overrides = [
        (
            "lightning_module.diffusion_module.corruption.discrete_corruptions."
            f"atomic_numbers.d3pm.schedule.num_steps="
            f"{inpainting_model_params['N_steps']}"
        ),
    ]

    if fix_cell:
        overrides.append(
            "~lightning_module.diffusion_module.corruption.sdes.cell"
        )

    return overrides


def _get_overrides(
    inpainting_model_params: dict[str, Any],
    predictor_corrector: str,
    fix_cell: bool = True,
    pretrained_name: str | None = None,
) -> tuple[list[str], list[str]]:
    """Get configuration overrides for sampling and model.

    Args:
        inpainting_model_params: Parameters for the inpainting model.
        predictor_corrector: Type of predictor-corrector to use.
        fix_cell: Whether to fix the cell during sampling.
        pretrained_name: Name of pretrained model, if any.

    Returns:
        Tuple of (sampling_config_overrides, config_overrides).
    """
    sampling_overrides = _get_sampling_config_overrides(
        inpainting_model_params, predictor_corrector, fix_cell
    )

    model_overrides = _get_model_config_overrides(
        inpainting_model_params, fix_cell, pretrained_name
    )

    return sampling_overrides, model_overrides


def _run_inpainting(
    predictor_corrector: str,
    structures_dl: DataLoader,
    inpainting_model_params: dict[str, Any],
    fix_cell: bool = True,
    record_trajectories: bool = False,
    pretrained_name: str | None = None,
    model_path: str | None = None,
) -> tuple[list[Structure], list, list | None]:
    """Run the inpainting process using MatterGen.

    Args:
        predictor_corrector: Type of predictor-corrector to use.
        structures_dl: DataLoader containing structures to inpaint.
        inpainting_model_params: Parameters for the inpainting model.
        fix_cell: Whether to fix the cell during sampling.
        record_trajectories: Whether to record trajectories.
        pretrained_name: Name of pretrained model, if any.
        model_path: Path to model checkpoint.

    Returns:
        Tuple of (inpainted_structures, trajectories, mean_trajectories).
        mean_trajectories is None if not recorded.
    """
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

    if len(reconstructed_structures) == 2:
        print("Not returning mean trajectories.")
        return (reconstructed_structures[0], reconstructed_structures[1], None)
    elif len(reconstructed_structures) == 3:
        print("Returning mean trajectories as well.")
        return reconstructed_structures
    else:
        raise ValueError(
            f"Unexpected number of outputs from inpainting: "
            f"{len(reconstructed_structures)}"
        )


def _prepare_structures(
    structures: list[Structure],
    batch_size: int = 64,
    seed: int = 1234,
) -> DataLoader:
    """Prepare structures for inpainting by converting them to a DataLoader.

    Args:
        structures: List of structures to prepare.
        batch_size: Batch size for the DataLoader.
        seed: Random seed for shuffling.

    Returns:
        DataLoader containing prepared structures.
    """
    np.random.seed(seed)

    structures_numpy, properties = structures_to_numpy(structures)
    dataset = CrystalDataset(
        **structures_numpy,
        properties=properties,
        transforms=[symmetrize_lattice, set_chemical_system_string],
    )

    return create_dataloader(dataset, batch_size)


def _extract_outputs(
    inpainted_structures: list[Structure],
    trajectories: list,
    mean_trajectories: list | None,
    labels: list[str],
    record_trajectories: bool,
) -> dict[str, Any]:
    """Extract and organize inpainting outputs.

    Args:
        inpainted_structures: List of inpainted structures.
        trajectories: List of trajectories.
        mean_trajectories: List of mean trajectories, if available.
        labels: Labels for the structures.
        record_trajectories: Whether trajectories were recorded.

    Returns:
        Dictionary containing organized outputs.
    """
    outputs = {
        "structures": BatchedStructures(
            structures={
                labels[i]: s for i, s in enumerate(inpainted_structures)
            }
        ),
    }

    if record_trajectories:
        outputs["trajectories"] = {
            labels[i]: t for i, t in enumerate(trajectories)
        }

        if mean_trajectories:
            outputs["mean_trajectories"] = {
                labels[i]: t for i, t in enumerate(mean_trajectories)
            }

        # Load recorded scores if available
        scores_path = Path("recorded_scores.pt")
        if scores_path.exists():
            outputs["scores"] = {
                "pos_scores": torch.load("recorded_scores.pt").numpy(),
                "atomic_numbers": torch.load(
                    "recorded_atomic_numbers.pt"
                ).numpy(),
                "pos_batch_idx": torch.load("recorded_batch_idx.pt").numpy(),
            }

    return outputs


def run_inpainting_pipeline(
    structures: dict[str, Structure],
    config: InpaintingPipelineParams | dict[str, Any],
) -> dict[str, Any]:
    """Run the inpainting experiment using MatterGen.

    Args:
        structures: Input structures for inpainting.
        config: Configuration for the inpainting process.

    Returns:
        Dictionary containing inpainted structures, trajectories, and scores.
    """
    if isinstance(structures, BatchedStructures):
        structures = structures.get_structures("pymatgen")

    labels, structures = map(list, zip(*structures.items()))

    prepared_structures = _prepare_structures(
        structures,
        batch_size=config["inpainting_model_params"].get("batch_size", 64),
    )

    inpainted_structures, trajectories, mean_trajectories = _run_inpainting(
        structures_dl=prepared_structures, **config
    )

    return _extract_outputs(
        inpainted_structures,
        trajectories,
        mean_trajectories,
        labels,
        config["record_trajectories"],
    )


def run_mpi_parallel_inpainting_pipeline(
    structures: dict[str, Structure],
    config: InpaintingPipelineParams | dict[str, Any],
) -> dict[str, Any] | None:
    """Run the inpainting experiment using MatterGen with MPI parallelization.

    Args:
        structures: Input structures for inpainting.
        config: Configuration for the inpainting process.

    Returns:
        Dictionary containing inpainted structures, trajectories, and scores
        on rank 0, None on other ranks.
    """
    import mpi4py.MPI

    if isinstance(structures, BatchedStructures):
        structures = structures.get_structures("pymatgen")

    labels, structures = map(list, zip(*structures.items()))

    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.rank
    nranks = comm.size

    # Distribute structures across ranks
    chunks = np.array_split(np.arange(len(structures)), nranks)

    # Set CUDA device if available
    if torch.cuda.is_available():
        if torch.cuda.device_count() == nranks:
            torch.cuda.set_device(rank)
        else:
            warnings.warn(
                f"CUDA is available, but the number of GPUs "
                f"({torch.cuda.device_count()}) does not match the number of "
                f"MPI ranks ({nranks})."
            )

    # Process local chunk
    chunk_idx = chunks[rank]
    local_structures = [structures[i] for i in chunk_idx]
    print(f"Rank {rank} processing {len(local_structures)} structures.")

    prepared_structures = _prepare_structures(
        local_structures,
        batch_size=config["inpainting_model_params"].get("batch_size", 64),
    )

    rank_results = _run_inpainting(structures_dl=prepared_structures, **config)

    # Gather results on rank 0
    all_results = comm.gather(rank_results, root=0)

    if rank == 0:
        print("Gathering results from all ranks...")

        # Combine results from all ranks
        all_inpainted_structures = []
        all_trajectories = []
        all_mean_trajectories = []

        for inpainted, traj, mean_traj in all_results:
            all_inpainted_structures.extend(inpainted)
            all_trajectories.extend(traj)
            if mean_traj:
                all_mean_trajectories.extend(mean_traj)

        return _extract_outputs(
            all_inpainted_structures,
            all_trajectories,
            all_mean_trajectories if all_mean_trajectories else None,
            labels,
            config["record_trajectories"],
        )

    return None
