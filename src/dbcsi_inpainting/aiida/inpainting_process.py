from pathlib import Path
from typing import Any, Dict, List, Tuple
from torch.utils.data import DataLoader
from pymatgen.core import Structure
from dbcsi_inpainting.generate_inpainting import (
    generate_reconstructed_structures,
)
from dbcsi_inpainting.aiida.config_schema import InpaintingPipelineParams
import numpy as np
from dbcsi_inpainting.utils.data_utils import create_dataloader

from dbcsi_inpainting.aiida.data import BatchedStructures, InpaintingStructure
from mattergen.common.data.dataset import CrystalDataset, structures_to_numpy
from mattergen.common.data.transform import (
    symmetrize_lattice,
    set_chemical_system_string,
)
import torch
import warnings


DBSCI_INPAINTING_BASE = "dbcsi_inpainting.custom_predictor_corrector"
GUIDED_PREDICTOR_CORRECTOR_MAPPING = {
    "baseline": "mattergen.diffusion.sampling.classifier_free_guidance.GuidedPredictorCorrector.from_pl_module",
    "baseline-with-noise": f"{DBSCI_INPAINTING_BASE}.CustomGuidedPredictorCorrector.from_pl_module",
    "baseline-store-scores": f"{DBSCI_INPAINTING_BASE}.AdditionalDataPredictorCorrector.from_pl_module",
    "repaint-v1": f"{DBSCI_INPAINTING_BASE}.CustomGuidedPredictorCorrectorRePaint.from_pl_module",
    "repaint-v2": f"{DBSCI_INPAINTING_BASE}.RePaintV2GuidedPredictorCorrector.from_pl_module",
    "TD": f"{DBSCI_INPAINTING_BASE}.TDPaintGuidedPredictorCorrector.from_pl_module",
}


def _get_overrides(
    inpainting_model_params: Dict[str, Any],
    predictor_corrector: str,
    fix_cell: bool = True,
    pretrained_name=None,
) -> Tuple[List[str], List[str]]:
    sampling_config_overrides = [
        f"sampler_partial.N={inpainting_model_params['N_steps']}",
        f"sampler_partial.n_steps_corrector={inpainting_model_params['n_corrector_steps']}",
        "~sampler_partial.predictor_partials.atomic_numbers",
        f"sampler_partial.corrector_partials.pos.snr={inpainting_model_params['coordinates_snr']}",
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
            + f"{inpainting_model_params['N_steps']}",
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
            f"+sampler_partial.n_resample_steps={inpainting_model_params['n_resample_steps']}"
        )
    if "jump_length" in inpainting_model_params:
        sampling_config_overrides.append(
            f"+sampler_partial.jump_length={inpainting_model_params['jump_length']}"
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
    if len(reconstructed_structures) == 2:
        print("Not returning mean trajectories.")
        return (reconstructed_structures[0], reconstructed_structures[1], None)
    elif len(reconstructed_structures) == 3:
        print("Returning mean trajectories as well.")
        return reconstructed_structures
    else:
        raise ValueError("Unexpected number of outputs from inpainting.")


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

    prepared_structures = __prepare_structures(
        structures,
        batch_size=config["inpainting_model_params"].get("batch_size", 64),
    )

    inpainted_structures, trajectories, mean_trajectories = _run_inpainting(
        structures_dl=prepared_structures, **config
    )

    inpainted_structures = {
        labels[i]: s for i, s in enumerate(inpainted_structures)
    }
    trajectories = {labels[i]: t for i, t in enumerate(trajectories)}

    if mean_trajectories:
        mean_trajectories = {
            labels[i]: t for i, t in enumerate(mean_trajectories)
        }

    outputs = {
        "structures": BatchedStructures(structures=inpainted_structures),
    }

    if config["record_trajectories"]:
        outputs.update(
            {
                "trajectories": trajectories,
            }
        )
        if mean_trajectories:
            outputs.update(
                {
                    "mean_trajectories": mean_trajectories,
                }
            )
        if Path("recorded_scores.pt").exists():
            recorded_scores = torch.load("recorded_scores.pt").numpy()
            atomic_numbers = torch.load("recorded_atomic_numbers.pt").numpy()
            pos_batch_idx = torch.load("recorded_batch_idx.pt").numpy()

            outputs["scores"] = {
                "pos_scores": recorded_scores,
                "atomic_numbers": atomic_numbers,
                "pos_batch_idx": pos_batch_idx,
            }

    return outputs


def run_mpi_parallel_inpainting_pipeline(
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
    import mpi4py

    if isinstance(structures, BatchedStructures):
        structures = structures.get_structures("pymatgen")
    labels, structures = map(list, zip(*structures.items()))

    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.rank
    nranks = comm.size
    N = len(structures)
    chunks = np.array_split(np.arange(N), comm.size)

    if torch.cuda.is_available() and torch.cuda.device_count() == nranks:
        torch.cuda.set_device(rank)
    elif torch.cuda.is_available():
        warnings.warn(
            f"CUDA is available, but the number of GPUs ({torch.cuda.device_count()}) does not match the number of MPI ranks ({nranks})."
        )

    chunk_idx = chunks[rank]
    local_structures = [structures[i] for i in chunk_idx]
    print(f"Rank {rank} processing {len(local_structures)} structures.")

    prepared_structures = __prepare_structures(
        local_structures,
        batch_size=config["inpainting_model_params"].get("batch_size", 64),
    )

    rank_inpainted_structures, rank_trajectories, rank_mean_trajectories = (
        _run_inpainting(structures_dl=prepared_structures, **config)
    )
    results_gathered = comm.gather(
        (rank_inpainted_structures, rank_trajectories, rank_mean_trajectories),
        root=0,
    )
    if rank == 0:
        print("Start gathering results from all ranks...")
        all_inpainted_structures = []
        all_trajectories = []
        all_mean_trajectories = []
        for res in results_gathered:
            all_inpainted_structures.extend(res[0])
            all_trajectories.extend(res[1])
            if res[2]:
                all_mean_trajectories.extend(res[2])

        print(len(all_mean_trajectories))
        inpainted_structures = {
            labels[i]: s for i, s in enumerate(all_inpainted_structures)
        }
        trajectories = {labels[i]: t for i, t in enumerate(all_trajectories)}
        if all_mean_trajectories:
            mean_trajectories = {
                labels[i]: t for i, t in enumerate(all_mean_trajectories)
            }

        outputs = {
            "structures": BatchedStructures(structures=inpainted_structures),
        }

        if config["record_trajectories"]:
            outputs.update(
                {
                    "trajectories": trajectories,
                }
            )
            if all_mean_trajectories:
                print("Returning mean trajectories as well. Updating outputs.")
                outputs.update(
                    {
                        "mean_trajectories": mean_trajectories,
                    }
                )

            if Path("recorded_scores.pt").exists():
                recorded_scores = torch.load("recorded_scores.pt").numpy()
                atomic_numbers = torch.load(
                    "recorded_atomic_numbers.pt"
                ).numpy()
                pos_batch_idx = torch.load("recorded_batch_idx.pt").numpy()

                outputs["scores"] = {
                    "pos_scores": recorded_scores,
                    "atomic_numbers": atomic_numbers,
                    "pos_batch_idx": pos_batch_idx,
                }
        print(outputs.keys())
        return outputs

    return None


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
