"""Utilities for relaxation tasks using MPI parallelization."""

import torch
import warnings
from dbcsi_inpainting.utils.relaxation_utils import relax_structures


def split_structures_by_len(
    structures: list, n_chunks: int
) -> tuple[list[list], list[list]]:
    """Split structures into chunks based on their lengths.

    Return a list of lists, where each inner list is a chunk of structures.
    Moreover, return a list of lists of indices corresponding to the structures in each chunk.
    """
    if not structures:
        return [], []

    n_chunks = min(n_chunks, len(structures))
    chunks = [[] for _ in range(n_chunks)]
    chunk_indices = [[] for _ in range(n_chunks)]
    chunk_lens = [0 for _ in range(n_chunks)]

    indexed_structures = sorted(
        enumerate(structures), key=lambda item: len(item[1]), reverse=True
    )

    for structure_idx, structure in indexed_structures:
        target_chunk = chunk_lens.index(min(chunk_lens))
        chunks[target_chunk].append(structure)
        chunk_indices[target_chunk].append(structure_idx)
        chunk_lens[target_chunk] += len(structure)

    return chunks, chunk_indices


def _relax_mpi_parallel(
    structures: list,
    keys: list[str],
    relax_inputs: dict,
) -> tuple[list, list]:
    """Initialize MPI parallel relaxation."""
    import mpi4py

    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.rank
    nranks = comm.size

    if (
        torch.cuda.is_available()
        and torch.cuda.device_count() == nranks
        and relax_inputs.get("device", "cpu") == "cuda"
    ):
        relax_inputs.update({"device": f"cuda:{rank}"})
        print(
            "Relax inputs updated for MPI parallel relaxation:", relax_inputs
        )
    elif (
        torch.cuda.is_available()
        and relax_inputs.get("device", "cpu") == "cuda"
    ):
        warnings.warn(
            "CUDA is available, but the number of GPUs "
            f"({torch.cuda.device_count()}) does not match the number of MPI "
            "ranks ({nranks})."
        )

    chunks, chunk_indices = split_structures_by_len(
        structures, n_chunks=nranks
    )

    if rank >= len(chunks):
        # If there are more ranks than chunks, some ranks do nothing
        return None, None, None, None, None

    local_structures = chunks[rank]
    local_indices = chunk_indices[rank]
    local_keys = [keys[i] for i in local_indices]

    (
        relaxed_structures,
        relaxed_energies,
        initial_energies,
        initial_forces,
        final_forces,
    ) = relax_structures(local_structures, **relax_inputs)
    print(f"Rank {rank} relaxed {len(local_structures)} structures.")
    results_gathered = comm.gather(
        (
            relaxed_structures,
            relaxed_energies,
            initial_energies,
            initial_forces,
            final_forces,
            local_keys,
        ),
        root=0,
    )

    if rank == 0:
        print("Start gathering results from all ranks...")
        all_relaxed_structures = []
        all_relaxed_energies = []
        all_initial_energies = []
        all_initial_forces = []
        all_final_forces = []
        all_local_keys = []
        for res in results_gathered:
            all_relaxed_structures.extend(res[0])
            all_relaxed_energies.extend(res[1])
            all_local_keys.extend(res[5])

            # These are optional
            if res[2]:
                all_initial_energies.extend(res[2])
            if res[3]:
                all_initial_forces.extend(res[3])
            if res[4]:
                all_final_forces.extend(res[4])

        all_relaxed_structures_d = {
            key: relaxed_structure
            for key, relaxed_structure in zip(
                all_local_keys, all_relaxed_structures
            )
        }
        all_relaxed_energies_d = {
            key: relaxed_energy
            for key, relaxed_energy in zip(
                all_local_keys, all_relaxed_energies
            )
        }

        all_initial_energies_d = None
        all_initial_forces_d = None
        all_final_forces_d = None

        if all_initial_energies:
            all_initial_energies_d = {
                key: initial_energy
                for key, initial_energy in zip(
                    all_local_keys, all_initial_energies
                )
            }
        if all_initial_forces:
            all_initial_forces_d = {
                key: initial_force
                for key, initial_force in zip(
                    all_local_keys, all_initial_forces
                )
            }
        if all_final_forces:
            all_final_forces_d = {
                key: final_force
                for key, final_force in zip(all_local_keys, all_final_forces)
            }
        print(f"Rank {rank} gathered results from all ranks.")
        return (
            all_relaxed_structures_d,
            all_relaxed_energies_d,
            all_initial_energies_d,
            all_initial_forces_d,
            all_final_forces_d,
        )

    return None, None, None, None, None  # If not root, return None
