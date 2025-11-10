# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.constraints import FixAtoms
from ase.optimize import BFGS, FIRE
import tqdm


def relax_atoms_mattersim(
    atoms: list[Atoms],
    device: str,
    load_path: str = None,
    **kwargs,
) -> tuple[list[Atoms], np.ndarray]:
    from mattersim.applications.batch_relax import BatchRelaxer
    from mattersim.forcefield.potential import Potential

    potential = Potential.from_checkpoint(
        device=device, load_path=load_path, load_training_state=False
    )

    batch_relaxer = BatchRelaxer(potential=potential, **kwargs)
    relaxation_trajectories = batch_relaxer.relax(atoms)

    relaxed_atoms = [t[-1] for t in relaxation_trajectories.values()]
    total_energies = np.array([a.info["total_energy"] for a in relaxed_atoms])

    return relaxed_atoms, total_energies


def _relax_atoms_mlip(
    atoms: Atoms,
    fmax: float,
    steps: int,
    optimizer: str,
    filter: str = None,
    **kwargs,
) -> float:
    if filter is not None:
        raise NotImplementedError("Filter not implemented yet.")

    opt_cls = {"bfgs": BFGS, "fire": FIRE}.get(optimizer.lower())
    if opt_cls is None:
        raise ValueError("Unsupported optimizer. Use bfgs or fire.")

    opt = opt_cls(atoms)
    opt.run(fmax=fmax, steps=steps)

    return atoms, float(atoms.get_potential_energy())


def _load_calculator(
    mlip: str,
    device: str,
    load_path: str | None,
    default_dtype: str,
) -> Calculator:
    if mlip == "mace":
        try:
            from mace.calculators import mace_mp
        except ImportError as e:
            raise RuntimeError(
                "MACE not installed. pip install mace-torch"
            ) from e
        return mace_mp(
            model=load_path,
            device=device,
            default_dtype=default_dtype,
            enable_cueq=True,
        )
    if mlip == "nequip":
        try:
            from nequip.ase import NequIPCalculator
        except ImportError as e:
            raise RuntimeError(
                "NequIP not installed. pip install nequip"
            ) from e
        return NequIPCalculator.from_compiled_model(
            compile_path=load_path,
            device=device,
        )
    raise ValueError(
        f"Unsupported mlip: {mlip}. Use 'mattersim', 'mace', or 'nequip'."
    )


def _run_relaxations(
    atoms: list[Atoms],
    calculator: Calculator,
    optimizer: str,
    max_n_steps: int,
    fmax: float,
    return_initial_energies: bool,
    return_initial_forces: bool,
    return_final_forces: bool,
    **relax_kwargs,
):
    relaxed_atoms = []
    final_energies = []
    initial_energies = [] if return_initial_energies else None
    initial_forces = [] if return_initial_forces else None
    final_forces = [] if return_final_forces else None

    for atom in tqdm.tqdm(atoms, miniters=50, mininterval=5):
        atom_to_opt = atom.copy()
        atom_to_opt.calc = calculator

        if return_initial_energies:
            initial_energies.append(float(atom_to_opt.get_potential_energy()))
        if return_initial_forces:
            initial_forces.append(
                float(np.max(np.linalg.norm(atom_to_opt.get_forces(), axis=1)))
            )

        relaxed_atom, final_energy = _relax_atoms_mlip(
            atoms=atom_to_opt,
            optimizer=optimizer,
            steps=max_n_steps,
            fmax=fmax,
            **relax_kwargs,
        )

        if return_final_forces:
            final_forces.append(
                float(
                    np.max(np.linalg.norm(relaxed_atom.get_forces(), axis=1))
                )
            )

        relaxed_atom.calc = None
        relaxed_atoms.append(relaxed_atom)
        final_energies.append(final_energy)

    return (
        relaxed_atoms,
        final_energies,
        initial_energies,
        initial_forces,
        final_forces,
    )


def relax_atoms_mlips(
    atoms: list[Atoms],
    mlip: str,
    device: str,
    load_path: str = None,
    default_dtype: str = "float32",
    return_initial_energies: bool = False,
    return_initial_forces: bool = False,
    return_final_forces: bool = False,
    **kwargs,
):
    optimizer = kwargs.pop("optimizer", "BFGS")
    max_n_steps = kwargs.pop("max_n_steps", 500)
    fmax = kwargs.pop("fmax", 0.05)

    if mlip == "mattersim":
        if any(
            (
                return_initial_energies,
                return_initial_forces,
                return_final_forces,
            )
        ):
            raise ValueError(
                "mattersim relaxation does not support returning initial/final energies/forces.\n"
                "Run them separately after relaxation if needed."
            )
        relaxed_atoms, total_energies = relax_atoms_mattersim(
            atoms,
            device=device,
            load_path=load_path,
            max_n_steps=max_n_steps,
            optimizer=optimizer,
            fmax=fmax,
            **kwargs,
        )
        return relaxed_atoms, total_energies, None, None, None

    calculator = _load_calculator(
        mlip=mlip,
        device=device,
        load_path=load_path,
        default_dtype=default_dtype,
    )

    return _run_relaxations(
        atoms=atoms,
        calculator=calculator,
        optimizer=optimizer,
        max_n_steps=max_n_steps,
        fmax=fmax,
        return_initial_energies=return_initial_energies,
        return_initial_forces=return_initial_forces,
        return_final_forces=return_final_forces,
        **kwargs,
    )


def relax_structures(
    structures: Structure | list[Structure],
    device: str,
    mlip: str = "mattersim",
    load_path: str = None,
    elements_to_relax: list[str] = None,
    return_initial_energies: bool = False,
    return_initial_forces: bool = False,
    return_final_forces: bool = False,
    **kwargs,
) -> tuple[list[Structure], np.ndarray]:
    if isinstance(structures, Structure):
        structures = [structures]

    atoms = [AseAtomsAdaptor.get_atoms(s) for s in structures]

    if elements_to_relax is not None:
        for a in atoms:
            c = FixAtoms(
                mask=[atom.symbol not in elements_to_relax for atom in a]
            )
            a.set_constraint(c)

    (
        relaxed_atoms,
        total_energies,
        initial_energies,
        initial_forces,
        final_forces,
    ) = relax_atoms_mlips(
        atoms,
        device=device,
        load_path=load_path,
        mlip=mlip,
        return_initial_energies=return_initial_energies,
        return_initial_forces=return_initial_forces,
        return_final_forces=return_final_forces,
        **kwargs,
    )
    relaxed_structures = [
        AseAtomsAdaptor.get_structure(a) for a in relaxed_atoms
    ]
    return (
        relaxed_structures,
        total_energies,
        initial_energies,
        initial_forces,
        final_forces,
    )
