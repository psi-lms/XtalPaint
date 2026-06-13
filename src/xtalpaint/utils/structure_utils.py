"""Utility functions for structure processing."""

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from xtalpaint.data import BatchedStructures
from xtalpaint.utils import _is_batched_structure
from xtalpaint.utils.data_utils import get_structure_keys


def check_for_nan_positions(structure: Structure) -> bool:
    """Check if a pymatgen Structure has NaN values in its atomic positions."""
    positions = structure.cart_coords
    return np.isnan(positions).any()


def refine_structures(
    structures: "BatchedStructures | dict[str, Structure]",
    symprec: float,
    primitive: bool = False,
) -> BatchedStructures:
    """Refine structures to standard conventional (or primitive) cells.

    Args:
        structures: Input structures.
        symprec: Symmetry precision passed to SpacegroupAnalyzer.
        primitive: If ``True``, return the primitive cell instead of the
            conventional cell.

    Returns:
        BatchedStructures with refined structures. Structures for which
        refinement raises an exception are kept as-is.
    """
    if _is_batched_structure(structures):
        structures_dict: dict[str, Structure] = structures.get_structures(
            strct_type="pymatgen"
        )
    else:
        structures_dict = dict(structures)

    refined: dict[str, Structure] = {}
    for k, s in structures_dict.items():
        analyzer = SpacegroupAnalyzer(s, symprec=symprec)
        try:
            result = analyzer.get_refined_structure()
        except Exception:
            result = s

        if primitive:
            try:
                result = SpacegroupAnalyzer(
                    result, symprec=symprec
                ).get_primitive_structure()
            except Exception:
                pass

        refined[k] = result

    return BatchedStructures(refined)


def filter_unique_structures(
    structures: "BatchedStructures | dict[str, Structure]",
    symprec: float = 0.1,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
) -> BatchedStructures:
    """Filter unique structures (per parent key) and space group.

    Groups samples by their parent structure key (splitting on ``_sample_``),
    then by space group number, then applies StructureMatcher within each
    sub-group to retain one representative per equivalence class. NaN
    structures are skipped. The first encountered structure in each equivalence
    class is kept as the representative.

    Args:
        structures: Inpainting samples, typically keyed as
            ``{base_key}_sample_{idx}``.
        symprec: Symmetry precision passed to SpacegroupAnalyzer.
        ltol: Fractional length tolerance for StructureMatcher.
        stol: Site tolerance for StructureMatcher.
        angle_tol: Angle tolerance in degrees for StructureMatcher.

    Returns:
        BatchedStructures containing one representative per unique structure.
    """
    if _is_batched_structure(structures):
        structures_dict: dict[str, Structure] = structures.get_structures(
            strct_type="pymatgen"
        )
    else:
        structures_dict = dict(structures)

    base_keys, _ = get_structure_keys(structures_dict)

    groups: dict[str, list[tuple[str, Structure]]] = {}
    for full_key, base_key in zip(structures_dict.keys(), base_keys):
        groups.setdefault(base_key, []).append(
            (full_key, structures_dict[full_key])
        )

    structure_matcher = StructureMatcher(
        ltol=ltol, stol=stol, angle_tol=angle_tol
    )
    unique: dict[str, Structure] = {}

    for members in groups.values():
        sg_groups: dict[int, list[tuple[str, Structure]]] = {}
        for full_key, structure in members:
            if check_for_nan_positions(structure):
                continue
            try:
                sg_num = SpacegroupAnalyzer(
                    structure, symprec=symprec
                ).get_space_group_number()
            except Exception:
                sg_num = -1
            sg_groups.setdefault(sg_num, []).append((full_key, structure))

        for sg_members in sg_groups.values():
            representatives: list[tuple[str, Structure]] = []
            for full_key, structure in sg_members:
                if not any(
                    structure_matcher.fit(structure, rep_strct)
                    for _, rep_strct in representatives
                ):
                    representatives.append((full_key, structure))
            for rep_key, rep_strct in representatives:
                unique[rep_key] = rep_strct

    return BatchedStructures(unique)
