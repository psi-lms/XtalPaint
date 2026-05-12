"""Utility functions for structure processing."""

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from xtalpaint.data import BatchedStructures
from xtalpaint.utils import _is_batched_structure


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
