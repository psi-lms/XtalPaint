from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd
from prettytable import PrettyTable
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure


def evaluate_results(
    structures_regen: List[Structure],
    relaxed_structures: List[Structure],
    structures_subset: List[Structure],
    output_dir: Path,
    param_str: str,
) -> Tuple[float, float]:
    """Evaluate regenerated structures against the originals."""
    matcher = StructureMatcher()
    results: List[Tuple[str, str, Any, Any]] = []

    assert [
        structures_regen[i].composition.iupac_formula
        == relaxed_structures[i].composition.iupac_formula
        for i in range(len(structures_regen))
    ]
    assert [
        structures_regen[i].composition.iupac_formula
        == structures_subset[i].composition.iupac_formula
        for i in range(len(structures_regen))
    ]

    for i in range(len(structures_regen)):
        formula = structures_subset[i].composition.alphabetical_formula
        volume = structures_subset[i].volume
        ref_key = f"{i} - {formula} - {volume:.0f}"
        strct_index = structures_regen[i].properties["structure_id"]

        assert strct_index == i

        # ref_struct = mapping[ref_key]
        ref_struct = structures_subset[i]
        assert (
            ref_struct.composition.iupac_formula
            == structures_regen[i].composition.iupac_formula
        )
        matches = matcher.fit(structures_regen[i], ref_struct)
        matches_relaxed = matcher.fit(relaxed_structures[i], ref_struct)
        results.append(
            (
                ref_key,
                structures_regen[i].composition.iupac_formula,
                matches,
                matches_relaxed,
            )
        )

    df = pd.DataFrame(
        results,
        columns=[
            "ref_key",
            "composition",
            "Matches",
            "Matches after relaxation",
        ],
    )
    df.to_csv(output_dir / f"{param_str}_df_results.csv")

    df_sum = df.groupby("ref_key").sum()
    table = PrettyTable(["Formula", "Matches", "Matches after relaxation"])
    for key, row in df_sum.iterrows():
        table.add_row([key, row["Matches"], row["Matches after relaxation"]])
    print(table)

    correctly_matched = df_sum["Matches"].gt(0).mean() * 100
    correctly_matched_relaxed = (
        df_sum["Matches after relaxation"].gt(0).mean() * 100
    )
    print(
        f"{correctly_matched}% correctly matched, after relaxation: {correctly_matched_relaxed}%"
    )
    return correctly_matched, correctly_matched_relaxed


def evaluate_sampling(
    structures_regen: List[Structure],
    structures_subset: List[Structure],
    results_path: Path,
    param_str: str,
) -> float:
    """
    Evaluate the reconstructed (and optionally relaxed) structures against the originals.
    Save a CSV summary and return the percentage of correctly matched structures.
    """
    matcher = StructureMatcher()
    results: List[Tuple[str, str, Any]] = []
    structures_w_H_mapping = {
        f"{s.composition.alphabetical_formula} {s.volume:.0f}": s
        for s in structures_subset
    }
    for i in range(len(structures_regen)):
        structure = structures_regen[i]
        formula = structure.composition.alphabetical_formula
        volume = structure.volume
        ref_key = f"{formula} {volume:.0f}"
        ref_strct = structures_w_H_mapping[ref_key]
        matches = matcher.fit(structures_regen[i], ref_strct)
        results.append((ref_key, structure.composition.iupac_formula, matches))

    df_results = pd.DataFrame(
        results, columns=["ref_key", "composition", "Matches"]
    )
    df_results.to_csv(
        results_path / f"{param_str}_df_results.csv", index=False
    )

    df_results_sum = df_results.groupby("ref_key").sum()
    table = PrettyTable(["Formula", "Matches"])
    for i, row in df_results_sum.iterrows():
        table.add_row([i, row["Matches"]])
    print(table)

    correctly_matched: float = df_results_sum["Matches"].gt(0).mean() * 100
    print(f"Correctly Matched: {correctly_matched}%")
    return correctly_matched
