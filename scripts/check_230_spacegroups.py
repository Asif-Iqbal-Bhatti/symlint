"""Sanity check across all 230 crystallographic space groups (Watanabe et al. PRL 117, 096404).

Computes the raw atomic insulator Wyckoff semigroup <2 * m_w> for all 230 space groups
up to nu_max = 200, compares it against 2 * m_min * N, and prints:
1. Every space group where the raw Wyckoff semigroup <2 * m_w> and 2 * m_min * N differ.
2. Verification of all benchmark space groups cited in Watanabe et al.
3. Full 230-space-group summary table (saved to results/spacegroup_semigroup_table.csv).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyxtal.symmetry import Group

from symlint.filling import (
     allowed_rule_string,
    allowed_set,
    atomic_insulator_semigroup,
    centering_divisor,
    primitive_generators,
    wyckoff_multiplicities_primitive,
)


def main(nu_max: int = 200) -> None:
    rows = []
    differing_groups = []

    for sg in range(1, 231):
        group = Group(sg)
        centering_letter, divisor = centering_divisor(sg)
        m_list = wyckoff_multiplicities_primitive(sg)
        m_min = m_list[0]
        generators = primitive_generators(sg)

        semigroup = atomic_insulator_semigroup(sg, nu_max=nu_max)
        multiples_of_2m_min = set(range(2 * m_min, nu_max + 1, 2 * m_min))

        only_in_semigroup = sorted(semigroup - multiples_of_2m_min)
        only_in_2m_min = sorted(multiples_of_2m_min - semigroup)
        differs = (semigroup != multiples_of_2m_min)

        row = {
            "sg_number": sg,
            "sg_symbol": group.symbol,
            "centering": centering_letter,
            "centering_divisor": divisor,
            "m_min": m_min,
            "2_m_min": 2 * m_min,
            "primitive_multiplicities": str(m_list),
            "generators_2m_w": str(generators),
            "semigroup_equals_2m_min_N": not differs,
            "in_semigroup_not_2m_min_N": str(only_in_semigroup[:10]),
            "in_2m_min_N_not_semigroup": str(only_in_2m_min[:10]),
            "allowed_rule_soc": allowed_rule_string(sg, mode="soc"),
            "allowed_rule_spinless": allowed_rule_string(sg, mode="spinless"),
        }
        rows.append(row)
        if differs:
            differing_groups.append(
                {
                    "SG": sg,
                    "Symbol": group.symbol,
                    "Centering": f"{centering_letter} (/{divisor})",
                    "m_w (prim)": m_list,
                    "2*m_w": generators,
                    "2*m_min*N": f"{2 * m_min}N",
                    "Semigroup <2*m_w>": sorted(semigroup)[:8],
                    "In <2*m_w> \\ 2*m_min*N (<=200)": only_in_semigroup,
                    "In 2*m_min*N \\ <2*m_w> (<=200)": only_in_2m_min,
                }
            )

    print("=" * 95)
    print(f"SANITY CHECK ACROSS ALL 230 SPACE GROUPS (1 <= nu <= {nu_max})")
    print("Comparing raw Wyckoff semigroup <2*m_w> vs. 2*m_min*N")
    print("=" * 95)
    print(f"Total space groups checked: 230")
    print(f"Space groups where <2*m_w> == 2*m_min*N: {230 - len(differing_groups)}")
    print(f"Space groups where <2*m_w> != 2*m_min*N: {len(differing_groups)}")
    print("-" * 95)

    for d in differing_groups:
        print(
            f"SG {d['SG']:3d} ({d['Symbol']:8s}) | Centering: {d['Centering']} | "
            f"m_w={d['m_w (prim)']} | 2*m_w={d['2*m_w']} | 2*m_min*N={d['2*m_min*N']}"
        )
        print(f"   First elements of <2*m_w>: {d['Semigroup <2*m_w>']}")
        print(f"   In <2*m_w> but NOT in 2*m_min*N (up to {nu_max}): {d['In <2*m_w> \\ 2*m_min*N (<=200)']}")
        print(f"   In 2*m_min*N but NOT in <2*m_w> (up to {nu_max}): {d['In 2*m_min*N \\ <2*m_w> (<=200)']}")
        print("-" * 95)

    # Check paper benchmark examples from Prompt item 5
    paper_examples = {
        "4N": [4, 7, 9],
        "6N": [144, 145],
        "8N (screw/glide)": [19, 29, 33, 76, 78],
        "12N": [169, 170],
        "8N (centered/other)": [73, 106, 110, 133, 135, 142, 206, 228],
    }
    print("\nBENCHMARK CHECK AGAINST EXAMPLES IN WATANABE ET AL. (PRL 117, 096404):")
    for expected_rule, sgs in paper_examples.items():
        base_rule = expected_rule.split()[0]
        statuses = []
        for sg in sgs:
            actual_rule = allowed_rule_string(sg, mode="soc")
            m_min = wyckoff_multiplicities_primitive(sg)[0]
            ok = (actual_rule == base_rule) and (
                allowed_set(sg, mode="soc", nu_max=nu_max) == set(range(2 * m_min, nu_max + 1, 2 * m_min))
            )
            statuses.append(f"SG {sg} ({Group(sg).symbol}): {actual_rule} [{'OK' if ok else 'MISMATCH'}]")
        print(f"  Expected {expected_rule:20s} -> " + ", ".join(statuses))

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out_csv = out_dir / "spacegroup_semigroup_table.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved full 230-group table to {out_csv}")


if __name__ == "__main__":
    main()
