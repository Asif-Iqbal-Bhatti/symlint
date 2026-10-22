"""Phase 1 unit tests for symlint core library and CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ase.build import bulk
from pymatgen.core import Lattice, Structure

from symlint.check import check_cif_string, check_file, check_structure
from symlint.cli import main as cli_main
from symlint.filling import (
    allowed_rule_string,
    allowed_set,
    atomic_insulator_semigroup,
    classify_filling,
    is_allowed,
    wyckoff_multiplicities_primitive,
)

DATA_DIR = Path(__file__).resolve().parent / "data"

# Complete Table II (Supplemental Material, page 6 of Watanabe et al., PRL 117, 096404 (2016))
# listing all 157 nonsymmorphic space groups and their band insulator rules with SOC.
# Any space group not in this dictionary is symmorphic and has rule "2N".
TABLE_II_NONSYMMORPHIC_SOC: dict[int, str] = {
    4: "4N", 7: "4N", 9: "4N", 11: "4N", 13: "4N", 14: "4N", 15: "4N", 17: "4N",
    18: "4N", 19: "8N", 20: "4N", 24: "4N", 26: "4N", 27: "4N", 28: "4N", 29: "8N",
    30: "4N", 31: "4N", 32: "4N", 33: "8N", 34: "4N", 36: "4N", 37: "4N", 39: "4N",
    40: "4N", 41: "4N", 43: "4N", 45: "4N", 46: "4N", 48: "4N", 49: "4N", 50: "4N",
    51: "4N", 52: "8N", 53: "4N", 54: "8N", 55: "4N", 56: "8N", 57: "8N", 58: "4N",
    59: "4N", 60: "8N", 61: "8N", 62: "8N", 63: "4N", 64: "4N", 66: "4N", 67: "4N",
    68: "4N", 70: "4N", 72: "4N", 73: "8N", 74: "4N", 76: "8N", 77: "4N", 78: "8N",
    80: "4N", 84: "4N", 85: "4N", 86: "4N", 88: "4N", 90: "4N", 91: "8N", 92: "8N",
    93: "4N", 94: "4N", 95: "8N", 96: "8N", 98: "4N", 100: "4N", 101: "4N", 102: "4N",
    103: "4N", 104: "4N", 105: "4N", 106: "8N", 108: "4N", 109: "4N", 110: "8N", 112: "4N",
    113: "4N", 114: "4N", 116: "4N", 117: "4N", 118: "4N", 120: "4N", 122: "4N", 124: "4N",
    125: "4N", 126: "4N", 127: "4N", 128: "4N", 129: "4N", 130: "8N", 131: "4N", 132: "4N",
    133: "8N", 134: "4N", 135: "8N", 136: "4N", 137: "4N", 138: "8N", 140: "4N", 141: "4N",
    142: "8N", 144: "6N", 145: "6N", 151: "6N", 152: "6N", 153: "6N", 154: "6N", 158: "4N",
    159: "4N", 161: "4N", 163: "4N", 165: "4N", 167: "4N", 169: "12N", 170: "12N", 171: "6N",
    172: "6N", 173: "4N", 176: "4N", 178: "12N", 179: "12N", 180: "6N", 181: "6N", 182: "4N",
    184: "4N", 185: "4N", 186: "4N", 188: "4N", 190: "4N", 192: "4N", 193: "4N", 194: "4N",
    198: "8N", 199: "4N", 201: "4N", 203: "4N", 205: "8N", 206: "8N", 208: "4N", 210: "4N",
    212: "8N", 213: "8N", 214: "4N", 218: "4N", 219: "4N", 220: "4N \\ {4}", 222: "4N",
    223: "4N", 224: "4N", 226: "4N", 227: "4N", 228: "8N", 230: "8N",
}


def test_si_diamond_allowed() -> None:
    """Si diamond (SG 227, 2 atoms in primitive cell, nu=28) -> allowed."""
    si = bulk("Si", "diamond", a=5.431)
    res = check_structure(si)
    assert res["sg_number"] == 227
    assert res["num_primitive_atoms"] == 2
    assert res["nu"] == 28
    assert res["m_min"] == 2
    assert res["allowed_rule"] == "4N"
    assert res["verdict"] == "allowed"
    assert res["can_be_insulator"] is True
    assert res["symprec_sensitive"] is False


def test_nacl_rocksalt_allowed() -> None:
    """NaCl rocksalt (SG 225, 2 atoms in primitive cell, nu=28) -> allowed."""
    nacl = bulk("NaCl", "rocksalt", a=5.64)
    res = check_structure(nacl)
    assert res["sg_number"] == 225
    assert res["num_primitive_atoms"] == 2
    assert res["nu"] == 28
    assert res["m_min"] == 1
    assert res["allowed_rule"] == "2N"
    assert res["verdict"] == "allowed"
    assert res["can_be_insulator"] is True
    assert res["symprec_sensitive"] is False


def test_na_bcc_odd_filling() -> None:
    """Na bcc (SG 229, 1 atom in primitive cell, nu=11) -> odd_filling."""
    na = bulk("Na", "bcc", a=4.29)
    res = check_structure(na)
    assert res["sg_number"] == 229
    assert res["num_primitive_atoms"] == 1
    assert res["nu"] == 11
    assert res["m_min"] == 1
    assert res["verdict"] == "odd_filling"
    assert res["can_be_insulator"] is False
    assert res["symprec_sensitive"] is False


def test_sriro3_pbnm_symmetry_enforced() -> None:
    """SrIrO3 orthorhombic perovskite Pbnm/Pnma (SG 62, 4 f.u./prim cell, nu=556) -> symmetry_enforced."""
    # 1. Test via Materials Project mp-1105160 CIF
    res_mp = check_file(DATA_DIR / "SrIrO3_Pnma_mp1105160.cif")
    assert res_mp["sg_number"] == 62
    assert res_mp["num_primitive_atoms"] == 20
    assert res_mp["nu"] == 556
    assert res_mp["m_min"] == 4
    assert res_mp["allowed_rule"] == "8N"
    assert res_mp["verdict"] == "symmetry_enforced"
    assert res_mp["can_be_insulator"] is False
    assert res_mp["symprec_sensitive"] is False

    # 2. Test via experimental Pbnm structure (Longo et al. 1971 / Zhao et al. 2008)
    s_exp = Structure.from_spacegroup(
        "Pbnm",
        Lattice.orthorhombic(5.5617, 5.5909, 7.8821),
        ["Sr", "Ir", "O", "O"],
        [[-0.0067, 0.5293, 0.25], [0.5, 0.0, 0.0], [0.072, 0.485, 0.25], [0.714, 0.286, 0.038]],
    )
    res_exp = check_structure(s_exp)
    assert res_exp["sg_number"] == 62
    assert res_exp["num_primitive_atoms"] == 20
    assert res_exp["nu"] == 556
    assert res_exp["verdict"] == "symmetry_enforced"


def test_four_override_space_groups() -> None:
    """Verify the four override groups (199, 214, 220, 230) in both 'soc' and 'spinless' modes."""
    nu_max = 200
    mult_4 = set(range(4, nu_max + 1, 4))
    mult_8 = set(range(8, nu_max + 1, 8))

    # SOC mode
    assert allowed_set(199, mode="soc", nu_max=nu_max) == mult_4
    assert allowed_set(214, mode="soc", nu_max=nu_max) == mult_4
    assert allowed_set(220, mode="soc", nu_max=nu_max) == (mult_4 - {4})
    assert allowed_set(230, mode="soc", nu_max=nu_max) == mult_8

    # Spinless mode
    assert allowed_set(199, mode="spinless", nu_max=nu_max) == (mult_4 - {4})
    assert allowed_set(214, mode="spinless", nu_max=nu_max) == (mult_4 - {4})
    assert allowed_set(220, mode="spinless", nu_max=nu_max) == (mult_4 - {4, 8, 20})
    assert allowed_set(230, mode="spinless", nu_max=nu_max) == (mult_8 - {8})

    # Verify point membership checks at boundary fillings
    assert is_allowed(4, 199, mode="soc") is True
    assert is_allowed(4, 199, mode="spinless") is False
    assert is_allowed(4, 214, mode="soc") is True
    assert is_allowed(4, 214, mode="spinless") is False

    assert is_allowed(4, 220, mode="soc") is False
    assert is_allowed(8, 220, mode="soc") is True
    assert is_allowed(20, 220, mode="soc") is True
    assert is_allowed(8, 220, mode="spinless") is False
    assert is_allowed(12, 220, mode="spinless") is True
    assert is_allowed(16, 220, mode="spinless") is True
    assert is_allowed(20, 220, mode="spinless") is False
    assert is_allowed(24, 220, mode="spinless") is True

    assert is_allowed(8, 230, mode="soc") is True
    assert is_allowed(8, 230, mode="spinless") is False
    assert is_allowed(16, 230, mode="spinless") is True


def test_all_230_space_groups_match_watanabe_tables() -> None:
    """Verify all 230 space groups against Tables I, II, and III of Watanabe et al. (2016)."""
    assert len(TABLE_II_NONSYMMORPHIC_SOC) == 157

    differing_semigroups = []
    for sg in range(1, 231):
        m_list = wyckoff_multiplicities_primitive(sg)
        m_min = m_list[0]
        semigroup = atomic_insulator_semigroup(sg, nu_max=200)
        mult_2m_min = set(range(2 * m_min, 201, 2 * m_min))
        if semigroup != mult_2m_min:
            differing_semigroups.append(sg)

        expected_soc_rule = TABLE_II_NONSYMMORPHIC_SOC.get(sg, "2N")
        assert allowed_rule_string(sg, mode="soc") == expected_soc_rule

        # In spinless mode, S_G equals the atomic insulator semigroup for ALL 230 space groups
        assert allowed_set(sg, mode="spinless", nu_max=200) == semigroup

    # Only the 4 Wyckoff-mismatched space groups differ between <2*m_w> and 2*m_min*N
    assert differing_semigroups == [199, 214, 220, 230]


def test_symprec_sensitivity_detection() -> None:
    """Verify that a small coordinate rattle triggers symprec_sensitive=True."""
    si = bulk("Si", "diamond", a=5.431, cubic=True)
    # Displace atom 0 by 0.02 A along x: breaks Fd-3m at symprec=1e-3, kept at symprec=0.1
    positions = si.get_positions()
    positions[0, 0] += 0.02
    si.set_positions(positions)

    res = check_structure(si, symprec=1e-3, alt_symprec=0.1)
    assert res["sg_number"] != 227
    assert res["sg_number_alt"] == 227
    assert res["symprec_sensitive"] is True


def test_cli_check_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify `symlint check <file.cif>` CLI output in both text and JSON modes."""
    cif_path = str(DATA_DIR / "SrIrO3_Pnma_mp1105160.cif")
    rc = cli_main(["check", cif_path])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "Pnma (#62)" in captured
    assert "symmetry_enforced" in captured

    rc_json = cli_main(["check", "--json", cif_path])
    assert rc_json == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sg_number"] == 62
    assert payload["nu"] == 556
    assert payload["verdict"] == "symmetry_enforced"
