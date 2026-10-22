"""Phase 2 (JARVIS audit outputs) and Phase 3 (MCP tool and marimo demo) tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from symlint.mcp_server import can_be_insulator, mcp

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
DATA_DIR = Path(__file__).resolve().parent / "data"


def test_phase2_results_files_and_consistency() -> None:
    """Verify all Phase 2 output files exist and summary counts are internally consistent."""
    summary_path = RESULTS_DIR / "summary.json"
    parquet_path = RESULTS_DIR / "full_results.parquet"
    conflicts_path = RESULTS_DIR / "conflicts.csv"
    candidates_path = RESULTS_DIR / "candidates.csv"
    figure_path = RESULTS_DIR / "gap_distribution_by_verdict.png"

    assert summary_path.exists()
    assert parquet_path.exists()
    assert conflicts_path.exists()
    assert candidates_path.exists()
    assert figure_path.exists() and figure_path.stat().st_size > 10_000

    summary = json.loads(summary_path.read_text())
    total = summary["total_entries"]
    assert total == 93902

    mag = summary["magnetic_breakdown"]
    assert mag["nonmagnetic"] + mag["magnetic"] + mag["unknown_magmom_na"] == total

    vc = summary["verdict_counts_all"]
    assert vc["allowed"] + vc["symmetry_enforced"] + vc["odd_filling"] + vc["error"] == total

    flags = summary["flag_categories"]
    assert (
        flags["A_CONFLICT_symmetry_enforced"] + flags["A_CONFLICT_odd_filling"]
        == flags["A_CONFLICT_total"]
    )
    assert (
        flags["A_CONFLICT_with_f_elements"] + flags["A_CONFLICT_without_f_elements"]
        == flags["A_CONFLICT_total"]
    )

    conflicts_df = pd.read_csv(conflicts_path)
    candidates_df = pd.read_csv(candidates_path)
    assert len(conflicts_df) == flags["A_CONFLICT_total"]
    assert len(candidates_df) == flags["C_CANDIDATES_symmetry_enforced_gapless_nonmagnetic"]


def test_mcp_tool_can_be_insulator() -> None:
    """Verify the FastMCP `can_be_insulator` tool on CIF strings."""
    assert mcp.name == "symlint"

    # 1. SrIrO3 Pnma (SG 62) -> symmetry_enforced, can_be_insulator = False
    sriro3_cif = (DATA_DIR / "SrIrO3_Pnma_mp1105160.cif").read_text()
    res_sriro3 = json.loads(can_be_insulator(sriro3_cif))
    assert res_sriro3["can_be_insulator"] is False
    assert res_sriro3["verdict"] == "symmetry_enforced"
    assert res_sriro3["sg_number"] == 62
    assert res_sriro3["nu"] == 556
    assert res_sriro3["allowed_rule"] == "8N"
    assert "Watanabe" in res_sriro3["citation"]
    assert "096404" in res_sriro3["citation"]
    assert isinstance(res_sriro3["explanation"], str) and len(res_sriro3["explanation"]) > 20

    # 2. Si diamond (SG 227) -> allowed, can_be_insulator = True
    si_cif = (DATA_DIR / "Si_diamond.cif").read_text()
    res_si = json.loads(can_be_insulator(si_cif))
    assert res_si["can_be_insulator"] is True
    assert res_si["verdict"] == "allowed"
    assert res_si["sg_number"] == 227
    assert res_si["nu"] == 28

    # 3. Na bcc (SG 229) -> odd_filling, can_be_insulator = False
    na_cif = (DATA_DIR / "Na_bcc.cif").read_text()
    res_na = json.loads(can_be_insulator(na_cif))
    assert res_na["can_be_insulator"] is False
    assert res_na["verdict"] == "odd_filling"
    assert res_na["sg_number"] == 229
    assert res_na["nu"] == 11


def test_marimo_notebook_valid() -> None:
    """Verify that `notebooks/demo.py` is a valid marimo App."""
    import importlib.util

    demo_path = ROOT_DIR / "notebooks" / "demo.py"
    assert demo_path.exists()
    spec = importlib.util.spec_from_file_location("symlint_demo", demo_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "app")
