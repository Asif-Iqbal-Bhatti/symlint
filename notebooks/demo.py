import marimo

__generated_with = "0.9.0"
app = marimo.App(width="full", app_title="symlint: A Physics Guardrail for AI-Driven Materials Discovery")


@app.cell
def _():
    import json
    from pathlib import Path
    import marimo as mo
    import pandas as pd

    from symlint.check import check_cif_string
    from symlint.mcp_server import can_be_insulator, robustness_score

    root_dir = Path(__file__).resolve().parent.parent
    results_dir = root_dir / "results"
    tests_data_dir = root_dir / "tests" / "data"
    return (
        Path,
        can_be_insulator,
        check_cif_string,
        json,
        mo,
        pd,
        results_dir,
        robustness_score,
        root_dir,
        tests_data_dir,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # `symlint`: A Physics Guardrail for AI-Driven Materials Discovery
        **An exact symmetry theorem that tells databases, ML models, generative design, and AI agents, in milliseconds and without DFT, when a crystal cannot be an insulator, and turns that into shortlists of semimetals, exotic correlated insulators, and superconductor candidates.**

        Based on H. Watanabe, H. C. Po, M. P. Zaletel, and A. Vishwanath, *"Filling-Enforced Gaplessness in Band Structures of the 230 Space Groups"*, Phys. Rev. Lett. **117**, 096404 (2016) [arXiv:1603.05646].
        """
    )
    return


@app.cell
def _(mo, tests_data_dir):
    preset_files = {
        "SrIrO3 (Pnma, SG 62) [Symmetry-Enforced Gapless]": tests_data_dir / "SrIrO3_Pnma_mp1105160.cif",
        "Si (Fd-3m, SG 227) [Allowed Insulator]": tests_data_dir / "Si_diamond.cif",
        "NaCl (Fm-3m, SG 225) [Allowed Insulator]": tests_data_dir / "NaCl_rocksalt.cif",
        "Na (Im-3m, SG 229) [Odd Filling Kramers Metal]": tests_data_dir / "Na_bcc.cif",
    }

    cif_upload = mo.ui.file(
        filetypes=[".cif"],
        multiple=False,
        label="Upload a crystal structure (.cif)",
    )
    preset_selector = mo.ui.dropdown(
        options=list(preset_files.keys()),
        value="SrIrO3 (Pnma, SG 62) [Symmetry-Enforced Gapless]",
        label="Or select a benchmark structure",
    )
    mode_selector = mo.ui.radio(
        options=["soc", "spinless"],
        value="soc",
        label="Spin-orbit coupling mode",
    )
    symprec_slider = mo.ui.dropdown(
        options={"1e-3 A (default)": 1e-3, "1e-2 A": 1e-2, "1e-1 A": 1e-1, "1e-4 A": 1e-4},
        value="1e-3 A (default)",
        label="Primary spglib symprec",
    )

    mo.vstack(
        [
            mo.md("## 1. Live Crystal Structure Symmetry & Filling Linter"),
            mo.hstack([cif_upload, preset_selector, mode_selector, symprec_slider], justify="start"),
        ]
    )
    return cif_upload, mode_selector, preset_files, preset_selector, symprec_slider


@app.cell
def _(
    can_be_insulator,
    cif_upload,
    json,
    mo,
    mode_selector,
    preset_files,
    preset_selector,
    robustness_score,
    symprec_slider,
):
    if cif_upload.value:
        active_name = cif_upload.value[0].name
        cif_text = cif_upload.value[0].contents.decode("utf-8")
    else:
        active_name = preset_selector.value
        cif_path = preset_files[active_name]
        cif_text = cif_path.read_text()

    mcp_json_str = can_be_insulator(
        cif_string=cif_text,
        symprec=float(symprec_slider.value),
        mode=mode_selector.value,
    )
    verdict_data = json.loads(mcp_json_str)
    rob_json_str = robustness_score(
        cif_string=cif_text,
        symprec=float(symprec_slider.value),
        mode=mode_selector.value,
    )
    rob_data = json.loads(rob_json_str)

    badge_map = {
        "allowed": ("ALLOWED (Band Insulator Permitted by Symmetry)", "success"),
        "symmetry_enforced": ("SYMMETRY-ENFORCED GAPLESS (Forced Metal or Nodal Semimetal)", "danger"),
        "odd_filling": ("ODD FILLING (Trivial Kramers Metal)", "warn"),
    }
    badge_text, badge_kind = badge_map.get(verdict_data["verdict"], (verdict_data["verdict"], "neutral"))

    mo.vstack(
        [
            mo.callout(
                mo.md(
                    f"### Verdict for `{active_name}`: **{badge_text}**\n\n"
                    f"**Explanation:** {verdict_data['explanation']}\n\n"
                    f"- **Space Group:** `{verdict_data['sg_symbol']}` (#{verdict_data['sg_number']}) "
                    f"at `symprec={verdict_data['symprec']}`\n"
                    f"- **Primitive Cell Formula:** `{verdict_data['primitive_formula']}` "
                    f"({verdict_data['num_primitive_atoms']} atoms/primitive cell)\n"
                    f"- **Primitive Electron Count ($\\nu$):** `{verdict_data['nu']}`\n"
                    f"- **Smallest Primitive Wyckoff Multiplicity ($m_{{\\min}}$):** `{verdict_data['m_min']}` "
                    f"(generators $2m_w$: `{verdict_data['allowed_generators']}`)\n"
                    f"- **Allowed Band Insulator Set ($S_G$, mode={verdict_data['mode']}):** `{verdict_data['allowed_rule']}`\n"
                    f"- **Maximal Subgroup Distortion Robustness:** `{rob_data.get('summary', 'N/A')}`\n"
                    f"- **Symmetry Tolerance Check (0.1 A):** SG #{verdict_data['symprec_0_1_check']['sg_number']} "
                    f"(`{verdict_data['symprec_0_1_check']['sg_symbol']}`), "
                    f"$\\nu={verdict_data['symprec_0_1_check']['nu']}$, "
                    f"verdict=`{verdict_data['symprec_0_1_check']['verdict']}`, "
                    f"sensitive=`{verdict_data['symprec_0_1_check']['symprec_sensitive']}`"
                ),
                kind=badge_kind,
            ),
            mo.accordion(
                {
                    "MCP Tool `can_be_insulator` JSON Output": mo.md(f"```json\n{mcp_json_str}\n```"),
                    "MCP Tool `robustness_score` JSON Output": mo.md(f"```json\n{rob_json_str}\n```"),
                }
            ),
        ]
    )
    return active_name, badge_kind, badge_text, cif_text, mcp_json_str, rob_data, rob_json_str, verdict_data


@app.cell
def _(json, mo, pd, results_dir):
    summary_path = results_dir / "summary.json"
    conflicts_path = results_dir / "conflicts.csv"
    candidates_path = results_dir / "candidates.csv"
    module_a_path = results_dir / "moduleA_ml_violations.csv"
    module_d_path = results_dir / "moduleD_mc3d_epw_symmetry_enforced.csv"
    new_cand_path = results_dir / "new_candidates.csv"
    fig_path = results_dir / "gap_distribution_by_verdict.png"
    funnel_fig_path = results_dir / "moduleD_superconductors_funnel.png"

    summary_data = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    conflicts_df = pd.read_csv(conflicts_path) if conflicts_path.exists() else pd.DataFrame()
    candidates_df = pd.read_csv(candidates_path) if candidates_path.exists() else pd.DataFrame()
    module_a_df = pd.read_csv(module_a_path) if module_a_path.exists() else pd.DataFrame()
    module_d_df = pd.read_csv(module_d_path) if module_d_path.exists() else pd.DataFrame()
    new_cand_df = pd.read_csv(new_cand_path) if new_cand_path.exists() else pd.DataFrame()

    flags = summary_data.get("flags", summary_data.get("flag_categories", {}))
    mo.vstack(
        [
            mo.md("## 2. Database Census & Cross-Module Summary"),
            mo.hstack(
                [
                    mo.stat(
                        label="Total Structures Audited",
                        value=f"{summary_data.get('total_entries', 0):,}",
                        caption="JARVIS dft_3d",
                    ),
                    mo.stat(
                        label="CONFLICT (Nonmagnetic + Gap > 0.1 eV)",
                        value=f"{flags.get('CONFLICT_total', flags.get('A_CONFLICT_total', 0)):,}",
                        caption=f"Symmetry-enforced: {flags.get('CONFLICT_symmetry_enforced', 0):,} | Odd: {flags.get('CONFLICT_odd_filling', 0):,}",
                    ),
                    mo.stat(
                        label="MAGNETIC_EXCEPTION",
                        value=f"{flags.get('MAGNETIC_EXCEPTION_total', flags.get('B_MAGNETIC_EXCEPTION_total', 0)):,}",
                        caption="Magnetic entries with gap > 0.1 eV",
                    ),
                    mo.stat(
                        label="CANDIDATE (Forced-Gapless)",
                        value=f"{flags.get('CANDIDATE_symmetry_enforced_gapless_nonmagnetic', flags.get('C_CANDIDATES_symmetry_enforced_gapless_nonmagnetic', 0)):,}",
                        caption=f"Near hull (ehull <= 25 meV): {flags.get('CANDIDATE_on_hull_ehull_le_0_025', 0):,}",
                    ),
                    mo.stat(
                        label="SG_MISMATCH",
                        value=f"{flags.get('SG_MISMATCH_db_vs_spglib_1e-3', flags.get('D_SPG_MISMATCH_db_vs_spglib_1e-3', 0)):,}",
                        caption="DB spg_number != spglib (1e-3 A)",
                    ),
                ],
                justify="space-around",
            ),
            mo.hstack(
                [
                    mo.image(src=str(fig_path)) if fig_path.exists() else mo.md(""),
                    mo.image(src=str(funnel_fig_path)) if funnel_fig_path.exists() else mo.md(""),
                ]
            ),
        ]
    )
    return (
        candidates_df,
        candidates_path,
        conflicts_df,
        conflicts_path,
        fig_path,
        flags,
        funnel_fig_path,
        module_a_df,
        module_a_path,
        module_d_df,
        module_d_path,
        new_cand_df,
        new_cand_path,
        summary_data,
        summary_path,
    )


@app.cell
def _(candidates_df, conflicts_df, mo):
    verdict_filter = mo.ui.dropdown(
        options=["All CONFLICT entries", "symmetry_enforced only", "odd_filling only"],
        value="symmetry_enforced only",
        label="Filter CONFLICT Entries by Verdict",
    )
    robust_only = mo.ui.checkbox(
        value=True,
        label="Only show tolerance-insensitive conflicts (symprec 1e-3 and 0.1 agree)",
    )
    mo.vstack(
        [
            mo.md("## 3. Browse Results Across All Modules (`CONFLICT`, `CANDIDATE`, ALIGNN Violations, MC3D Superconductors, New MLIP Candidates)"),
            mo.hstack([verdict_filter, robust_only], justify="start"),
        ]
    )
    return robust_only, verdict_filter


@app.cell
def _(candidates_df, conflicts_df, mo, module_a_df, module_d_df, new_cand_df, robust_only, verdict_filter):
    filtered_conflicts = conflicts_df.copy()
    if not filtered_conflicts.empty:
        if verdict_filter.value == "symmetry_enforced only":
            filtered_conflicts = filtered_conflicts[filtered_conflicts["verdict"] == "symmetry_enforced"]
        elif verdict_filter.value == "odd_filling only":
            filtered_conflicts = filtered_conflicts[filtered_conflicts["verdict"] == "odd_filling"]
        if robust_only.value and "verdict_sensitive" in filtered_conflicts.columns:
            filtered_conflicts = filtered_conflicts[~filtered_conflicts["verdict_sensitive"]]

    display_cols = [
        "record_id",
        "jid",
        "formula",
        "sg_number",
        "sg_symbol",
        "db_spg_number",
        "nu",
        "allowed_rule",
        "verdict",
        "optb88vdw_bandgap",
        "mbj_bandgap",
        "abs_magmom",
        "ehull",
        "symprec_sensitive",
    ]
    avail_conflict_cols = [c for c in display_cols if c in filtered_conflicts.columns]
    avail_cand_cols = [c for c in display_cols if c in candidates_df.columns]

    mo.ui.tabs(
        {
            f"CONFLICT ({len(filtered_conflicts):,})": mo.ui.table(
                filtered_conflicts[avail_conflict_cols].head(500),
                selection=None,
            ),
            f"CANDIDATE ({len(candidates_df):,})": mo.ui.table(
                candidates_df[avail_cand_cols].head(500),
                selection=None,
            ),
            f"Module A: ALIGNN Violations ({len(module_a_df):,})": mo.ui.table(
                module_a_df.head(500),
                selection=None,
            ),
            f"Module D: MC3D Nonsymmorphic EPW Superconductors ({len(module_d_df):,})": mo.ui.table(
                module_d_df.head(500),
                selection=None,
            ),
            f"Step 8c: New MLIP-Validated Candidates ({len(new_cand_df):,})": mo.ui.table(
                new_cand_df,
                selection=None,
            ),
        }
    )
    return avail_cand_cols, avail_conflict_cols, display_cols, filtered_conflicts


if __name__ == "__main__":
    app.run()
