# symlint: a physics guardrail for AI driven materials discovery

**One line:** An exact symmetry theorem that tells databases, ML models, generative design and AI agents, in milliseconds and without DFT, when a crystal cannot be an insulator, and turns that into shortlists of semimetals and superconductor candidates.

*Team Forced Crossings (Asif Iqbal Bhatti, solo) | 4th Open Scientific Intelligence Hackathon 2026*

Every number below comes from `results/summary.json`, `results/timing.json` or the CSV files in `results/`.

---

## Two minute story

**1. Problem (0:00 to 0:20).** AI models and agents now propose and label crystals by the hundreds of thousands. They do not know exact band theory, so nothing stops a model from predicting a band gap for a crystal that, by symmetry, cannot have one.

**2. The rule (0:20 to 0:45).** Watanabe, Po, Zaletel and Vishwanath (PRL 117, 096404, 2016) proved that for a nonmagnetic crystal, the space group and the number of electrons per primitive cell decide whether a band insulator is allowed at all. symlint implements this for all 230 space groups, checks it against Table I of the paper, and needs only the structure: median 1.7 ms per crystal, no DFT. Example: SrIrO3 in Pnma has 556 electrons per primitive cell; Pnma needs a multiple of 8; 556 = 4 mod 8, so it cannot be a band insulator.

**3. What it finds (0:45 to 1:35).**
- **Database census (JARVIS dft_3d, 93,902 entries).** Among 70,658 nonmagnetic entries, 4,079 are reported gapless at a symmetry forbidden filling: forced gapless candidates. 992 nonmagnetic entries carry a gap above 0.1 eV at a forbidden filling; 991 of them keep that verdict at a looser symmetry tolerance. These are curation targets for the database, 817 of them containing lanthanides or actinides, where the f electron treatment is the likely cause.
- **ML band gap models (ALIGNN 2.0, official pretrained JARVIS models).** On 463 held out structures not in JARVIS whose filling forbids a gap, at least one of the two models predicts a gap above 0.1 eV for 27.0% of them (MBJ model 24.8%, OptB88vdW model 6.9%). The theorem is a free, exact output constraint.
- **Superconductors (Bercx et al., PRX Energy 2025, MC3D).** Among 4,528 nonmagnetic DFT metals, symmetry enforced metals pass the Tc > 1 K screen 41.2% of the time versus 31.8% for allowed metals (chi squared 23.3, p about 1e-6). 180 of the 945 EPW candidates are symmetry enforced, in nonsymmorphic groups, the shortlist for follow up.
- **Discovery with two MLIPs (MACE and GRACE, laptop GPU).** symlint pre-screens substitutions in nonsymmorphic prototypes before any GPU time is spent; 11 of 14 constructed Pnma and Cmcm structures, none of them in JARVIS, are dynamically stable in both potentials, and the two potentials agree on 27 of 29 crystals overall.

**4. Honest scope (1:35 to 1:50).** Nonmagnetic, band theory level only; magnetic entries are reported separately. "Forced gapless" means metal or nodal semimetal, not a proven topological material. MLIP stability is a screening estimate; DFT inputs with SOC are prepared for the top candidates.

**5. Ready to use (1:50 to 2:00).** Installable Python package with a CLI (`symlint check`, `symlint trace`), an MCP server so LLM agents can call `can_be_insulator`, a marimo notebook, and a provenance record for every result (997 records, each traceable with `symlint trace`).

---

## Example trace shown in the demo

```text
$ symlint trace CaZnCu_SG62
  Stage 1 [PASS] 1_symmetry_and_filling_check: SG 62 (Pnma), nu=316, rule=8N, verdict=symmetry_enforced
  Stage 2 [PASS] 2_subgroup_robustness_check: Survives 2/7 maximal subgroups (2/7 t-subgroups)
  Stage 3 [PASS] 3_mlip_relaxation_and_symmetry_recheck: SG 62 preserved=True, E_hull(MLIP)=0.0000 eV/atom
  Stage 4 [PASS] 4_dual_mlip_phonons_and_stability: w_min(MACE)=-0.00 THz, w_min(GRACE)=+0.00 THz, both_stable=True
```

## What is a prototype or next

- Module C (experimental insulators at forbidden filling): prototype only; composition to structure matching produces mostly artifacts.
- Agent design loop: a scripted demonstration of the MCP tools; an LLM driven loop is next.
- MLIP hull: approximate (unrelaxed competing phases); a proper hull is next.
- Cross-database census on Materials Project (and Alexandria): this run uses JARVIS only; entries flagged in both databases will be the strongest curation cases.
- Magnetic space groups (1,651) and 2D layer groups: future work. 21,519 magnetic JARVIS 3D entries were excluded from the census.
