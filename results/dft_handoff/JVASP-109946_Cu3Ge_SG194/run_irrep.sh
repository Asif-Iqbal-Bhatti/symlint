#!/usr/bin/env bash
# Confirm symmetry-enforced band touchings and little-group irreps from VASP WAVECAR
# Requires: pip install irrep
set -euo pipefail

echo "Running IrRep symmetry indicator & band-crossing analysis for JVASP-109946_Cu3Ge_SG194 (SG #194, nu=238)..."
irrep -code=vasp -fWAV=WAVECAR -fPOS=POSCAR -Ecut=100 -IBstart=1 -IBend=254 -spinor > irrep_output.txt
echo "Done. Inspect irrep_output.txt for little-group corepresentations at high-symmetry momenta."
