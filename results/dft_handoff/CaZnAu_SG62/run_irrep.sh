#!/usr/bin/env bash
# Confirm symmetry-enforced band touchings and little-group irreps from VASP WAVECAR
# Requires: pip install irrep
set -euo pipefail

echo "Running IrRep symmetry indicator & band-crossing analysis for CaZnAu_SG62 (SG #62, nu=516)..."
irrep -code=vasp -fWAV=WAVECAR -fPOS=POSCAR -Ecut=100 -IBstart=1 -IBend=532 -spinor > irrep_output.txt
echo "Done. Inspect irrep_output.txt for little-group corepresentations at high-symmetry momenta."
