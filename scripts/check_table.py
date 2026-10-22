"""Step 1 sanity check: compute S_G for all 230 space groups and compare <2*m_w> vs 2*m_min*N."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_230_spacegroups import main

if __name__ == "__main__":
    main(nu_max=200)
