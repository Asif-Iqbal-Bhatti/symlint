"""symlint: Symmetry Lint for materials databases.

Exact filling-enforced gaplessness checks for crystals across all 230 space groups,
based on H. Watanabe, H. C. Po, M. P. Zaletel, and A. Vishwanath,
"Filling-Enforced Gaplessness in Band Structures of the 230 Space Groups",
Phys. Rev. Lett. 117, 096404 (2016) [arXiv:1603.05646].
"""

from symlint.check import check_cif_string, check_file, check_structure
from symlint.filling import (
    OVERRIDE_SPACE_GROUPS,
    allowed_rule_string,
    allowed_set,
    atomic_insulator_semigroup,
    centering_divisor,
    classify_filling,
    is_allowed,
    wyckoff_multiplicities_primitive,
)

__version__ = "0.1.0"

__all__ = [
    "OVERRIDE_SPACE_GROUPS",
    "allowed_rule_string",
    "allowed_set",
    "atomic_insulator_semigroup",
    "centering_divisor",
    "check_cif_string",
    "check_file",
    "check_structure",
    "classify_filling",
    "is_allowed",
    "wyckoff_multiplicities_primitive",
]
