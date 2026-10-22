"""Filling-enforced gaplessness rules for the 230 crystallographic space groups.

Reference:
H. Watanabe, H. C. Po, M. P. Zaletel, and A. Vishwanath,
"Filling-Enforced Gaplessness in Band Structures of the 230 Space Groups",
Phys. Rev. Lett. 117, 096404 (2016), arXiv:1603.05646.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pyxtal.symmetry import Group

ModeType = Literal["soc", "spinless"]

CENTERING_DIVISORS: dict[str, int] = {
    "P": 1,
    "A": 2,
    "B": 2,
    "C": 2,
    "I": 2,
    "F": 4,
    "R": 3,  # Hexagonal setting (standard in ITA / PyXtal / spglib)
}

# The four body/face-centered space groups where the band insulator set (with SOC)
# or the Wyckoff semigroup differs from 2 * m_min * N.
OVERRIDE_SPACE_GROUPS: dict[int, dict[str, str]] = {
    199: {"soc": "4N", "spinless": "4N \\ {4}"},
    214: {"soc": "4N", "spinless": "4N \\ {4}"},
    220: {"soc": "4N \\ {4}", "spinless": "4N \\ {4, 8, 20}"},
    230: {"soc": "8N", "spinless": "8N \\ {8}"},
}


def _validate_sg_and_mode(sg_number: int, mode: str = "soc") -> None:
    if not isinstance(sg_number, int) or not (1 <= sg_number <= 230):
        raise ValueError(f"Space group number must be an int in [1, 230], got {sg_number!r}")
    if mode not in ("soc", "spinless"):
        raise ValueError(f"mode must be 'soc' or 'spinless', got {mode!r}")


@lru_cache(maxsize=256)
def centering_divisor(sg_number: int) -> tuple[str, int]:
    """Return (centering_letter, divisor) for converting conventional to primitive cell."""
    _validate_sg_and_mode(sg_number)
    group = Group(sg_number)
    centering_letter = group.symbol[0].upper()
    if centering_letter not in CENTERING_DIVISORS:
        raise ValueError(
            f"Unrecognized lattice centering '{centering_letter}' for SG {sg_number} ({group.symbol})"
        )
    return centering_letter, CENTERING_DIVISORS[centering_letter]


@lru_cache(maxsize=256)
def wyckoff_positions_primitive(sg_number: int) -> list[dict[str, int | str]]:
    """Return full Wyckoff table with conventional and primitive multiplicities."""
    _validate_sg_and_mode(sg_number)
    group = Group(sg_number)
    centering_letter, divisor = centering_divisor(sg_number)
    rows: list[dict[str, int | str]] = []
    for wp in group.Wyckoff_positions:
        mult = int(wp.multiplicity)
        if mult % divisor != 0:
            raise AssertionError(
                f"SG {sg_number} ({group.symbol}) Wyckoff {mult}{wp.letter} "
                f"not divisible by centering divisor {divisor} ({centering_letter})"
            )
        rows.append(
            {
                "letter": wp.letter,
                "multiplicity_conventional": mult,
                "multiplicity_primitive": mult // divisor,
            }
        )
    return rows


@lru_cache(maxsize=256)
def wyckoff_multiplicities_primitive(sg_number: int) -> list[int]:
    """Return sorted unique primitive Wyckoff multiplicities m_w for a space group.

    Uses pyxtal.symmetry.Group(sg_number).wyckoffs divided by lattice centering
    (P=1, A/B/C=2, I=2, F=4, R=3).
    """
    rows = wyckoff_positions_primitive(sg_number)
    return sorted({int(r["multiplicity_primitive"]) for r in rows})


@lru_cache(maxsize=256)
def primitive_generators(sg_number: int) -> list[int]:
    """Return the atomic insulator generators {2 * m_w} sorted ascending."""
    return [2 * m for m in wyckoff_multiplicities_primitive(sg_number)]


@lru_cache(maxsize=512)
def atomic_insulator_semigroup(sg_number: int, nu_max: int = 2000) -> set[int]:
    """Compute the raw atomic insulator set S_G up to nu_max.

    Defined as all positive integer fillings nu in [1, nu_max] that can be expressed
    as a nonnegative integer linear combination of {2 * m_w}.
    """
    _validate_sg_and_mode(sg_number)
    if nu_max < 1:
        return set()
    generators = primitive_generators(sg_number)
    reachable = bytearray(nu_max + 1)
    reachable[0] = 1
    for g in generators:
        for x in range(g, nu_max + 1):
            if reachable[x - g]:
                reachable[x] = 1
    return {x for x in range(1, nu_max + 1) if reachable[x]}


def _override_allowed_set(sg_number: int, mode: str, nu_max: int) -> set[int]:
    """Return the band insulator allowed set for the four override space groups."""
    if sg_number == 199:
        base = set(range(4, nu_max + 1, 4))
        return base if mode == "soc" else (base - {4})
    if sg_number == 214:
        base = set(range(4, nu_max + 1, 4))
        return base if mode == "soc" else (base - {4})
    if sg_number == 220:
        base = set(range(4, nu_max + 1, 4))
        return (base - {4}) if mode == "soc" else (base - {4, 8, 20})
    if sg_number == 230:
        base = set(range(8, nu_max + 1, 8))
        return base if mode == "soc" else (base - {8})
    raise ValueError(f"SG {sg_number} is not an override space group")


@lru_cache(maxsize=1024)
def allowed_set(sg_number: int, mode: str = "soc", nu_max: int = 2000) -> set[int]:
    """Return the set of band-insulator-allowed fillings S_G for 1 <= nu <= nu_max.

    Parameters
    ----------
    sg_number : int
        International space group number (1..230).
    mode : {"soc", "spinless"}, default "soc"
        Whether spin-orbit coupling is included ("soc") or neglected ("spinless").
    nu_max : int, default 2000
        Maximum electron count per primitive cell to generate in the returned set.
    """
    _validate_sg_and_mode(sg_number, mode)
    if sg_number in OVERRIDE_SPACE_GROUPS:
        return _override_allowed_set(sg_number, mode, nu_max)
    return atomic_insulator_semigroup(sg_number, nu_max=nu_max)


def is_allowed(nu: int, sg_number: int, mode: str = "soc") -> bool:
    """Fast membership check for whether primitive electron count nu is in S_G."""
    _validate_sg_and_mode(sg_number, mode)
    if nu <= 0 or nu % 2 != 0:
        return False
    if sg_number in OVERRIDE_SPACE_GROUPS:
        if sg_number in (199, 214):
            if nu % 4 != 0:
                return False
            return True if mode == "soc" else (nu != 4)
        if sg_number == 220:
            if nu % 4 != 0:
                return False
            excluded = {4} if mode == "soc" else {4, 8, 20}
            return nu not in excluded
        if sg_number == 230:
            if nu % 8 != 0:
                return False
            return True if mode == "soc" else (nu != 8)
    nu_bound = max(2000, int(nu))
    return int(nu) in allowed_set(sg_number, mode=mode, nu_max=nu_bound)


def allowed_rule_string(sg_number: int, mode: str = "soc") -> str:
    """Return a concise human-readable description of S_G (e.g. '8N', '4N \\ {4}')."""
    _validate_sg_and_mode(sg_number, mode)
    if sg_number in OVERRIDE_SPACE_GROUPS:
        return OVERRIDE_SPACE_GROUPS[sg_number][mode]
    m_min = wyckoff_multiplicities_primitive(sg_number)[0]
    return f"{2 * m_min}N"


def classify_filling(nu: int, sg_number: int, mode: str = "soc") -> str:
    """Classify primitive electron count nu in space group sg_number.

    Returns
    -------
    str
        - "odd_filling": nu is odd (trivial Kramers metal)
        - "symmetry_enforced": nu is even but not in S_G (nonsymmorphic filling-enforced gapless)
        - "allowed": nu is in S_G (band insulator is symmetry-allowed)
    """
    _validate_sg_and_mode(sg_number, mode)
    if nu <= 0:
        raise ValueError(f"Electron count nu must be positive, got {nu}")
    if nu % 2 == 1:
        return "odd_filling"
    if is_allowed(nu, sg_number, mode=mode):
        return "allowed"
    return "symmetry_enforced"
