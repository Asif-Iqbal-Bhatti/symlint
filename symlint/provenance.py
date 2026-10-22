"""Step 5b (Trust): Self-contained provenance tracking for symlint (no AiiDA/DB dependency).

Inspired by:
P. Kraus, E. Bainglass, F. F. Ramirez, E. S.ibille, L.udin, N. Marzari,
C. Battaglia, and G. Pizzi, "A bridge between trust and control: computational
workflows meet automated battery cycling", J. Mater. Chem. A 12, 10773 (2024),
DOI: 10.1039/d3ta06889g.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from symlint import __version__ as SYMLINT_VERSION
from symlint.check import to_spglib_cell

_GIT_COMMIT_CACHE: str | None = None


def get_git_commit() -> str:
    """Return the short git commit hash of the repository, or 'unknown'."""
    global _GIT_COMMIT_CACHE
    if _GIT_COMMIT_CACHE is not None:
        return _GIT_COMMIT_CACHE
    try:
        repo_dir = Path(__file__).resolve().parent.parent
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_dir),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        ).strip()
        _GIT_COMMIT_CACHE = out or "unknown"
    except Exception:
        _GIT_COMMIT_CACHE = "unknown"
    return _GIT_COMMIT_CACHE


def get_package_versions(include_mlip: bool = False) -> dict[str, str]:
    """Collect versions of core symmetry packages (and optional MLIP/DFT packages)."""
    pkgs = ["spglib", "pyxtal", "numpy", "ase", "pymatgen"]
    if include_mlip:
        pkgs.extend(["torch", "mace-torch", "tensorpotential", "phonopy", "alignn"])
    versions: dict[str, str] = {"symlint": SYMLINT_VERSION}
    for pkg in pkgs:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def canonical_structure_sha256(structure: Any) -> str:
    """Compute a deterministic SHA-256 hash of a crystal structure's canonical cell tuple."""
    try:
        lattice, frac_coords, numbers = to_spglib_cell(structure)
        lat_arr = np.round(np.asarray(lattice, dtype=float), 6)
        pos_arr = np.round(np.mod(np.asarray(frac_coords, dtype=float), 1.0), 6)
        num_arr = np.asarray(numbers, dtype=int)
        payload = {
            "lattice": lat_arr.tolist(),
            "frac_coords": pos_arr.tolist(),
            "atomic_numbers": num_arr.tolist(),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()
    except Exception:
        raw = repr(structure).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


def make_record_id(source_type: str, source_id: str, structure_sha256: str) -> str:
    """Generate a compact, deterministic, human-readable provenance record ID."""
    clean_src = "".join(c if c.isalnum() or c in "-_" else "_" for c in f"{source_type}-{source_id}")
    return f"sym-{clean_src}-{structure_sha256[:8]}"


def create_provenance_record(
    structure: Any,
    source: dict[str, Any],
    parameters: dict[str, Any],
    output: dict[str, Any],
    stage_trace: list[dict[str, Any]] | None = None,
    stopped_early: bool = False,
    stop_reason: str | None = None,
    record_id: str | None = None,
    include_mlip: bool = False,
    structure_sha256: str | None = None,
) -> dict[str, Any]:
    """Create a complete JSON-serializable provenance record for a symlint evaluation."""
    sha256_hex = structure_sha256 or canonical_structure_sha256(structure)
    src_type = str(source.get("dataset") or source.get("type") or "file")
    src_id = str(source.get("id") or source.get("path") or "struct")
    rec_id = record_id or make_record_id(src_type, src_id, sha256_hex)

    return {
        "record_id": rec_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "structure_sha256": sha256_hex,
        "source": source,
        "symlint_version": SYMLINT_VERSION,
        "git_commit": get_git_commit(),
        "package_versions": get_package_versions(include_mlip=include_mlip),
        "parameters": parameters,
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
        "stage_trace": stage_trace or [],
        "output": output,
    }


def write_provenance_jsonl(
    records: list[dict[str, Any]],
    jsonl_path: Path | str = Path("results/provenance.jsonl"),
    append: bool = True,
) -> int:
    """Write or append provenance records to `results/provenance.jsonl`, deduplicating by `record_id`."""
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict[str, Any]] = {}
    if append and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    existing[str(obj["record_id"])] = obj
                except Exception:
                    continue

    for rec in records:
        existing[str(rec["record_id"])] = rec

    with open(path, "w", encoding="utf-8") as f:
        for rec in existing.values():
            f.write(json.dumps(rec, default=str) + "\n")

    return len(existing)


def find_provenance_record(
    query: str,
    jsonl_path: Path | str = Path("results/provenance.jsonl"),
) -> dict[str, Any] | None:
    """Find a provenance record by exact `record_id`, prefix, or `source.id`."""
    path = Path(jsonl_path)
    if not path.exists():
        return None
    q = query.strip()
    partial_match: dict[str, Any] | None = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            rid = str(rec.get("record_id", ""))
            sid = str(rec.get("source", {}).get("id", ""))
            if rid == q or sid == q:
                return rec
            if partial_match is None and (q in rid or q in sid):
                partial_match = rec
    return partial_match


def format_provenance_trace(rec: dict[str, Any]) -> str:
    """Format a provenance record and its staged control trace for CLI/README display."""
    lines = [
        f"PROVENANCE & CONTROL TRACE: {rec.get('record_id')}",
        "=" * 78,
        f"Timestamp (UTC)   : {rec.get('timestamp_utc')}",
        f"Structure SHA-256 : {rec.get('structure_sha256')}",
        f"Source            : {json.dumps(rec.get('source', {}))}",
        f"Code Version      : symlint {rec.get('symlint_version')} (git {rec.get('git_commit')})",
        f"Package Versions  : {json.dumps(rec.get('package_versions', {}))}",
        f"Parameters        : {json.dumps(rec.get('parameters', {}))}",
        f"Stopped Early     : {rec.get('stopped_early')} (reason: {rec.get('stop_reason')})",
    ]
    stages = rec.get("stage_trace") or []
    if stages:
        lines.append("-" * 78)
        lines.append(f"Staged Control Pipeline ({len(stages)} stages executed):")
        for idx, st in enumerate(stages, start=1):
            status = "PASS" if st.get("passed", True) else "STOP"
            lines.append(
                f"  Stage {idx} [{status}] {st.get('stage')}: {st.get('summary')}"
            )
    lines.append("-" * 78)
    lines.append("Final Output:")
    lines.append(json.dumps(rec.get("output", {}), indent=2, default=str))
    return "\n".join(lines)
