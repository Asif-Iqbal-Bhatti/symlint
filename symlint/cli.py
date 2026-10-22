"""Command-line interface for symlint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from symlint.check import check_file


def _format_human_verdict(res: dict) -> str:
    lines = [
        f"File:                  {res.get('file', '<structure>')}",
        f"Formula (primitive):   {res['primitive_formula']} ({res['num_primitive_atoms']} atoms/prim cell)",
        f"Space group:           {res['sg_symbol']} (#{res['sg_number']}) [symprec={res['symprec']}]",
        f"Electron count (nu):   {res['nu']}",
        f"Smallest Wyckoff m_min:{res['m_min']} (primitive multiplicities: {res['wyckoff_multiplicities_primitive']})",
        f"Allowed generators:    {res['allowed_generators']}",
        f"Allowed set S_G ({res['mode']}): {res['allowed_rule']}",
        f"Verdict:               {res['verdict']} (can_be_insulator={res['can_be_insulator']})",
        f"Tolerance check (0.1): SG #{res['sg_number_alt']} ({res['sg_symbol_alt']}), "
        f"nu={res['nu_alt']}, verdict={res['verdict_alt']}, sensitive={res['symprec_sensitive']}",
        f"Explanation:           {res['explanation']}",
        f"Reference:             {res['citation']}",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="symlint",
        description="Symmetry Lint: exact filling-enforced gaplessness check for crystal structures.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "check",
        help="Check one or more structure files (CIF, POSCAR, etc.) for filling-enforced gaplessness.",
    )
    check_parser.add_argument(
        "files",
        nargs="+",
        help="Path(s) to crystal structure file(s), e.g., file.cif",
    )
    check_parser.add_argument(
        "--symprec",
        type=float,
        default=1e-3,
        help="Primary spglib symmetry tolerance in Angstroms (default: 1e-3).",
    )
    check_parser.add_argument(
        "--alt-symprec",
        type=float,
        default=0.1,
        help="Secondary spglib symmetry tolerance for sensitivity check (default: 0.1).",
    )
    check_parser.add_argument(
        "--mode",
        choices=["soc", "spinless"],
        default="soc",
        help="Spin-orbit coupling mode: 'soc' (default) or 'spinless'.",
    )
    check_parser.add_argument(
        "--json",
        action="store_true",
        help="Print output as JSON.",
    )

    subparsers.add_parser(
        "mcp",
        help="Start the FastMCP server exposing the `can_be_insulator` tool over stdio.",
    )

    trace_parser = subparsers.add_parser(
        "trace",
        help="Print the provenance and staged-control chain for a record_id in results/provenance.jsonl.",
    )
    trace_parser.add_argument(
        "record_id",
        help="Provenance record_id (or source ID such as JVASP-109946 or CaZnCu_SG62).",
    )
    trace_parser.add_argument(
        "--file",
        default="results/provenance.jsonl",
        help="Path to provenance JSONL file (default: results/provenance.jsonl).",
    )
    trace_parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON record.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "mcp":
        from symlint.mcp_server import main as mcp_main

        mcp_main()
        return 0

    if args.command == "trace":
        from symlint.provenance import find_provenance_record, format_provenance_trace

        rec = find_provenance_record(args.record_id, jsonl_path=Path(args.file))
        if rec is None:
            print(f"No provenance record matching '{args.record_id}' found in {args.file}.", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(rec, indent=2, default=str))
        else:
            print(format_provenance_trace(rec))
        return 0

    if args.command == "check":
        results = []
        for fpath in args.files:
            res = check_file(
                Path(fpath),
                symprec=args.symprec,
                mode=args.mode,
                alt_symprec=args.alt_symprec,
            )
            results.append(res)
        if args.json:
            payload = results[0] if len(results) == 1 else results
            print(json.dumps(payload, indent=2))
        else:
            print("\n\n".join(_format_human_verdict(r) for r in results))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
