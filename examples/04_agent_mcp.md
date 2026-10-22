# Using symlint from an AI agent (MCP)

symlint ships an MCP server, so any MCP client (Claude Desktop, Claude Code, other agent frameworks) can call it as a tool.

## Tools

| Tool | Input | Output |
|---|---|---|
| `can_be_insulator` | CIF string, optional `symprec`, `mode` (`soc` or `spinless`) | verdict JSON, one sentence explanation, citation |
| `robustness_score` | CIF string | how many maximal symmetry-lowering distortions keep the filling forbidden |
| `relax_and_stability` | CIF string | MLIP relaxation and phonon check (needs `pip install "symlint[mlip]"` and ideally a GPU) |

## Start the server

```bash
symlint mcp          # or: symlint-mcp
```

It speaks MCP over stdio.

## Register it with a client

Claude Code:

```bash
claude mcp add symlint -- symlint-mcp
```

Claude Desktop or any client that reads an `mcpServers` JSON block:

```json
{
  "mcpServers": {
    "symlint": { "command": "symlint-mcp" }
  }
}
```

If `symlint-mcp` is not on the client's PATH (for example, it lives in a virtual environment), give the full path, e.g. `/path/to/.venv/bin/symlint-mcp`.

## Call the tool from Python without a client

```python
from pathlib import Path
from symlint.mcp_server import can_be_insulator

print(can_be_insulator(Path("tests/data/SrIrO3_Pnma_mp1105160.cif").read_text()))
```
