# Vendored: FreeCAD MCP (neka-nat/freecad-mcp)

- source: https://github.com/neka-nat/freecad-mcp, commit `78bf0230a03324b56c9a0b7f55150fb5be2a0d42` (2026-10-13)
- licence: MIT (`LICENSE`, unchanged)
- copied unmodified: `addon/` (the FreeCAD addon that serves the RPC calls inside the GUI session), `src/` (the MCP
  server), `pyproject.toml`, `uv.lock`, `LICENSE`, `README.md`; left out: `assets/`, `docs/`, `examples/`, `scripts/`,
  `tests/`
- the task's image installs the addon into FreeCAD's `Mod/` and the server into `/opt/freecad-mcp` (`uv sync --frozen`)
