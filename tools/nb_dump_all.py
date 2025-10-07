#!/usr/bin/env python3
# filename: tools/nb_dump_all.py
import json, glob
from pathlib import Path

out = Path("out/notebooks_all_source.md")
with out.open("w", encoding="utf-8") as f:
    for nb_path in sorted(glob.glob("notebooks/*.ipynb")):
        nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
        f.write(f"\n\n# ==== {nb_path} ====\n\n")
        for i, cell in enumerate(nb.get("cells", []), 1):
            ct = cell.get("cell_type")
            src = "".join(cell.get("source", []))
            if not src.strip():
                continue
            f.write(f"\n\n## cell {i} [{ct}]\n\n")
            if ct == "markdown":
                f.write(src.strip() + "\n")
            elif ct == "code":
                f.write("```python\n" + src.rstrip() + "\n```\n")
print(f"Wrote: {out}")
