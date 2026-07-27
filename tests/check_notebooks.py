"""Validate the notebooks parse and that their code cells compile.

A notebook with a syntax error costs an hour on Thursday and looks like a
mysterious kernel failure. Checking is free.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAIL = []

for nb_path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    md = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
    print(f"{nb_path.name}: {len(nb['cells'])} cells ({len(code)} code, {len(md)} md)")

    for i, cell in enumerate(code):
        src = "".join(cell["source"])
        # Shell escapes and magics are not Python. Neither are their backslash
        # continuation lines, which is easy to miss and shows up as a bogus
        # "unexpected indent" on the line after a multi-line !command.
        lines, in_shell = [], False
        for line in src.splitlines():
            is_shell = line.lstrip().startswith(("!", "%")) or in_shell
            in_shell = is_shell and line.rstrip().endswith("\\")
            lines.append("pass" if is_shell else line)
        cleaned = "\n".join(lines)
        try:
            compile(cleaned, f"{nb_path.name}:cell{i}", "exec")
        except SyntaxError as e:
            FAIL.append(f"{nb_path.name} cell {i}: {e}")
            print(f"  FAIL cell {i}: {e}")

print()
if FAIL:
    print(f"{len(FAIL)} broken cell(s)")
    sys.exit(1)
print("all notebook code cells compile")
