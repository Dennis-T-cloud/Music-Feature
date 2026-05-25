"""Execute task2.ipynb end-to-end without requiring Jupyter.

This runner keeps the notebook as the source of truth, but applies the
handoff settings needed on the Windows GPU machine.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)

    with open("task2.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)

    namespace = {
        "__name__": "__main__",
        "__file__": str(Path("task2.ipynb").resolve()),
        "display": print,
    }

    skip_cells = {
        26,  # Debug training; full training below replaces it.
    }

    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code" or idx in skip_cells:
            continue

        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        source = source.replace("RUN_FULL_TRAINING = False", "RUN_FULL_TRAINING = True")
        source = source.replace("epochs=10,", "epochs=20,")
        source = source.replace('output_path=f"/tmp/_eval_candidate.mid"', 'output_path="_eval_candidate.mid"')

        print(f"\n=== Executing notebook cell {idx} ===", flush=True)
        exec(compile(source, f"task2.ipynb cell {idx}", "exec"), namespace)


if __name__ == "__main__":
    main()
