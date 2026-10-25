#!/usr/bin/env python
"""Run the Kaggle notebook's own cells against faked mounts, before queueing a run.

    python -m scripts.rehearse_kernel --limit 10
    python -m scripts.rehearse_kernel --kernel-dir kaggle/analog-baseline --limit 3

A queued Kaggle run costs 30 minutes before it even reaches the first import, and a
missing mount, a stale `dataset_sources` or a bad path fails exactly there. This stages a
fake `/kaggle/input` — one directory per dataset, the competition files symlinked — and
**executes the notebook's actual cell sources** against it, so what is tested is the
bytes that will run rather than a paraphrase of them.

The pipeline cell is re-run with a molecule limit, since the point is to prove the wiring
works, not to reproduce the full inference.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from casmi import data as D

ROOT = Path(__file__).resolve().parents[1]


def stage_mounts(into: Path, shape: str = "expanded") -> None:
    """One directory per dataset, as Kaggle lays them out.

    Both library shapes have to be rehearsed, because which one arrives is Kaggle's
    decision: `kaggle_dataset.py` uploads with `-r zip`, so the dataset holds
    `casmi.zip`, and Kaggle may serve that as-is or expand it. The notebook handles
    both; only testing one leaves the other to fail in a queued run.
    """
    src = into / "casmi-src"
    src.mkdir(parents=True)
    if shape == "zip":
        import zipfile
        with zipfile.ZipFile(src / "casmi.zip", "w") as z:
            for f in sorted((ROOT / "src" / "casmi").rglob("*.py")):
                z.write(f, Path("casmi") / f.relative_to(ROOT / "src" / "casmi"))
    else:
        shutil.copytree(ROOT / "src" / "casmi", src / "casmi",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    comp = into / "enveda-CASMI26-molecule-id-mass-spectra"
    comp.mkdir(parents=True)
    for name in ("train.parquet", "test.parquet", "sample_submission.csv"):
        target = D.RAW / name
        if not target.exists():
            raise SystemExit(f"{target} is missing; cannot rehearse")
        (comp / name).symlink_to(target.resolve())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kernel-dir", default="kaggle/analog-baseline", type=Path)
    ap.add_argument("--shape", choices=("expanded", "zip"), default="expanded",
                    help="which library mount shape to fake")
    ap.add_argument("--bootstrap-only", action="store_true",
                    help="run only the mount-finding and import cells")
    ap.add_argument("--limit", type=int, default=5,
                    help="test molecules to actually rank")
    args = ap.parse_args()

    meta = json.loads((args.kernel_dir / "kernel-metadata.json").read_text())
    notebook = json.loads((args.kernel_dir / meta["code_file"]).read_text())
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    print(f"{args.kernel_dir}: {len(cells)} code cell(s), id {meta['id']}")
    print(f"  internet {'ON — WRONG' if meta['enable_internet'] else 'off'}, "
          f"gpu {'on' if meta['enable_gpu'] else 'off'}, "
          f"sources {meta['dataset_sources']}")
    assert not meta["enable_internet"], "a scored run must have internet off"

    with tempfile.TemporaryDirectory(prefix="casmi-rehearse-") as tmp:
        mounts = Path(tmp)
        stage_mounts(mounts, args.shape)
        os.environ["KAGGLE_INPUT"] = str(mounts)
        print(f"  faked mounts at {mounts}\n")

        scope: dict = {"__name__": "__main__"}
        for i, cell in enumerate(cells):
            source = "".join(cell["source"])
            # The pipeline cell is the slow one. Rehearse the wiring on a few molecules;
            # proving it ranks 400 is what the real run is for.
            if "run_submission" in source:
                source = source.replace(
                    'out_path="submission.csv",',
                    f'out_path="submission.csv", limit_molecules={args.limit},')
            # With a limit the submission legitimately covers fewer molecules than the
            # test file, so the completeness assertion is rehearsed against what was
            # actually produced. The real run checks against all 400.
            source = source.replace("S.check_submission(submission, test_ids)",
                                    "S.check_submission(submission, submission.molecule_id)")
            if args.bootstrap_only and "run_submission" in source:
                print("(stopping before the pipeline cell)")
                break
            print(f"----- cell {i} -----", flush=True)
            exec(compile(source, f"<cell {i}>", "exec"), scope)

    print("\nrehearsal passed: mounts found, library imported, submission written "
          "and checked")


if __name__ == "__main__":
    main()
