"""Publish the library, or a model package, as a Kaggle dataset.

A scored notebook runs with internet off. It cannot install, clone or download anything,
so a mounted dataset is the only way our code reaches it. This stages the files in a temp
directory with the metadata Kaggle wants and pushes them.

    python -m scripts.kaggle_dataset --source                  # src/casmi
    python -m scripts.kaggle_dataset --model out/fpnet_v2      # a trained checkpoint
    python -m scripts.kaggle_dataset --source --dry-run

The first push of a slug creates the dataset; every later one adds a version, so the
notebook's `dataset_sources` never has to change. Both are private — a dataset is the one
place competition data could leak by accident, and nothing here reads `data/`.

The CLI quirks below are carried over from the RSNA project's version of this script,
where each one cost a debugging session.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

#: The CLI installed beside this interpreter rather than whatever is on PATH, so the
#: script works when called as `.venv/bin/python -m scripts.kaggle_dataset` without the
#: virtualenv activated.
KAGGLE = str(Path(sys.executable).with_name("kaggle"))

OWNER = "mathysgouverneur"
SOURCE_SLUG = "casmi-src"
MODEL_SLUG = "casmi-fpnet"


def owns(slug: str) -> bool:
    """Whether the account already has this dataset.

    Asked by listing our own datasets rather than by querying the slug: `datasets status`
    and `datasets files` both answer with an HTTP error printed to stdout and an exit
    code of 0, so neither can be tested. Paged rather than asked for a big page, because
    `--page-size` does not exist and an unrecognised flag makes the CLI print usage to
    stdout and exit — which reads as an empty listing, so every dataset looks new and
    `create` gets attempted on slugs that already exist.
    """
    ref = f"{OWNER}/{slug}"
    for page in range(1, 26):
        listing = subprocess.run(
            [KAGGLE, "datasets", "list", "-m", "-v", "-p", str(page)],
            capture_output=True, text=True)
        rows = [line for line in listing.stdout.splitlines()
                if line.strip() and not line.startswith("ref,")]
        if not rows or rows[0].startswith("No datasets"):
            return False
        if any(line.split(",")[0] == ref for line in rows):
            return True
    return False


def run(cmd: list[str], dry_run: bool = False) -> None:
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    if dry_run:
        return
    if subprocess.run(cmd).returncode != 0:
        raise SystemExit("command failed")


def stage_source(into: Path) -> tuple[str, str]:
    """`src/casmi` at the dataset root, so the notebook can `import casmi`."""
    shutil.copytree(Path("src/casmi"), into / "casmi",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    modules = sorted(p.name for p in (into / "casmi").rglob("*.py"))
    print(f"{len(modules)} module(s) staged: {', '.join(modules)}")
    return SOURCE_SLUG, "CASMI source"


def stage_model(into: Path, run_dir: Path) -> tuple[str, str]:
    """A trained checkpoint plus whatever describes it.

    The checkpoint already carries its config, its bit selection and its conditioning
    vocabularies, so nothing about the model has to be written in the notebook — the
    same reasoning as the RSNA weights package's manifest.
    """
    checkpoint = run_dir / "fpnet.pt"
    if not checkpoint.is_file():
        raise SystemExit(f"{run_dir} holds no fpnet.pt")
    shutil.copy2(checkpoint, into / "fpnet.pt")
    for extra in ("run.json", "history.csv", "fpnet_eval.csv"):
        if (run_dir / extra).is_file():
            shutil.copy2(run_dir / extra, into / extra)
    size = sum(f.stat().st_size for f in into.iterdir()) / 1e6
    print(f"{run_dir.name}: {size:.0f} MB staged")
    return MODEL_SLUG, "CASMI FPNet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="store_true",
                        help="publish src/casmi as the code dataset")
    parser.add_argument("--model", type=Path,
                        help="publish this training run's checkpoint")
    parser.add_argument("--slug", default=None, help="override the dataset slug")
    parser.add_argument("--message", default="update", help="version message")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if bool(args.source) == bool(args.model):
        raise SystemExit("pass exactly one of --source or --model")

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp)
        slug, title = (stage_source(staged) if args.source
                       else stage_model(staged, args.model))
        slug = args.slug or slug

        (staged / "dataset-metadata.json").write_text(json.dumps({
            "title": title,
            "id": f"{OWNER}/{slug}",
            "licenses": [{"name": "unknown"}],
        }, indent=1) + "\n", encoding="utf-8")

        # -r zip: directories are *skipped* by default, so without this the source
        #         dataset uploads empty.
        # -t:     without it Kaggle rewrites anything it reads as tabular into CSV,
        #         which would quietly reshape run.json or history.csv.
        common = ["-p", str(staged), "-r", "zip", "-t"]
        if owns(slug):
            run([KAGGLE, "datasets", "version", *common,
                 "-m", args.message, "-d"], args.dry_run)
        else:
            run([KAGGLE, "datasets", "create", *common], args.dry_run)

    print(f"\nmounted at /kaggle/input/{slug} once Kaggle finishes processing it")


if __name__ == "__main__":
    main()
