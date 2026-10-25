"""Push a notebook to Kaggle, stamped with the commit that produced it.

Kaggle records a leaderboard score against a *kernel version*. Git records what the code
was. Neither knows about the other, so this prepends a cell printing the commit into the
run's own log — provenance without anyone having to remember.

Notebooks know nothing about this. The stamp is added to a staged copy, so the same
mechanism works for our own notebook and for a vendored third-party one that has to stay
byte-identical to what its author published.

    python -m scripts.kaggle_push
    python -m scripts.kaggle_push --kernel-dir kaggle/analog-baseline --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

KAGGLE = str(Path(sys.executable).with_name("kaggle"))


def run(cmd: list[str], dry_run: bool = False) -> None:
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    if dry_run:
        return
    done = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    print(done.stdout, end="", flush=True)
    if done.returncode != 0:
        raise SystemExit("command failed")
    # The Kaggle CLI prints its own failures and still exits 0 — a refused push came
    # back as a success and the script then printed a run URL for a kernel version that
    # was never created. A push whose output says it failed did fail, whatever the exit
    # status claims.
    if "error" in done.stdout.lower():
        raise SystemExit("the CLI reported an error and exited 0; nothing was pushed")


def git_stamp() -> str:
    """Short commit, marked dirty when the tree does not match it.

    A stamp claiming a clean commit while uncommitted edits were pushed would be worse
    than no stamp at all.
    """
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"
    dirty = subprocess.run(["git", "diff", "--quiet"]).returncode != 0
    return f"{sha}-dirty" if dirty else sha


def stamp_cell(stamp: str) -> dict:
    source = (
        "# Prepended by scripts/kaggle_push.py. Ties this run to the commit that\n"
        "# produced it: Kaggle knows kernel versions, git knows code, this joins them.\n"
        f'print("run stamp: {stamp}")\n'
    )
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": source.splitlines(keepends=True)}


def stage(kernel_dir: Path, stamp: str) -> Path:
    """Build the directory that is actually pushed, leaving the repo untouched."""
    meta = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    notebook_path = kernel_dir / meta["code_file"]
    if not notebook_path.is_file():
        raise SystemExit(f"{kernel_dir} holds no {meta['code_file']}")

    staging = Path(tempfile.mkdtemp(prefix="casmi-kernel-"))
    (staging / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"].insert(0, stamp_cell(stamp))
    (staging / meta["code_file"]).write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return staging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-dir", default="kaggle/analog-baseline", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stamp = git_stamp()
    print(f"commit stamp: {stamp}\n")
    if stamp.endswith("-dirty"):
        print("WARNING: the working tree is dirty, so this run cannot be reproduced\n"
              "         from a commit alone. Commit first if the result matters.\n")

    staging = stage(args.kernel_dir, stamp)
    try:
        run([KAGGLE, "kernels", "push", "-p", str(staging)], args.dry_run)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    kernel_id = json.loads(
        (args.kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))["id"]
    print(f"\nPushed {stamp}. Run: https://www.kaggle.com/code/{kernel_id}")
    print("Submit from that page once it completes (the CLI cannot), then record the "
          "score in docs/experiments.md.")


if __name__ == "__main__":
    main()
