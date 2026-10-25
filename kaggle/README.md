# The Kaggle side

What Kaggle is told to run, kept in git instead of in a browser. The structure and most of
the hard-won details are carried over from the RSNA project's `kaggle/`.

```
kaggle/
  analog-baseline/   the scored notebook — analogue propagation, no external data
```

Each directory holds a `kernel-metadata.json` (the slug, what to mount, GPU, internet off)
and the notebook that runs. `kaggle kernels push -p DIR` needs a directory holding both,
which is why this exists as a folder at all. Pushing to the same `id` creates a **new
version** of the same kernel, never a new kernel.

## What `notebook.ipynb` is

A **launcher**, not a pipeline. It mounts two things and calls one function:

| mount | what it is |
|---|---|
| `casmi-src` | our library, published by `scripts/kaggle_dataset.py --source`. Internet is off in a scored run, so a mounted dataset is the only channel |
| the competition | `train.parquet` and `test.parquet` |

The chain itself is [`casmi.submit.run_submission`](../src/casmi/submit.py), which
`scripts/rehearse_kernel.py` calls too: **one implementation**, so a leaderboard score is
reproducible locally. Nothing about the method is written in the notebook.

The notebook locates each mount **by content** rather than by path, because Kaggle names a
mount after its dataset and a rename would otherwise break it in silence. It also handles
both mount shapes — the package expanded as staged, or `casmi.zip` — since the upload
archives directories rather than skipping them and which arrives is Kaggle's decision.

It carries **no `try/except`**. A run that swallows its failure writes the ethanol fallback
and reports COMPLETE, which looks exactly like a method that learnt nothing.

## The analogue baseline, and why it goes first

Analogue propagation alone: match each test spectrum against the whole training library,
take the best-matching *other* molecules whose structures are known, and rank the
mass-window candidates by weighted structural similarity to those.

**It needs no external data.** Reference library, candidate bank and fingerprints all come
out of `train.parquet` — no attached dataset, no weights, nothing under a non-commercial
licence. Four of the public stack's core datasets are CC-BY-NC-SA and the hosts have ruled
that such data can get a solution removed from the leaderboard, so a first submission that
touches none of it is worth having.

**It tests the measuring instrument, not the method.** Offline it scores 0.682 on our
molecule-level validation split and 0.434 at the family-disjoint ceiling of 0.50 we believe
reproduces the real test ([`../docs/measurement.md`](../docs/measurement.md) §10). We
predict a board score near **0.43**. A score there confirms the calibration and licenses
hundreds of offline comparisons; a score far from it says the calibration is wrong, which
is worth learning now rather than in six weeks. The agreement with the public 0.433 is
currently between two different methods on two different test sets — suggestive, and
**not** a verification.

## Rehearsing before pushing

```bash
python -m scripts.rehearse_kernel --limit 5
```

Fakes `/kaggle/input` as one directory per dataset with the competition files symlinked,
then **executes the notebook's own cell sources** against it — so what is tested is the
bytes that will run, not a paraphrase. It catches a missing mount, a bad import or a stale
`dataset_sources` in minutes instead of in a queued run. The pipeline cell is re-run with
a molecule limit; proving it ranks all 400 is what the real run is for.

The notebook reads its mount root from `KAGGLE_INPUT`, defaulting to `/kaggle/input`. The
variable is unset on Kaggle.

## Pushing

```bash
python -m scripts.kaggle_dataset --source          # publish src/casmi first
python -m scripts.kaggle_push                      # then the kernel
python -m scripts.kaggle_push --dry-run
```

The push never modifies the repo: it copies the kernel to a temp directory, prepends a
cell printing the current commit, and pushes that. The run then states its own provenance
in its Kaggle log — the only thing joining a leaderboard score to a commit. A dirty tree
is reported, because a stamp claiming a clean commit while uncommitted edits were pushed
would be worse than no stamp.

Then **submit from the kernel's page** — the CLI cannot — and record the result in
[`../docs/experiments.md`](../docs/experiments.md).

### Two CLI traps, both carried over

- `kaggle` **prints its failures and still exits 0.** A refused push reads as a success
  and the script then prints a run URL for a version that was never created, so
  `kaggle_push` greps its own output for `error`.
- `datasets status` and `datasets files` also answer with an HTTP error on stdout and exit
  0, so neither can be used to ask whether a slug exists. `kaggle_dataset.owns()` pages
  through our own dataset listing instead.

## If RDKit is missing from the image

The notebook imports RDKit at the top and asserts it can parse ethanol, so a missing
dependency fails in seconds rather than after an hour of parquet reading. The competition
ships an RDKit wheel as a dataset (`metric/rdkit-2026-3-3-wheel`, which ten public
notebooks attach) and the fix is to add it to `dataset_sources` and install from the local
wheel. It is not attached by default: the dataset's licence is undeclared, and RDKit is in
most Kaggle images already.
