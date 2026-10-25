# Enveda CASMI 2026 — molecule identification from mass spectra

<https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra>

| | |
|---|---|
| deadline | **2026-12-14 23:59 UTC** |
| category | Featured, $50,000 |
| teams | 3855 |
| leaderboard top | **0.504** (2026-10-23) |
| best *verified* public notebook | **0.433** |
| rules accepted | yes |
| data | on disk at `/data/mgr/enveda-casmi/raw` |

Given an MS/MS spectrum, name the molecule. Up to 25 ranked SMILES per molecule, scored
by **mean reciprocal rank** on the tautomer-canonical InChIKey-14 — so stereochemistry is
ignored, and because `1/rank` decays fast, [the entire tail below rank 3 is worth
≤0.002](docs/competition.md#where-the-score-actually-lives). The game is slot 1.

## Start here

| | |
|---|---|
| [`docs/competition.md`](docs/competition.md) | The task, the data columns, the metric's exact identity rule, and the licence rulings that can get a solution disqualified |
| [`docs/public-meta.md`](docs/public-meta.md) | What the leaderboard is actually doing, the six mechanisms the gains cluster around, the documented nulls, and the candidate angles |
| [`docs/measurement.md`](docs/measurement.md) | What we can tell apart, and what the task measurably is |
| [`docs/leviers.md`](docs/leviers.md) | Every lever that could gain points, what each is worth, and the measured nulls |
| [`docs/strategie.md`](docs/strategie.md) | The plan, the licence constraint, and what would falsify it |
| [`docs/experiments.md`](docs/experiments.md) | Every number measured, and **what test set it was measured on** |
| [`kaggle/README.md`](kaggle/README.md) | What Kaggle is told to run, and how to rehearse it first |
| [`notebooks/external/README.md`](notebooks/external/README.md) | The 17 vendored public notebooks, with vote counts and why each is kept |

Two facts from those documents worth having before anything else:

**Nobody near the top is generating molecules.** The official tutorial trains a
spectrum→SMILES transformer; the leaderboard retrieves candidates from structure databases
and ranks them. Generation survives only as a fallback channel.

**The test set is a different instrument from every public reference library.**
`test.parquet` is 100% timsTOF. In `train.parquet`, only `enveda-180` and
`enveda-np-examples` contain timsTOF spectra — 45.5% of the corpus — and GNPS, MassBank,
MoNA, RIKEN, Pluskal, MSDIAL, SpectraVerse and Masaryk contain **none**. Median peak
count: test 230, timsTOF train 138, everything else 20. The official tutorial excludes
`enveda-180` in a one-line filter with no comment, and the whole field inherited it, so
every public notebook matches spectra against references from the wrong instrument at a
tenth of the peak density. [Measured here.](docs/competition.md#the-finding-that-matters-most-the-test-set-is-a-different-instrument)

## Layout

```
data/raw -> /data/mgr/enveda-casmi/raw   competition files, mode 700, gitignored
data/external/                           small third-party inputs, versioned + PROVENANCE
data/candidates/                          COCONUT / PubChem / model weights, gitignored
data/processed/                           derived, gitignored, safe to delete
docs/                                     what we know and how we measured it
notebooks/external/                       vendored public kernels, one dir each
notebooks/                                our own exploration
scripts/                                  entry points
src/casmi/                                the package
kaggle/                                   notebooks we push to Kaggle, one dir each
```

See [`data/README.md`](data/README.md) for why `raw/` is a symlink and not a directory.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then `pip install -e .` so `import casmi` resolves, and `python -m pytest tests -q`
(48 tests, ~16 s) to check the install.

Python 3.10.12, stack verified working: numpy 2.2.6, pandas 2.3.3, pyarrow 25.0.1,
**rdkit 2026.09.1**, matchms 0.33.1, scikit-learn 1.7.2, **torch 2.14.1+cu126** on the
box's NVIDIA L40 (47.6 GB). See `requirements.txt` for the two pip traps that build hides.

RDKit's version is a trap worth knowing about: the public field pins **2026.03.3** in
submission notebooks, because a Python 3.12-vs-3.13 wheel mismatch *silently zeroed*
ICEBERG/GLACIER scoring for days. 2026.09.1 is fine for local work; a submission notebook
must pin the competition wheel.

The Kaggle CLI is in this venv (`.venv/bin/kaggle`), authenticated from
`~/.kaggle/kaggle.json`. This project keeps its own venv rather than borrowing
RSNA-project's, which is pinned to a CUDA torch build nothing here needs.

## State, 2026-10-23

The task, as measured rather than assumed: **rank ~78 same-mass candidates, of which ~34
share the right answer's molecular formula, for a molecule whose spectrum the library has
never seen.** Floor 0.085 at random; best verified public stack 0.433.

| | |
|---|---|
| metric reimplemented, tautomer-canonical, 48 tests | ✅ |
| resolution quantified — **±0.047** on the board, **0.0024** paired on 4,000 molecules | ✅ |
| validation split matching the test profile within 1.5 pts on every axis | ✅ |
| pool coverage — a measured dead end, which is the result | ✅ |
| retrieval baseline, three brackets | ✅ |
| adduct parser — +29 % reference spectra, a 1.8 ppm bug fixed | ✅ |
| formula identification — 51.5 % top-1, **30 %** of its prize, soft ordering only | ✅ |
| the two retrieval constants fitted — **+0.0066 ± 0.0039** paired | ✅ |
| FPNet, spectrum → fingerprint, 32.2 M params on 1.13 M timsTOF spectra | 🔄 training |
| learned cross-instrument similarity | ⏳ queued behind the GPU |

Entry points: `scripts/noise_floor.py`, `build_splits.py`, `eval_baseline.py`,
`eval_formula.py`, `tune_retrieval.py`, `train_fpnet.py`.

## Next

The ordered list with values attached is [`docs/leviers.md`](docs/leviers.md). The top of
it, after reading the public FPNet's actual code: **condition the model on instrument,
adduct and collision energy** and train on all 2.54 M spectra, rather than discarding 55 %
of the corpus for domain purity as the current run does.

### Open questions, recorded before they get forgotten

- Are the test molecules actually *drawn from* `enveda-180`? Not established — it is an
  inference from the 0.3 % coverage figure plus the board sitting at 0.50, and the whole
  plan rests on it. The first submission is the cheapest check.
- Do formula, fingerprint and spectral signals **add or overlap**? Open, and the harness
  can answer it to ±0.001.
- What separates the 34 same-formula isomers? Nothing built so far touches this, and it is
  where the residual score lives.
