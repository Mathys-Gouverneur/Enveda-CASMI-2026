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
kaggle/                                   notebooks we push to Kaggle
```

See [`data/README.md`](data/README.md) for why `raw/` is a symlink and not a directory.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Python 3.10.12, and the stack verified working: numpy 2.2.6, pandas 2.3.3, pyarrow 25.0.1,
**rdkit 2026.09.1**, matchms 0.33.1, scikit-learn 1.7.2, torch 2.14.1+cpu.

RDKit's version is a trap worth knowing about: the public field pins **2026.03.3** in
submission notebooks, because a Python 3.12-vs-3.13 wheel mismatch *silently zeroed*
ICEBERG/GLACIER scoring for days. 2026.09.1 is fine for local work; a submission notebook
must pin the competition wheel.

The Kaggle CLI is in this venv (`.venv/bin/kaggle`), authenticated from
`~/.kaggle/kaggle.json`. This project keeps its own venv rather than borrowing
RSNA-project's, which is pinned to a CUDA torch build nothing here needs.

## State, 2026-10-23

Done:

- repository, venv, chemistry stack, directory layout
- `/data/mgr/enveda-casmi/` mode `700` — `drwx------`, same as `/data/mgr/rsna-knee`
- all three competition files downloaded, schemas and submission format documented
- 17 public notebooks vendored and indexed; the public meta read and written up
- the instrument-gap finding above, measured

Nothing is blocked.

## Next

**Size the instrument gap.** Hold out timsTOF molecules, then compare retrieval accuracy
with public-library references against Enveda references. One number, and it decides
whether the gap is the opening or a curiosity.

It matters because of how this leaderboard is shaped: 400 test molecules means
[one molecule is worth 0.0025](docs/public-meta.md#where-the-headroom-is), ranks 9–16 all
sit inside 0.004, and every constant in the public stack was fitted by submission against
those same 400. A trustworthy offline measurement is worth more here than any single
mechanism — and a timsTOF-only held-out split is both test-faithful and far larger than
the public LB everyone is tuning against.

Open questions worth recording now, before they get forgotten:

- Are the test molecules actually *drawn from* `enveda-180`? Not established.
- What does a cosine match lose across the instrument gap? Not measured.
- `test.parquet` has no `molecular_formula` column; train does. Formula has to come from
  precursor + adduct, and same-formula isomer separation is where the public stack spends
  most of its effort.
