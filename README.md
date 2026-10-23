# Enveda CASMI 2026 — molecule identification from mass spectra

<https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra>

| | |
|---|---|
| deadline | **2026-12-14 23:59 UTC** |
| category | Featured, $50,000 |
| teams at setup | 3855 |
| **rules accepted** | **no — nothing can be downloaded or submitted until they are** |

## What the competition serves

Read off `kaggle competitions files`, 2026-10-23:

| file | size |
|---|---|
| `train.parquet` | **3.03 GB** |
| `test.parquet` | 4.8 MB |
| `sample_submission.csv` | 43 KB |

Three files, parquet, 3 GB in all — a different shape of problem from the RSNA knee work:
no DICOM, no per-study pixel pipeline, and the whole corpus fits in memory on this
machine. The 1500:1 ratio between train and test is the first thing to understand.

## Setup state

- repository created at `/working/mgr/Repositories/Enveda_CASMI`, `git init` done
- **nothing downloaded**: the rules must be accepted on the website first
- the Kaggle CLI used throughout the RSNA project is at
  `/working/mgr/Repositories/RSNA-project/.venv/bin/kaggle` and is already authenticated

## Carried over from RSNA, deliberately

Those three months produced measurement rules that cost real submissions to learn. They
live in that project's memory directory and do **not** follow a new working directory, so
the ones that generalise are restated here:

- **Propagate every stage between a measurement and the metric before quoting a
  projection.** An out-of-fold gain of +0.0088 a column was announced as "+0.002 to
  +0.004" of macro and measured 0.000, because the blend weight and the column count were
  never multiplied through.
- **A factor you never varied cannot support an inference.** "Every top submission keeps
  ACL" was offered as evidence when ACL was present in all eleven scored submissions.
- **Rank candidates on a paired per-fold delta plus a sign test**, never on a small
  held-out set: 58 radiologist-read studies inverted the sign of a real effect.
- **Verify a mount by its contents, not by the fact that you pushed it.** Two nine-hour
  runs were lost to a code dataset that was never republished.
- **Never delete a trained run.** Rank-averaging two or three of them was the largest
  out-of-fold lever of the project, at no training cost.
