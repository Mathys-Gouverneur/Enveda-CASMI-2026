# Experiments

One row per thing measured, newest first. The point of the table is that a number is
useless without knowing *what it was measured on* — this competition's whole public record
is a cautionary tale about that, so every row names its test set.

## Reading the columns

| column | meaning |
|---|---|
| **where** | `board` = the real Kaggle leaderboard, 400 molecules, ±0.047 at 95 % · `val@T` = our validation split at family-disjoint ceiling `T` (`T=1.0` is molecule-level, `T=0.50` is the setting we believe reproduces the real test) |
| **n** | molecules scored |
| **commit** | the stamp the run printed, joining the score to the code |

A `val@1.00` number and a `board` number are **not comparable**: the former is optimistic
by roughly 0.26 for anything that leans on structural neighbourhood
([`measurement.md`](measurement.md) §9).

## Submissions to the board

| date | what | where | MRR | commit | note |
|---|---|---|---|---|---|
| — | *analogue baseline, pending* | board | *predict ~0.43* | — | tests the calibration, not the method |

**Budget: 5 submissions per day.** What they are good for, and not:

- **Good**: calibrating our offline split against reality; paired A/B of two variants when
  the expected effect exceeds ~0.01; checking that the offline and notebook pipelines
  agree.
- **Not good**: tuning constants. 400 molecules at ±0.047 is exactly what made the public
  field overfit — a dozen of its published "gains" are inside that noise.
- A comparison between *our own* two submissions is **paired** (the board scores everyone
  on the same 400 molecules), which resolves ~0.005–0.02 rather than ±0.047. That is the
  one thing the board does better than its absolute number suggests.

## Offline measurements

| what | where | n | MRR | note |
|---|---|---|---|---|
| analogue propagation, `top_k=10`, `p=3` | val@1.00 | 300 | **0.682** | strongest single mechanism |
| " | val@0.80 | 300 | 0.669 | |
| " | val@0.70 | 300 | 0.631 | |
| " | val@0.60 | 300 | 0.547 | |
| " | **val@0.50** | 300 | **0.434** | the public best verified is 0.433 on the board |
| " | val@0.45 | 300 | 0.355 | |
| " | val@0.40 | 300 | 0.263 | |
| FPNetV2, epoch 0 of 5 | val@1.00 | 800 | 0.628 | selection subset; training in flight |
| FPNet v1, 10 epochs, **last** checkpoint | val@1.00 | 4000 | 0.210 | degraded: loss peaked at epoch 2 and the script saved the last. Fixed since; the true peak is not recoverable |
| formula soft ordering | val@1.00 | 4000 | 0.123 | +0.037 over random, 30 % of the formula prize |
| random over the mass window | val@1.00 | 4000 | 0.085 | the floor every method must beat |
| spectral retrieval, molecule's own spectra present | val@1.00 | 4000 | 0.650 | optimistic bracket, not the test's regime |
| spectral retrieval, `held_out` | val@1.00 | 4000 | 0.000 | zero *by construction*, not by difficulty |

## Lessons that cost a measurement

| | |
|---|---|
| Reference exclusion keyed on `(molecule, precursor)` removed every collision energy of a molecule — 348k spectra instead of 11,743 — and depressed `in_library` MRR from 0.65 to 0.13 **without raising anything**. Spectra are identified by a content hash now. |
| A hand-written adduct table had `[M+NH4]+` off by one electron mass. Replacing it with a parser also recovered the 29 % of references that are dimers. |
| Sub-formula reachability on a 1 mDa grid accumulated rounding along the sum, so caffeine's own formula landed 2 mDa from its exact mass. Exact float enumeration instead. |
| The formula score as a *hard gate* needs 41 % accuracy merely to break even and captures 4 % of its prize; as a **soft ordering** it captures 30 %. Confirmed against a strong ranker at all seven ceilings. |
| Our validation split was optimistic by ~0.26 because splitting by molecule leaves whole chemical families intact. The ceiling knob exists to fix that. |
