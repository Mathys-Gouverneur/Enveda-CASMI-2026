# Strategy

Full inventory of levers, with what each is worth: [`leviers.md`](leviers.md).

Written 2026-10-23, after the measurements in [`measurement.md`](measurement.md). Those
measurements changed the plan twice, so this document states what we now believe and
what would have to be false for it to be wrong.

Deadline **2026-12-14** — about seven weeks.

## What the task turned out to be

Not "search a huge chemical space". The measurements narrow it hard:

- The test's chemical space (Enveda timsTOF) is **0.3 % covered** by public structure
  databases, against 25.4 % for the public spectral libraries. A test molecule absent
  from `train.parquet`'s structures is effectively unfindable.
- So the candidate bank that matters is **train's own 275,810 structures**, and
  specifically the **182,009 Enveda structures inside the test's mass band**.
- The mass constraint alone leaves a median of **78 candidates**, of which ~90 % share
  the right answer's molecular formula.
- The test's spectra are mostly **not** in the spectral library: plain cosine retrieval
  gets 0.650 when they are, and the best verified public stack gets 0.433 on the real
  test.

> **The task is: rank ~78 same-mass candidates, mostly same-formula, for a molecule whose
> spectrum the library has never seen. Floor 0.085 (random). Public best 0.433.**

That is a *ranking* problem on a fixed, modest candidate set — not a retrieval or
generation problem. It is a much narrower target than the public stack's architecture
implies.

## The licence constraint, which settles a strategic question

Audited with `kaggle datasets metadata` (see `scripts/` and the table below). The core of
the public stack is published **non-commercially**:

| dataset | licence | notebooks using it |
|---|---|---|
| `prvsiyan/coconut-casmi26-candidates` | **CC-BY-NC-SA-4.0** | 11 |
| `prvsiyan/casmi26-fp-models-v2` (the FPNet weights) | **CC-BY-NC-SA-4.0** | 11 |
| `prvsiyan/casmi26-ranker-features` | **CC-BY-NC-SA-4.0** | 11 |
| `prvsiyan/chebi-lipidmaps-casmi26` | **CC-BY-NC-SA-4.0** | 6 |
| `ahmedberatozer/*` — ICEBERG, GLACIER, pools, models | `other` (undeclared) | 10 each |
| `metric/rdkit-2026-3-3-wheel` | `unknown` | 10 |
| `franciscoangulo/...pubchemlite-cc0` | CC0-1.0 | 4 |
| `megayak/casmi26-simulated-ranker-rows` | CC0-1.0 | 10 |
| `hengck23/...dreams...` | MIT | 6 |

The hosts have ruled that external data under a commercial-use-restricting licence can get
a solution **removed from the leaderboard entirely**, not merely denied a prize. So:

- **We cannot ship the public stack's candidate banks, fingerprint models, or ranker
  features.** Forking a public notebook and submitting it inherits the exposure.
- But **upstream is clean**: COCONUT is CC BY 4.0, ChEBI is CC BY, PubChemLite is CC0.
  The NC label belongs to the Kaggle repackagings — partly because the bit-selection
  derives from FPNet checkpoints. Rebuilding from upstream removes the problem.
- All four upstream sources are reachable from this machine (checked).

This turns "rebuild from the bricks versus fork the public stack" from a taste question
into a requirement. We may use the NC datasets **locally, as an oracle to compare
against**; nothing derived from them goes into a submission.

Two further operational notes: the `ahmedberatozer` datasets carrying ICEBERG and GLACIER
are licence-**undeclared**, which is not the same as permissive — the models themselves
are fine when trained on MassSpecGym, so we should obtain the checkpoints from their
upstream release rather than from a Kaggle repackaging. And all three candidate-bank
pickles were inspected with `pickletools` before loading: they contain only the standard
numpy array-reconstruction triple and no other imports.

## What we are not going to do, and why

Each of these is measured, by us or by a competitor who published the number:

| | evidence |
|---|---|
| Expand the candidate pool with external databases | +27 to +34 candidates in the window, **zero** recall gain. Public null: 0.335 → 0.205 when flooded. |
| Rank-1 locking | A competitor: local validation 0.63 → 0.89, board **0.413 → 0.409**. |
| Fixed-slot interleaving of PubChem candidates | Displaced 95.2 % of true rank 4–10 candidates to the tail. |
| Work on ranks 3–25 | The whole tail is worth ≤0.002. |
| Output-side tautomer dedup alone | ≤0.001 — the official scorer already dedups. |
| De novo generation as the answer | The official tutorial's paradigm; the board does not use it. |
| Tune constants by submission | 400 molecules, ±0.047 at 95 %. We cannot see what we would be tuning. |

## The plan

### Done

1. **Metric reimplemented** with the tautomer-canonical identity; 48 tests pin it.
2. **Resolution quantified**: ±0.047 on the board; **0.0024** detectable on a
   4,000-molecule paired comparison.
3. **Validation split** that matches the test profile to within 1.5 points on every axis.
4. **Pool coverage measured** — and found to be a dead end, which is itself the result.
5. **Retrieval baseline** with the three brackets: 0.085 random, 0.650 in-library.
6. **Adduct parser** recovering 29 % more reference spectra, and a 1.8 ppm bug fixed.
7. **Formula identification from fragments** — 51.5 % top-1, capturing **30 %** of the
   formula prize, and only as a soft ordering: a hard gate needs 41 % accuracy merely to
   break even. The score belongs in a ranker as a feature, not in front of it as a filter.
8. **The two retrieval constants fitted** rather than guessed: neutral losses weight **0**
   (the 0.5 placeholder was costing 0.0036) and **log-sum-exp** pooling over a molecule's
   spectra rather than the maximum. +0.0066 ± 0.0039 paired — an effect no submission
   could see.

### In flight

**FPNet**, the spectrum → fingerprint predictor, training on all 1,129,508 timsTOF
spectra excluding validation molecules. 32.2 M parameters, 5,716 output bits, 10 epochs at
~26 min each. Evaluated on the `held_out` split against the 0.085 random floor — the first
method that can score at all in the regime the test actually poses.

### Next, in order

1. **Learned cross-instrument similarity.** The 2,288 molecules measured on both
   instruments are a supervised dataset for it; the 183 k molecules × multiple collision
   energies are a much larger contrastive one. Unexploited, as far as the public record
   shows. Needs the GPU, so it queues behind FPNet.

2. **Compose the signals.** Formula ordering, FPNet fingerprint agreement and spectral
   similarity are three independent views of the same candidate list. Whether they add or
   overlap is an open question, and the harness can answer it to ±0.001. The formula
   measurement predicts they should compose: formula says *what the molecule is made of*,
   FPNet says *which substructures it contains*, and neither sees *how it breaks*.

3. **Same-formula isomer separation**, the part nothing above touches. 34 candidates
   remain in the median true-formula group. This needs a signal that sees fragmentation
   mechanism — forward simulators, obtained from their upstream releases rather than the
   licence-undeclared Kaggle repackagings.

4. **Rebuild the candidate bank from upstream** (COCONUT CC BY, ChEBI CC BY, PubChemLite
   CC0) — not for recall, which we have shown it does not add, but so that nothing in the
   pipeline traces back to an NC source, and so that gating decisions are ours.

5. **A first submission**, to check that the offline and online pipelines agree. Worth
   little as a measurement (±0.047), a lot as a wiring check.

### What would falsify this

The plan rests on one inference: that the test molecules' structures are in
`train.parquet`. It follows from the 0.3 % coverage figure plus the board sitting at 0.50,
but it is an inference, not a measurement — we cannot see the test labels.

If it is wrong, and a material share of test molecules are novel structures absent from
train, then candidate ranking has a hard ceiling and de novo generation stops being a
curiosity. The cheapest check is the first submission: if our offline `held_out` number
and the board disagree sharply in the same direction, this is the assumption to revisit
first.
