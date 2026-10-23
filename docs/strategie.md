# Strategy

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

1. **Metric reimplemented** with the tautomer-canonical identity, 40 tests pinning it.
2. **Resolution quantified**: ±0.047 on the board; **0.0024** detectable on a
   4,000-molecule paired comparison.
3. **Validation split** that matches the test profile to within 1.5 points on every axis.
4. **Pool coverage measured** — and found to be a dead end, which is itself the result.
5. **Retrieval baseline** with the three brackets: 0.085 random, 0.650 in-library.
6. **Adduct parser** recovering 29 % more reference spectra, and a 1.8 ppm bug fixed.

### Next, in order

1. **A spectrum → fingerprint predictor trained on the timsTOF half.** This is the pivot:
   it scores a candidate the library has no spectrum for, which is the regime the test is
   in and where retrieval scores zero. 1.15 M in-domain spectra, 183 k molecules, and
   nobody else is training on them because the tutorial's filter removed them.
   Measured against the 0.085 floor on the `held_out` split.

2. **Same-formula reranking.** ~90 % of a window shares the answer's formula, so this is
   where the remaining score is. Needs a signal that sees *how* a molecule breaks, not
   just what it is made of — fragment-formula annotation first, since it is cheap and
   licence-free, then forward simulators from upstream checkpoints.

3. **Learned cross-instrument similarity.** The 2,288 molecules measured on both
   instruments are a supervised dataset for it, and the 183 k molecules × multiple
   collision energies are a much larger contrastive one. Unexploited by anyone, as far as
   the public record shows.

4. **Multi-spectrum fusion.** Three spectra per molecule at different collision energies.
   The current baseline pools them by taking each candidate's best score, which is a
   placeholder, not a method.

5. **Rebuild the candidate bank from upstream** (COCONUT CC BY, ChEBI CC BY, PubChemLite
   CC0) — not for recall, which we have shown it does not add, but so that nothing in the
   pipeline traces back to an NC source, and so that gating decisions are ours.

6. **A first submission** early, to calibrate our local numbers against the board. One
   submission is worth little as a measurement (±0.047) but a lot as a check that the
   offline and online pipelines agree.

### What would falsify this

The plan rests on one inference: that the test molecules' structures are in
`train.parquet`. It follows from the 0.3 % coverage figure plus the board sitting at 0.50,
but it is an inference, not a measurement — we cannot see the test labels.

If it is wrong, and a material share of test molecules are novel structures absent from
train, then candidate ranking has a hard ceiling and de novo generation stops being a
curiosity. The cheapest check is the first submission: if our offline `held_out` number
and the board disagree sharply in the same direction, this is the assumption to revisit
first.
