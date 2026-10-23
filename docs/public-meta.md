# What the public leaderboard is actually doing

Synthesised on 2026-10-23 from the 17 notebooks in [`../notebooks/external/`](../notebooks/external/),
above all `alexchilton/casmi26-public-lineage-0-34-to-0-44` — a notebook-by-notebook
lineage map that labels every number as *claimed by the author*, *board reading*, or
*measured on re-run*. It is the single most useful document in the competition and it has
**0 votes**.

Scores below are the author's claims unless marked otherwise. Treat all of them as soft:
three numbers in the whole public record were verified by re-running the notebook.

## The one structural fact

**Nobody near the top is generating molecules.** The official tutorial trains a
spectrum-encoder → SMILES-decoder transformer and samples 25 candidates — de novo
generation. The leaderboard does something else entirely: **retrieve candidates from
structure databases, then rank them**. Generation appears only as a minor fallback channel
for molecules no database covers.

That gap is the first thing to decide about (see "Angles" below).

## The pipeline everything above 0.40 shares

The public meta converged on one stack. Reading `metric-audit-and-safe-tail`, which
documents it stage by stage:

1. **Candidate pool** — COCONUT (natural products) + ChEBI/LIPID MAPS + the training
   structures ≈ 711k, plus a separate channel over ~105 M PubChem structures.
2. **Evidence per candidate** — library matching including adduct-shifted matches,
   mass-shifted analog propagation, FPNet fingerprint-prediction models, MetFrag-lite
   fragment matching.
3. **Two independent rankers** — gradient-boosted (LightGBM), one family descended from
   `prvsiyan`'s analog-propagation baseline, one being `ahmedberatozer`'s own engine.
4. **Popularity prior** — `0.25 · (log1p(#PubChem substances) + log1p(#PubMed papers))`
   added to the z-scored ranker output. Claimed +0.004 on the PubChem channel, +0.010 on
   the main pool re-rank.
5. **Forward-model rescoring** — ICEBERG and GLACIER *predict* each candidate's spectrum
   and compare it with the measured one, re-ranking **within same-formula groups**. This
   is where isomers separate, and most wrong answers share the right answer's formula.
   GLACIER restricted to `[M+H]+` spectra only: claimed +0.006.
6. **Reciprocal-rank fusion** of the two engines — `1/(3+r_main) + 0.6/(3+r_engine2)`,
   keep top 40, then one last same-formula simulator re-rank, then top 25.
7. **Metric-safe dedup and tail backfill** — dedup on the metric's own tautomer-canonical
   identity *after* ranking, fill vacant slots from ranked overflow.

## The six mechanisms the gains cluster around

In rough order of documented value:

| mechanism | documented effect |
|---|---|
| **PubChem-join** — rank PubChem-only candidates through the real engine's features instead of interleaving them at fixed slots | **+0.011, the largest single step in the public record** (`huseyinemreaksoy`, 0.409 → 0.420) |
| **Engine fusion** — reciprocal-rank-sum of two independently built engines | `seyitkaangunes`, 0.366 → 0.382; the hinge the whole lineage pivots on |
| **Pool popularity re-rank** | +0.010 |
| **Forward-model rescoring** (ICEBERG/GLACIER on same-formula groups) | 0.382 → 0.386; GLACIER-on-`[M+H]+`-only +0.006 |
| **Tail backfill + metric-identity dedup** | ≤ 0.002 — the tail is nearly worthless, see the metric note |
| **Rank-1 / library-hit shields** | **mixed, probably negative.** `YahiaZakaria20` locked rank 1 and went 0.63 → 0.89 on offline validation but **0.413 → 0.409 on the board** |

That last row is the most instructive line in the public record: the locked structures
are well-documented training compounds, so offline validation rewarded exactly what the
hidden test does not contain. It is a worked example of the validation trap this
competition sets.

## Documented nulls — things measured and found not to work

Worth as much as the positives, and cheaper:

- **Ungated pool flooding.** `prvsiyan` added PubChem's top-50 isomers by `f·z` with no
  gate: **0.335 → 0.205**. Candidate recall is not the bottleneck; precision is.
- **Forced isomer slots.** `xhhuang`'s v36 reports its own forced insertions displaced
  **95.2%** of true rank 4–10 candidates to the tail.
- **Output-side tautomer dedup** on a 0.342 submission touched 7 of 400 rows — ceiling
  about 0.001.
- **Six of seven** same-formula isomer-separation features posted by `alexchilton` in
  forum thread 742055 were null or negative. The exception was ICEBERG as a ranker
  feature: +0.0015, and only in the stratum with more than 100 candidates.

## Two traps that cost the field real time

- **The RDKit wheel mismatch.** A Python 3.12-vs-3.13 wheel incompatibility *silently
  zeroed* ICEBERG/GLACIER scoring in notebooks that did not pin the right image. Days of
  the public record are confounded by it, and some contradictory score claims are
  probably this. Everything serious now pins **RDKit 2026.03.3**. Our venv has
  2026.09.1, which is fine locally but is **not** what a submission notebook should use.
- **The visible test overlaps training.** Any score read off it is a smoke check. Several
  conflicting public claims come from authors reading that number as a validation
  estimate.

## Where the headroom is

The public ceiling is ~0.44–0.46 claimed; the actual leaderboard top on 2026-10-23 is
**0.504**, and the best *verified* public number is **0.435**. So the top teams hold
roughly **0.07** over the reproducible public stack — a large gap, which means the public
notebooks are not simply the leaderboard with noise on top.

Two further observations about the shape of the field:

- Scores are **densely packed**: ranks 9–16 all sit at 0.471–0.475. With 400 test
  molecules, **one rank-1 flip is worth 0.0025** — so those eight places are separated by
  about one or two molecules. Most apparent movement on this leaderboard is noise, and
  any improvement under ~0.01 cannot be distinguished from it by a single submission.
- The whole public stack is an **ensemble of hand-tuned weights** (`POP_MU 0.25`,
  `alpha 0.6`, `lib_max < 0.9`, slots 2/4/6/8/10, top-40, top-60, top-80…) fitted by
  submission against a 400-molecule public set. That is heavy leaderboard overfitting, and
  it is the most likely explanation for the public/private gap being different from what
  the public scores suggest.

## Angles — the candidate directions, with the case for and against each

Not a plan yet; the data has to be on disk and `enveda-180` understood before choosing.
Ordered by how much they differ from what everyone else is doing.

### A. Build the retrieval+ranking stack properly, from the pieces, not by forking

*(Second, not first — angle B below is measured and this one is a prerequisite for C.)*

Reimplement stages 1–7 ourselves against a **validation set that is not leaky**, so that
every weight in that long list of constants is fitted on something other than 400 public
molecules. The public notebooks give the mechanisms away for free; what they cannot give
us is a trustworthy measurement.

*For*: the mechanisms are proven, and the field's weakest point is measurement, not
method. *Against*: large build, and the attached-dataset licence audit has to be done
carefully.

### B. The instrument gap — `enveda-180`, and it is bigger than it looked

**Measured 2026-10-23, and this is now the strongest angle.** The test set is **100%
timsTOF**. Exactly two libraries in `train.parquet` contain any timsTOF spectrum:
`enveda-180` (1.15 M spectra, 182,941 molecules) and `enveda-np-examples` (1,184). Every
public library — GNPS, MassBank, MoNA, RIKEN, Pluskal, MSDIAL, SpectraVerse, Masaryk —
has **zero**. Median peak count: test 230, timsTOF train 138, everything else 20. Full
table in [`competition.md`](competition.md#the-finding-that-matters-most-the-test-set-is-a-different-instrument).

The official tutorial excludes `enveda-180` in a bare one-line filter with no comment, and
the whole field inherited it. So:

- Every public notebook does cosine / modified-cosine library matching against reference
  spectra **measured on different instruments at a tenth of the test's peak density**.
- The 45.5% of the corpus that shares the test's instrument is the part everyone discards.
- A held-out split restricted to timsTOF molecules is both the test-faithful validation
  set *and* far larger than the 400-molecule public LB everyone is tuning against.

The name misleads — `enveda-180` is not 180 molecules, it is 182,941.

*For*: measured, specific, unexploited, and it attacks both the field's method weakness
and its measurement weakness at once. *Against*: it does not yet establish that test
molecules are *drawn from* `enveda-180`, nor what the instrument gap costs in retrieval
accuracy — both measurable, neither measured.

**First experiment**: hold out timsTOF molecules; compare retrieval accuracy using
public-library references against Enveda references. That single number sizes the whole
angle.

### C. Attack rank 1 specifically, on the molecules where the stack is weakest

Since the tail is worth ≤0.002, the only question that matters is which molecule gets
slot 1. Stratify: where does the public pipeline put the right answer at rank 2–5, and
what distinguishes those cases? The ICEBERG-as-feature result (+0.0015 only when
candidates > 100) suggests the gains are strongly conditional, and nobody is exploiting
the conditioning.

*For*: directly targets where the score is. *Against*: needs a working baseline first.

### D. Revisit generation as a *candidate source*, not an answer

The tutorial's de novo model is scored as an answer and loses badly. But its samples could
feed the pool for molecules no database covers — the case the retrieval stack cannot
reach. Generation then only needs recall, and the existing ranker handles precision.

*For*: genuinely unexplored in the public record, and plausibly where part of the top
teams' 0.07 lives. *Against*: the measured null on pool flooding warns that adding
candidates without a gate is actively harmful.

### E. Do not bother with

Fixed-slot interleaving, output-side tautomer dedup, tail work below rank 3, rank-1
locking — all measured, all worth ≤0.002 or negative.
