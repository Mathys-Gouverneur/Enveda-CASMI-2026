# Measurement: what we can tell apart, and what the task actually is

Everything here is measured on `data/raw/` with the code in `src/casmi/` and
`scripts/`. Reproduce with:

```bash
python scripts/noise_floor.py          # no data needed
python scripts/build_splits.py --n-val 4000
python scripts/eval_baseline.py
python -m pytest tests -q
```

## 1. Resolution: the leaderboard cannot see what we need to see

Simulated from reciprocal-rank distributions at the two score levels the board actually
shows (`scripts/noise_floor.py`). The per-molecule reciprocal rank has sd ≈ 0.48, which
over 400 molecules gives:

| | |
|---|---|
| standard error of a score on 400 molecules | **0.024** |
| 95% interval | **±0.047** |

Three consequences, and they are the reason the measurement work came before any model.

- **The top of the leaderboard is inside its own noise.** Ranks 1 to 16 span 0.504 to
  0.473 — 0.031, well under ±0.047. The public board is not ordering those teams by
  skill; it cannot.
- **A public→private shift of ±0.05 per team is the expected scale**, not an upset. Any
  plan that assumes our public score transfers is mis-specified.
- **But a *paired* comparison is far sharper**, because both arms see the same molecules
  and the molecule-to-molecule variance cancels:

| molecules fixed | true effect | paired SE on 400 | power | paired SE on 4,000 | power |
|---|---|---|---|---|---|
| 0.25 % | +0.0024 | 0.0024 | 0.26 | **0.0008** | **0.94** |
| 0.5 % | +0.0048 | 0.0034 | 0.33 | 0.0011 | 1.00 |
| 1 % | +0.0097 | 0.0048 | 0.57 | 0.0015 | 1.00 |

A 4,000-molecule validation set resolves an effect of **0.0024** that 400 molecules
cannot see at all. That is a 10× advantage in resolution, and it comes from the half of
the corpus the field discards.

## 2. The validation split

`scripts/build_splits.py` — molecule-level, timsTOF-only, calibrated to the test profile.
4,000 molecules, 11,743 query spectra:

| | ours | test |
|---|---|---|
| spectra per molecule | 2.94 | 3.00 |
| median peaks | 243 | 230 |
| neutral mass range | 246.1–439.1 | 246.1–439.1 |
| `[M+H]+` | 78.8 % | 79.1 % |
| `[M-H]-` | 17.4 % | 15.9 % |
| `[M+CH2O2-H]-` | 2.6 % | 2.6 % |

Split **by molecule**, never by spectrum: a molecule carries 6 spectra in the median and
up to 1,468, so a spectrum-level split puts the same structure on both sides.

### The spectrum hash, and the bug it caught

Excluding "these exact spectra" from a reference library needs a per-spectrum identity,
and `train.parquet` has none. The first attempt keyed on `(inchikey14, precursor_mz)` —
which removes **every collision energy of the molecule**, not the three query spectra. It
silently deleted 348,000 reference spectra instead of 11,743, turning the `in_library`
split into something close to `held_out` and depressing its MRR from 0.65 to 0.13.

`data.spectrum_uid` now hashes the spectrum's own bytes. The lesson generalises: a split
bug does not raise an exception, it just quietly returns a worse number, and a worse
number is indistinguishable from a worse method.

## 3. The three regimes, and which one the test is

| regime | MRR | top-1 | answer in 25 |
|---|---|---|---|
| `in_library` — the molecule's other spectra are in the library | **0.650** | 59.2 % | 74.0 % |
| `held_out` — no spectrum of the molecule survives | 0.000 | 0.0 % | 0.0 % |
| `held_out`, random ranking of the structure mass window | **0.085** | 2.9 % | 42.4 % |
| *best verified public stack, on the real test* | *0.433* | | |

The middle row is **0 by construction**, not by difficulty: if no spectrum of the
molecule is in the reference library, spectral retrieval cannot return its key at any
rank. Reporting it as a difficulty measurement would be a category error.

The row that binds is the third: the true structure *is* in the candidate bank's mass
window (median **78** candidates at ±10 ppm), and we rank that window at random. **0.085
is the floor every ranking model must beat**, not zero.

And the comparison that matters: the public stack's 0.433 sits between the random floor
and our `in_library` bracket. If the test were the easy regime, plain cosine matching
would already deliver ~0.65 and the leaderboard would not be at 0.50. So:

> **The test is mostly: structure present in the candidate bank, spectra absent from the
> spectral library.** That is precisely the regime where fingerprint predictors and
> forward simulators earn their place, and where plain retrieval scores zero.

## 4. Pool coverage: a wider candidate pool is a liability here

The naive form of this measurement is a trap. Validation molecules come from
`train.parquet`, so train structures contain them by construction (100 %) — and the
`COCONUT` and `ChEBI+LipidMaps` banks as published on Kaggle were **deduplicated against
train** by their publishers, giving them 0 % by construction. Both numbers are artefacts.
(Verified: their stored keys are correct InChIKey-14 of their own SMILES, and their masses
agree with ours to 0.0 ppm — the zero overlap is filtering, not a convention mismatch.)

The question with an answer: **how much of each chemical space do public structure
databases cover?** Measured against PubChemLite, the one external bank here not filtered
against train:

| chemical space | molecules | in PubChemLite |
|---|---|---|
| **Enveda / timsTOF — the test's provenance** | 183,191 | **473 (0.3 %)** |
| public spectral libraries | 95,157 | 24,203 (**25.4 %**) |
| Enveda only, absent from public libraries | 180,653 | 191 (0.1 %) |
| shared Enveda + public | 2,538 | 282 (11.1 %) |

**An 85× difference in coverage.** The test's own chemical space is the one public
databases do not have. Two conclusions:

1. A test molecule absent from train's structures is very likely **unfindable** with any
   database we can legally ship. Since the board sits at 0.50, the test molecules are
   very likely *in* train.
2. Adding external banks adds candidates without adding recall:

| pool | median window at ±10 ppm | recall |
|---|---|---|
| train structures | **78** | 1.000 |
| + PubChemLite | 105 | 1.000 |
| + COCONUT | 112 | 1.000 |
| + ChEBI/LipidMaps | 83 | 1.000 |

Candidates added, recall unchanged — one more wrong answer to outrank, for nothing. This
is what the public null was measuring all along: flooding the pool with PubChem isomers
took a public notebook from **0.335 to 0.205**. We now have the reason, not just the
observation.

Also worth noting: 182,009 of 183,191 Enveda molecules (99.4 %) fall inside the test's
mass range, against 59 % of the public-library structures. Enveda's library was built in
the test's own mass band — another sign the test is drawn from it.

## 5. Adduct arithmetic: 29 % of the reference library was being discarded

A lookup table of monomer adducts covered only **71.3 %** of timsTOF spectra. The
remainder is dominated by **dimers** — `[2M+Na]+` alone is 188,158 spectra, `[2M+H]+`
89,487, `[2M-H]-` 52,933 — where the neutral mass is `(m/z − shift)/2`.

`data.parse_adduct` now parses the general form `[nM ± A]z±`. Coverage went to **100 %**,
the reference library from 811k to 1,143k spectra, and `in_library` MRR from 0.630 to
**0.650**. The test set itself contains no dimers, so these matter purely as references —
which is exactly what a retrieval baseline is short of.

Writing the parser also exposed an error in the hand-written table it replaced:
`[M+NH4]+` was off by one electron mass, because 18.033823 is already the NH4⁺ *ion* mass
and the table subtracted the electron from it a second time. 0.55 mDa — about 1.8 ppm on a
300 Da molecule, enough to shift a ±5 ppm window. The literal values now live in
`tests/test_casmi.py`, where a disagreement fails loudly.

## 6. Formula identification: worth 30 % of its prize, and only as a soft signal

`scripts/eval_formula.py`. The prize is bounded first: ranking the mass window at random
scores 0.085, ranking only the **true-formula** subset at random scores 0.210. So a
perfect formula signal is worth up to **0.125** — it more than doubles the floor. But
note what it does *not* do: 34 isomers remain in the median true-formula group, against 77
in the mass window. A 2.3× reduction, not a solution. Same-formula separation stays the
core problem.

The signal is free of any licence or external data: **every fragment is a sub-formula of
the molecule**, so a candidate formula can be scored on how much of the spectrum's
intensity it explains as sub-formulas of itself.

### The bias that made the first version useless

Raw explained intensity gave **32.2 %** formula top-1 accuracy and a *negative* gain on
the task. The cause is structural, not a bug: a formula with more atoms explains more
peaks for free. Caffeine admits 1,485 sub-formula masses across 194 Da — one every
0.13 Da — while a 400 Da formula admits 11,718, one every 0.034 Da. Against a ±4 mDa
tolerance with a ±1 H allowance, the larger formula matches most of any spectrum by
chance. **The score was ranking formulas by density, not by correctness.**

`formula.chance_match_rate` computes the by-chance hit rate exactly — the measure of the
union of tolerance intervals around every reachable mass, over the mass range — and the
adjusted score is `explained − chance`:

| | raw | chance-adjusted |
|---|---|---|
| formula top-1 | 32.2 % | **51.5 %** |
| top-2 / top-3 | 57.9 / 77.2 % | **72.5 / 83.7 %** |

### Hard gate versus soft ordering

Paired on identical permutations (one RNG seeded per molecule, so every condition sees the
same draw):

| design | MRR | share of the 0.125 prize |
|---|---|---|
| mass window (base) | 0.0854 | — |
| hard gate on top-1 formula | 0.0898 | 4 % |
| hard gate on top-2 | 0.0985 | 10 % |
| **soft ordering by formula score** | **0.1229** | **30 %** |
| oracle: true formula known | 0.2102 | 100 % |

A hard gate must clear `base/oracle` = **41 %** formula accuracy merely to break even,
because being wrong costs the molecule its entire score. At 51.5 % it barely clears it.
Soft ordering — visit formula groups in score order, never exclude anything — captures
seven times more.

Two conclusions worth carrying:

- **The formula score belongs inside a ranker as a feature, not in front of it as a
  filter.** This is why the public stack reranks same-formula groups rather than gating on
  formula; we now have the arithmetic for why.
- Without the chance adjustment, soft ordering captures 20 % and the hard gate is
  **−6 %**. The correction is most of what makes the signal usable.

### An implementation note that cost a debugging cycle

Sub-formula reachability was first marked on a 1 mDa grid, rounding each element's mass to
the grid independently. The error accumulates along the sum: caffeine's own formula landed
**2 mDa** from where its exact mass puts it, so a formula did not contain itself, and the
drift grew with molecule size. Exact float enumeration of all `prod(nᵢ+1)` sub-formula
masses cannot drift and costs at most a few hundred thousand floats in this mass range —
11 ms for `C30H50N5O10S2`. Two tests now assert that a formula contains its own parts and
nothing more.

## 7. The two retrieval constants, fitted instead of guessed

`scripts/tune_retrieval.py`, swept on the 4,000-molecule `in_library` split. Both of these
were placeholders in the first baseline, with comments saying so.

**MRR by neutral-loss weight × fusion rule:**

| loss weight | logsumexp | max | mean | sum | rrf |
|---|---|---|---|---|---|
| **0.00** | **0.6562** | 0.6509 | 0.6456 | 0.6456 | 0.6122 |
| 0.25 | 0.6540 | 0.6507 | 0.6464 | 0.6464 | 0.6110 |
| 0.50 | 0.6532 | 0.6496 | 0.6462 | 0.6462 | 0.6113 |
| 1.00 | 0.6480 | 0.6452 | 0.6437 | 0.6437 | 0.6078 |
| 2.00 | 0.6433 | 0.6419 | 0.6402 | 0.6402 | 0.6023 |

- **The neutral-loss view does not help; it hurts.** Weight 0 wins for every fusion rule
  and MRR falls monotonically as the weight rises. The 0.5 placeholder cost 0.0036.
- **Log-sum-exp beats the maximum** (0.6562 vs 0.6509). The argument for the maximum was
  sound — a reference recorded at one collision energy matches one query spectrum and not
  the others, so averaging punishes the right answer — and a soft maximum keeps that while
  still crediting a candidate that matches several of a molecule's spectra. Plain mean and
  sum are worse, and reciprocal-rank fusion is clearly worst here, which is worth noting
  because RRF is the public stack's blending tool of choice.
- Combined against the placeholders: **+0.0066 ± 0.0039** (95 %, paired, n = 4,000) →
  real.

That last line is the measurement thesis in one number. An effect of 0.0066 is invisible
to a leaderboard submission (±0.047) and comfortably resolved by a paired offline
comparison. The harness paid for itself here.

**Scope, which matters.** This is the `in_library` regime, where the correct answer is the
same molecule at another collision energy, so the fragment cosine already matches it
almost exactly. Neutral losses exist to match *analogues* — a molecule and its methylated
derivative share losses but not fragment masses — which is the `held_out` regime, where
retrieval scores 0 and this experiment is blind. Read it as "neutral losses do not help
same-molecule retrieval", not as "neutral losses are useless". The fusion-rule ranking
should transfer; the loss weight must be refitted once a ranker scores in `held_out`.

## 8. Metric reimplementation

`src/casmi/metric.py`, 40 tests in `tests/test_casmi.py`. The three rules that decide
whether a local number matches the board:

1. **InChIKey-14**, so stereochemistry is ignored — two enantiomers are one answer.
2. **Tautomer-canonical**, not raw. A public measurement puts the disagreement at 10.4 %
   of keys, understating a raw-key MRR by ~0.007. `metric.measure_tautomer_churn`
   re-measures it on our own structures rather than inheriting the number.
3. **Duplicates do not consume a slot.** The scorer keeps the first occurrence of each
   identity, so 25 spellings of ethanol followed by the answer scores 0.5, not 0.04. This
   is why output-side dedup is worth ≤0.001: the scorer already does it.

One caveat on the numbers above: the retrieval evaluation compares raw `inchikey14` on
both sides, which is self-consistent because both come from `train.parquet`. Tautomer
canonicalisation becomes mandatory the moment an external database's keys enter the
comparison, since those were computed by someone else.

## 9. Analogue propagation, and the split bug it exposed

`src/casmi/analog.py`, `scripts/eval_analog.py`. This section corrects an error in the
validation design above, so read it before trusting any number that depends on the
spectral library.

### The mechanism

In the `held_out` regime the molecule has no reference spectrum, so direct retrieval
scores 0. Analogue propagation reaches it anyway: match the query against the **whole**
library, take the best-matching *other* molecules, and rank mass-window candidates by
structural similarity to those. This is the root of the entire public lineage.

Measured on 200 validation molecules, `top_k = 10`, `p = 3`:

| | MRR |
|---|---|
| random over the mass window | 0.077 |
| formula soft ordering | 0.116 |
| **analogue propagation** | **0.714** |
| analogue + formula (hard formula-group ordering first) | 0.402 |

The last row is the hard-gate lesson again: ordering by formula group *first* pushes the
true answer behind an entire wrong group whenever the formula call is wrong, which costs
more than the formula signal adds. Formula has to be a feature beside the analogue score,
not a sort key before it.

### 0.714 is not a leak, and that is the problem

0.714 is higher than our `in_library` cosine bracket (0.650) and far above the best
verified public stack (0.433), so it demanded an explanation before it could be believed.

The first hypothesis — structural near-duplicates hiding behind different InChIKey-14
values — is **wrong**. Measured over the 200 molecules, the best analogue's structural
Tanimoto to the true answer tops out at 0.964, only 1.5 % exceed 0.95, and none exceeds
0.99. There are no duplicates.

What the diagnostic found instead: the library is **densely populated with close
relatives**. Median structural Tanimoto to the best analogue is **0.72**, and median
spectral cosine to it is **0.887**. Enveda measured families of related natural products,
so nearly every validation molecule has a distinct but near-identical cousin in train.

Stratifying by that similarity explains the whole number:

| structural Tanimoto to the best available analogue | n | MRR | top-1 |
|---|---|---|---|
| ≤ 0.40 | 28 | **0.273** | 10.7 % |
| 0.40–0.50 | 18 | 0.729 | 61.1 % |
| 0.50–0.60 | 24 | 0.623 | 45.8 % |
| 0.60–0.70 | 26 | 0.815 | 65.4 % |
| 0.70–0.80 | 44 | 0.888 | 81.8 % |
| 0.80–1.00 | 60 | 0.781 | 60.0 % |

Performance is almost entirely a function of whether a close relative exists. Restricted
to molecules with no relative above Tanimoto 0.5, analogue propagation scores **0.452** —
which lands beside the best verified public stack (0.433) and the current leaderboard top
(0.504).

### What this means, and what has to change

> **Our validation split is optimistic by roughly 0.26 MRR for any method that leans on
> structural neighbourhood.** Splitting by molecule leaves whole families intact; the real
> test behaves like the "no close relative" stratum.

The inference runs the other way too, and it is reassuring about the data rather than
about our split: the public leaderboard's 0.433 is almost exactly what this mechanism
scores when relatives are absent. That is independent evidence that the real test was
constructed to exclude the family overlap our split preserves.

Consequences, in order:

1. **The split must become family-disjoint**, not merely molecule-disjoint — a scaffold or
   similarity-threshold split, which is standard practice in molecular ML and which this
   work skipped. Every library-dependent number above is to be re-measured on it.
2. **Formula identification is barely affected** (it never touches the library), so its
   +0.037 stands. The `in_library` retrieval numbers and the fusion-rule ranking are
   measured within one split and remain valid as *relative* comparisons.
3. **FPNet is partly affected**: it trains on spectra of the validation molecules'
   relatives, so its 0.210 is optimistic too, though less directly.
4. **Analogue propagation is still the strongest single mechanism we have**, and the
   stratification says where it fails — exactly the molecules with no documented family,
   which is where the remaining score lives.

## 10. Calibrating our validation against the real leaderboard

`scripts/build_neighbours.py`, then `scripts/eval_analog.py --family-ceiling ...`.

§9 showed our molecule-level split is too easy because it leaves chemical families intact.
The fix is not one harder split but a **tunable one**. `build_neighbours.py` computes every
(validation molecule, bank structure) pair with Tanimoto ≥ 0.35 — 10.6 M pairs in 120 s —
and the evaluation then drops analogues at or above a ceiling, simulating a library from
which the molecule's family is absent.

Note what is filtered and what is not. The **reference spectral library** loses the
relatives; the **candidate bank keeps everything**, because the true answer has to stay
findable and the real test's candidate pool is not family-filtered either. The filter uses
the label, which is legitimate: it constructs a harder split, not a method, and nothing
downstream of the split can see it.

Measured on 300 molecules, `top_k = 10`, `p = 3`:

| family ceiling | median best relative left | MRR, analogue propagation |
|---|---|---|
| 1.00 — our molecule-level split | 0.695 | 0.682 |
| 0.80 | 0.638 | 0.669 |
| 0.70 | 0.578 | 0.631 |
| 0.60 | 0.494 | 0.547 |
| **0.50** | 0.432 | **0.4341** |
| 0.45 | 0.388 | 0.355 |
| 0.40 | 0.360 | 0.263 |

> **At ceiling 0.50 analogue propagation scores 0.4341. The best verified public stack
> scores 0.433 on the real test.**

A match to 0.001. Three things follow.

**We now have a calibrated validation.** The real test behaves like a family-disjoint
split at ceiling ≈ 0.50, so our split can be set to reproduce the real difficulty instead
of understating it by 0.25. Every future comparison should be run at that ceiling, and
ideally reported across the curve — a method that only helps where close relatives exist
is a method that will not transfer.

**One well-tuned mechanism reproduces the entire public stack.** Their two engines,
forward simulators, popularity prior, PubChem join and rank fusion arrive at the same place
as analogue propagation alone. Either their stacking adds almost nothing on top of the
analogue signal, or they are leaving a great deal on the table elsewhere. Both readings
argue for building up from a measured baseline rather than forking their pipeline.

**Formula must be a feature, not a sort key.** The combination column is worse than
analogue propagation alone at *every one of the seven ceilings* — 0.371 against 0.682 at
ceiling 1.0, 0.242 against 0.434 at 0.50. Ordering by formula group first pushes the true
answer behind an entire wrong group whenever the formula call is wrong, and at 51.5 %
accuracy that happens often enough to cost more than the signal is worth. §6 reached this
conclusion from the gate-versus-soft-ordering arithmetic; this confirms it against a strong
ranker instead of against random.
