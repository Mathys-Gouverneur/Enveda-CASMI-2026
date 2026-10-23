# Levers: everything that could gain points, and what each is worth

Written 2026-10-23. Companion to [`strategie.md`](strategie.md), which says what to do
next; this one is the full inventory, including the things we decided against.

## How to read the evidence column

| label | meaning |
|---|---|
| **measured** | measured by us, on the 4,000-molecule timsTOF validation split, paired |
| *published* | a number a competitor published, with the caveats in [`public-meta.md`](public-meta.md) — only three numbers in the entire public record were verified by re-running a notebook |
| estimated | our judgement, unmeasured, stated as such |

**Nothing in this document adds up.** The brackets below are 0.085 at the floor and 0.433
for the best verified public stack; that 0.35 is not a sum of the deltas listed here.
Mechanisms overlap — formula identification, fingerprint prediction and fragment scoring
are three views of the same evidence, and a ranker given all three gets less than the sum.
Treat the numbers as ordering, not as a budget.

And the resolution caveat that governs everything: a single submission has a 95 % interval
of **±0.047**. A paired offline comparison on 4,000 molecules resolves **0.0024**. Every
lever below is to be accepted or rejected offline, never on the board.

## The budget: where score can come from

| bracket | MRR | what it is |
|---|---|---|
| random over the mass window | **0.085** | the floor. Not zero — zero is what pure retrieval scores in this regime, which is a definition, not a difficulty |
| + formula soft ordering | **0.123** | measured, ours |
| true formula known (oracle) | 0.205 | ceiling of all formula work |
| best verified public stack | 0.433 | the target |
| current leaderboard top | 0.504 | inside its own noise against the top 16 |
| plain cosine, same molecule in the library | 0.650 | optimistic bracket, not the test's regime |

The task restated: **rank ~78 same-mass candidates, of which ~34 share the right answer's
molecular formula, for a molecule whose spectrum the library has never seen.**

---

## Tier 1 — measured by us, already banked or one step away

| lever | value | evidence |
|---|---|---|
| **Formula soft ordering** — rank candidates by formula score, never filter on it | **+0.037** (0.085 → 0.123), 30 % of the formula prize | **measured** |
| **Chance-corrected formula scoring** — subtract the by-chance match rate | formula top-1 32 % → **51.5 %**; the uncorrected version is worth −6 % | **measured** |
| **Dimer adducts as references** — `[2M+Na]+` alone is 188k spectra | +0.020 on in-library retrieval; 29 % more reference spectra | **measured** |
| **Log-sum-exp over a molecule's spectra** instead of the maximum | +0.005 | **measured** |
| **Drop the neutral-loss cosine** in same-molecule retrieval | +0.004 | **measured**, and scoped — see Tier 3 |

## Tier 2 — mechanisms competitors published, for us to re-derive

Re-derive rather than import: the datasets carrying their versions are CC-BY-NC-SA, and
host rulings make that a disqualification risk ([`strategie.md`](strategie.md)).

| lever | value | note |
|---|---|---|
| **Rank external candidates through the real engine**, never insert them at fixed slots | *+0.011*, the largest single published step | The general principle matters more than the PubChem specifics: never bypass the ranker. |
| **Fuse two independently built engines by reciprocal rank** | *+0.016* | Two engines fail differently; rank fusion ignores calibration mismatch. Note our own measurement that RRF is the *worst* way to pool one engine's spectra — it is a tool for combining engines, not spectra. |
| **Popularity prior** — `log1p(#PubChem substances) + log1p(#PubMed papers)` | *+0.010* | Not chemistry. A bet on the test's composition: if the private set holds more obscure compounds it reverses. Cheap, and worth having behind a switch. |
| **Forward simulators on same-formula groups** (ICEBERG / GLACIER) | *+0.006* for restricting GLACIER to `[M+H]+` | The only family that sees *how* a molecule fragments. Obtain checkpoints from their upstream MassSpecGym release, not the licence-undeclared Kaggle repackaging. |
| **Cover the simulator's blind spot** — ICEBERG where GLACIER has no score | *+0.005 claimed* | 21 % of test spectra are not `[M+H]+` (16 % negative mode, 2.6 % formate) and get no GLACIER score at all. A named, measured hole. |
| **In-silico fragmentation scoring** (MetFrag-style) | in their 31-feature ranker as 4 features | Finer-grained than our formula score: it scores the *structure*, we score the *formula*. Complementary, not redundant. |

## Tier 3 — architecture gaps against the public FPNet

Read off their code, not their prose. Our FPNet is close to a subset of theirs.

| lever | expected | why |
|---|---|---|
| **Condition on instrument, adduct, collision energy, ionisation mode** (learned embeddings + a global token) | estimated **large** | The highest-expected-value item in this document. It lets one model train on all 2.54 M spectra while learning the instrument correction, instead of our discarding 55 % of the corpus to get domain purity. Ironic given the instrument finding is ours: they put it in the model, we put it in the data selection, and theirs is probably the better place. |
| **Neutral-loss embedding per peak** (a second sinusoidal channel) | estimated moderate | **Our measurement does not contradict this.** We measured that a neutral-loss *cosine* hurts in the `in_library` regime, where the answer is the same molecule and fragment masses already match. Neutral losses exist to relate a molecule to its *analogues*, which is the regime that matters and where our experiment is blind. Our omission was a side effect of a scoped result, not a decision. |
| **FPNet logits as ranker features, not as the final score** | estimated **large** | They feed `candidate_fp @ predicted_logits` to LightGBM in six normalisations at once — raw, z-score, rank-normalised, delta-to-max, bit-count-normalised, and an is-argmax indicator — rather than trusting any one calibration. Ranking directly by Tanimoto, as we planned, is strictly weaker. |
| **Agreement features between evidence sources** | estimated moderate | Their most subtle group: `library × (1 − model_rank)`, `analogue × (1 − model_rank)`, whether the library's best and the model's best are the same candidate, and the *correlation* between library score and model rank. Concordance between sources is a signal distinct from each source. |
| **`log(n_candidates)` as a feature** | estimated small but free | Stratum awareness inside the model. The published result that ICEBERG-as-feature is worth +0.0015 *only* where candidates exceed 100 says the gains are conditional; this is how you let the model exploit that. |
| **CLS token concatenated with the masked mean** | estimated small | Two pooled views instead of one. |
| **Merge a molecule's spectra into one pseudo-spectrum** before inference (peaks within 0.005 Da collapsed, max intensity kept) | estimated moderate | A different multi-spectrum strategy from ours: they merge spectra then score once, we score each then pool. Theirs gives the model the whole molecule at once. Both are testable against each other to ±0.001. |
| **Ensemble several checkpoints** | estimated small–moderate | They load multiple FPNet checkpoints plus a merged-spectra variant. Cheap, reliable, no new ideas required. |
| **Revisit peak budget** | estimated small | They use 128 peaks, we use 256, and the test's median spectrum has 230. Someone is leaving information on the table; measurable. |

## Tier 4 — unexploited, where our own edge is

| lever | expected | why nobody has it |
|---|---|---|
| **The measurement harness itself** | **multiplies everything else** | Not a score lever, a *selection* lever. Competitors cannot distinguish a 0.01 gain from noise on 400 molecules; we resolve 0.0024 on 4,000. Every other row in this document is accepted or rejected with it. This is the single most valuable asset we have. |
| **Learned cross-instrument similarity** | estimated moderate–large | 2,288 molecules are measured on *both* the Enveda timsTOF and a public instrument — a directly supervised dataset for learning what the instrument change does to a spectrum. Nobody in the public record uses it. |
| **Contrastive pretraining on collision-energy pairs** | estimated moderate | 183 k molecules × multiple energies is a large self-supervised positive-pair set: the same molecule at two energies must embed together. Free supervision, untouched. |
| **Same-formula isomer separation beyond simulators** | estimated large, hard | 34 candidates remain in the median true-formula group and this is where the residual score lives. Nothing we have built touches it: formula identification stops at the group boundary, fingerprints struggle (theophylline vs theobromine score 0.84 Tanimoto to each other), and simulators are slow. An open problem, which is exactly why it is where the top 0.07 may hide. |
| **Rank-2 hedging** | estimated small | Rank 1 is worth 1.0 and rank 2 is 0.5, so when rank 1 is uncertain the rank-2 pick should be maximally *different* from it, not its near-duplicate. Nobody optimises expected MRR explicitly. |
| **Stratified rankers** | estimated moderate | Fit separate models per stratum (adduct, candidate count, mass band) instead of one global ranker with a stratum feature. Our validation set is large enough to fit per-stratum models honestly; the board is not. |
| **Generation as a candidate source, gated** | estimated small here | Worth stating because it is the tutorial's whole approach: for a molecule no database covers, a generator could supply candidates and the ranker would judge them. But our coverage measurement says test molecules are almost certainly *in* train, so the case barely arises — and the published null on pool flooding (0.335 → 0.205) warns that ungated additions are actively harmful. **Low priority, contrary to appearances.** |

## Tier 5 — enabling, not scoring

| lever | why |
|---|---|
| **Rebuild the candidate banks from upstream** (COCONUT CC BY, ChEBI CC BY, PubChemLite CC0) | Not for recall — we measured it adds none. So that nothing in a submission traces to a CC-BY-NC-SA source, which host rulings make a removal risk. |
| **Pin RDKit 2026.03.3 in any submission notebook** | A Python 3.12-vs-3.13 wheel mismatch silently zeroed ICEBERG/GLACIER scoring across the field for days. Several contradictory public claims are probably this. |
| **Audit every attached dataset's licence before submitting** | Four of the stack's core datasets are CC-BY-NC-SA and seven are licence-undeclared. |
| **One early submission** | Worth little as a measurement (±0.047), a lot as a check that the offline and online pipelines agree. The one thing that could falsify our central assumption that test structures are in train. |
| **Runtime budget** | Notebook-only, internet disabled, simulators slow. Only the top-k candidates can be simulated. |

## Measured nulls — do not spend time here

| | evidence |
|---|---|
| Expanding the candidate pool with external databases | **measured**: +27 to +34 candidates in the window, **zero** recall gain. *Published*: 0.335 → 0.205 when flooded |
| Hard gating on the predicted formula | **measured**: +0.004 against +0.037 for soft ordering; needs 41 % accuracy just to break even |
| Locking rank 1 | *published*: local validation 0.63 → 0.89, board **0.413 → 0.409** |
| Inserting candidates at fixed slots | *published*: displaced 95.2 % of true rank 4–10 candidates to the tail |
| Anything below rank 3 | *published*: the whole tail is worth ≤0.002 |
| Output-side tautomer dedup alone | *published*: ≤0.001 — the official scorer already dedups |
| Tuning constants by submission | **measured**: ±0.047. You cannot see what you are tuning |

## Priority order

Ranked by expected value per unit of work, not by expected value alone.

1. **Instrument / adduct / collision-energy conditioning**, and retrain on all 2.54 M
   spectra. Largest expected gain, and the current run is the clean control for it.
2. **FPNet logits into a ranker** with the formula score, library score, fragment score and
   agreement features. Turns three separate signals into one that can exploit their
   concordance.
3. **Neutral-loss embedding per peak.** Cheap, and our reason for omitting it does not hold.
4. **Compose and measure.** Formula + FPNet + spectral similarity are three views; whether
   they add or overlap is open and we can answer it to ±0.001.
5. **Forward simulators on same-formula groups**, from upstream checkpoints — the only
   thing that attacks the 34 remaining isomers.
6. **Learned cross-instrument similarity** on the 2,288 paired molecules. Our most original
   bet; queue it behind the above because it is research, not engineering.
7. **Popularity prior, behind a switch.** Cheap, published at +0.010, and philosophically a
   bet we should be able to turn off.
8. **Rebuild banks from upstream** and audit licences, before any submission.
