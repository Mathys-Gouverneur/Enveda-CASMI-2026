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

## 6. Metric reimplementation

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
