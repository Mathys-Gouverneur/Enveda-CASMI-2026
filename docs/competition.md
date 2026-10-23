# The task, the data, and the metric

Measured against `data/raw/` on 2026-10-23 unless a line says otherwise. Where a public
notebook's claim disagrees with the file, the file wins and the disagreement is noted —
there are two, and both matter.

## The task

Given an MS/MS spectrum — a list of fragment `m/z` values with intensities, plus the
precursor `m/z` — name the molecule that produced it. Output is **up to 25 SMILES per
molecule, ranked**.

Scoring is **per molecule, not per spectrum**: the test file holds 1,213 spectra for 400
molecules, 3.0 spectra each, and a molecule's spectra all vote into one ranked list.

`sample_submission.csv` is 400 rows, two columns:

| molecule_id | smiles |
|---|---|
| `m_005e53` | `CCO;CCO;CCO;` … 25 times |

So the 25 ranked candidates are **one semicolon-joined string in a single cell**, not 25
rows and not 25 columns.

## The metric: mean reciprocal rank

The official implementation is a public Kaggle notebook,
<https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank> (Apache-2.0). Per
molecule the score is `1/rank` of the first correct candidate, `0` if none of the 25 is
correct; the leaderboard is the mean over molecules.

Two properties of the identity it compares matter more than they look:

- **Stereochemistry is ignored.** Candidates are compared on **InChIKey-14** — the first
  block of the InChIKey, the skeleton hash. Two spellings of the same skeleton occupy two
  slots and win nothing.
- **It is the tautomer-canonical key, not the raw one.** `alexchilton` measured 4,000
  COCONUT keys and found **10.43% differ** between the raw and tautomer-canonical forms.
  An offline MRR computed on raw keys is therefore understated by roughly **0.007** —
  more than most of the public leaderboard steps are worth. Any local validation has to
  canonicalise tautomers the way the metric does, or it is measuring the wrong thing.

### Where the score actually lives

`1/rank` falls away fast: rank 1 is worth 1.0, rank 2 is 0.5, rank 3 is 0.33, and
ranks 11–25 are together worth less than rank 2 alone. `lszlst`'s rank-mass probes put
**the entire tail from rank 3 down at no more than 0.002** of the score.

So the whole game is **rank 1, and secondarily rank 2**. Work on slots 3–25 is nearly
free score-wise — which cuts both ways: it is not worth optimising, but it is also not
worth risking, and several public notebooks lost ground by letting forced insertions at
fixed slots displace good candidates.

## The data

| file | size | what |
|---|---|---|
| `train.parquet` | 3.03 GB | **2,539,608** labelled spectra, 275,810 unique `inchikey14` |
| `test.parquet` | 4.8 MB | 1,213 unlabelled spectra, **400 molecules** |
| `sample_submission.csv` | 43 KB | 400 rows, `molecule_id` + semicolon-joined `smiles` |

The tutorial's "all 1.39 M spectra" is **not** the corpus size — it is what is left after
its own `enveda-180` filter and train/val split. The file holds 2.54 M.

`train.parquet` carries 18 columns, considerably more than the six the tutorial reads:

| column | what |
|---|---|
| `ingest_lib` | **source library — the most important column in the file, see below** |
| `inchikey14` | the label, as the metric's skeleton hash |
| `inchikey` | the full key, with stereo |
| `normalized_smiles` | the label, as a structure |
| `molecular_formula` | formula — the metric's isomer problem lives inside formula groups |
| `adduct`, `adduct_orig` | `[M+H]+` etc.; GLACIER is restricted to `[M+H]+` by the public stack |
| `ionization_mode` | positive / negative |
| `instrument_type` | **see below** |
| `precursor_mz`, `precursor_error_ppm` | precursor mass/charge and its error |
| `ms2_mzs`, `ms2_normalized_intensities` | the spectrum |
| `num_peaks`, `base_peak_intensity` | spectrum summaries |
| `collision_energy_ev`, `collision_energy_orig`, `collision_energy_orig_units` | energy |

`test.parquet` has the same spectrum and instrument columns plus `molecule_id` and
`spectrum_id`, and no labels. It notably **lacks** `molecular_formula` — formula has to be
inferred from the precursor and adduct, which is one of the stated reasons same-formula
isomer separation is where the public stack spends its effort.

3 GB on a box with this much RAM means the whole corpus sits in memory; the bottleneck is
candidate search, not I/O.

## The finding that matters most: the test set is a different instrument

**`test.parquet` is 100% timsTOF.** In `train.parquet`, exactly two libraries contain any
timsTOF spectrum at all:

| `ingest_lib` | spectra | molecules | timsTOF | median peaks |
|---|---|---|---|---|
| **`enveda-180`** | 1,153,785 | 182,941 | **100%** | 138 |
| **`enveda-np-examples`** | 1,184 | 250 | **100%** | 142 |
| `pluskal_ms2` | 527,581 | 46,821 | 0% | 21 |
| `riken` | 347,171 | 15,892 | 0% | 9 |
| `gnps` | 220,849 | 45,750 | 0% | 46 |
| `massbank` | 101,727 | 9,180 | 0% | 11 |
| `mona` | 92,416 | 11,681 | 0% | 57 |
| `spectraverse` | 50,933 | 9,631 | 0% | 47 |
| `msdial` | 40,765 | 9,127 | 0% | 11 |
| `drug_plus` | 2,545 | 2,539 | 0% | 108 |
| `masaryk` | 652 | 416 | 0% | 9 |

Every public library — GNPS, MassBank, MoNA, RIKEN, Pluskal, MSDIAL, SpectraVerse,
Masaryk — contains **zero** timsTOF spectra. The two Enveda libraries are 45.5% of the
corpus between them.

Peak density follows the same split, and the test set sits on the Enveda side of it:

```
median peaks:   test 230   |   timsTOF train 138   |   everything else 20
```

Two consequences.

- **The official tutorial's filter throws away the only training data that shares the
  test's instrument**, and it does so in a bare
  `filters=[('ingest_lib', '!=', 'enveda-180')]` with no comment. The whole field
  inherited that line.
- The public leaderboard does library matching — cosine and modified-cosine similarity
  against reference spectra — and its reference libraries were measured on *different
  instruments at a tenth of the peak density*. Whatever that costs, everyone is paying it
  equally, and nobody in the public record appears to have noticed.

The name is misleading: `enveda-180` is **not** 180 molecules. It is 182,941 molecules
over 1.15 M spectra — the largest and densest library in the file.

### What this does not yet establish

That the test molecules are *drawn from* `enveda-180`, or how much the instrument gap
actually costs a cosine match. Both are measurable and neither is measured yet. The first
real experiment should be: hold out timsTOF molecules, and compare retrieval accuracy
with the public-library references against retrieval with the Enveda references.

## Submission mechanics

- Notebook-only, **internet disabled**. Every one of the 17 vendored notebooks runs
  offline; external data has to come in as attached Kaggle datasets.
- The visible test file is a placeholder that overlaps training, so a score on it is an
  operational smoke check, **not** a validation estimate. Real scoring is a hidden rerun.
- Several public notebooks carry a "commit switch" that runs a handful of molecules when
  it detects the placeholder, so that saving a version stays quick.

## Licence exposure — a disqualification risk, not a penalty

The hosts have ruled that external data or models under licences restricting commercial
use or redistribution can get a solution **removed from the leaderboard entirely**.
Rulings collected from the forum threads cited in `alexchilton`'s lineage map:

| allowed | not allowed |
|---|---|
| GNPS, MassBank, MoNA — recent releases (thread 745228) | newly built NIST-matched derivative datasets (745832) |
| CC BY-SA records (745228) | NIST-gated ICEBERG/GLACIER checkpoints (742991, 744338) |
| ICEBERG / GLACIER trained on MassSpecGym (742991) | |
| DreaMS weights (745604) | |
| GNPS-NIST14-MATCHES, grandfathered (745832) | |

`train.parquet` is labelled CC BY-NC 4.0, but the hosts treat it as an open compilation
and do not read the label as a commercial-use bar; **test data carries its own separate
noncommercial restriction** (thread 743236).

The sharp edge: the "spectral-library-massbankharmonized" dataset is CC BY-NC-SA, and
public notebooks differ on whether they filter it to CC BY/CC0 records. `lszlst`'s
`b432-massbank` deliberately removes that filter to measure the coverage cost. **Any
notebook we copy from has to be audited for which datasets it attaches**, because
inheriting one of those attachments inherits the exposure.
