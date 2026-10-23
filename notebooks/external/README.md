# Vendored public notebooks

Seventeen public Kaggle kernels, snapshotted **2026-10-23**, one directory each, holding
the `.ipynb` and the `kernel-metadata.json` as the API returned them. Nothing here is
ours and nothing here is modified.

Committed rather than re-downloaded on demand: public kernels are revised in place and
sometimes deleted, so the copy that counts is this one. 3.4 MB in all.

Re-snapshot or add one with:

```
scripts/pull_external_notebook.sh <owner>/<kernel-slug> [local-dir-name]
```

## Reading the score column

**Every score in a notebook title is the author's own claim.** Across the public record
only three numbers were ever verified by re-running the notebook, and the public test set
has 400 molecules — so one molecule is worth 0.0025 and these claims are not separable
from noise at the 0.01 level. The vote column is the better signal of whether anyone else
got the notebook to work: the two highest-claimed notebooks here have **1 and 2 votes**.

See [`../../docs/public-meta.md`](../../docs/public-meta.md) for the synthesis.

## Tier A — understanding the problem

| directory | author | votes | why it is here |
|---|---|---|---|
| [`casmi-denovo-tutorial`](casmi-denovo-tutorial/) | `inversion` (Kaggle) | 385 | The official tutorial: a spectrum-encoder → SMILES-decoder transformer. **Read first** — it is the only authoritative statement of the data columns, the submission shape and the per-molecule pooling rule. Note that it solves the task by *generation*, which is not what the leaderboard does. |
| [`public-lineage-034-to-044`](public-lineage-034-to-044/) | `alexchilton` | **0** | A notebook-by-notebook map of which public notebook descends from which, what each changed, and what it scored — with every number labelled *claimed* / *board reading* / *measured on re-run*. The most valuable document in the competition, and nobody has upvoted it. **Read second.** |
| [`metric-audit-and-safe-tail`](metric-audit-and-safe-tail/) | `obstacledeveloper` | 47 | The reference 0.433 stack, documented stage by stage in prose, with the per-stage leaderboard deltas. The clearest description of the pipeline everything above 0.40 shares. **Measured 0.433 on verbatim re-run** — the highest verified public number. |
| [`the-data-behind-spectra-to-structures`](the-data-behind-spectra-to-structures/) | `dmitriigluzdov` | 136 | EDA of the corpus itself. |
| [`from-spectra-to-structures`](from-spectra-to-structures/) | `dmitriigluzdov` | 162 | The companion pipeline walkthrough. |
| [`inchikey-churn-metric-cost`](inchikey-churn-metric-cost/) | `dariushafshar` | 8 | Measures that 10.4% of pool keys differ between the raw and tautomer-canonical InChIKey — i.e. that an offline MRR on raw keys is understated by ~0.007. Matters for any local validation. |
| [`13-steps-to-0456`](13-steps-to-0456/) | `varshithakolupuri` | 2 | Claims to attribute a gain to each of 13 steps. Low votes; useful as a checklist, not as evidence. |
| [`fast-spectral-cosine-baseline`](fast-spectral-cosine-baseline/) | `haideptry` | 262 | The simplest thing that works: spectral cosine library matching. Good floor to measure against. |
| [`analog-propagation-baseline`](analog-propagation-baseline/) | `prvsiyan` | 151 | **The root of the entire leaderboard lineage**: library matching plus mass-shifted analog propagation. Also carries the measured null on ungated pool flooding (0.335 → 0.205). |

## Tier B — the public state of the art

| directory | author | votes | claimed |
|---|---|---|---|
| [`spectraforge-apex-lb-0458`](spectraforge-apex-lb-0458/) | `mayankraaz` | 2 | 0.458 — highest claim found; essentially unvetted |
| [`pubchemlite-n1a-explained-lb-0456`](pubchemlite-n1a-explained-lb-0456/) | `omerfarukmerey` | 1 | 0.456, with its reasoning written out |
| [`e006-n1a-chebi-lb-0452`](e006-n1a-chebi-lb-0452/) | `seyitkaangunes` | **35** | 0.452 — **the highest claim that the community actually corroborated**, and by the author who introduced engine fusion |
| [`e006-pubchem-promo-lb-0446`](e006-pubchem-promo-lb-0446/) | `seyitkaangunes` | 62 | 0.446 |
| [`forensics-lab-apex-dossier-lb-0442`](forensics-lab-apex-dossier-lb-0442/) | `saddamaditya` | 72 | 0.442 |
| [`iceberg-where-glacier-is-blind-lb-0438`](iceberg-where-glacier-is-blind-lb-0438/) | `samanyu1808` | 57 | 0.438 — one clean idea: ICEBERG-only rerank for molecules with no `[M+H]+` spectrum, where GLACIER is blind by construction |
| [`closing-the-evidence-gap`](closing-the-evidence-gap/) | `flexonafft` | 89 | not stated |
| [`evgendvorkin-casmi-2026`](evgendvorkin-casmi-2026/) | `evgendvorkin` | 188 | not stated — origin of the "two-ranker" blend |

## What the metadata gives away

Read off the seventeen `kernel-metadata.json` files:

- **Internet is disabled in every single one.** All external data arrives as attached
  Kaggle datasets.
- The ten strongest notebooks attach the **same 14–17 datasets**. The public meta is not
  really seventeen approaches; it is one stack with retuned constants. The shared set:

  | dataset | used by |
  |---|---|
  | `dmitriigluzdov/casmi26-pubchem-popularity-prior` | 13 |
  | `prvsiyan/casmi26-ranker-features` | 11 |
  | `prvsiyan/casmi26-fp-models-v2` | 11 |
  | `prvsiyan/coconut-casmi26-candidates` | 11 |
  | `ahmedberatozer/casmi26-{iceberg,glacier,fpnet-full1,pubchem-tier,v2-pool,v3-models,v4b-models}` | 10 each |
  | `megayak/casmi26-simulated-ranker-rows` | 10 |
  | `metric/rdkit-2026-3-3-wheel` | 10 |
  | `prvsiyan/chebi-lipidmaps-casmi26` | 6 |
  | `hengck23/hengck23-dreams-enveda-casmi26` | 6 |

- **`metric/rdkit-2026-3-3-wheel` in ten of them is not incidental.** A Python
  3.12-vs-3.13 RDKit wheel mismatch silently zeroed ICEBERG/GLACIER scoring across the
  field for days; pinning that wheel is the fix.

Before attaching any of these: audit the licence. External data under a
commercial-use-restricting licence can get a solution **removed from the leaderboard**,
and at least one dataset in circulation (`spectral-library-massbankharmonized`,
CC BY-NC-SA) is used with the licence filter deliberately removed in one public
ablation. Details in [`../../docs/competition.md`](../../docs/competition.md).
