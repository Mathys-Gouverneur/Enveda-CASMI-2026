"""The submission pipeline: one implementation, run locally and on Kaggle.

`kaggle/analog-baseline/notebook.ipynb` mounts this library and calls `run_submission`.
`scripts/predict.py` calls the same function against our own validation split. One
implementation, so a leaderboard score is reproducible locally — the lesson carried over
from the RSNA project's `kaggle/README.md`.

What this baseline is, and why it is the right thing to submit first:

**Analogue propagation, and nothing else.** Match each test spectrum against the whole
training library; the best matches are *other* molecules, whose structures are known;
rank the mass-window candidates by weighted structural similarity to those. Measured at
0.682 on our molecule-level validation split and **0.434** at the family-disjoint ceiling
of 0.50 that we believe reproduces the real test's difficulty
(`docs/measurement.md` §10).

**It needs no external data at all.** Reference library, candidate bank and fingerprints
all come out of `train.parquet`. No attached dataset, no model weights, nothing under a
non-commercial licence — so it carries none of the disqualification exposure that four of
the public stack's core datasets do (`docs/strategie.md`).

**So it tests the measuring instrument.** We predict ~0.43. A board score near that
confirms the calibration and licenses hundreds of offline comparisons; a score far from
it says the calibration is wrong, which is worth knowing now rather than in six weeks.
That hypothesis is **not yet verified** — the agreement with the public 0.433 is between
two different methods on two different test sets, which is suggestive and no more.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import analog as AN
from . import data as D
from . import fingerprint as FP
from . import metric as M

#: Fitted on the validation split, not inherited: `top_k = 10` beat 20 and 80, and
#: `p = 3` beat 1 and 6 (`out/analog_eval.csv`).
TOP_K = 10
P_SIM = 3.0
PPM = 10.0

#: Reference library restricted to the test's own instrument. Every number we have was
#: measured this way; whether adding the other 55% of the corpus helps is a question for
#: the harness, not for a submission.
TIMSTOF_ONLY = True

FALLBACK_SMILES = "CCO"
"""Every molecule_id must carry at least one guess, or the row scores nothing anyway but
the file may be rejected. Ethanol is the sample submission's own filler."""


def _log(message: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {message}", flush=True)


def run_submission(train_path, test_path, out_path, top_k: int = TOP_K,
                   p: float = P_SIM, ppm: float = PPM,
                   timstof_only: bool = TIMSTOF_ONLY,
                   limit_molecules: int | None = None) -> pd.DataFrame:
    """Read train and test parquet, write `submission.csv`, return the frame.

    Deliberately free of `try/except`: a swallowed failure would write the ethanol
    fallback and report success, which looks exactly like a method that learnt nothing.
    """
    t0 = time.time()
    train_path, test_path, out_path = Path(train_path), Path(test_path), Path(out_path)

    # ---- the candidate bank, out of train.parquet itself ---------------------------
    meta = pd.read_parquet(train_path,
                           columns=["inchikey14", "normalized_smiles",
                                    "molecular_formula", "ingest_lib"])
    bank = meta.drop_duplicates("inchikey14").drop(columns="ingest_lib")
    masses = {f: D.formula_mass(f) for f in bank.molecular_formula.unique()}
    bank["neutral_mass"] = bank.molecular_formula.map(masses)
    bank = (bank.dropna(subset=["neutral_mass"])
                .sort_values("neutral_mass").reset_index(drop=True))
    del meta
    _log(f"bank: {len(bank):,} structures", t0)

    X, ok = FP.fingerprint_matrix(bank.normalized_smiles)
    bank, X = bank[ok].reset_index(drop=True), X[ok]
    bank_mass = bank.neutral_mass.to_numpy(dtype=float)
    bank_key = bank.inchikey14.to_numpy(dtype=object)
    smiles_of = dict(zip(bank.inchikey14, bank.normalized_smiles))
    row_of = {k: i for i, k in enumerate(bank_key)}
    _log(f"fingerprinted {len(bank):,} structures ({(~ok).sum()} unparseable)", t0)

    # ---- the reference library ------------------------------------------------------
    spec_cols = ["ingest_lib", "inchikey14", "adduct", "precursor_mz",
                 "ms2_mzs", "ms2_normalized_intensities"]
    filters = [("ingest_lib", "in", list(D.TIMSTOF_LIBS))] if timstof_only else None
    ref = pd.read_parquet(train_path, columns=spec_cols, filters=filters)
    ref = ref[ref.inchikey14.isin(row_of)].reset_index(drop=True)
    _log(f"reference library: {len(ref):,} spectra over "
         f"{ref.inchikey14.nunique():,} molecules", t0)

    # ---- the queries ----------------------------------------------------------------
    query = pd.read_parquet(test_path)
    query["neutral_mass"] = D.neutral_mass(query.precursor_mz, query.adduct)
    unmapped = int(query.neutral_mass.isna().sum())
    if unmapped:
        # Not silently dropped: a molecule with no usable mass still needs a row, and we
        # want to see in the log that it happened.
        print(f"WARNING: {unmapped} test spectra have an unparseable adduct", flush=True)
    molecules = list(dict.fromkeys(query.molecule_id))
    if limit_molecules:
        molecules = molecules[:limit_molecules]
        query = query[query.molecule_id.isin(set(molecules))].reset_index(drop=True)
    _log(f"test: {len(query):,} spectra over {len(molecules):,} molecules", t0)

    # ---- analogues ------------------------------------------------------------------
    analogues = AN.find_analogues(query, ref, top_k=top_k, verbose=True)
    _log(f"analogues found for {len(analogues):,} molecules", t0)

    # ---- propagate and rank ---------------------------------------------------------
    rows = []
    for i, mol in enumerate(molecules):
        g = query[query.molecule_id == mol]
        mass = g.neutral_mass.dropna()
        ranked: list[str] = []
        if len(mass):
            lo, hi = D.mass_window(bank_mass, float(mass.median()), ppm)
            keys = list(dict.fromkeys(bank_key[lo:hi]))
            a = [x for x in analogues.get(mol, []) if x[0] in row_of]
            if keys and a:
                scores = AN.propagate(
                    a, np.array(keys, dtype=object),
                    cand_fp=X[[row_of[k] for k in keys]],
                    analog_fp=X[[row_of[k] for k, _ in a]], p=p)
                order = np.argsort(-scores)
                ranked = [keys[j] for j in order]
            elif keys:
                ranked = keys

        # Emit SMILES, deduplicated on the metric's own tautomer-canonical identity so a
        # slot is never spent twice on one answer. The official scorer dedups too, so
        # this is worth little (<=0.001, measured publicly) -- it is here because it is
        # free and correct, not as a gain.
        out, seen = [], set()
        for key in ranked:
            smi = smiles_of.get(key)
            if smi is None:
                continue
            ident = M.key14(smi)
            if ident is None or ident in seen:
                continue
            seen.add(ident)
            out.append(smi)
            if len(out) == M.N_SLOTS:
                break
        rows.append({"molecule_id": mol,
                     "smiles": ";".join(out) if out else FALLBACK_SMILES})
        if i and i % 100 == 0:
            _log(f"ranked {i:,}/{len(molecules):,}", t0)

    submission = pd.DataFrame(rows)
    submission.to_csv(out_path, index=False)
    _log(f"wrote {out_path} — {len(submission)} rows", t0)
    return submission


def check_submission(submission: pd.DataFrame, expected_molecules) -> None:
    """Assert the shape Kaggle expects. Loud, because a bad file scores 0 silently."""
    assert list(submission.columns) == ["molecule_id", "smiles"], submission.columns
    assert submission.molecule_id.is_unique, "one row per molecule"
    expected = list(dict.fromkeys(expected_molecules))
    missing = set(expected) - set(submission.molecule_id)
    assert not missing, f"{len(missing)} molecules missing from the submission"
    counts = submission.smiles.str.split(";").apply(len)
    assert counts.max() <= M.N_SLOTS, f"a row carries {counts.max()} candidates"
    assert (counts >= 1).all(), "a row carries no candidate"
    print(f"submission.csv: {submission.shape}, "
          f"{counts.min()}-{counts.max()} candidates per molecule "
          f"(median {counts.median():.0f}), "
          f"{submission.smiles.eq(FALLBACK_SMILES).sum()} fallback rows")
