"""The submission pipeline: one implementation, run locally and on Kaggle.

`kaggle/analog-baseline/notebook.ipynb` mounts this library and calls `run_submission`.
`scripts/predict.py` calls the same function against our own validation split. One
implementation, so a leaderboard score is reproducible locally — the lesson carried over
from the RSNA project's `kaggle/README.md`.

What the board has already told us, which is the only calibrated information we have:

| what was submitted | our validation said | the board said |
|---|---|---|
| analogue propagation alone | 0.434 at family ceiling 0.50 | **0.080** |

0.080 is the random floor. The pipeline is **not** at fault: scored against the visible
`test.parquet` — whose 1,213 spectra turn out to be *exact copies* of training spectra,
cosine 1.000 for all 400 molecules — this same code gets MRR 1.0000 and top-1 100 %. So
Kaggle re-ran the notebook against hidden data, and on that data a library-dependent
method carries no information.

Which means our validation measured an easier task than the real one, in a way the family
ceiling did not fix: validation molecules were drawn from `train`, so **the answer was
always in the candidate bank** and a relative was usually in the library. The real test
grants neither.

So every performance number in this repository is suspect, and `method="fpnet"` is the
next experiment rather than the next improvement: FPNet needs **no spectral library at
all**, so if it scores well above 0.080 the failure was library dependence, and if it
scores 0.08 too then something more basic is wrong and that is worth knowing before
building anything else. Both branches print `DIAG` lines into the scoring run's log —
unparseable adducts, window sizes, analogue-match quality — so the next submission
explains itself whatever its score.
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


def _z(v: np.ndarray) -> np.ndarray:
    """Per-molecule z-score, so signals on different scales can be summed."""
    v = np.asarray(v, dtype=np.float64)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 1e-12 else np.zeros_like(v)


def _fpnet_logits(checkpoint, query: pd.DataFrame):
    """Predicted fingerprint logits per query spectrum, and the bits they cover.

    Imported lazily: the analogue method needs neither torch nor a checkpoint, and a
    submission that does not use FPNet should not fail because torch is absent.
    """
    import torch

    from . import fpnet as N

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    bits = np.asarray(ck["bits"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ck.get("v2"):
        cfg = N.FPNetV2Config(**ck["cfg"])
        model = N.FPNetV2(cfg)
        enc_adduct, enc_instr = ck["adduct_vocab"], ck["instrument_vocab"]
    else:
        cfg = N.FPNetConfig(**ck["cfg"])
        model = N.FPNet(cfg)
        enc_adduct = enc_instr = None
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    print(f"DIAG fpnet v2={bool(ck.get('v2'))} best_epoch={ck.get('best_epoch')} "
          f"bits={len(bits)} device={device.type}", flush=True)

    mzs = query.ms2_mzs.to_numpy()
    its = query.ms2_normalized_intensities.to_numpy()
    prec = query.precursor_mz.to_numpy(dtype=np.float32)
    extra = None
    if enc_adduct is not None:
        ce, present = N.mean_collision_energy(query.collision_energy_ev.to_numpy())
        extra = (N.encode_vocab(query.adduct, enc_adduct),
                 N.encode_vocab(query.instrument_type, enc_instr), ce, present,
                 (query.ionization_mode.to_numpy() != "negative").astype(np.float32))

    out = []
    with torch.no_grad():
        for s in range(0, len(query), 128):
            sl = slice(s, min(s + 128, len(query)))
            mz, it, pad = N.collate_spectra(mzs[sl], its[sl], cfg.max_peaks,
                                            precursors=None if extra else prec[sl])
            args = [mz.to(device), it.to(device), pad.to(device)]
            if extra:
                args += [torch.from_numpy(prec[sl]).to(device)]
                args += [torch.from_numpy(e[sl]).to(device) for e in extra]
            out.append(model(*args).float().cpu().numpy())
    return np.concatenate(out), bits

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
                   limit_molecules: int | None = None,
                   method: str = "analog", fpnet_checkpoint=None) -> pd.DataFrame:
    """Read train and test parquet, write `submission.csv`, return the frame.

    `method` picks what ranks the mass window:

    - `analog` — spectral match against the whole library, then structural similarity to
      the matched structures. **Scored 0.080 on the board**, at the random floor, while
      scoring 1.0000 on the visible test (whose spectra are exact copies of training
      spectra) and 0.434 on our validation. The pipeline is correct; the signal does not
      transfer, which is what a library-dependent method does when the library holds no
      useful relative.
    - `fpnet` — a predicted fingerprint compared against each candidate's exact one.
      **Needs no spectral library at all**, so it is the direct test of whether the
      failure above is library dependence or something more basic.
    - `combined` — both, as a weighted sum of per-molecule z-scores.

    Deliberately free of `try/except`: a swallowed failure would write the ethanol
    fallback and report success, which looks exactly like a method that learnt nothing.
    """
    if method not in ("analog", "fpnet", "combined"):
        raise ValueError(f"unknown method {method!r}")
    if method in ("fpnet", "combined") and fpnet_checkpoint is None:
        raise ValueError(f"method {method!r} needs fpnet_checkpoint")
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

    # ---- the reference library, only when a method needs it -------------------------
    # Skipped for `fpnet`: reading 1.15M spectra with their peak arrays costs seconds and
    # gigabytes, and that method never consults a spectral library -- which is the whole
    # reason it is worth submitting.
    ref = None
    if method in ("analog", "combined"):
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

    # ---- diagnostics the board cannot give us ---------------------------------------
    # The 0.080 run proved the pipeline correct and the signal absent, but could not say
    # which. These print into the scoring run's own log, so the next submission answers
    # the question whatever its score turns out to be.
    print(f"DIAG unparseable_adduct={unmapped} "
          f"adducts={sorted(query.adduct.astype(str).unique())[:12]}", flush=True)

    analogues = {}
    if method in ("analog", "combined"):
        analogues = AN.find_analogues(query, ref, top_k=top_k, verbose=True)
        best = np.array([a[0][1] if a else 0.0 for a in analogues.values()])
        # On the visible test this is 1.000 for every molecule (its spectra ARE training
        # spectra); on our validation the median is 0.884. Whatever it is on the hidden
        # test is the number that explains the score.
        print(f"DIAG best_analogue_cosine median={np.median(best):.3f} "
              f"p10={np.percentile(best,10):.3f} p90={np.percentile(best,90):.3f} "
              f"frac_above_0.8={float((best>0.8).mean()):.3f}", flush=True)
        _log(f"analogues found for {len(analogues):,} molecules", t0)

    fp_logits = None
    if method in ("fpnet", "combined"):
        fp_logits, fp_bits = _fpnet_logits(fpnet_checkpoint, query)
        _log(f"FPNet logits {fp_logits.shape} over {len(fp_bits):,} bits", t0)

    # ---- rank -------------------------------------------------------------------------
    q_mol = query.molecule_id.to_numpy(dtype=object)
    window_sizes, empty_windows = [], 0
    rows = []
    for i, mol in enumerate(molecules):
        g = query[query.molecule_id == mol]
        mass = g.neutral_mass.dropna()
        ranked: list[str] = []
        if len(mass):
            lo, hi = D.mass_window(bank_mass, float(mass.median()), ppm)
            keys = list(dict.fromkeys(bank_key[lo:hi]))
            window_sizes.append(len(keys))
            if not keys:
                empty_windows += 1
            cand_fp = X[[row_of[k] for k in keys]] if keys else None
            score = np.zeros(len(keys))

            if keys and method in ("analog", "combined"):
                a = [x for x in analogues.get(mol, []) if x[0] in row_of]
                if a:
                    score = score + _z(AN.propagate(
                        a, np.array(keys, dtype=object), cand_fp=cand_fp,
                        analog_fp=X[[row_of[k] for k, _ in a]], p=p))
            if keys and method in ("fpnet", "combined"):
                spectra_rows = np.flatnonzero(q_mol == mol)
                lg = fp_logits[spectra_rows]
                s_fp = FP.modified_tanimoto_logit(lg, cand_fp[:, fp_bits]).T.max(axis=0)
                score = score + _z(s_fp)

            if keys:
                ranked = [keys[j] for j in np.argsort(-score)]
        else:
            empty_windows += 1

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

    print(f"DIAG window_size median={int(np.median(window_sizes)) if window_sizes else 0} "
          f"empty_windows={empty_windows}/{len(molecules)} method={method}", flush=True)

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
