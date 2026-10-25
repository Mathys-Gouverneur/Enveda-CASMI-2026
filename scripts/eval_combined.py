#!/usr/bin/env python
"""Do our three signals compose, or do they overlap?

    python scripts/eval_combined.py --n-val 300 --family-ceiling 0.5

Three independent views of the same candidate list, each measured alone:

| signal | what it sees | uses the spectral library? |
|---|---|---|
| analogue propagation | structural similarity to the best-matching *other* molecules | **yes** |
| formula consistency | whether a candidate's formula can explain the fragments | no |
| FPNet | agreement between a predicted and an exact fingerprint | no |

Only the first is affected by the family-disjointness ceiling, which filters the library —
so the ceiling changes the baseline this measures against, not the other two signals.
Re-measuring formula "at ceiling 0.5" would return its ceiling-1.0 number unchanged;
what the ceiling changes is whether it still *adds* anything once analogue propagation is
no longer being flattered by a library full of near-twins.

Signals are combined as a weighted sum of per-molecule z-scores. Not a trained ranker:
with three signals a sweep is both interpretable and enough to answer whether they are
complementary, and a ranker is only worth building once they are known to be.
"""
from __future__ import annotations

import argparse
import itertools
import pathlib
import time

import numpy as np
import pandas as pd

from casmi import analog as AN
from casmi import data as D
from casmi import formula as FO
from casmi import metric as M


def z(v: np.ndarray) -> np.ndarray:
    """Per-molecule z-score, so signals on different scales can be added."""
    v = np.asarray(v, dtype=np.float64)
    s = v.std()
    return (v - v.mean()) / s if s > 1e-12 else np.zeros_like(v)


def rr_of(order_keys, true_key: str) -> float:
    seen: list[str] = []
    for k in order_keys:
        if k in seen:
            continue
        seen.append(k)
        if k == true_key:
            return 1.0 / len(seen) if len(seen) <= M.N_SLOTS else 0.0
        if len(seen) >= M.N_SLOTS:
            break
    return 0.0


def fpnet_scores(checkpoint: pathlib.Path, query: pd.DataFrame, bank_fp: np.ndarray,
                 windows: dict, bank_row: dict) -> dict[str, np.ndarray]:
    """Per-candidate FPNet log-likelihood, one vector per molecule."""
    import torch

    from casmi import fingerprint as FP
    from casmi import fpnet as N
    import sys
    sys.path.insert(0, str(D.ROOT / "scripts"))
    from train_fpnet import Conditioning, predict_logits          # noqa: E402

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    bits = ck["bits"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ck.get("v2"):
        cfg = N.FPNetV2Config(**ck["cfg"])
        model = N.FPNetV2(cfg)
        cond = Conditioning.__new__(Conditioning)
        cond.adduct, cond.instrument = ck["adduct_vocab"], ck["instrument_vocab"]
    else:
        cfg = N.FPNetConfig(**ck["cfg"])
        model = N.FPNet(cfg)
        cond = None
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    print(f"  FPNet: {'v2' if ck.get('v2') else 'v1'}, "
          f"best_epoch {ck.get('best_epoch', 'n/a')}, {len(bits):,} bits, {device}")

    logits = predict_logits(model, query, cfg.max_peaks, device, cond)
    Y = bank_fp[:, bits]
    q_mol = query.molecule_id.to_numpy(dtype=object)
    out = {}
    for mol, keys in windows.items():
        rows = np.flatnonzero(q_mol == mol)
        if not len(rows) or not keys:
            out[mol] = np.zeros(len(keys))
            continue
        cand = Y[[bank_row[k] for k in keys]]
        s = FP.modified_tanimoto_logit(logits[rows], cand).T      # (n_spectra, n_cand)
        out[mol] = s.max(axis=0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=str(D.PROCESSED / "splits"))
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--ppm", type=float, default=10.0)
    ap.add_argument("--family-ceiling", type=float, default=0.5)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--p", type=float, default=3.0)
    ap.add_argument("--fpnet", default=str(D.ROOT / "out" / "fpnet_v1" / "fpnet.pt"))
    ap.add_argument("--neighbours", default=str(D.PROCESSED / "neighbours.npz"))
    ap.add_argument("--out", default=str(D.ROOT / "out"))
    args = ap.parse_args()

    sdir, odir = pathlib.Path(args.splits), pathlib.Path(args.out)
    query = pd.read_parquet(sdir / "query.parquet")
    truth = dict(pd.read_csv(sdir / "truth.csv").itertuples(index=False, name=None))
    keep = sorted(truth)[: args.n_val]
    query = query[query.molecule_id.isin(set(keep))].reset_index(drop=True)
    truth = {m: truth[m] for m in keep}
    val_keys = set(truth.values())
    print(f"{len(truth):,} molecules, {len(query):,} spectra, "
          f"family ceiling {args.family_ceiling}")

    bank = D.load_structures()
    zf = np.load(D.PROCESSED / "bank_fingerprints.npz", allow_pickle=False)
    X, ok = zf["X"], zf["ok"]
    bank, X = bank[ok].reset_index(drop=True), X[ok]
    bm = bank.neutral_mass.to_numpy(dtype=float)
    bk = bank.inchikey14.to_numpy(dtype=object)
    bf = bank.molecular_formula.to_numpy(dtype=object)
    row_of = {k: i for i, k in enumerate(bk)}
    key_to_formula = dict(zip(bk, bf))

    # ---- candidate windows, and the formula signal ---------------------------------
    grouped = {m: g for m, g in query.groupby("molecule_id", sort=False)}
    windows, formula_score = {}, {}
    t0 = time.time()
    for mol, g in grouped.items():
        lo, hi = D.mass_window(bm, float(np.median(g.neutral_mass.to_numpy(float))),
                               args.ppm)
        keys = list(dict.fromkeys(bk[lo:hi]))
        windows[mol] = keys
        spectra = list(zip(g.ms2_mzs.to_numpy(), g.ms2_normalized_intensities.to_numpy(),
                           g.precursor_mz.to_numpy(dtype=float)))
        per_formula = FO.score_formulas_for_molecule(
            spectra, list(dict.fromkeys(bf[lo:hi])))
        formula_score[mol] = np.array(
            [per_formula.get(key_to_formula.get(k), 0.0) for k in keys])
    print(f"formula signal in {time.time()-t0:.0f}s")

    # ---- the analogue signal, at the calibrated ceiling -----------------------------
    cols = ["inchikey14", "adduct", "precursor_mz",
            "ms2_mzs", "ms2_normalized_intensities"]
    tims = D.load_train(columns=cols, timstof_only=True)
    ref = tims[~tims.inchikey14.isin(val_keys)]
    ref = ref[ref.inchikey14.isin(row_of)].reset_index(drop=True)

    relatives: dict[str, dict[str, float]] = {}
    if args.family_ceiling < 1.0:
        nz = np.load(args.neighbours, allow_pickle=True)
        vk, bkk = nz["val_keys"], nz["bank_keys"]
        for r, c, t in zip(nz["row"], nz["col"], nz["tanimoto"]):
            if t < 0.999:
                relatives.setdefault(str(vk[r]), {})[str(bkk[c])] = float(t)

    t0 = time.time()
    found = AN.find_analogues(query, ref,
                              top_k=args.top_k * 8 if relatives else args.top_k)
    print(f"analogue search in {time.time()-t0:.0f}s")

    analog_score = {}
    for mol, keys in windows.items():
        rel = relatives.get(truth[mol], {})
        a = [x for x in found.get(mol, [])
             if x[0] in row_of and rel.get(x[0], 0.0) < args.family_ceiling][:args.top_k]
        analog_score[mol] = AN.propagate(
            a, np.array(keys, dtype=object),
            cand_fp=X[[row_of[k] for k in keys]],
            analog_fp=X[[row_of[k] for k, _ in a]] if a else None, p=args.p)

    # ---- the FPNet signal ------------------------------------------------------------
    fp_score = {m: np.zeros(len(k)) for m, k in windows.items()}
    ck = pathlib.Path(args.fpnet)
    if ck.is_file():
        fp_score = fpnet_scores(ck, query, X, windows, row_of)
    else:
        print(f"  (no checkpoint at {ck}; FPNet column will be zero)")

    # ---- combine ---------------------------------------------------------------------
    rng = np.random.default_rng(0)
    combos = {
        "random": (0.0, 0.0, 0.0),
        "analogue": (1.0, 0.0, 0.0),
        "formula": (0.0, 1.0, 0.0),
        "fpnet": (0.0, 0.0, 1.0),
        "analogue + formula": (1.0, 0.3, 0.0),
        "analogue + fpnet": (1.0, 0.0, 0.3),
        "analogue + formula + fpnet": (1.0, 0.3, 0.3),
        "analogue + fpnet (heavy)": (1.0, 0.0, 1.0),
        "analogue + formula + fpnet (heavy)": (1.0, 1.0, 1.0),
    }
    rows = []
    for name, (wa, wf, wp) in combos.items():
        rr = []
        for mol, keys in windows.items():
            if not keys:
                rr.append(0.0)
                continue
            s = (wa * z(analog_score[mol]) + wf * z(formula_score[mol])
                 + wp * z(fp_score[mol]))
            if wa == wf == wp == 0.0:
                s = rng.random(len(keys))
            order = np.array(keys, dtype=object)[np.argsort(-s)]
            rr.append(rr_of(order, truth[mol]))
        rr = np.array(rr)
        rows.append({"signals": name, "mrr": rr.mean(),
                     "top1": float((rr == 1).mean())})
        print(f"  {name:38s} MRR {rr.mean():.4f}   top-1 {(rr==1).mean()*100:5.1f} %")

    df = pd.DataFrame(rows)
    df.to_csv(odir / f"combined_ceiling{args.family_ceiling}.csv", index=False)
    best_single = df[df.signals.isin(["analogue", "formula", "fpnet"])].mrr.max()
    best_combo = df[~df.signals.isin(["random", "analogue", "formula", "fpnet"])].mrr.max()
    print(f"""
{'='*72}
best single signal   {best_single:.4f}
best combination     {best_combo:.4f}   ({best_combo - best_single:+.4f})

A combination that beats every single signal means they see different things and a
ranker is worth building. One that does not means they overlap, and the effort belongs
elsewhere.""")


if __name__ == "__main__":
    main()
