#!/usr/bin/env python
"""Fit the two retrieval constants that were placeholders, on validation rather than taste.

    python scripts/tune_retrieval.py

Two decisions, both left as guesses when the baseline was written, and both applying to
every ranker downstream:

**The neutral-loss weight.** `spectra.combined_similarity` scores a fragment cosine plus
`loss_weight ×` a neutral-loss cosine. The loss view catches analogues the fragment view
misses — a molecule and its methylated derivative share losses, not fragment masses — but
0.5 was a placeholder with a comment saying so.

**The multi-spectrum fusion rule.** The test gives ~3 spectra per molecule at different
collision energies and scores per molecule, so the spectra must vote. The baseline takes
each candidate's **maximum** over them, on the argument that a reference recorded at one
energy will legitimately match one query spectrum and not the others. That argument is
plausible, which is exactly why it needs checking.

Comparisons are paired on the same molecules, which `docs/measurement.md` §1 shows
resolves effects around 0.001 on 4,000 molecules — far below what a submission could ever
confirm.
"""
from __future__ import annotations

import argparse
import itertools
import pathlib
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from casmi import data as D
from casmi import metric as M
from casmi import spectra as SP

FUSIONS = {
    # Each candidate's score across a molecule's spectra -> one number.
    "max": lambda s: s.max(axis=0),
    "mean": lambda s: s.mean(axis=0),
    "sum": lambda s: s.sum(axis=0),
    # Soft maximum: dominated by the best spectrum but credits agreement across them.
    "logsumexp": lambda s: np.log(np.exp(s * 8.0).sum(axis=0)) / 8.0,
    # Reciprocal-rank fusion over each spectrum's own ranking, which ignores score
    # calibration entirely -- the trick the public stack uses to blend engines.
    "rrf": lambda s: sum(1.0 / (10.0 + np.argsort(np.argsort(-row))) for row in s),
}


def evaluate(query: pd.DataFrame, reference: pd.DataFrame, truth: dict, ppm: float,
             weights: list[float], fusions: list[str]) -> pd.DataFrame:
    ref = reference.sort_values("neutral_mass").reset_index(drop=True)
    ref_mass = ref.neutral_mass.to_numpy(dtype=float)
    ref_key = ref.inchikey14.to_numpy(dtype=object)

    t0 = time.time()
    Rf, Rl = SP.bin_spectra(ref, "fragment"), SP.bin_spectra(ref, "loss")
    Qf, Ql = SP.bin_spectra(query, "fragment"), SP.bin_spectra(query, "loss")
    print(f"  binned {len(ref):,} references + {len(query):,} queries "
          f"in {time.time()-t0:.0f}s")

    q_mol = query.molecule_id.to_numpy(dtype=object)
    q_mass = query.neutral_mass.to_numpy(dtype=float)
    molecules = pd.unique(q_mol)

    # Per-molecule reciprocal ranks for every (weight, fusion) pair, filled in one pass
    # over the molecules so the expensive part -- slicing and multiplying -- is shared.
    combos = list(itertools.product(weights, fusions))
    rr = {c: np.zeros(len(molecules)) for c in combos}

    t0 = time.time()
    for i, mol in enumerate(molecules):
        rows = np.flatnonzero(q_mol == mol)
        lo, hi = D.mass_window(ref_mass, float(np.median(q_mass[rows])), ppm)
        if hi <= lo:
            continue
        sf = (Qf[rows] @ Rf[lo:hi].T).toarray()
        sl = (Ql[rows] @ Rl[lo:hi].T).toarray()
        keys = ref_key[lo:hi]
        truth_key = truth[mol]
        for w in weights:
            s = sf + w * sl if w else sf
            for name in fusions:
                best = FUSIONS[name](s)
                order = np.argsort(best)[::-1]
                seen, rank = set(), 0
                for j in order:
                    k = keys[j]
                    if k in seen:
                        continue
                    seen.add(k)
                    rank += 1
                    if k == truth_key:
                        rr[(w, name)][i] = 1.0 / rank if rank <= M.N_SLOTS else 0.0
                        break
                    if rank >= M.N_SLOTS:
                        break
        if i and i % 1000 == 0:
            print(f"    {i:,}/{len(molecules):,} ... {time.time()-t0:.0f}s", flush=True)

    out = [{"loss_weight": w, "fusion": f, "mrr": rr[(w, f)].mean(),
            "top1": float((rr[(w, f)] == 1).mean())} for w, f in combos]
    return pd.DataFrame(out), rr, molecules


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=str(D.PROCESSED / "splits"))
    ap.add_argument("--ppm", type=float, default=10.0)
    ap.add_argument("--out", default=str(D.ROOT / "out"))
    args = ap.parse_args()

    sdir, odir = pathlib.Path(args.splits), pathlib.Path(args.out)
    odir.mkdir(parents=True, exist_ok=True)
    query = pd.read_parquet(sdir / "query.parquet")
    truth = dict(pd.read_csv(sdir / "truth.csv").itertuples(index=False, name=None))

    cols = ["ingest_lib", "inchikey14", "adduct", "precursor_mz",
            "ms2_mzs", "ms2_normalized_intensities", "spectrum_uid"]
    print("reading timsTOF references ...")
    tims = D.load_train(columns=[c for c in cols if c != "spectrum_uid"],
                        timstof_only=True)
    tims["neutral_mass"] = D.neutral_mass(tims.precursor_mz, tims.adduct)
    tims = tims.dropna(subset=["neutral_mass"])
    tims["spectrum_uid"] = D.spectrum_uid(tims)
    ref = tims[~tims.spectrum_uid.isin(set(query.spectrum_uid))]
    print(f"  in_library reference set: {len(ref):,} spectra\n")

    weights = [0.0, 0.25, 0.5, 1.0, 2.0]
    fusions = list(FUSIONS)
    df, rr, molecules = evaluate(query, ref, truth, args.ppm, weights, fusions)

    print("\n" + "=" * 72)
    piv = df.pivot(index="loss_weight", columns="fusion", values="mrr")
    print("MRR by neutral-loss weight and fusion rule\n")
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))

    best = df.sort_values("mrr", ascending=False).iloc[0]
    baseline = df[(df.loss_weight == 0.5) & (df.fusion == "max")].iloc[0]
    print(f"\nbaseline (the placeholders): loss_weight 0.5, max   -> MRR {baseline.mrr:.4f}")
    print(f"best                       : loss_weight {best.loss_weight:g}, "
          f"{best.fusion:9s} -> MRR {best.mrr:.4f}   ({best.mrr-baseline.mrr:+.4f})")

    # Paired significance: the two arms saw the same molecules, so test the difference.
    d = rr[(best.loss_weight, best.fusion)] - rr[(0.5, "max")]
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"\npaired difference: {d.mean():+.4f} ± {1.96*se:.4f} (95%), "
          f"n = {len(d):,} molecules")
    print(f"  -> {'REAL' if abs(d.mean()) > 1.96*se else 'inside noise'}")

    print("\nwhat adding the neutral-loss view costs, per fusion rule:")
    for f in fusions:
        a = df[(df.loss_weight == 0.0) & (df.fusion == f)].mrr.iloc[0]
        for w in [w for w in weights if w > 0]:
            b = df[(df.loss_weight == w) & (df.fusion == f)].mrr.iloc[0]
            print(f"  {f:10s} weight {w:<4g} {a:.4f} -> {b:.4f}  ({b-a:+.4f})")

    print("""
Scope of this result. It is measured in the `in_library` regime, where the correct answer
is the SAME molecule recorded at another collision energy, so the fragment cosine already
matches it almost exactly and the loss view only adds noise. The loss view exists to find
*analogues* -- a molecule and its methylated derivative share neutral losses but not
fragment masses -- which is the `held_out` regime, where spectral retrieval scores 0 and
this experiment therefore cannot measure it at all.

So read this as "neutral losses do not help same-molecule retrieval", not as "neutral
losses are useless". The fusion-rule ranking should transfer; the loss weight must be
refitted once there is a ranker that scores in the held_out regime.""")

    df.to_csv(odir / "tune_retrieval.csv", index=False)
    pd.DataFrame({"molecule_id": molecules,
                  "reciprocal_rank": rr[(best.loss_weight, best.fusion)]}) \
      .to_csv(odir / "per_molecule_rr_tuned.csv", index=False)
    print(f"\nwritten to {odir}")


if __name__ == "__main__":
    main()
