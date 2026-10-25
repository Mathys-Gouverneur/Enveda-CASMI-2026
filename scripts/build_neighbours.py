#!/usr/bin/env python
"""Structural neighbours of every validation molecule, so splits can be family-disjoint.

    python scripts/build_neighbours.py --min-tanimoto 0.35

`docs/measurement.md` §9 established the problem: splitting by molecule leaves whole
chemical families intact, and analogue propagation scores 0.714 on such a split against
0.452 when no relative above Tanimoto 0.5 is available — the latter matching the public
leaderboard. Performance is almost entirely a function of how close the nearest available
relative is, so a split that ignores structural proximity measures the wrong thing.

Rather than commit to one threshold, this computes the **neighbour table once**: every
(validation molecule, bank structure) pair with Tanimoto at or above `--min-tanimoto`.
Any downstream evaluation can then exclude relatives above whatever ceiling it wants and
report MRR *as a function* of that ceiling, which is strictly more informative than a
single split and needs no recomputation.

The full comparison is 4,000 × 275,809 Tanimoto coefficients over 6,311-bit fingerprints.
As a blocked float32 matmul that is a few 10^12 FLOPs, which numpy does on this box in
about a minute; the GPU is not needed and is usually busy training.
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import pandas as pd

from casmi import data as D


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=str(D.PROCESSED / "splits"))
    ap.add_argument("--min-tanimoto", type=float, default=0.35,
                    help="keep pairs at or above this; below it nothing downstream cares")
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--out", default=str(D.PROCESSED / "neighbours.npz"))
    args = ap.parse_args()

    truth = pd.read_csv(pathlib.Path(args.splits) / "truth.csv")
    bank = D.load_structures()
    cache = D.PROCESSED / "bank_fingerprints.npz"
    if not cache.exists():
        raise SystemExit("fingerprint cache missing; run scripts/train_fpnet.py once")
    z = np.load(cache, allow_pickle=False)
    X, ok = z["X"], z["ok"]
    bank, X = bank[ok].reset_index(drop=True), X[ok]
    row_of = {k: i for i, k in enumerate(bank.inchikey14)}

    val_keys = [k for k in truth.inchikey14 if k in row_of]
    print(f"{len(val_keys):,} validation structures against {len(bank):,} bank structures")

    Xb = X.astype(np.float32)
    nb = Xb.sum(axis=1)
    Xv = Xb[[row_of[k] for k in val_keys]]
    nv = Xv.sum(axis=1)

    rows, cols, vals = [], [], []
    t0 = time.time()
    for s in range(0, len(val_keys), args.block):
        e = min(s + args.block, len(val_keys))
        inter = Xv[s:e] @ Xb.T                                    # (block, n_bank)
        tan = inter / (nv[s:e, None] + nb[None, :] - inter + 1e-9)
        r, c = np.nonzero(tan >= args.min_tanimoto)
        rows.append(r + s)
        cols.append(c)
        vals.append(tan[r, c].astype(np.float32))
        if s and (s // args.block) % 4 == 0:
            print(f"  {e:,}/{len(val_keys):,}  {time.time()-t0:.0f}s  "
                  f"{sum(len(v) for v in vals):,} pairs so far", flush=True)

    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    print(f"\n{len(vals):,} pairs at Tanimoto >= {args.min_tanimoto} "
          f"in {time.time()-t0:.0f}s")

    np.savez_compressed(args.out,
                        val_keys=np.array(val_keys, dtype=object),
                        bank_keys=bank.inchikey14.to_numpy(dtype=object),
                        row=rows, col=cols, tanimoto=vals,
                        min_tanimoto=args.min_tanimoto)
    print(f"written to {args.out}")

    # What each ceiling does to the **reference spectral library**. The candidate bank is
    # never filtered: the true answer has to stay findable, and the real test's candidate
    # pool is not family-filtered either. Exclusion is also *per molecule* downstream —
    # when scoring molecule m we drop m's own relatives, not everyone's — so these counts
    # are the per-molecule average, not a single global removal.
    print("\n=== what each family-disjoint ceiling removes, per validation molecule ===")
    print(f"  {'ceiling':>8}  {'relatives dropped (median)':>26}  "
          f"{'best relative left (median)':>28}")
    order = np.argsort(rows, kind="stable")
    r_sorted, v_sorted = rows[order], vals[order]
    starts = np.searchsorted(r_sorted, np.arange(len(val_keys)))
    ends = np.searchsorted(r_sorted, np.arange(len(val_keys)), side="right")
    for ceil in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4):
        dropped, best_left = [], []
        for i in range(len(val_keys)):
            t = v_sorted[starts[i]:ends[i]]
            t = t[t < 0.999]                       # the molecule itself is never a relative
            dropped.append(int((t >= ceil).sum()))
            left = t[t < ceil]
            best_left.append(float(left.max()) if left.size else 0.0)
        print(f"  {ceil:>8.2f}  {np.median(dropped):>26.0f}  {np.median(best_left):>28.3f}")
    print("\n  The last column is the knob that matters: it is the quantity MRR tracks."
          "\n  Our molecule-level split sits at ceiling 1.00; the public leaderboard's"
          "\n  0.433 corresponds to roughly the 0.45-0.50 row.")


if __name__ == "__main__":
    main()
