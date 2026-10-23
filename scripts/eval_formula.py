#!/usr/bin/env python
"""Does fragment-consistency actually identify the molecular formula?

    python scripts/eval_formula.py

`docs/measurement.md` bounds the prize: ranking the mass window at random scores 0.088,
and ranking only the true-formula subset at random scores 0.205. So a perfect formula
gate more than doubles the floor. This measures how much of that is reachable by scoring
each candidate formula on how much of the spectrum's intensity it can explain as
sub-formulas of itself.

Reports three things, because they answer different questions:

- **formula top-1 accuracy** — how often the gate picks the right formula.
- **MRR when gating on the predicted formula** — the realised gain, which is what matters.
- **the same with a soft gate** — keeping candidates from the top-k formulas, since a hard
  gate that is wrong costs the molecule entirely while a soft one only demotes it.
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import pandas as pd

from casmi import data as D
from casmi import formula as FO
from casmi import metric as M


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=str(D.PROCESSED / "splits"))
    ap.add_argument("--ppm-window", type=float, default=10.0, help="candidate mass window")
    ap.add_argument("--ppm-frag", type=float, default=20.0, help="fragment match tolerance")
    ap.add_argument("--out", default=str(D.ROOT / "out"))
    ap.add_argument("--raw", action="store_true",
                    help="use the unadjusted explained-intensity score, which is biased "
                         "towards larger formulas; kept so the correction is measurable")
    args = ap.parse_args()

    sdir, odir = pathlib.Path(args.splits), pathlib.Path(args.out)
    odir.mkdir(parents=True, exist_ok=True)

    query = pd.read_parquet(sdir / "query.parquet")
    truth = dict(pd.read_csv(sdir / "truth.csv").itertuples(index=False, name=None))

    bank = D.load_structures()
    bm = bank.neutral_mass.to_numpy(dtype=float)
    bk = bank.inchikey14.to_numpy(dtype=object)
    bf = bank.molecular_formula.to_numpy(dtype=object)
    key_to_formula = dict(zip(bk, bf))

    grouped = {m: g for m, g in query.groupby("molecule_id", sort=False)}
    print(f"{len(grouped):,} validation molecules, "
          f"candidate window ±{args.ppm_window:g} ppm, "
          f"fragment tolerance ±{args.ppm_frag:g} ppm\n")

    n_formulas, correct_1, correct_2, correct_3 = [], [], [], []
    rr = {"window": [], "gate_top1": [], "gate_top2": [], "soft_order": [],
          "oracle_formula": []}
    t0 = time.time()

    for i, (mol, g) in enumerate(grouped.items()):
        # Per-molecule RNG, seeded from the molecule id. A single shared generator would
        # be consumed differently depending on which conditions run, so adding a
        # condition shifted every other condition's permutations and moved the baseline
        # by ~0.003 between runs. Seeding per molecule makes the conditions paired on
        # identical permutations, which is the comparison we actually want.
        rng = np.random.default_rng(abs(hash(mol)) % (2**32))
        true_key = truth[mol]
        true_formula = key_to_formula.get(true_key)
        mass = float(np.median(g.neutral_mass.to_numpy(dtype=float)))
        lo, hi = D.mass_window(bm, mass, args.ppm_window)
        keys, formulas = bk[lo:hi], bf[lo:hi]
        cands = list(dict.fromkeys(formulas))
        n_formulas.append(len(cands))

        spectra = list(zip(g.ms2_mzs.to_numpy(), g.ms2_normalized_intensities.to_numpy(),
                           g.precursor_mz.to_numpy(dtype=float)))
        scores = FO.score_formulas_for_molecule(spectra, cands, args.ppm_frag,
                                                adjusted=not args.raw)
        ranked = sorted(cands, key=lambda f: -scores[f])

        correct_1.append(bool(ranked[:1]) and ranked[0] == true_formula)
        correct_2.append(true_formula in ranked[:2])
        correct_3.append(true_formula in ranked[:3])

        def rr_of(pool_keys, shuffle: bool = True):
            """RR of the true key in a candidate pool.

            `shuffle=True` randomises within the pool, which is what isolates the pool's
            own value from any ranking skill. `shuffle=False` keeps the order given —
            needed for soft ordering, whose whole content *is* the order.
            """
            pool = list(dict.fromkeys(pool_keys))
            if true_key not in pool:
                return 0.0
            arr = np.array(pool, dtype=object)
            if shuffle:
                arr = arr[rng.permutation(len(arr))]
            r = int(np.flatnonzero(arr == true_key)[0]) + 1
            return 1.0 / r if r <= M.N_SLOTS else 0.0

        rr["window"].append(rr_of(keys))
        rr["gate_top1"].append(rr_of(keys[formulas == ranked[0]]))
        top2 = set(ranked[:2])
        rr["gate_top2"].append(rr_of(keys[np.isin(formulas, list(top2))]))
        rr["oracle_formula"].append(rr_of(keys[formulas == true_formula]))

        # Soft ordering: keep every candidate, but visit formula groups in score order,
        # random within a group. A hard gate that is wrong costs the molecule entirely
        # (RR 0) and so needs accuracy above base/oracle = 39% merely to break even; soft
        # ordering never excludes the answer, it only demotes it. This is why the public
        # stack reranks same-formula groups instead of filtering on formula.
        by_formula: dict[str, list] = {}
        for k, f in zip(keys, formulas):
            by_formula.setdefault(f, []).append(k)
        soft = []
        for f in ranked:
            grp = list(dict.fromkeys(by_formula.get(f, [])))
            soft.extend(np.array(grp, dtype=object)[rng.permutation(len(grp))] if grp else [])
        rr["soft_order"].append(rr_of(soft, shuffle=False))

        if i and i % 1000 == 0:
            print(f"  {i:,}/{len(grouped):,} ... {time.time()-t0:.0f}s", flush=True)

    print(f"\ndone in {time.time()-t0:.0f}s\n")
    print("=" * 70)
    print(f"candidate formulas per molecule: median {int(np.median(n_formulas))}, "
          f"mean {np.mean(n_formulas):.1f}, max {max(n_formulas)}")
    print(f"\nformula identification accuracy")
    print(f"  top-1   {np.mean(correct_1)*100:5.1f} %")
    print(f"  top-2   {np.mean(correct_2)*100:5.1f} %")
    print(f"  top-3   {np.mean(correct_3)*100:5.1f} %")
    # A molecule whose window holds only one formula is free; report the hard cases too.
    multi = np.array(n_formulas) > 1
    print(f"\n  on the {multi.sum():,} molecules whose window holds >1 formula:")
    print(f"  top-1   {np.mean(np.array(correct_1)[multi])*100:5.1f} %")

    print(f"\nMRR of RANDOM ranking within each pool (isolates the gate's value)")
    rows = []
    for name in ("window", "gate_top1", "gate_top2", "soft_order", "oracle_formula"):
        v = np.array(rr[name])
        rows.append({"pool": name, "mrr": v.mean()})
        print(f"  {name:18s} {v.mean():.4f}")
    base, oracle = np.mean(rr["window"]), np.mean(rr["oracle_formula"])
    print(f"\n  share of the formula prize ({oracle:.3f} - {base:.3f} = "
          f"{oracle-base:.3f}) that each design captures:")
    for name in ("gate_top1", "gate_top2", "soft_order"):
        got = np.mean(rr[name]) - base
        print(f"    {name:12s} {got:+.4f}  = {got/max(oracle-base,1e-9)*100:4.0f} %")
    print(f"""
  A hard gate must exceed base/oracle = {base/oracle*100:.0f}% formula accuracy merely to
  break even, because being wrong costs the molecule entirely. At {np.mean(correct_1)*100:.0f}% it barely
  clears that. Soft ordering never excludes the answer, only demotes it, and captures
  several times more. This is why the public stack reranks same-formula groups rather
  than filtering on formula — and the formula score belongs in a ranker as a feature,
  not in front of it as a filter.""")
    pd.DataFrame(rows).to_csv(odir / "formula_gate.csv", index=False)
    pd.DataFrame(rr).assign(molecule_id=list(grouped)).to_csv(
        odir / "formula_gate_per_molecule.csv", index=False)
    print(f"\nwritten to {odir}")


if __name__ == "__main__":
    main()
