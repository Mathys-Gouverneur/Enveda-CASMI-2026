#!/usr/bin/env python
"""Does analogue propagation beat the floor in the regime the test is actually in?

    python scripts/eval_analog.py --n-val 1000

The `held_out` regime removes every spectrum of the validation molecules, so direct
spectral retrieval scores exactly 0 and ranking the mass window at random scores 0.085.
Analogue propagation is how the public stack scores anyway: match the query against the
whole library, take the best-matching *other* molecules, and rank candidates by structural
similarity to those.

Sweeps the two parameters rather than inheriting them — `top_k` analogues and the
similarity exponent `p`, which the public stack sets to 3 — and compares against:

- the random floor over the same mass window,
- the formula soft ordering already measured at 0.123,
- and the combination, which is the first real test of whether our signals compose.
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import pandas as pd

from casmi import analog as AN
from casmi import data as D
from casmi import formula as FO
from casmi import metric as M


def rr_of(ranked_keys, true_key: str) -> float:
    seen: list[str] = []
    for k in ranked_keys:
        if k in seen:
            continue
        seen.append(k)
        if k == true_key:
            return 1.0 / len(seen) if len(seen) <= M.N_SLOTS else 0.0
        if len(seen) >= M.N_SLOTS:
            break
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=str(D.PROCESSED / "splits"))
    ap.add_argument("--n-val", type=int, default=1000,
                    help="validation molecules to use; the analogue search is the "
                         "expensive part and scales linearly in query spectra")
    ap.add_argument("--ppm", type=float, default=10.0)
    ap.add_argument("--top-k", type=int, nargs="+", default=[20, 80])
    ap.add_argument("--p", type=float, nargs="+", default=[1.0, 3.0, 6.0])
    ap.add_argument("--family-ceiling", type=float, nargs="+", default=[1.0],
                    help="drop analogues whose structural Tanimoto to the true answer is "
                         "at or above this, simulating a library from which the "
                         "molecule's family is absent. 1.0 keeps today's split.")
    ap.add_argument("--neighbours", default=str(D.PROCESSED / "neighbours.npz"))
    ap.add_argument("--out", default=str(D.ROOT / "out"))
    args = ap.parse_args()

    sdir, odir = pathlib.Path(args.splits), pathlib.Path(args.out)
    odir.mkdir(parents=True, exist_ok=True)

    query = pd.read_parquet(sdir / "query.parquet")
    truth = dict(pd.read_csv(sdir / "truth.csv").itertuples(index=False, name=None))
    keep = sorted(truth)[: args.n_val]
    query = query[query.molecule_id.isin(set(keep))].reset_index(drop=True)
    truth = {m: truth[m] for m in keep}
    val_keys = set(truth.values())
    print(f"{len(truth):,} molecules, {len(query):,} query spectra")

    # ---- structure bank and its cached fingerprints -------------------------------
    bank = D.load_structures()
    cache = D.PROCESSED / "bank_fingerprints.npz"
    if not cache.exists():
        raise SystemExit("run scripts/train_fpnet.py once to build the fingerprint cache")
    z = np.load(cache, allow_pickle=False)
    X, ok = z["X"], z["ok"]
    bank, X = bank[ok].reset_index(drop=True), X[ok]
    bm = bank.neutral_mass.to_numpy(dtype=float)
    bk = bank.inchikey14.to_numpy(dtype=object)
    bf = bank.molecular_formula.to_numpy(dtype=object)
    row_of = {k: i for i, k in enumerate(bk)}
    key_to_formula = dict(zip(bk, bf))
    print(f"bank: {len(bank):,} structures, fingerprints {X.shape}")

    # ---- held_out reference library ------------------------------------------------
    cols = ["ingest_lib", "inchikey14", "adduct", "precursor_mz",
            "ms2_mzs", "ms2_normalized_intensities"]
    print("reading the held_out reference library ...")
    tims = D.load_train(columns=cols, timstof_only=True)
    ref = tims[~tims.inchikey14.isin(val_keys)]
    ref = ref[ref.inchikey14.isin(row_of)].reset_index(drop=True)
    print(f"  {len(ref):,} reference spectra over {ref.inchikey14.nunique():,} molecules"
          f" — none of them a validation molecule")

    # Per-molecule relative sets, for the family-disjoint ceilings. Filtering analogues
    # by their similarity to the true answer uses the label -- legitimate, because it
    # constructs a harder *split*, not a method: it simulates a reference library from
    # which this molecule's chemical family is absent. No real system could do this, and
    # nothing downstream of the split sees it.
    relatives: dict[str, dict[str, float]] = {}
    if max(args.family_ceiling) < 1.0 or len(args.family_ceiling) > 1:
        nz = np.load(args.neighbours, allow_pickle=True)
        vk, bkk = nz["val_keys"], nz["bank_keys"]
        for r, c, t in zip(nz["row"], nz["col"], nz["tanimoto"]):
            if t < 0.999:
                relatives.setdefault(str(vk[r]), {})[str(bkk[c])] = float(t)
        print(f"neighbour table: {len(relatives):,} molecules with relatives "
              f">= {float(nz['min_tanimoto']):.2f}")

    t0 = time.time()
    # Search wide: the ceilings discard the closest analogues, so there must be enough
    # left underneath to still fill top_k.
    search_k = min(max(args.top_k) * 8, 400) if relatives else max(args.top_k)
    analogues = AN.find_analogues(query, ref, top_k=search_k, verbose=True)
    print(f"analogue search: {time.time()-t0:.0f}s")

    # sanity: the true answer must never appear among its own analogues
    leaked = sum(truth[m] in {k for k, _ in a} for m, a in analogues.items())
    print(f"leak check — true key among its own analogues: {leaked} (must be 0)")
    assert leaked == 0

    # ---- score ---------------------------------------------------------------------
    rng = np.random.default_rng(0)
    rows = []
    per_mol_best: dict[str, float] = {}

    grouped = {m: g for m, g in query.groupby("molecule_id", sort=False)}
    windows, formula_order = {}, {}
    for mol, g in grouped.items():
        mass = float(np.median(g.neutral_mass.to_numpy(dtype=float)))
        lo, hi = D.mass_window(bm, mass, args.ppm)
        keys = list(dict.fromkeys(bk[lo:hi]))
        windows[mol] = keys
        cands = list(dict.fromkeys(bf[lo:hi]))
        spectra = list(zip(g.ms2_mzs.to_numpy(), g.ms2_normalized_intensities.to_numpy(),
                           g.precursor_mz.to_numpy(dtype=float)))
        sc = FO.score_formulas_for_molecule(spectra, cands)
        formula_order[mol] = {f: i for i, f in enumerate(sorted(cands, key=lambda f: -sc[f]))}

    # baselines
    base_rand = np.mean([
        rr_of(np.array(windows[m], dtype=object)[rng.permutation(len(windows[m]))], truth[m])
        for m in truth])
    base_formula = np.mean([
        rr_of(sorted(windows[m], key=lambda k: (formula_order[m].get(key_to_formula.get(k), 99),
                                                rng.random())), truth[m])
        for m in truth])
    print(f"\nbaselines on these {len(truth)} molecules:")
    print(f"  random over the mass window   {base_rand:.4f}")
    print(f"  formula soft ordering         {base_formula:.4f}")

    for ceil in args.family_ceiling:
      for top_k in args.top_k:
        for p in args.p:
            rr, rr_combo, best_left = [], [], []
            for mol in truth:
                keys = windows[mol]
                if not keys:
                    rr.append(0.0); rr_combo.append(0.0)
                    best_left.append(0.0); continue
                cand_fp = X[[row_of[k] for k in keys]]
                # Analogue keys always resolve: the reference library was filtered to
                # structures present in the bank before the search.
                rel = relatives.get(truth[mol], {})
                avail = [x for x in analogues[mol]
                         if x[0] in row_of and rel.get(x[0], 0.0) < ceil]
                a = avail[:top_k]
                bl = max((rel.get(k, 0.0) for k, _ in a), default=0.0)
                best_left.append(bl)
                an_fp = X[[row_of[k] for k, _ in a]] if a else None
                s = AN.propagate(a, np.array(keys, dtype=object),
                                 cand_fp=cand_fp, analog_fp=an_fp, p=p)
                order = np.argsort(-s)
                rr.append(rr_of(np.array(keys, dtype=object)[order], truth[mol]))
                # combination: formula group first, analogue score within
                fo = formula_order[mol]
                combo = sorted(range(len(keys)),
                               key=lambda i: (fo.get(key_to_formula.get(keys[i]), 99), -s[i]))
                rr_combo.append(rr_of(np.array(keys, dtype=object)[combo], truth[mol]))
            mrr, mrr_c = float(np.mean(rr)), float(np.mean(rr_combo))
            rows.append({"family_ceiling": ceil, "top_k": top_k, "p": p,
                         "mrr_analog": mrr, "mrr_analog_plus_formula": mrr_c,
                         "median_best_relative": float(np.median(best_left))})
            print(f"  ceiling {ceil:4.2f}  top_k {top_k:3d}  p {p:4.1f}  ->  "
                  f"analogue {mrr:.4f}   + formula {mrr_c:.4f}   "
                  f"(best relative left, median {np.median(best_left):.3f})")
            if mrr_c > max(per_mol_best.values(), default=-1):
                per_mol_best = {"mrr": mrr_c}

    df = pd.DataFrame(rows)
    df.to_csv(odir / "analog_eval.csv", index=False)
    best = df.sort_values("mrr_analog_plus_formula", ascending=False).iloc[0]
    print(f"""
{'='*72}
random floor                 {base_rand:.4f}
formula soft ordering        {base_formula:.4f}
analogue propagation         {best.mrr_analog:.4f}   (top_k {best.top_k:.0f}, p {best.p:g})
analogue + formula           {best.mrr_analog_plus_formula:.4f}

This is the regime where direct spectral retrieval scores 0.000, so every point here
comes from evidence reaching a molecule the library has never seen. Whether formula and
analogue signals compose or overlap is the question the last two rows answer.""")
    print(f"\nwritten to {odir}")


if __name__ == "__main__":
    main()
