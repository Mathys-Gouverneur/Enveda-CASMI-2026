#!/usr/bin/env python
"""The two numbers that decide what to build next.

    python scripts/eval_baseline.py

**1. Pool coverage — the ceiling.** No ranker can recover a structure that is not in the
pool, so coverage bounds every method that follows. The naive form of this measurement is
a trap, and the script reports the diagnostic instead:

Validation molecules are drawn from train.parquet, so train structures contain them by
construction (100%, a control) — while `COCONUT` and `ChEBI+LipidMaps` as published on
Kaggle have been *deduplicated against train*, giving them 0% by construction too. Both
numbers are artefacts of how the banks were built, not facts about chemistry.

The question that does have an answer: **how much of each chemical space do public
structure databases cover at all?** Measured against PubChemLite, the one external bank
here that was not filtered against train, the answer separates sharply — and it decides
whether a bigger candidate pool is a lever or a liability.

**2. Mass-gated spectral retrieval — the floor.** The simplest thing that works: keep
reference spectra within ±ppm of the query's neutral mass, score them by binned cosine,
pool a molecule's spectra by taking each candidate's best score. Reported in both
regimes, because they bracket the real task:

- `in_library`  — other spectra of the same molecule remain in the reference library.
- `held_out`    — no spectrum of a validation molecule survives, so only analogues are
                  left and spectral matching has to generalise across structures.

Writes per-molecule reciprocal ranks to out/, which `noise_floor.py --scores` then turns
into a resolution figure for this actual score level rather than a simulated one.
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from casmi import data as D
from casmi import metric as M
from casmi import spectra as SP

CANDIDATES = pathlib.Path("/data/mgr/enveda-casmi/candidates")

#: Candidate banks on disk, with the licence each is published under. The licence column
#: is not bookkeeping: the hosts have ruled that external data restricting commercial use
#: can get a solution removed from the leaderboard, so an NC bank is usable for local
#: measurement and must never be attached to a submission. Upstream COCONUT and ChEBI are
#: themselves CC BY — the NC label belongs to these Kaggle repackagings, which is why
#: rebuilding them from upstream is on the plan.
BANKS = {
    "PubChemLite": ("franciscoangulo-casmi26-pubchemlite-cc0-candidate-fingerprints",
                    "pubchemlite", "CC0-1.0", True),
    "COCONUT": ("prvsiyan-coconut-casmi26-candidates", "coco", "CC-BY-NC-SA-4.0", False),
    "ChEBI+LipidMaps": ("prvsiyan-chebi-lipidmaps-casmi26", "bio", "CC-BY-NC-SA-4.0", False),
}


def load_bank(folder: str, stem: str) -> tuple[np.ndarray, np.ndarray] | None:
    d = CANDIDATES / folder
    mass_f, meta_f = d / f"{stem}_mass.npy", d / f"{stem}_meta.pkl"
    if not (mass_f.exists() and meta_f.exists()):
        return None
    mass = np.load(mass_f, allow_pickle=False)
    # Safe to unpickle: inspected with pickletools, these carry only the standard numpy
    # array-reconstruction triple (_reconstruct / ndarray / dtype) and no other imports.
    keys = np.asarray(pickle.load(open(meta_f, "rb"))["keys"], dtype=object)
    order = np.argsort(mass, kind="stable")
    return mass[order], keys[order]


def recall_in_window(bank_mass: np.ndarray, bank_keys: np.ndarray,
                     query_mass: np.ndarray, truth_keys: np.ndarray,
                     ppm: float) -> dict:
    hit, sizes = np.zeros(len(query_mass), dtype=bool), np.zeros(len(query_mass), dtype=int)
    for i, (m, k) in enumerate(zip(query_mass, truth_keys)):
        lo, hi = D.mass_window(bank_mass, m, ppm)
        sizes[i] = hi - lo
        if hi > lo:
            hit[i] = k in set(bank_keys[lo:hi])
    return {"recall": float(hit.mean()), "median_window": int(np.median(sizes)),
            "mean_window": float(sizes.mean()), "max_window": int(sizes.max())}


def retrieve(query: pd.DataFrame, reference: pd.DataFrame, ppm: float,
             loss_weight: float, n_keep: int = M.N_SLOTS) -> dict[str, list[str]]:
    """Mass-gated cosine retrieval, one ranked candidate list per query molecule."""
    ref = reference.sort_values("neutral_mass").reset_index(drop=True)
    ref_mass = ref.neutral_mass.to_numpy(dtype=float)
    ref_key = ref.inchikey14.to_numpy(dtype=object)

    t0 = time.time()
    Rf = SP.bin_spectra(ref, "fragment")
    Rl = SP.bin_spectra(ref, "loss") if loss_weight else None
    print(f"    binned {len(ref):,} reference spectra in {time.time()-t0:.0f}s "
          f"({Rf.nnz/1e6:.0f}M nonzeros)")

    Qf = SP.bin_spectra(query, "fragment")
    Ql = SP.bin_spectra(query, "loss") if loss_weight else None
    q_mol = query.molecule_id.to_numpy(dtype=object)
    q_mass = query.neutral_mass.to_numpy(dtype=float)

    out: dict[str, list[str]] = {}
    t0 = time.time()
    for mol in pd.unique(q_mol):
        rows = np.flatnonzero(q_mol == mol)
        lo, hi = D.mass_window(ref_mass, float(np.median(q_mass[rows])), ppm)
        if hi <= lo:
            out[mol] = []
            continue
        s = (Qf[rows] @ Rf[lo:hi].T).toarray()
        if loss_weight:
            s = s + loss_weight * (Ql[rows] @ Rl[lo:hi].T).toarray()
        best = s.max(axis=0)                       # a molecule's spectra vote by maximum
        ranked, seen = [], set()
        for j in np.argsort(best)[::-1]:
            k = ref_key[lo + j]
            if k in seen:
                continue
            seen.add(k)
            ranked.append(k)
            if len(ranked) == n_keep:
                break
        out[mol] = ranked
    print(f"    retrieved {len(out):,} molecules in {time.time()-t0:.0f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=str(D.PROCESSED / "splits"))
    ap.add_argument("--ppm", type=float, default=10.0)
    ap.add_argument("--loss-weight", type=float, default=0.5)
    ap.add_argument("--out", default=str(D.ROOT / "out"))
    args = ap.parse_args()

    sdir, odir = pathlib.Path(args.splits), pathlib.Path(args.out)
    odir.mkdir(parents=True, exist_ok=True)

    query = pd.read_parquet(sdir / "query.parquet")
    truth = dict(pd.read_csv(sdir / "truth.csv").itertuples(index=False, name=None))
    val_keys = set(truth.values())
    print(f"validation: {len(truth):,} molecules, {len(query):,} query spectra\n")

    per_mol = query.drop_duplicates("molecule_id").set_index("molecule_id")
    q_mass = per_mol.neutral_mass.to_numpy(dtype=float)
    q_truth = np.array([truth[m] for m in per_mol.index], dtype=object)

    # --------------------------------------------------------------- 1. pool coverage
    print("=" * 78)
    print("1. POOL COVERAGE — can a candidate bank hold the answer at all?")
    print("=" * 78)

    # (a) The diagnostic that actually answers something: how much of each chemical
    #     space do public structure databases cover? Only PubChemLite can be used --
    #     the other two banks were deduplicated against train by their publishers.
    meta_libs = D.load_train(columns=["ingest_lib", "inchikey14"])
    is_env = meta_libs.ingest_lib.str.startswith("enveda")
    env_mols = set(meta_libs[is_env].inchikey14)
    pub_mols = set(meta_libs[~is_env].inchikey14)

    pcl = load_bank(*BANKS["PubChemLite"][:2])
    if pcl is not None:
        pcl_keys = set(pcl[1])
        print(f"\n  Coverage by PubChemLite ({len(pcl_keys):,} structures, CC0, the one")
        print(f"  external bank here not filtered against train):\n")
        print(f"    {'chemical space':<42} {'molecules':>10} {'covered':>16}")
        for name, s in [("Enveda / timsTOF  (the test's provenance)", env_mols),
                        ("public spectral libraries", pub_mols),
                        ("Enveda only, absent from public libs", env_mols - pub_mols),
                        ("shared Enveda + public", env_mols & pub_mols)]:
            n = len(s & pcl_keys)
            print(f"    {name:<42} {len(s):>10,} {n:>8,} ({n/len(s)*100:>4.1f} %)")
        print("\n  The test's own chemical space is the one public databases do not have.")
        print("  So a molecule absent from train's structures is very likely unfindable,")
        print("  the test molecules are therefore very likely IN train, and a wider pool")
        print("  mostly adds wrong answers -- which is what the public null on pool")
        print("  flooding (0.335 -> 0.205) was measuring all along.")

    # (b) Window sizes, which do matter: they say how many candidates a ranker must sort.
    print("\n  Mass-window sizes and recall (train bank = control, see the docstring):\n")
    rows = []
    banks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    train_struct = D.load_structures()
    banks["train structures (control)"] = (
        train_struct.neutral_mass.to_numpy(dtype=float),
        train_struct.inchikey14.to_numpy(dtype=object))

    for name, (folder, stem, lic, clean) in BANKS.items():
        b = load_bank(folder, stem)
        if b is None:
            print(f"  (missing: {name})")
            continue
        banks[f"{name} [{lic}]"] = b

    for ppm in (5.0, 10.0, 20.0):
        for name, (bm, bk) in banks.items():
            r = recall_in_window(bm, bk, q_mass, q_truth, ppm)
            rows.append({"bank": name, "ppm": ppm, "n_bank": len(bm), **r})

    rec = pd.DataFrame(rows)
    for ppm in (5.0, 10.0, 20.0):
        print(f"\n  ±{ppm:g} ppm")
        sub = rec[rec.ppm == ppm][["bank", "n_bank", "recall", "median_window", "max_window"]]
        print("    " + sub.to_string(index=False, float_format=lambda v: f"{v:.3f}")
              .replace("\n", "\n    "))

    # What widening the pool actually costs, given that coverage says it adds little:
    # every extra structure in the mass window is one more wrong answer to outrank.
    print("\n  What each bank adds to the window a ranker must sort through:\n")
    base = banks["train structures (control)"]
    for name, (bm, bk) in banks.items():
        if name.startswith("train"):
            continue
        um = np.concatenate([base[0], bm])
        uk = np.concatenate([base[1], bk])
        order = np.argsort(um, kind="stable")
        r = recall_in_window(um[order], uk[order], q_mass, q_truth, args.ppm)
        b = recall_in_window(base[0], base[1], q_mass, q_truth, args.ppm)
        print(f"    train + {name:<40} median window "
              f"{b['median_window']:>4d} -> {r['median_window']:<4d} "
              f"(recall {b['recall']:.3f} -> {r['recall']:.3f})")
    print("\n  Candidates added, recall unchanged. That is the shape of a liability.")
    rec.to_csv(odir / "pool_recall.csv", index=False)

    # ------------------------------------------------------- 2. retrieval baseline
    print("\n" + "=" * 78)
    print("2. MASS-GATED SPECTRAL RETRIEVAL — the floor")
    print("=" * 78)

    cols = ["ingest_lib", "inchikey14", "adduct", "precursor_mz",
            "ms2_mzs", "ms2_normalized_intensities"]
    print("\nreading reference spectra ...")
    tims = D.load_train(columns=cols, timstof_only=True)
    tims["neutral_mass"] = D.neutral_mass(tims.precursor_mz, tims.adduct)
    tims = tims.dropna(subset=["neutral_mass"])

    # Exclude exactly the query spectra -- by content hash, not by (molecule, precursor),
    # which would drop every collision energy of the molecule and collapse `in_library`
    # into `held_out`.
    print("hashing reference spectra ...")
    tims["spectrum_uid"] = D.spectrum_uid(tims)
    queried = set(query.spectrum_uid)
    assert queried <= set(tims.spectrum_uid), "query spectra not found in the timsTOF frame"

    results = []
    regimes = {
        # in_library: the molecule's OTHER spectra remain available as references.
        "in_library": tims[~tims.spectrum_uid.isin(queried)],
        # held_out: no spectrum of any validation molecule survives. For *pure spectral
        # retrieval* this scores 0 by construction -- the true key is simply not in the
        # reference library, so no ranking of it is possible. It is reported to make that
        # explicit: it is the regime a candidate structure bank plus a fingerprint or
        # forward model exists to serve, and the number to beat is 0, not the in_library
        # score.
        "held_out": tims[~tims.inchikey14.isin(val_keys)],
    }

    for regime, ref in regimes.items():
        print(f"\n  --- {regime}: {len(ref):,} reference spectra, "
              f"{ref.inchikey14.nunique():,} molecules ---")
        preds = retrieve(query, ref, args.ppm, args.loss_weight)
        per = M.mrr(preds, truth, already_keys=True)
        rr = np.array(list(per.values()))
        top1 = float(np.mean([bool(p) and p[0] == truth[m] for m, p in preds.items()]))
        in_25 = float(np.mean([truth[m] in p for m, p in preds.items()]))
        print(f"    MRR {rr.mean():.4f}   top-1 {top1*100:.1f} %   "
              f"answer in the 25 {in_25*100:.1f} %")
        results.append({"regime": regime, "mrr": rr.mean(), "top1": top1, "in25": in_25})
        pd.DataFrame({"molecule_id": list(per), "reciprocal_rank": list(per.values())}) \
          .to_csv(odir / f"per_molecule_rr_{regime}.csv", index=False)

    # ------------------------------------------------- 3. the floor that actually binds
    # held_out retrieval scores 0 by construction, so it says nothing about difficulty.
    # The meaningful floor for that regime is: the true structure IS in the candidate
    # bank's mass window, and we rank the window at random. Everything a model adds has
    # to be measured against this, not against zero.
    print("\n" + "=" * 78)
    print("3. THE FLOOR THAT BINDS — random ranking of the structure mass window")
    print("=" * 78)
    rng = np.random.default_rng(0)
    bm, bk = banks["train structures (control)"]
    rr_rand = []
    for m, k in zip(q_mass, q_truth):
        lo, hi = D.mass_window(bm, m, args.ppm)
        win = bk[lo:hi]
        uniq = list(dict.fromkeys(win))                 # distinct identities, as scored
        if k not in uniq:
            rr_rand.append(0.0)
            continue
        order = rng.permutation(len(uniq))
        rank = int(np.flatnonzero(np.array(uniq, dtype=object)[order] == k)[0]) + 1
        rr_rand.append(1.0 / rank if rank <= M.N_SLOTS else 0.0)
    rr_rand = np.array(rr_rand)
    print(f"\n  candidates in the window (median)   {int(np.median([D.mass_window(bm, m, args.ppm)[1] - D.mass_window(bm, m, args.ppm)[0] for m in q_mass[:500]]))}")
    print(f"  random ranking MRR                  {rr_rand.mean():.4f}")
    results.append({"regime": "held_out, random over structure window",
                    "mrr": float(rr_rand.mean()), "top1": float((rr_rand == 1).mean()),
                    "in25": float((rr_rand > 0).mean())})

    print("\n" + "=" * 78)
    print(pd.DataFrame(results).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"""
How to read these three numbers:

  in_library   {results[0]['mrr']:.3f}  the molecule's other spectra are in the library, so plain
               cosine matching does most of the work. An optimistic bracket.
  held_out     0.000  by construction: the true key is not in the spectral library at
               all, so retrieval cannot return it. Not a difficulty measurement.
  random       {rr_rand.mean():.3f}  the true structure IS in the candidate bank's mass window
               and we guess. This is the floor a ranking model has to beat.

The best verified public stack scores 0.433 on the real test. That sits between the
random floor and our in_library bracket, which says the test is mostly *not* the easy
regime: the test molecules' structures are in the bank, but their spectra are largely
not in the library. That is exactly the regime fingerprint predictors and forward
simulators exist for, and the one where plain retrieval scores zero.""")
    pd.DataFrame(results).to_csv(odir / "retrieval_baseline.csv", index=False)


if __name__ == "__main__":
    main()
