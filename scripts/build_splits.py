#!/usr/bin/env python
"""Build and persist the timsTOF validation splits.

    python scripts/build_splits.py --n-val 4000

Writes to data/processed/splits/ (gitignored, regenerable):

    query.parquet       the query spectra, test-shaped, with a synthetic molecule_id
    truth.csv           molecule_id -> inchikey14
    val_molecules.txt   the held-out structures, so a reference library can exclude them

The reference libraries are not written: they are the complement of `val_molecules.txt`
inside train.parquet, which is cheaper to recompute than to store twice.
"""
from __future__ import annotations

import argparse

import pandas as pd

from casmi import data as D
from casmi import splits as S


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-val", type=int, default=4000,
                    help="validation molecules; far above the test's 400 on purpose")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(D.PROCESSED / "splits"))
    args = ap.parse_args()

    out = __import__("pathlib").Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # instrument_type / ionization_mode / collision_energy_ev are carried so the query
    # frame can be fed to a conditioned model without re-reading the 3 GB file. They do
    # not affect which molecules are selected.
    cols = ["ingest_lib", "inchikey14", "adduct", "precursor_mz", "num_peaks",
            "collision_energy_orig", "instrument_type", "ionization_mode",
            "collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities"]
    print("reading the timsTOF half of train.parquet ...")
    tims = D.load_train(columns=cols, timstof_only=True)
    print(f"  {len(tims):,} spectra, {tims.inchikey14.nunique():,} molecules")

    sp = S.make_splits(tims, n_val_molecules=args.n_val, seed=args.seed)
    inl = sp["in_library"]

    print(f"\n{inl}")
    print(f"{sp['held_out']}")

    q = inl.query
    print(f"\nquery profile against the test profile:")
    print(f"  {'':24s} {'ours':>10s} {'test':>10s}")
    print(f"  {'molecules':24s} {q.molecule_id.nunique():>10,} "
          f"{D.TEST_PROFILE['n_molecules']:>10,}")
    print(f"  {'spectra per molecule':24s} {len(q)/q.molecule_id.nunique():>10.2f} "
          f"{D.TEST_PROFILE['spectra_per_molecule']:>10.2f}")
    print(f"  {'median peaks':24s} {q.num_peaks.median():>10.0f} "
          f"{D.TEST_PROFILE['median_peaks']:>10d}")
    print(f"  {'neutral mass min':24s} {q.neutral_mass.min():>10.1f} "
          f"{D.TEST_PROFILE['mass_min']:>10.1f}")
    print(f"  {'neutral mass max':24s} {q.neutral_mass.max():>10.1f} "
          f"{D.TEST_PROFILE['mass_max']:>10.1f}")
    print(f"\n  adduct mix:")
    ours = q.adduct.value_counts(normalize=True)
    for a, share in sorted(D.TEST_PROFILE["adduct_mix"].items(), key=lambda kv: -kv[1]):
        print(f"    {a:16s} {ours.get(a, 0.0)*100:>6.1f} %   test {share*100:>5.1f} %")

    # A content hash, so the evaluation can remove exactly these spectra from a
    # reference library and nothing else. See data.spectrum_uid for why.
    print("\nhashing query spectra for exact exclusion ...")
    q = q.assign(spectrum_uid=D.spectrum_uid(q))
    assert q.spectrum_uid.is_unique, "spectrum hash collided, which should not happen"

    q.to_parquet(out / "query.parquet", index=False)
    pd.DataFrame({"molecule_id": list(inl.truth), "inchikey14": list(inl.truth.values())}) \
      .to_csv(out / "truth.csv", index=False)
    (out / "val_molecules.txt").write_text("\n".join(sorted(set(inl.truth.values()))) + "\n")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
