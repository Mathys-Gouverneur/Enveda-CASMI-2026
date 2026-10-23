#!/usr/bin/env python
"""Train the spectrum -> fingerprint model, then rank candidates with it.

    python scripts/train_fpnet.py --epochs 10               # v1, the control
    python scripts/train_fpnet.py --v2 --epochs 10          # conditioned, all instruments

Two configurations, and the pair is the experiment:

**v1** trains on the timsTOF half alone. Domain purity bought by discarding 55% of the
corpus, because a model that cannot see which instrument produced a spectrum has no way
to use the rest.

**v2** (`--v2`) conditions on instrument, adduct, collision energy and polarity through a
global token, adds a per-peak neutral-loss channel, and pools CLS together with the mean.
That lets one model train on all 2.54M spectra and *learn* the instrument correction, so
`--v2` implies `--all-instruments`. This is what the public FPNet does, and running both
answers whether instrument knowledge belongs in the data selection or in the model.

What it does, and why in this order:

1. Builds a training set whose molecules are **not** in the validation split, so nothing
   leaks. Targets are exact RDKit fingerprints of the molecule.
2. Selects the bits worth predicting (dropping the near-constant ones) and stores the
   selection with the checkpoint — a model and its bit selection are one artefact.
3. Trains with per-bit BCE, balanced by positive frequency.
4. Evaluates where it counts: on the **`held_out`** split, ranking the candidate mass
   window from train's structure bank by predicted-vs-exact fingerprint agreement.

The number to beat is **0.085**, the MRR of ranking that window at random
(`docs/measurement.md` §3). Plain spectral retrieval scores 0.000 in this regime, so this
is the first method that can score at all on the task the test actually poses.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from casmi import data as D
from casmi import fingerprint as FP
from casmi import fpnet as N
from casmi import metric as M
from casmi import spectra as SP


class Conditioning:
    """The metadata tensors FPNetV2 reads, and the vocabularies they index into.

    Built once from the training frame and then applied to any frame, so the validation
    and test spectra are encoded against exactly the vocabulary the model was fitted
    with. Index 0 is the out-of-vocabulary bucket; every adduct and instrument present in
    test.parquet also occurs in train, so the test never lands there.
    """

    def __init__(self, df: pd.DataFrame, n_adducts: int, n_instruments: int):
        self.adduct = N.build_vocab(df.adduct, n_adducts)
        self.instrument = N.build_vocab(df.instrument_type, n_instruments)

    def encode(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        ce, present = N.mean_collision_energy(df.collision_energy_ev.to_numpy())
        return {
            "adduct": N.encode_vocab(df.adduct, self.adduct),
            "instrument": N.encode_vocab(df.instrument_type, self.instrument),
            "ce": ce,
            "ce_present": present,
            # +1 positive, 0 negative; anything else is treated as positive, which is the
            # majority -- there are exactly two values in both train and test.
            "polarity": (df.ionization_mode.to_numpy() != "negative").astype(np.float32),
        }


class SpectrumFingerprintSet(Dataset):
    """Spectra with their molecule's fingerprint as target."""

    def __init__(self, df: pd.DataFrame, Y: np.ndarray, key_to_row: dict[str, int],
                 max_peaks: int, cond: Conditioning | None = None):
        self.mzs = df.ms2_mzs.to_numpy()
        self.its = df.ms2_normalized_intensities.to_numpy()
        self.prec = df.precursor_mz.to_numpy(dtype=np.float32)
        self.rows = np.array([key_to_row[k] for k in df.inchikey14], dtype=np.int64)
        self.Y = Y
        self.max_peaks = max_peaks
        self.cond = cond.encode(df) if cond is not None else None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        return i

    def collate(self, idx):
        idx = np.asarray(idx)
        # v2 reads the precursor through the global token, so it is not prepended as a
        # peak; v1 has nowhere else to put it.
        mz, it, pad = N.collate_spectra(
            self.mzs[idx], self.its[idx], self.max_peaks,
            precursors=None if self.cond is not None else self.prec[idx])
        y = torch.from_numpy(self.Y[self.rows[idx]].astype(np.float32))
        if self.cond is None:
            return (mz, it, pad), y
        c = self.cond
        extra = (torch.from_numpy(self.prec[idx]),
                 torch.from_numpy(c["adduct"][idx]),
                 torch.from_numpy(c["instrument"][idx]),
                 torch.from_numpy(c["ce"][idx]),
                 torch.from_numpy(c["ce_present"][idx]),
                 torch.from_numpy(c["polarity"][idx]))
        return (mz, it, pad, *extra), y


@torch.no_grad()
def predict_logits(model, df, max_peaks, device, cond: Conditioning | None = None,
                   batch: int = 256) -> np.ndarray:
    model.eval()
    out = []
    mzs, its = df.ms2_mzs.to_numpy(), df.ms2_normalized_intensities.to_numpy()
    prec = df.precursor_mz.to_numpy(dtype=np.float32)
    enc = cond.encode(df) if cond is not None else None
    for s in range(0, len(df), batch):
        sl = slice(s, min(s + batch, len(df)))
        mz, it, pad = N.collate_spectra(mzs[sl], its[sl], max_peaks,
                                        precursors=None if enc else prec[sl])
        args = [mz.to(device), it.to(device), pad.to(device)]
        if enc is not None:
            args += [torch.from_numpy(prec[sl]).to(device),
                     torch.from_numpy(enc["adduct"][sl]).to(device),
                     torch.from_numpy(enc["instrument"][sl]).to(device),
                     torch.from_numpy(enc["ce"][sl]).to(device),
                     torch.from_numpy(enc["ce_present"][sl]).to(device),
                     torch.from_numpy(enc["polarity"][sl]).to(device)]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            z = model(*args)
        out.append(z.float().cpu().numpy())
    return np.concatenate(out)


def rank_with_fingerprints(query: pd.DataFrame, logits: np.ndarray,
                           bank_mass: np.ndarray, bank_keys: np.ndarray,
                           bank_fp: np.ndarray, ppm: float, scoring: str,
                           n_keep: int = M.N_SLOTS) -> dict[str, list[str]]:
    """Rank each molecule's candidate mass window by fingerprint agreement."""
    q_mol = query.molecule_id.to_numpy(dtype=object)
    q_mass = query.neutral_mass.to_numpy(dtype=float)
    probs = 1.0 / (1.0 + np.exp(-logits)) if scoring == "tanimoto" else None

    out: dict[str, list[str]] = {}
    for mol in pd.unique(q_mol):
        rows = np.flatnonzero(q_mol == mol)
        lo, hi = D.mass_window(bank_mass, float(np.median(q_mass[rows])), ppm)
        if hi <= lo:
            out[mol] = []
            continue
        cand_fp, cand_keys = bank_fp[lo:hi], bank_keys[lo:hi]
        if scoring == "tanimoto":
            s = FP.tanimoto_dense(probs[rows], cand_fp)          # (n_spectra, n_cand)
            # Log-sum-exp measured best for pooling a molecule's spectra, and Tanimoto
            # shares the cosine's [0,1] scale, so the fitted beta transfers.
            best = SP.pool_logsumexp(s)
        else:
            s = FP.modified_tanimoto_logit(logits[rows], cand_fp).T
            # Log-likelihoods are unbounded and on a different scale entirely, where a
            # beta fitted on [0,1] cosines would saturate. Max until pooling is refitted
            # for this scoring rule on its own.
            best = s.max(axis=0)
        ranked, seen = [], set()
        for j in np.argsort(best)[::-1]:
            k = cand_keys[j]
            if k in seen:
                continue
            seen.add(k)
            ranked.append(k)
            if len(ranked) == n_keep:
                break
        out[mol] = ranked
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=str(D.PROCESSED / "splits"))
    ap.add_argument("--out", default=str(D.ROOT / "out" / "fpnet"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-peaks", type=int, default=256)
    ap.add_argument("--max-train-spectra", type=int, default=400_000)
    ap.add_argument("--ppm", type=float, default=10.0)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--v2", action="store_true",
                    help="conditioned model: per-peak neutral loss, a global token "
                         "carrying precursor/energy/polarity/adduct/instrument, and "
                         "CLS+mean pooling. Implies --all-instruments unless overridden.")
    ap.add_argument("--all-instruments", action="store_true",
                    help="train on all 2.54M spectra rather than the timsTOF half. Only "
                         "sensible with --v2, which can learn the instrument correction; "
                         "v1 has no way to tell the instruments apart.")
    args = ap.parse_args()
    if args.v2 and not args.all_instruments:
        args.all_instruments = True

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    sdir, odir = pathlib.Path(args.splits), pathlib.Path(args.out)
    odir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}"
          f"{' — ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")

    # ------------------------------------------------------------------ data
    query = pd.read_parquet(sdir / "query.parquet")
    truth = dict(pd.read_csv(sdir / "truth.csv").itertuples(index=False, name=None))
    val_keys = set(truth.values())

    cols = ["ingest_lib", "inchikey14", "adduct", "precursor_mz",
            "ms2_mzs", "ms2_normalized_intensities"]
    if args.v2:
        cols += ["instrument_type", "ionization_mode", "collision_energy_ev"]
    scope = "all instruments" if args.all_instruments else "timsTOF only"
    print(f"reading spectra ({scope}) ...")
    tims = D.load_train(columns=cols, timstof_only=not args.all_instruments)
    train_df = tims[~tims.inchikey14.isin(val_keys)]
    print(f"  {len(train_df):,} training spectra over "
          f"{train_df.inchikey14.nunique():,} molecules (validation molecules excluded)")
    if len(train_df) > args.max_train_spectra:
        train_df = train_df.sample(n=args.max_train_spectra, random_state=args.seed)
        print(f"  capped to {len(train_df):,} spectra")

    # ------------------------------------------------- fingerprints for every structure
    print("\nbuilding the structure bank and its exact fingerprints ...")
    bank = D.load_structures()
    # Cached: 275k structures take minutes to fingerprint and the result is a pure
    # function of the structure list, so recomputing it every run is wasted time.
    cache = D.PROCESSED / "bank_fingerprints.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if z["n"] == len(bank) and z["width"] == FP.FP_WIDTH:
            Xfull, ok = z["X"], z["ok"]
            print(f"  loaded cached fingerprints from {cache.name}")
        else:
            cache.unlink()
    if not cache.exists():
        t0 = time.time()
        Xfull, ok = FP.fingerprint_matrix(bank.normalized_smiles)
        np.savez_compressed(cache, X=Xfull, ok=ok, n=len(bank), width=FP.FP_WIDTH)
        print(f"  fingerprinted {len(bank):,} structures in {time.time()-t0:.0f}s "
              f"({(~ok).sum()} unparseable), cached")
    bank, Xfull = bank[ok].reset_index(drop=True), Xfull[ok]
    print(f"  {len(bank):,} usable structures")

    # Bit selection on the TRAINING molecules only -- selecting on the full bank would
    # let the validation molecules influence which bits we predict.
    train_keys = set(train_df.inchikey14)
    is_train_struct = bank.inchikey14.isin(train_keys).to_numpy()
    bits = FP.select_bits(Xfull[is_train_struct])
    print(f"  predicting {len(bits):,} of {FP.FP_WIDTH} bits "
          f"(frequency in [0.5%, 99.5%] among training molecules)")

    Y = Xfull[:, bits]
    key_to_row = {k: i for i, k in enumerate(bank.inchikey14)}
    train_df = train_df[train_df.inchikey14.isin(key_to_row)]

    bank_mass = bank.neutral_mass.to_numpy(dtype=float)
    bank_keys = bank.inchikey14.to_numpy(dtype=object)

    # ------------------------------------------------------------------ model
    cond = None
    if args.v2:
        cfg = N.FPNetV2Config(n_bits=len(bits), d_model=args.d_model,
                              n_layers=args.n_layers, max_peaks=args.max_peaks)
        cond = Conditioning(train_df, cfg.n_adducts, cfg.n_instruments)
        model = N.FPNetV2(cfg).to(device)
        print(f"\nconditioning: {len(cond.adduct)} adducts + OOV, "
              f"{len(cond.instrument)} instruments + OOV")
    else:
        cfg = N.FPNetConfig(n_bits=len(bits), d_model=args.d_model,
                            n_layers=args.n_layers, max_peaks=args.max_peaks)
        model = N.FPNet(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{'FPNetV2' if args.v2 else 'FPNet'}: {n_params/1e6:.1f}M parameters, "
          f"{cfg.n_layers} layers, d_model {cfg.d_model}, {cfg.n_bits:,} output bits")

    ds = SpectrumFingerprintSet(train_df, Y, key_to_row, args.max_peaks, cond)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4,
                    collate_fn=ds.collate, drop_last=True, persistent_workers=True)

    pw = N.pos_weight_from(Y[is_train_struct]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(dl), pct_start=0.1)
    scaler_dtype = torch.bfloat16

    print(f"\ntraining {args.epochs} epochs x {len(dl)} steps")
    for epoch in range(args.epochs):
        model.train()
        running, seen, t0 = 0.0, 0, time.time()
        for step, (inputs, y) in enumerate(dl):
            inputs = tuple(t.to(device, non_blocking=True) for t in inputs)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=scaler_dtype, enabled=device.type == "cuda"):
                loss = loss_fn(model(*inputs), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item() * len(y)
            seen += len(y)
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step:5d}/{len(dl)}  "
                      f"loss {running/max(seen,1):.4f}  "
                      f"{seen/(time.time()-t0):.0f} spectra/s", flush=True)
        print(f"  epoch {epoch} done: loss {running/seen:.4f} in {time.time()-t0:.0f}s")

    torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__, "bits": bits,
                "v2": args.v2,
                "adduct_vocab": cond.adduct if cond else None,
                "instrument_vocab": cond.instrument if cond else None},
               odir / "fpnet.pt")
    print(f"\nsaved {odir/'fpnet.pt'}")

    # ------------------------------------------------------------------ evaluation
    print("\n" + "=" * 78)
    print("EVALUATION on the held_out regime — the floor to beat is 0.085 (random)")
    print("=" * 78)
    logits = predict_logits(model, query, args.max_peaks, device, cond)

    # Per-bit quality first: a ranking failure could be the model or the scoring rule.
    probs = 1.0 / (1.0 + np.exp(-logits))
    q_rows = np.array([key_to_row[truth[m]] for m in query.molecule_id], dtype=np.int64)
    Ytrue = Y[q_rows]
    from sklearn.metrics import roc_auc_score
    keep = (Ytrue.sum(axis=0) > 0) & (Ytrue.sum(axis=0) < len(Ytrue))
    auc = roc_auc_score(Ytrue[:, keep].ravel(), probs[:, keep].ravel())
    print(f"\n  per-bit ROC AUC (micro, {keep.sum():,} usable bits): {auc:.4f}")

    rows = []
    for scoring in ("tanimoto", "logit"):
        preds = rank_with_fingerprints(query, logits, bank_mass, bank_keys, Y,
                                       args.ppm, scoring)
        per = M.mrr(preds, truth, already_keys=True)
        rr = np.array(list(per.values()))
        top1 = float(np.mean([bool(p) and p[0] == truth[m] for m, p in preds.items()]))
        in25 = float(np.mean([truth[m] in p for m, p in preds.items()]))
        print(f"  {scoring:>10s}: MRR {rr.mean():.4f}   top-1 {top1*100:5.1f} %   "
              f"in 25 {in25*100:5.1f} %")
        rows.append({"scoring": scoring, "mrr": float(rr.mean()), "top1": top1,
                     "in25": in25, "auc": float(auc)})
        pd.DataFrame({"molecule_id": list(per), "reciprocal_rank": list(per.values())}) \
          .to_csv(odir / f"per_molecule_rr_fpnet_{scoring}.csv", index=False)

    pd.DataFrame(rows).to_csv(odir / "fpnet_eval.csv", index=False)
    (odir / "run.json").write_text(json.dumps(vars(args) | {"n_params": n_params,
                                                            "n_bits": int(len(bits))},
                                              indent=2))
    best = max(rows, key=lambda r: r["mrr"])
    print(f"\n  best: {best['scoring']} at MRR {best['mrr']:.4f} "
          f"against a random floor of 0.085 "
          f"({best['mrr']/0.085:.1f}x)")
    print("\n  Compare on the same molecules with scripts/noise_floor.py --scores "
          f"{odir}/per_molecule_rr_fpnet_{best['scoring']}.csv")


if __name__ == "__main__":
    main()
