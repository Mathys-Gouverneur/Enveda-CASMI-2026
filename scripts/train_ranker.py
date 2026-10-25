#!/usr/bin/env python
"""Fit a ranker over the three signals, and test whether it beats adding them up.

    python scripts/train_ranker.py --features out/features_c05.parquet

`eval_combined.py` showed the signals compose — a weighted sum of z-scores took MRR from
0.434 to 0.459 at the calibrated ceiling. Two things it could not do:

**Say whether that is real.** A weighted sum reports one number per configuration, so
there is nothing to pair. This keeps per-molecule reciprocal ranks and tests the
difference the way `docs/measurement.md` §1 says it has to be tested.

**Use the signals conditionally.** A fixed weight applies everywhere, but the public
record says the gains are conditional — ICEBERG as a feature was worth +0.0015 *only*
where candidates exceeded 100. A ranker can learn "trust FPNet when the analogue score is
weak", which no single weight can express.

Features beyond the three raw signals, each with a reason:

| family | why |
|---|---|
| rank-normalised, z-scored, delta-to-max | the public FPNet's logits go into LightGBM in six normalisations at once, because no single calibration is trustworthy |
| `log(n_candidates)` | stratum awareness: the published conditional gains say difficulty matters |
| agreement between sources | whether two signals pick the *same* candidate is information neither carries alone |

Grouped cross-validation by molecule, because a candidate list is one observation and
splitting inside it would leak.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

from casmi import metric as M

SIGNALS = ("analog", "formula", "fpnet")


def rank_norm(v: np.ndarray) -> np.ndarray:
    """Position in [0, 1], best first. Immune to the scale a signal happens to use."""
    order = np.argsort(np.argsort(-v))
    return order / max(len(v) - 1, 1)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Per-candidate features, labels, and group sizes, grouped by molecule."""
    frames, labels, groups, molecules = [], [], [], []
    for mol, g in df.groupby("molecule_id", sort=False):
        g = g.reset_index(drop=True)
        cols = {}
        raw = {s: g[s].to_numpy(dtype=float) for s in SIGNALS}
        for s, v in raw.items():
            sd = v.std()
            cols[f"{s}_z"] = (v - v.mean()) / sd if sd > 1e-12 else np.zeros_like(v)
            cols[f"{s}_rank"] = rank_norm(v)
            cols[f"{s}_delta_max"] = v - v.max()
            cols[f"{s}_is_max"] = (v == v.max()).astype(float)
        # Agreement: do two signals crown the same candidate? Carried as a per-row
        # indicator so the ranker can learn to trust a consensus.
        for a, b in (("analog", "formula"), ("analog", "fpnet"), ("formula", "fpnet")):
            cols[f"agree_{a}_{b}"] = cols[f"{a}_is_max"] * cols[f"{b}_is_max"]
            cols[f"prod_{a}_{b}"] = cols[f"{a}_rank"] * cols[f"{b}_rank"]
        cols["log_n_cand"] = np.full(len(g), np.log(len(g)))
        frames.append(pd.DataFrame(cols))
        labels.append(g.is_true.to_numpy().astype(int))
        groups.append(len(g))
        molecules.append(mol)
    return (pd.concat(frames, ignore_index=True), np.concatenate(labels),
            np.array(groups)), molecules


def mrr_from_scores(scores: np.ndarray, labels: np.ndarray,
                    groups: np.ndarray) -> np.ndarray:
    """Per-molecule reciprocal rank from a flat score vector and group sizes."""
    out, at = [], 0
    for n in groups:
        s, y = scores[at:at + n], labels[at:at + n]
        at += n
        if not y.any():
            out.append(0.0)
            continue
        rank = int(np.flatnonzero(np.argsort(-s) == int(np.argmax(y)))[0]) + 1
        out.append(1.0 / rank if rank <= M.N_SLOTS else 0.0)
    return np.array(out)


def paired(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    """Report the difference the only way it can honestly be reported."""
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    verdict = "REAL" if abs(d.mean()) > 1.96 * se else "inside noise"
    print(f"  {name_a} - {name_b}: {d.mean():+.4f} ± {1.96*se:.4f} (95%, paired, "
          f"n={len(d)})  -> {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(pathlib.Path("out/features_c05.parquet")))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    print(f"{df.molecule_id.nunique():,} molecules, {len(df):,} candidate rows, "
          f"{int(df.is_true.sum()):,} positives")

    (Xf, y, groups), molecules = build_features(df)
    print(f"{Xf.shape[1]} features\n")

    # Baselines on the same groups, so every comparison below is paired.
    flat = pd.concat([g.reset_index(drop=True) for _, g in
                      df.groupby("molecule_id", sort=False)], ignore_index=True)
    base = {}
    for name, score in (
            ("analogue alone", flat.analog_z.to_numpy()),
            ("sum of z-scores", (flat.analog_z + 0.3 * flat.formula_z
                                 + 0.3 * flat.fpnet_z).to_numpy())):
        base[name] = mrr_from_scores(score, y, groups)
        print(f"  {name:22s} MRR {base[name].mean():.4f}")

    # ---- grouped cross-validation -----------------------------------------------------
    import lightgbm as lgb

    rng = np.random.default_rng(args.seed)
    fold_of = rng.integers(0, args.folds, size=len(groups))
    starts = np.concatenate([[0], np.cumsum(groups)])
    oof = np.zeros(len(y))

    for fold in range(args.folds):
        tr_g = np.flatnonzero(fold_of != fold)
        te_g = np.flatnonzero(fold_of == fold)
        tr_rows = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in tr_g])
        te_rows = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in te_g])

        model = lgb.LGBMRanker(
            objective="lambdarank", n_estimators=300, learning_rate=0.05,
            num_leaves=15, min_child_samples=40, subsample=0.9, colsample_bytree=0.8,
            random_state=args.seed, verbose=-1,
            # The metric only ever looks at the first correct answer, so optimising the
            # top of the list is the objective, not a truncation of it.
            eval_at=[1, 5], label_gain=[0, 1])
        model.fit(Xf.iloc[tr_rows], y[tr_rows], group=groups[tr_g])
        oof[te_rows] = model.predict(Xf.iloc[te_rows])

    ranker = mrr_from_scores(oof, y, groups)
    print(f"\n  {'ranker (out-of-fold)':22s} MRR {ranker.mean():.4f}")

    print("\npaired comparisons:")
    paired(ranker, base["analogue alone"], "ranker", "analogue")
    paired(ranker, base["sum of z-scores"], "ranker", "weighted sum")
    paired(base["sum of z-scores"], base["analogue alone"], "weighted sum", "analogue")

    # Which features the ranker actually used, fitted once on everything for the report.
    full = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05,
                          num_leaves=15, min_child_samples=40, random_state=args.seed,
                          verbose=-1, label_gain=[0, 1])
    full.fit(Xf, y, group=groups)
    imp = (pd.Series(full.feature_importances_, index=Xf.columns)
           .sort_values(ascending=False))
    print("\ntop features by split count:")
    for name, value in imp.head(12).items():
        print(f"  {name:24s} {value:5d}")

    out = pathlib.Path(args.features).with_name("ranker_per_molecule.csv")
    pd.DataFrame({"molecule_id": molecules, "ranker": ranker,
                  "analogue": base["analogue alone"],
                  "weighted_sum": base["sum of z-scores"]}).to_csv(out, index=False)
    print(f"\nper-molecule scores -> {out}")


if __name__ == "__main__":
    main()
