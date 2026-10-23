#!/usr/bin/env python
"""How big a difference has to be before it is real.

The leaderboard scores 400 molecules, and every weight in the public stack was tuned by
submission against them. This says what that buys in resolution.

Three questions that get confused with each other, with very different answers:

1. **How well does a public score estimate true skill?** Unpaired — the private test is a
   *different* molecule sample. Governed by `sd(RR)/sqrt(400)`, and it is the size of the
   shake-up to expect between public and private.
2. **Can the public board rank two teams?** Paired: the public leaderboard scores
   everyone on the **same** 400 molecules, so the molecule-to-molecule variance largely
   cancels and the resolution is much finer than (1). This is why the board can order
   teams it cannot pin down absolutely.
3. **Can we tell our own method B from method A offline?** Paired too, and on a
   validation set we choose the size of. This is the lever: resolution improves as
   `1/sqrt(n)`, and we have 183k timsTOF molecules against the board's 400.

    python scripts/noise_floor.py
    python scripts/noise_floor.py --scores out/per_molecule_rr.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

# Reciprocal ranks are not continuous: a molecule scores 1, 1/2, 1/3 ... 1/25, or 0.
RR_VALUES = np.array([1.0] + [1.0 / r for r in range(2, 26)] + [0.0])


def plausible_rr(mrr_target: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Per-molecule reciprocal ranks whose mean is about `mrr_target`.

    Shape taken from what the public record implies: most of the mass at rank 1 or at
    zero, a thin middle. The rank-1 share is solved for to hit the target mean, so the
    variance is a consequence rather than an assumption.
    """
    mid_share = 0.18
    n_mid = int(n * mid_share)
    mid = rng.choice(RR_VALUES[1:-1], size=n_mid, p=np.ones(24) / 24)
    p1 = float(np.clip((mrr_target - mid.mean() * mid_share) / (1 - mid_share), 0.0, 1.0))
    head = rng.random(n - n_mid) < p1
    return np.concatenate([mid, np.where(head, 1.0, 0.0)])


def unpaired(rr: np.ndarray, n_test: int, draws: int, rng: np.random.Generator) -> dict:
    """Spread of the score over independent `n_test`-molecule samples."""
    means = rr[rng.integers(0, len(rr), size=(draws, n_test))].mean(axis=1)
    se = float(means.std(ddof=1))
    return {"mrr": float(rr.mean()), "sd_rr": float(rr.std(ddof=1)), "se": se,
            "ci95": 1.96 * se,
            "p2.5": float(np.percentile(means, 2.5)),
            "p97.5": float(np.percentile(means, 97.5))}


def paired_min_detectable(rr: np.ndarray, n_test: int, fractions, draws: int,
                          rng: np.random.Generator) -> pd.DataFrame:
    """Power to detect a genuine improvement, scored on the same molecules.

    Method B is method A with a fraction `f` of molecules *fixed* — promoted to rank 1.
    That is the realistic shape of progress here (a mechanism that rescues some
    molecules), and unlike a symmetric perturbation it has a non-zero true effect.
    """
    rows = []
    for f in fractions:
        n_flip = max(1, int(round(len(rr) * f)))
        gain = np.zeros(len(rr))
        # Fix the molecules that are not already correct: promoting a rank-1 molecule to
        # rank 1 gains nothing, and pretending otherwise would overstate the effect.
        improvable = np.flatnonzero(rr < 1.0)
        picked = rng.choice(improvable, size=min(n_flip, len(improvable)), replace=False)
        gain[picked] = 1.0 - rr[picked]

        d = gain[rng.integers(0, len(rr), size=(draws, n_test))].mean(axis=1)
        se = float(d.std(ddof=1))
        rows.append({
            "molecules_fixed_%": f * 100,
            f"n_fixed_per_{n_test}": f * n_test,
            "true_effect": float(gain.mean()),
            "se_paired": se,
            # one-sided 95%: the effect must clear 1.645 SE to be called real
            "power_95": float((d > 1.645 * se).mean()) if se > 0 else 1.0,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", help="CSV with a reciprocal_rank column, from a real run")
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--n-pool", type=int, default=200_000)
    ap.add_argument("--draws", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    if args.scores:
        rr = pd.read_csv(args.scores).reciprocal_rank.to_numpy(dtype=float)
        sources = [(f"measured, {len(rr)} molecules", rr)]
    else:
        print("No --scores given: simulating the reciprocal-rank distribution at the two")
        print("score levels the leaderboard shows — the verified public 0.433 and the top 0.504.\n")
        sources = [(f"simulated, MRR~{t:.2f}", plausible_rr(t, args.n_pool, rng))
                   for t in (0.433, 0.504)]

    for name, rr in sources:
        u = unpaired(rr, args.n_test, args.draws, rng)
        print(f"=== {name} ===")
        print(f"  MRR                                {u['mrr']:.4f}")
        print(f"  sd of per-molecule RR              {u['sd_rr']:.4f}")
        print(f"\n  (1) public score as an estimate of true skill, on {args.n_test} molecules")
        print(f"      standard error                 {u['se']:.4f}")
        print(f"      95% interval                   ±{u['ci95']:.4f}  "
              f"[{u['p2.5']:.3f}, {u['p97.5']:.3f}]")
        print(f"      -> expect a public/private shift of this order, per team")

        print(f"\n  (3) paired A/B of our own methods, same molecules:")
        tab = paired_min_detectable(rr, args.n_test, (0.0025, 0.005, 0.01, 0.025, 0.05),
                                    args.draws, rng)
        print("      " + tab.to_string(index=False, float_format=lambda v: f"{v:.4f}")
              .replace("\n", "\n      "))

        bigger = paired_min_detectable(rr, 4000, (0.0025, 0.005, 0.01), args.draws, rng)
        print(f"\n      the same comparisons on a 4,000-molecule validation set:")
        print("      " + bigger.to_string(index=False, float_format=lambda v: f"{v:.4f}")
              .replace("\n", "\n      "))
        print()

    print("What this settles:")
    print("  * A single submission cannot confirm a gain below ~0.01: the per-molecule")
    print("    spread is too wide for 400 molecules to pin the score down absolutely.")
    print("  * Public->private movement of ±0.05 is the expected scale, not a surprise.")
    print("    The ~0.03 separating the top 16 teams is inside that.")
    print("  * But a PAIRED offline comparison resolves far smaller effects, and the")
    print("    resolution scales with the validation set we choose. That is the whole")
    print("    argument for building the validation set before the model.")


if __name__ == "__main__":
    main()
