"""A three-way split, and the bookkeeping that makes a number quotable.

Everything measured so far was fitted, selected and reported on one set of molecules.
That is the mistake this project exists to avoid — a six-configuration sweep whose maximum
gets written down as a result is the same error as tuning constants against a
400-molecule leaderboard, only smaller and better hidden.

So the 4,000 validation molecules are split three ways, disjointly and deterministically:

| set | size | what may touch it |
|---|---|---|
| `fit` | 2,000 | fitting weights, ranker parameters, `top_k`, `p`, thresholds — anything continuous |
| `select` | 1,000 | choosing *between* configurations, and choosing a training epoch |
| `report` | 1,000 | read once per decision. Never for tuning, never for choosing |

The rule that gives the numbers their meaning: **a configuration is chosen on `select`,
then evaluated once on `report`, and the `report` number is the quotable one.** A maximum
over `select` is a selection artefact; the same configuration's `report` score is an
estimate.

`Evaluation.quote` enforces the accounting rather than trusting anyone to remember: it
records every read of `report` with the decision that justified it, so the count of reads
is visible and a second read for the same decision is refused.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

#: Fractions of the validation molecules. `fit` is largest because continuous fitting is
#: the hungriest; `report` only ever estimates a handful of chosen configurations.
SHARES = {"fit": 0.50, "select": 0.25, "report": 0.25}


def assign_sets(molecule_ids, shares: dict[str, float] = SHARES) -> dict[str, str]:
    """molecule_id -> set name, by hashing the id.

    Hashed rather than shuffled with a seed: the assignment is then a pure function of the
    id, so it is identical across runs, machines and reorderings of the input, and adding
    molecules never reshuffles the ones already assigned. A seeded permutation would
    silently move molecules between sets the moment the input order changed.
    """
    bounds, acc = [], 0.0
    for name, share in shares.items():
        acc += share
        bounds.append((name, acc))

    out = {}
    for mid in molecule_ids:
        h = hashlib.blake2b(str(mid).encode(), digest_size=8).digest()
        u = int.from_bytes(h, "big") / float(1 << 64)
        for name, upper in bounds:
            if u < upper:
                out[str(mid)] = name
                break
        else:
            out[str(mid)] = bounds[-1][0]
    return out


@dataclass
class Evaluation:
    """The three sets, plus a ledger of every read of `report`."""

    sets: dict[str, str]
    ledger_path: Path | None = None
    _ledger: list[dict] = field(default_factory=list)

    @classmethod
    def from_truth(cls, truth_csv, ledger_path=None) -> "Evaluation":
        ids = pd.read_csv(truth_csv).molecule_id
        ev = cls(assign_sets(ids), Path(ledger_path) if ledger_path else None)
        if ev.ledger_path and ev.ledger_path.exists():
            ev._ledger = json.loads(ev.ledger_path.read_text())
        return ev

    def members(self, name: str) -> list[str]:
        if name not in SHARES:
            raise ValueError(f"unknown set {name!r}; expected one of {list(SHARES)}")
        return sorted(m for m, s in self.sets.items() if s == name)

    def mask(self, molecule_ids, name: str) -> np.ndarray:
        want = set(self.members(name))
        return np.array([str(m) in want for m in molecule_ids])

    def summary(self) -> pd.Series:
        return pd.Series(self.sets).value_counts().reindex(list(SHARES))

    # ------------------------------------------------------------------ the accounting

    def quote(self, decision: str, scores: dict[str, float], detail: str = "") -> float:
        """Read `report` once, for a named decision, and record that it happened.

        `scores` maps molecule_id to reciprocal rank over the whole validation set; only
        the `report` members are used. Refuses a second read for a decision already
        quoted, because re-reading `report` until a number looks right is how a held-out
        set stops being held out.
        """
        if any(e["decision"] == decision for e in self._ledger):
            raise RuntimeError(
                f"`report` has already been read for decision {decision!r}. "
                "Pick a new decision name, or use `select` — re-reading the report set "
                "for the same question is what makes it stop being held out.")
        members = self.members("report")
        values = [scores[m] for m in members if m in scores]
        if not values:
            raise ValueError("no report-set molecules in `scores`")
        mrr = float(np.mean(values))
        self._ledger.append({"decision": decision, "detail": detail,
                             "n": len(values), "mrr": mrr})
        if self.ledger_path:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            self.ledger_path.write_text(json.dumps(self._ledger, indent=1) + "\n")
        return mrr

    @property
    def reads(self) -> int:
        """How many times `report` has been read. Every read spends some of its value."""
        return len(self._ledger)

    def ledger(self) -> pd.DataFrame:
        return pd.DataFrame(self._ledger)


def paired_difference(a: dict[str, float], b: dict[str, float],
                      molecules=None) -> dict[str, float]:
    """Paired mean difference with a 95% interval, over the molecules both scored.

    Paired because the alternative throws away the only advantage a shared validation set
    has: `docs/measurement.md` §1 measures a 10x resolution gain from pairing, and an
    unpaired test on these sample sizes sees almost nothing.
    """
    keys = sorted(set(a) & set(b)) if molecules is None else [
        m for m in molecules if m in a and m in b]
    if not keys:
        raise ValueError("no molecules in common")
    d = np.array([a[m] - b[m] for m in keys], dtype=float)
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("inf")
    return {"n": len(d), "mean": float(d.mean()), "se": float(se),
            "ci95": float(1.96 * se),
            "real": bool(abs(d.mean()) > 1.96 * se)}
