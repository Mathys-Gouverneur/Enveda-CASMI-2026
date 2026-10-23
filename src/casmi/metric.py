"""The competition metric, reimplemented so a local number means something.

Mean reciprocal rank over molecules, comparing the **tautomer-canonical InChIKey-14**.
Three details decide whether a local MRR matches the leaderboard, and all three are easy
to get wrong:

1. **InChIKey-14, not the full key.** Stereochemistry is ignored, so two enantiomers are
   one answer and spending two slots on them wins nothing.
2. **Tautomer-canonical, not raw.** A public measurement on 4,000 COCONUT keys found
   10.43% of keys differ between the two forms, which understates a raw-key MRR by about
   0.007 — more than most published leaderboard steps are worth. `key14` canonicalises by
   default; `measure_tautomer_churn` re-measures the 10.43% on our own structures rather
   than trusting the number.
3. **Duplicates do not consume a slot.** The scorer keeps the first occurrence of each
   identity, so a list with the right answer at raw position 4 behind three spellings of
   ethanol scores 0.5, not 0.25.

Canonicalisation costs ~1.3 ms/molecule, which is why `key14` is cached and the batch
helpers exist.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Mapping, Sequence

import numpy as np

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

N_SLOTS = 25
"""Candidates the submission may carry per molecule."""

_TAUTOMER = rdMolStandardize.TautomerEnumerator()


@lru_cache(maxsize=1_000_000)
def key14(smiles: str, canonical_tautomer: bool = True) -> str | None:
    """InChIKey-14 of a SMILES, or None if RDKit cannot parse it.

    `canonical_tautomer=False` gives the raw key — only useful for measuring what the
    canonicalisation changes, never for scoring.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if canonical_tautomer:
        try:
            mol = _TAUTOMER.Canonicalize(mol)
        except Exception:
            # A handful of structures make the enumerator fail or blow its limits. Falling
            # back to the raw key is better than dropping the candidate: a raw key still
            # matches a raw key, so the pair is only lost if exactly one side failed.
            pass
    try:
        return Chem.MolToInchiKey(mol).split("-")[0]
    except Exception:
        return None


def keys14(smiles: Iterable[str], canonical_tautomer: bool = True) -> list[str | None]:
    """`key14` over an iterable, sharing the cache across calls."""
    return [key14(s, canonical_tautomer) for s in smiles]


def dedup_keys(candidate_keys: Sequence[str | None], n_slots: int = N_SLOTS) -> list[str]:
    """The identities the scorer actually sees: first occurrence of each, capped.

    Drops None (unparseable) and repeats, exactly as the official scorer does, so that
    position in this list is the rank that counts.
    """
    seen: list[str] = []
    for k in candidate_keys:
        if k is None or k in seen:
            continue
        seen.append(k)
        if len(seen) == n_slots:
            break
    return seen


def reciprocal_rank(
    candidates: Sequence[str],
    truth: str,
    n_slots: int = N_SLOTS,
    already_keys: bool = False,
) -> float:
    """1/rank of the first candidate matching `truth`, else 0.

    `candidates` are SMILES, or InChIKey-14s when `already_keys=True`. `truth` is always
    an InChIKey-14 — canonicalise it with `key14` before calling if you hold a SMILES.
    """
    keys = list(candidates) if already_keys else keys14(candidates)
    for rank, k in enumerate(dedup_keys(keys, n_slots), start=1):
        if k == truth:
            return 1.0 / rank
    return 0.0


def mrr(
    predictions: Mapping[str, Sequence[str]],
    truth: Mapping[str, str],
    n_slots: int = N_SLOTS,
    already_keys: bool = False,
) -> dict[str, float]:
    """Per-molecule reciprocal rank. Keys of `truth` define the scored set.

    A molecule absent from `predictions` scores 0 rather than being skipped — the
    leaderboard has no opt-out, and silently dropping it would flatter the score.
    """
    return {
        mid: reciprocal_rank(predictions.get(mid, []), t, n_slots, already_keys)
        for mid, t in truth.items()
    }


def score(
    predictions: Mapping[str, Sequence[str]],
    truth: Mapping[str, str],
    n_slots: int = N_SLOTS,
    already_keys: bool = False,
) -> float:
    """The leaderboard number: mean reciprocal rank over molecules."""
    per = mrr(predictions, truth, n_slots, already_keys)
    return float(np.mean(list(per.values()))) if per else 0.0


# --------------------------------------------------------------------------- submission

def to_submission_rows(predictions: Mapping[str, Sequence[str]],
                       n_slots: int = N_SLOTS) -> list[dict[str, str]]:
    """Rows for submission.csv: one per molecule, candidates joined by ';'.

    The format is unusual enough to be worth a helper — all 25 candidates live in a
    single cell of the `smiles` column, not in 25 rows or 25 columns.
    """
    return [{"molecule_id": mid, "smiles": ";".join(list(c)[:n_slots])}
            for mid, c in predictions.items()]


def from_submission_rows(df) -> dict[str, list[str]]:
    """Inverse of `to_submission_rows`, for reading a submission back."""
    return {r.molecule_id: str(r.smiles).split(";") for r in df.itertuples()}


# ------------------------------------------------------------------- what the metric costs

def measure_tautomer_churn(smiles: Sequence[str]) -> dict[str, float]:
    """How often the raw and tautomer-canonical InChIKey-14 disagree, on our structures.

    A public notebook measured 10.43% on 4,000 COCONUT keys. This re-measures it here,
    because the number sizes the bias of any raw-key validation and is cheap to check
    rather than inherit.
    """
    raw = keys14(smiles, canonical_tautomer=False)
    can = keys14(smiles, canonical_tautomer=True)
    both = [(r, c) for r, c in zip(raw, can) if r is not None and c is not None]
    if not both:
        return {"n": 0, "churn": float("nan"), "unparsed": len(smiles)}
    changed = sum(r != c for r, c in both)
    return {
        "n": len(both),
        "churn": changed / len(both),
        "unparsed": len(smiles) - len(both),
    }
