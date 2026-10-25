"""Analogue propagation: how spectral evidence reaches a molecule with no spectrum.

The gap this fills is the one `measurement.md` §3 opens. In the `held_out` regime — the
one the test is actually in — the molecule has no reference spectrum, so direct spectral
retrieval cannot return its key at any rank and scores exactly 0. Yet the public
leaderboard reaches 0.433 largely on spectral evidence. The bridge is this:

1. Match the query spectrum against the **whole** library. The best matches are *other*
   molecules — analogues, related by a methylation, a hydroxylation, a ring change.
2. Those matches have **known structures**.
3. Score each mass-window candidate by structural similarity to those analogue
   structures, weighted by how well each analogue matched spectrally.

So a molecule never seen by the library still gets scored, through its neighbours. This is
the root of the entire public lineage (`prvsiyan/analog-propagation-casmi-2026-baseline`),
and the mechanism behind "mass-shifted analog propagation" in the lineage map.

Two implementation facts the data forced:

**No mass gate on the analogue search.** Analogues differ in mass by construction, so
gating on the query's mass defeats the purpose. And it would not help anyway: the corpus
spans 133-1109 Da but clusters so tightly that even a ±30 Da window already contains 51 %
of the library, and ±100 Da contains 98 %.

**Weight by `similarity ** p`, not by similarity.** A spectral match at 0.9 is far more
trustworthy than two at 0.45, and a linear weight treats them as equal. The public stack
uses `p = 3`; it is a free parameter to fit, not a constant to inherit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import spectra as SP

DEFAULT_P = 3.0
DEFAULT_TOP_K = 80


def find_analogues(query: pd.DataFrame, reference: pd.DataFrame, top_k: int = DEFAULT_TOP_K,
                   block: int = 256, loss_weight: float = 0.0,
                   exclude_keys: set[str] | None = None,
                   verbose: bool = False) -> dict[str, list[tuple[str, float]]]:
    """Per query molecule, the `top_k` best-matching reference structures.

    Returns `{molecule_id: [(reference inchikey14, similarity), ...]}` sorted best first,
    deduplicated by structure keeping its best-matching spectrum.

    The full query × reference product would be 11,743 × 1.15 M floats — 54 GB — so it is
    computed in row blocks and reduced to the top-k inside each block.
    """
    Qf = SP.bin_spectra(query, "fragment")
    Rf = SP.bin_spectra(reference, "fragment")
    Ql = SP.bin_spectra(query, "loss") if loss_weight else None
    Rl = SP.bin_spectra(reference, "loss") if loss_weight else None

    ref_key = reference.inchikey14.to_numpy(dtype=object)
    q_mol = query.molecule_id.to_numpy(dtype=object)
    molecules = pd.unique(q_mol)

    # Per molecule, accumulate the best similarity seen for each reference structure.
    best: dict[str, dict[str, float]] = {m: {} for m in molecules}
    row_of = {m: np.flatnonzero(q_mol == m) for m in molecules}

    RfT = Rf.T.tocsc()
    RlT = Rl.T.tocsc() if loss_weight else None
    for start in range(0, len(query), block):
        stop = min(start + block, len(query))
        s = (Qf[start:stop] @ RfT).toarray()
        if loss_weight:
            s += loss_weight * (Ql[start:stop] @ RlT).toarray()
        # Top-k per query spectrum, before collapsing to structures: a molecule's
        # several spectra may each nominate different analogues.
        k = min(top_k * 4, s.shape[1])
        idx = np.argpartition(-s, k - 1, axis=1)[:, :k]
        for i in range(stop - start):
            mol = q_mol[start + i]
            d = best[mol]
            for j in idx[i]:
                key = ref_key[j]
                if exclude_keys is not None and key in exclude_keys:
                    continue
                v = float(s[i, j])
                if v > d.get(key, -1.0):
                    d[key] = v
        if verbose and start and (start // block) % 10 == 0:
            print(f"    analogues: {stop:,}/{len(query):,} query spectra", flush=True)

    return {m: sorted(d.items(), key=lambda kv: -kv[1])[:top_k] for m, d in best.items()}


def propagate(analogues: list[tuple[str, float]], candidate_keys: np.ndarray,
              fp_of: dict[str, np.ndarray] | None = None,
              cand_fp: np.ndarray | None = None, analog_fp: np.ndarray | None = None,
              p: float = DEFAULT_P, reduce: str = "max") -> np.ndarray:
    """Score candidates by weighted structural similarity to the analogues.

    `cand_fp` (n_candidates × bits) and `analog_fp` (n_analogues × bits) are binary
    fingerprint matrices; pass them directly to avoid rebuilding per molecule.
    `reduce` is `"max"` (the single best-supported analogue) or `"mean"`.
    """
    if cand_fp is None or analog_fp is None or len(analogues) == 0:
        return np.zeros(len(candidate_keys), dtype=np.float32)

    cf = cand_fp.astype(np.float32)
    af = analog_fp.astype(np.float32)
    inter = cf @ af.T
    tan = inter / (cf.sum(1)[:, None] + af.sum(1)[None, :] - inter + 1e-9)

    w = np.clip(np.array([s for _, s in analogues], dtype=np.float32), 0, None) ** p
    weighted = tan * w[None, :]
    return weighted.max(axis=1) if reduce == "max" else (
        weighted.sum(axis=1) / (w.sum() + 1e-9))
