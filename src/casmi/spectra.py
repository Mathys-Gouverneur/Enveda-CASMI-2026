"""Spectral representation and similarity.

Binned sparse vectors rather than peak-list alignment. A proper modified cosine walks
two sorted peak lists with a ppm tolerance and solves an assignment problem; at
1,000 x 100,000 comparisons that is far too slow in Python, while a 0.01 Da bin plus a
sparse matmul does the same job for a rounding error and runs in seconds.

Two views of the same spectrum, both cheap:

- **fragment view**, binned on m/z — what the fragments weigh.
- **neutral-loss view**, binned on `precursor - m/z` — what was *lost* to make each
  fragment. This is the trick that makes an analogue match: a molecule and its methylated
  derivative have fragments at different m/z but the same losses, so the fragment view
  misses the relation and the loss view catches it. It is a cheap stand-in for the
  modified cosine the public stack uses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

BIN_WIDTH = 0.01
MAX_MZ = 1000.0


def _l2_normalise(X: sp.csr_matrix) -> sp.csr_matrix:
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A.ravel()
    norms[norms == 0] = 1.0
    return sp.diags(1.0 / norms) @ X


def bin_spectra(df: pd.DataFrame, view: str = "fragment", bin_width: float = BIN_WIDTH,
                max_mz: float = MAX_MZ, sqrt_intensity: bool = True) -> sp.csr_matrix:
    """Spectra -> L2-normalised sparse matrix, one column per bin.

    `sqrt_intensity` dampens the dominant peaks. Without it a single base peak carries
    most of the vector norm and the cosine stops seeing the rest of the pattern; it is
    the standard weighting in the field for exactly that reason.
    """
    if view not in ("fragment", "loss"):
        raise ValueError(f"view must be 'fragment' or 'loss', got {view!r}")

    n_bins = int(max_mz / bin_width)
    rows, cols, vals = [], [], []

    mz_all = df.ms2_mzs.to_numpy()
    it_all = df.ms2_normalized_intensities.to_numpy()
    prec_all = df.precursor_mz.to_numpy(dtype=float)

    for i, (mz, it, prec) in enumerate(zip(mz_all, it_all, prec_all)):
        mz = np.asarray(mz, dtype=float)
        it = np.asarray(it, dtype=float)
        if view == "loss":
            # The precursor peak itself becomes loss 0; fragments heavier than the
            # precursor are noise or isotope artefacts and are dropped.
            mz = prec - mz
            keep = (mz >= 0) & (mz < max_mz)
        else:
            keep = (mz >= 0) & (mz < max_mz)
        if not keep.any():
            continue
        v = it[keep]
        rows.append(np.full(int(keep.sum()), i, dtype=np.int32))
        cols.append((mz[keep] / bin_width).astype(np.int32))
        vals.append(np.sqrt(v) if sqrt_intensity else v)

    if not rows:
        return sp.csr_matrix((len(df), n_bins))

    X = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(df), n_bins), dtype=np.float32)
    # Duplicate bins (two peaks in one bucket) must add, which csr_matrix does on build.
    X.sum_duplicates()
    return _l2_normalise(X)


def cosine(Q: sp.csr_matrix, R: sp.csr_matrix, block: int = 2048) -> np.ndarray:
    """Dense cosine matrix between query and reference spectra, in row blocks.

    Blocked because the product is dense: 5,000 x 200,000 float32 is 4 GB, and the
    caller usually wants to reduce it per block rather than hold it.
    """
    out = np.empty((Q.shape[0], R.shape[0]), dtype=np.float32)
    Rt = R.T.tocsc()
    for start in range(0, Q.shape[0], block):
        stop = min(start + block, Q.shape[0])
        out[start:stop] = (Q[start:stop] @ Rt).toarray()
    return out


def combined_similarity(query: pd.DataFrame, reference: pd.DataFrame,
                        loss_weight: float = 0.0, **kw) -> np.ndarray:
    """Fragment cosine plus `loss_weight` x neutral-loss cosine.

    **Default 0, measured.** `scripts/tune_retrieval.py` swept this on 4,000 validation
    molecules: MRR falls monotonically as the weight rises, for every fusion rule, and
    the earlier placeholder of 0.5 cost 0.0036.

    That verdict is scoped to same-molecule retrieval, which is what the `in_library`
    split measures — there the fragment cosine already matches the answer almost exactly
    and the loss view only adds noise. Neutral losses exist to match *analogues*, which
    is the `held_out` regime where retrieval scores 0 and the experiment is blind. Refit
    this once a ranker scores in that regime.
    """
    s = cosine(bin_spectra(query, "fragment", **kw), bin_spectra(reference, "fragment", **kw))
    if loss_weight:
        s = s + loss_weight * cosine(bin_spectra(query, "loss", **kw),
                                     bin_spectra(reference, "loss", **kw))
    return s


#: Sharpness of the log-sum-exp pooling. Soft enough to credit agreement across a
#: molecule's spectra, sharp enough to stay dominated by its best match.
LSE_BETA = 8.0


def pool_logsumexp(sim: np.ndarray) -> np.ndarray:
    """Soft maximum over a molecule's spectra, the measured-best fusion rule."""
    return np.log(np.exp(sim * LSE_BETA).sum(axis=0)) / LSE_BETA


def best_per_molecule(sim: np.ndarray, query_molecule: np.ndarray,
                      reference_key: np.ndarray, n_keep: int = 25,
                      fusion=pool_logsumexp) -> dict[str, list[tuple[str, float]]]:
    """Pool a molecule's spectra into one ranked candidate list.

    Scoring is per molecule, so a molecule's several spectra have to vote together.

    The default is **log-sum-exp**, not the maximum, because that is what measured best:
    `scripts/tune_retrieval.py` on 4,000 molecules gives 0.6562 against 0.6509 for max,
    0.6456 for mean/sum and 0.6122 for reciprocal-rank fusion. The reasoning for taking a
    maximum was sound — a reference recorded at one collision energy matches one query
    spectrum well and the others poorly, so averaging punishes the right answer — but a
    soft maximum keeps that property while still crediting a candidate that matches
    several of the molecule's spectra.
    """
    results: dict[str, list[tuple[str, float]]] = {}
    for mol in pd.unique(query_molecule):
        block = sim[query_molecule == mol]
        best = fusion(block)
        # Collapse references that are the same structure, keeping its best score.
        order = np.argsort(best)[::-1]
        seen: dict[str, float] = {}
        for j in order:
            k = reference_key[j]
            if k not in seen:
                seen[k] = float(best[j])
                if len(seen) == n_keep:
                    break
        results[mol] = list(seen.items())
    return results
