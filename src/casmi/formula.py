"""Molecular formula identification from the fragments, and its use as a gate.

Why this earns its place, measured on our validation split at ±10 ppm:

| pool | median candidates | MRR at random |
|---|---|---|
| mass window | 77 | 0.088 |
| mass window ∩ true formula | **34** | **0.205** |

Knowing the formula more than doubles the floor. It does **not** solve the problem — 34
isomers remain, which is the same-formula separation that forward simulators exist for —
but it is a free multiplier on whatever ranker sits downstream, and it needs no external
database, no model weights and no licence.

The signal: **every fragment of a molecule is a sub-formula of that molecule.** A peak at
152.0570 Da can only come from a candidate formula that contains some sub-formula of that
exact mass. Score each candidate formula by how much of the spectrum's intensity it can
explain, and the wrong formulas in the mass window lose. This is the core of SIRIUS and
of MetFrag-style scoring, implemented directly rather than imported, so nothing here
carries a licence.

For a formula with element counts (nC, nH, nN, nO, …), the achievable sub-formula masses
are all sums `i·mC + j·mH + …` with `0 ≤ i ≤ nC`. They are enumerated **exactly, in
floating point**, and matched with `searchsorted`.

The first attempt marked reachable masses on a 1 mDa grid instead, which is faster but
wrong: rounding each element's mass to the grid independently lets the error accumulate
along the sum. Caffeine's own formula landed 2 mDa away from where its exact mass says it
should be — larger than the instrument tolerance we match within, and growing with
molecule size, so bigger molecules drifted further. Exact enumeration costs
`prod(nᵢ + 1)` floats, which for this competition's 246-439 Da molecules is at most a few
hundred thousand, and buys arithmetic that cannot drift.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .data import _ELEMENT_MASS, _FORMULA_TOKEN, PROTON

#: Refuse to enumerate beyond this many sub-formula masses. Never reached in this
#: competition's mass range; a guard against a pathological formula, not a tuning knob.
MAX_SUBFORMULAS = 20_000_000


def parse_formula(formula: str) -> dict[str, int] | None:
    """Element -> count, or None if an element is unknown."""
    out: dict[str, int] = {}
    for element, count in _FORMULA_TOKEN.findall(formula):
        if element not in _ELEMENT_MASS:
            return None
        out[element] = out.get(element, 0) + (int(count) if count else 1)
    return out or None


@lru_cache(maxsize=4096)
def reachable_masses(formula: str) -> np.ndarray | None:
    """Sorted array of every sub-formula mass inside `formula`, exactly.

    Cached on the formula string: a mass window holds a handful of distinct formulas and
    the same ones recur across the validation set, so this runs far fewer times than it
    is called. The cache is sized for formulas, not for molecules.
    """
    counts = parse_formula(formula)
    if counts is None:
        return None

    size = 1
    for n in counts.values():
        size *= n + 1
    if size > MAX_SUBFORMULAS:
        return None

    masses = np.zeros(1, dtype=np.float64)
    # Heaviest element first: the running array stays sorted-ish and uniques collapse
    # earlier, which keeps the intermediate allocations smaller.
    for element, n in sorted(counts.items(), key=lambda kv: -_ELEMENT_MASS[kv[0]]):
        add = np.arange(n + 1, dtype=np.float64) * _ELEMENT_MASS[element]
        masses = np.unique((masses[:, None] + add[None, :]).ravel())
    return masses


def explained_intensity(mzs, intensities, formula: str, precursor_mz: float,
                        ppm: float = 20.0, allow_h_shift: int = 1) -> float:
    """Fraction of a spectrum's intensity explainable as sub-formulas of `formula`.

    `allow_h_shift` permits ±n hydrogens per fragment: fragmentation routinely transfers
    hydrogens, and an ion is a sub-formula plus or minus a proton depending on polarity,
    so demanding an exact neutral sub-formula match rejects most real fragments. One
    hydrogen is the usual allowance.
    """
    reach = reachable_masses(formula)
    if reach is None or reach.size == 0:
        return 0.0

    mzs = np.asarray(mzs, dtype=float)
    its = np.asarray(intensities, dtype=float)
    keep = (mzs > 0) & (mzs <= precursor_mz + 0.5)
    mzs, its = mzs[keep], its[keep]
    if its.size == 0 or its.sum() <= 0:
        return 0.0

    # Work in neutral-fragment space: an observed ion carries a proton one way or another.
    neutral = mzs - PROTON
    tol = np.maximum(neutral * ppm / 1e6, 0.002)
    h = _ELEMENT_MASS["H"]

    matched = np.zeros(len(neutral), dtype=bool)
    for shift in range(-allow_h_shift, allow_h_shift + 1):
        target = neutral + shift * h
        # Nearest reachable sub-formula mass, vectorised over all peaks at once.
        idx = np.searchsorted(reach, target)
        left = reach[np.clip(idx - 1, 0, len(reach) - 1)]
        right = reach[np.clip(idx, 0, len(reach) - 1)]
        nearest = np.where(np.abs(target - left) <= np.abs(right - target), left, right)
        matched |= np.abs(target - nearest) <= tol
    return float(its[matched].sum() / its.sum())


@lru_cache(maxsize=4096)
def chance_match_rate(formula: str, upper_mass: float, ppm: float = 20.0,
                      allow_h_shift: int = 1) -> float:
    """Fraction of the mass axis that `formula` would match *by chance*.

    The correction that makes this signal usable at all. Raw explained intensity is
    biased towards whichever candidate formula has the most atoms: caffeine admits 1,485
    sub-formula masses across 194 Da — one every 0.13 Da — while a 400 Da formula admits
    11,718, one every 0.034 Da. Against a ±4 mDa tolerance and a ±1 H allowance, the
    larger formula explains most of any spectrum for free, so the unadjusted score ranks
    formulas by density rather than by correctness. Measured: 32 % top-1 accuracy and a
    *negative* gain on the real task.

    This computes the hit rate exactly — the measure of the union of the tolerance
    intervals around every reachable mass (and its ±H shifts), over the mass range — so
    the adjusted score is "explained more than its size alone would predict".
    """
    reach = reachable_masses(formula)
    if reach is None or reach.size == 0 or upper_mass <= 0:
        return 1.0

    h = _ELEMENT_MASS["H"]
    centres = np.concatenate([reach + s * h for s in range(-allow_h_shift, allow_h_shift + 1)])
    centres = centres[(centres >= 0) & (centres <= upper_mass)]
    if centres.size == 0:
        return 0.0
    centres.sort()

    tol = np.maximum(centres * ppm / 1e6, 0.002)
    lo, hi = centres - tol, centres + tol
    # Merge overlapping intervals so overlap is not counted twice.
    covered, cur_lo, cur_hi = 0.0, lo[0], hi[0]
    for a, b in zip(lo[1:], hi[1:]):
        if a <= cur_hi:
            cur_hi = max(cur_hi, b)
        else:
            covered += cur_hi - cur_lo
            cur_lo, cur_hi = a, b
    covered += cur_hi - cur_lo
    return float(min(covered / upper_mass, 1.0))


def adjusted_score(mzs, intensities, formula: str, precursor_mz: float,
                   ppm: float = 20.0, allow_h_shift: int = 1) -> float:
    """Explained intensity above what the formula's size alone would explain.

    `explained - chance`, which is the quantity that discriminates. Positive means the
    spectrum's peaks fall on this formula's sub-formulas more than coincidence accounts
    for.
    """
    obs = explained_intensity(mzs, intensities, formula, precursor_mz, ppm, allow_h_shift)
    exp = chance_match_rate(formula, float(precursor_mz), ppm, allow_h_shift)
    return obs - exp


def rank_formulas(mzs, intensities, precursor_mz: float, candidates: list[str],
                  ppm: float = 20.0, adjusted: bool = True) -> list[tuple[str, float]]:
    """Candidate formulas, best first. `adjusted=False` gives the raw, biased score."""
    f_score = adjusted_score if adjusted else explained_intensity
    scored = [(f, f_score(mzs, intensities, f, precursor_mz, ppm)) for f in candidates]
    return sorted(scored, key=lambda t: -t[1])


def score_formulas_for_molecule(spectra: list[tuple], candidates: list[str],
                                ppm: float = 20.0, adjusted: bool = True
                                ) -> dict[str, float]:
    """Pool several spectra of one molecule into one score per candidate formula.

    The mean over spectra, not the maximum: a wrong formula can happen to explain one
    low-information spectrum well, but explaining *all* of a molecule's collision
    energies is much harder to do by accident. This is the opposite choice from spectral
    retrieval, where the maximum is right because references are recorded at one energy.
    """
    f_score = adjusted_score if adjusted else explained_intensity
    out: dict[str, float] = {}
    for f in candidates:
        vals = [f_score(mz, it, f, prec, ppm) for mz, it, prec in spectra]
        out[f] = float(np.mean(vals)) if vals else 0.0
    return out
