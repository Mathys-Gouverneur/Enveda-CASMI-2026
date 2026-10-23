"""Loading the competition files, and the facts about them worth encoding once.

The numbers in `TEST_PROFILE` are measured, not assumed — `notebooks/01-...ipynb` shows
the measurement. They live here because every split and every evaluation has to be
calibrated against them, and a constant that drifts from the file it describes is worse
than no constant.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

TRAIN = RAW / "train.parquet"
TEST = RAW / "test.parquet"
SAMPLE_SUBMISSION = RAW / "sample_submission.csv"

#: The two libraries measured on the test set's instrument. Everything else in
#: train.parquet is a different instrument at roughly a tenth of the peak density.
TIMSTOF_LIBS = ("enveda-180", "enveda-np-examples")

#: Measured on test.parquet, 2026-10-23.
TEST_PROFILE = {
    "n_spectra": 1213,
    "n_molecules": 400,
    "spectra_per_molecule": 3.0,
    "instrument": "timsTOF",          # 100% of it
    "median_peaks": 230,
    "mass_min": 246.1,                # neutral mass, derived from precursor + adduct
    "mass_max": 439.1,
    "adduct_mix": {                   # share of spectra
        "[M+H]+": 0.791, "[M-H]-": 0.159, "[M+CH2O2-H]-": 0.026,
        "[M+Na]+": 0.018, "[M+NH4]+": 0.003, "[M+K]+": 0.002,
    },
}

#: Monoisotopic masses, for deriving a neutral mass from a precursor m/z.
_ELEMENT_MASS = {
    "C": 12.0, "H": 1.0078250319, "N": 14.0030740052, "O": 15.9949146221,
    "S": 31.97207069, "P": 30.97376151, "Cl": 34.96885271, "Br": 78.9183376,
    "F": 18.99840320, "I": 126.904473, "Na": 22.98976928, "K": 38.96370649,
    "Si": 27.9769265, "Se": 79.9165218, "B": 11.0093055, "As": 74.9215942,
    "Fe": 55.9349421, "D": 2.0141017780,
}
PROTON = 1.00727646
ELECTRON = 0.00054858

_FORMULA_TOKEN = __import__("re").compile(r"([A-Z][a-z]?)(\d*)")


def formula_mass(formula: str) -> float:
    """Neutral monoisotopic mass of a molecular formula, NaN if an element is unknown."""
    total = 0.0
    for element, count in _FORMULA_TOKEN.findall(formula):
        if element not in _ELEMENT_MASS:
            return float("nan")
        total += _ELEMENT_MASS[element] * (int(count) if count else 1)
    return total


# [2M+Na]+ , [M+CH2O2-H]- , [M-2H2O+H]+ ... multiplicity, body, charge.
_ADDUCT_RE = __import__("re").compile(r"^\[(\d*)M(.*?)\](\d*)([+-])$")
_TERM_RE = __import__("re").compile(r"([+-])(\d*)([A-Za-z0-9]+)")


def parse_adduct(adduct: str) -> tuple[int, float, int, int] | None:
    """`(multiplicity, formula shift, charge count, charge sign)` or None if unparseable.

    Parsed rather than tabulated because a third of the timsTOF spectra are **dimers**
    -- `[2M+Na]+` alone is 188k of them -- and a lookup table of the singly-charged
    monomer adducts silently drops 29% of the reference library. The test set has no
    dimers, so these matter only as references, but a reference is exactly what a
    retrieval baseline is short of.
    """
    m = _ADDUCT_RE.match(adduct.strip())
    if m is None:
        return None
    mult = int(m.group(1) or 1)
    body, charge = m.group(2), int(m.group(3) or 1)
    sign = 1 if m.group(4) == "+" else -1

    shift = 0.0
    if body:
        terms = _TERM_RE.findall(body)
        # Every character of the body must be consumed, or we are guessing.
        if sum(len(s) + len(n) + len(f) for s, n, f in terms) != len(body):
            return None
        for s, n, formula in terms:
            fm = formula_mass(formula)
            if not np.isfinite(fm):
                return None
            shift += (1 if s == "+" else -1) * int(n or 1) * fm
    return mult, shift, charge, sign


def adduct_shift(adduct: str) -> float:
    """m/z shift of a monomer, singly-charged adduct. NaN otherwise."""
    p = parse_adduct(adduct)
    if p is None or p[0] != 1 or p[2] != 1:
        return float("nan")
    _, shift, _, sign = p
    return shift - sign * ELECTRON


#: The adducts present in test.parquet, with their m/z shifts **derived from the parser**
#: so there is one source of truth. An earlier hand-written table here had `[M+NH4]+`
#: wrong by one electron mass: 18.033823 is already the NH4+ *ion* mass, and the table
#: subtracted the electron from it a second time. Literal expected values live in
#: tests/test_casmi.py, where a disagreement fails loudly instead of silently shifting a
#: mass window.
TEST_ADDUCTS = ("[M+H]+", "[M-H]-", "[M+Na]+", "[M+K]+", "[M+NH4]+",
                "[M+CH2O2-H]-", "[M+Cl]-")
ADDUCT_SHIFT = {a: adduct_shift(a) for a in TEST_ADDUCTS}


def neutral_mass(precursor_mz, adduct) -> np.ndarray:
    """Neutral monoisotopic mass of M from the observed precursor m/z and its adduct.

    For `[nM + A]z±`:  m/z = (n·M + shift(A) ∓ z·mₑ) / z,  so
    M = (m/z·z − shift(A) ± z·mₑ) / n.  NaN where the adduct cannot be parsed, rather
    than a guess — a wrong shift moves a molecule into the wrong mass window silently,
    which is worse than dropping it.
    """
    adduct = pd.Series(adduct, dtype=object)
    parsed = {a: parse_adduct(str(a)) for a in adduct.dropna().unique()}

    mult = adduct.map(lambda a: parsed.get(a)[0] if parsed.get(a) else np.nan)
    shift = adduct.map(lambda a: parsed.get(a)[1] if parsed.get(a) else np.nan)
    charge = adduct.map(lambda a: parsed.get(a)[2] if parsed.get(a) else np.nan)
    sign = adduct.map(lambda a: parsed.get(a)[3] if parsed.get(a) else np.nan)

    mz = np.asarray(precursor_mz, dtype=float)
    z = charge.to_numpy(dtype=float)
    return ((mz * z - shift.to_numpy(dtype=float) + sign.to_numpy(dtype=float) * z * ELECTRON)
            / mult.to_numpy(dtype=float))


# ------------------------------------------------------------------------------- loading

SPECTRUM_COLUMNS = ["ms2_mzs", "ms2_normalized_intensities", "precursor_mz"]

META_COLUMNS = [
    "ingest_lib", "inchikey14", "molecular_formula", "normalized_smiles",
    "adduct", "instrument_type", "ionization_mode", "num_peaks", "precursor_mz",
]


def load_train(columns: list[str] | None = None, timstof_only: bool = False,
               filters: list | None = None) -> pd.DataFrame:
    """Read train.parquet. `timstof_only` keeps just the test's instrument.

    Pushed down to the parquet reader rather than filtered afterwards: the file is 3 GB
    and the timsTOF half is 1.15 M rows, so doing it here halves the read.
    """
    f = list(filters or [])
    if timstof_only:
        f.append(("ingest_lib", "in", list(TIMSTOF_LIBS)))
    return pd.read_parquet(TRAIN, columns=columns, filters=f or None)


def load_test(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(TEST, columns=columns)


def load_structures(timstof_only: bool = False) -> pd.DataFrame:
    """One row per unique structure: inchikey14, smiles, formula, neutral mass.

    This is the candidate bank that comes free with the competition data — no external
    database, no licence question. Sorted by mass so a window is a `searchsorted` pair.
    """
    meta = load_train(columns=["inchikey14", "normalized_smiles", "molecular_formula",
                              "ingest_lib"], timstof_only=timstof_only)
    s = meta.drop_duplicates("inchikey14").drop(columns="ingest_lib")
    masses = {f: formula_mass(f) for f in s.molecular_formula.unique()}
    s["neutral_mass"] = s.molecular_formula.map(masses)
    return (s.dropna(subset=["neutral_mass"])
             .sort_values("neutral_mass")
             .reset_index(drop=True))


def spectrum_uid(df: pd.DataFrame) -> pd.Series:
    """A stable per-spectrum identifier, derived from the spectrum's own bytes.

    train.parquet has no spectrum id, and row position is not safe to rely on across
    reads with different column sets or filters. Excluding "these exact spectra" from a
    reference library therefore needs a content hash: identifying rows by
    (inchikey14, precursor_mz) instead silently removes every collision energy of the
    molecule, which turns an `in_library` split into a `held_out` one and quietly
    destroys the measurement.
    """
    import hashlib

    out = []
    for mz, it, prec in zip(df.ms2_mzs.to_numpy(), df.ms2_normalized_intensities.to_numpy(),
                            df.precursor_mz.to_numpy(dtype=float)):
        h = hashlib.blake2b(digest_size=12)
        h.update(np.asarray(mz, dtype=np.float64).tobytes())
        h.update(np.asarray(it, dtype=np.float64).tobytes())
        h.update(np.float64(prec).tobytes())
        out.append(h.hexdigest())
    return pd.Series(out, index=df.index, name="spectrum_uid")


def mass_window(sorted_masses: np.ndarray, target: float, ppm: float = 5.0
                ) -> tuple[int, int]:
    """Half-open index range of `sorted_masses` within ±ppm of `target`."""
    tol = target * ppm / 1e6
    return (int(np.searchsorted(sorted_masses, target - tol, side="left")),
            int(np.searchsorted(sorted_masses, target + tol, side="right")))
