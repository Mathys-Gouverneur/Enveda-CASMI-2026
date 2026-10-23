"""Validation splits that resemble the test set instead of the training set.

Four rules, each of which the obvious split gets wrong:

1. **Split by molecule, never by spectrum.** A molecule carries 6 spectra in the median
   and up to 1,468. A random split over spectra puts the same structure on both sides and
   the resulting number is meaningless.
2. **Restrict to timsTOF.** The test set is 100% timsTOF and only `enveda-180` and
   `enveda-np-examples` carry that instrument. Validating on public-library spectra
   measures a different problem.
3. **Match the test profile.** 3 spectra per molecule, neutral mass 246-439 Da, and the
   test's adduct mix. A validation molecule with 40 spectra at every collision energy is
   an easier question than anything the test asks.
4. **Say whether the answer is in the reference library.** This is the axis the public
   record never separates, and it changes the problem completely:

   - `in_library`: other spectra of the same molecule stay in the reference library, so
     spectral matching can win outright.
   - `held_out`: every spectrum of the validation molecules is removed from the reference
     library, so the answer can only come from a structure database plus a model that
     scores candidates it has never seen a spectrum for.

   The real test is somewhere between the two, and we do not know where. Reporting both
   brackets it; reporting only the first flatters every method that leans on retrieval.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data as D


@dataclass
class Split:
    """Molecule-level split with its reference library already chosen."""

    query: pd.DataFrame          #: spectra to identify, test-shaped
    reference: pd.DataFrame      #: spectra usable as a labelled library
    regime: str                  #: "in_library" | "held_out"
    truth: dict[str, str] = field(default_factory=dict)  #: molecule_id -> inchikey14

    def __repr__(self) -> str:
        return (f"Split({self.regime}: {self.query.molecule_id.nunique()} molecules, "
                f"{len(self.query)} query spectra, {len(self.reference):,} reference spectra)")


def _subsample_to_profile(df: pd.DataFrame, rng: np.random.Generator,
                          per_molecule: int) -> pd.DataFrame:
    """Keep at most `per_molecule` spectra per molecule, preferring distinct energies.

    The test gives 3 spectra per molecule at different collision energies. Taking 3 at
    random from a molecule that has 40 would often draw near-duplicates and overstate how
    much multi-spectrum fusion helps, so distinct energies come first.
    """
    out = []
    for _, g in df.groupby("inchikey14", sort=False):
        if len(g) <= per_molecule:
            out.append(g)
            continue
        # one spectrum per collision energy first, then fill at random
        by_energy = g.drop_duplicates("collision_energy_orig")
        if len(by_energy) >= per_molecule:
            pick = by_energy.sample(n=per_molecule, random_state=int(rng.integers(1 << 31)))
        else:
            rest = g.drop(by_energy.index)
            pick = pd.concat([by_energy, rest.sample(n=per_molecule - len(by_energy),
                                                     random_state=int(rng.integers(1 << 31)))])
        out.append(pick)
    return pd.concat(out)


def make_splits(spectra: pd.DataFrame, n_val_molecules: int = 4000, seed: int = 0,
                match_mass_range: bool = True, match_adducts: bool = True,
                per_molecule: int | None = None) -> dict[str, Split]:
    """Build the `in_library` and `held_out` splits from timsTOF spectra.

    `spectra` must carry at least inchikey14, adduct, precursor_mz,
    collision_energy_orig and the two spectrum columns. `n_val_molecules` is deliberately
    far above the test's 400: a bigger validation set is the whole point of using the
    half of the corpus everyone else discards, and 400-molecule draws for the noise floor
    are taken from it afterwards by `lb_sized_draws`.
    """
    rng = np.random.default_rng(seed)
    df = spectra.copy()

    if "neutral_mass" not in df:
        df["neutral_mass"] = D.neutral_mass(df.precursor_mz, df.adduct)

    # Candidate validation molecules: those that could plausibly be a test question.
    ok = df.neutral_mass.notna()
    if match_mass_range:
        lo, hi = D.TEST_PROFILE["mass_min"], D.TEST_PROFILE["mass_max"]
        ok &= df.neutral_mass.between(lo, hi)
    if match_adducts:
        ok &= df.adduct.isin(D.TEST_PROFILE["adduct_mix"])
    eligible = df[ok]

    # A molecule needs >= 2 spectra to appear in the in_library regime at all: one to ask
    # with, one to leave in the library. Molecules with a single spectrum would make the
    # two regimes identical and silently bias the comparison.
    counts = eligible.inchikey14.value_counts()
    pool = counts[counts >= 2].index.to_numpy()
    if len(pool) < n_val_molecules:
        raise ValueError(f"only {len(pool)} eligible molecules, asked for {n_val_molecules}")
    val_mols = rng.choice(pool, size=n_val_molecules, replace=False)
    val_set = set(val_mols)

    per_molecule = per_molecule or int(round(D.TEST_PROFILE["spectra_per_molecule"]))
    val_spectra = eligible[eligible.inchikey14.isin(val_set)]
    query = _subsample_to_profile(val_spectra, rng, per_molecule).copy()

    # A synthetic molecule_id, so the split is scored by exactly the code that scores a
    # submission: the real test identifies molecules, not spectra.
    ids = {k: f"v_{i:05d}" for i, k in enumerate(sorted(val_set))}
    query["molecule_id"] = query.inchikey14.map(ids)
    truth = {ids[k]: k for k in val_set}

    queried_index = set(query.index)
    # in_library: the molecule's OTHER spectra remain available as references.
    ref_in = df.drop(index=[i for i in queried_index if i in df.index])
    # held_out: no spectrum of any validation molecule survives.
    ref_held = df[~df.inchikey14.isin(val_set)]

    return {
        "in_library": Split(query, ref_in, "in_library", truth),
        "held_out": Split(query, ref_held, "held_out", truth),
    }


def lb_sized_draws(truth: dict[str, str], n: int = 400, draws: int = 500,
                   seed: int = 0) -> list[list[str]]:
    """Random subsets of validation molecules the size of the real test set.

    The leaderboard scores 400 molecules. Re-scoring a prediction on many 400-molecule
    draws of our own validation set gives the spread the leaderboard itself has, which is
    the only honest way to say whether a difference between two methods is real.
    """
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(truth))
    if len(ids) < n:
        raise ValueError(f"{len(ids)} molecules available, need {n} per draw")
    return [list(rng.choice(ids, size=n, replace=False)) for _ in range(draws)]
