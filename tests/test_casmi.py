"""Invariants that must not drift. Run with `.venv/bin/python -m pytest tests -q`.

These are not exhaustive unit tests; they pin the handful of facts that, if they broke
quietly, would make every downstream number wrong without any error — the metric's
identity rule, the adduct arithmetic, and the spectrum hash that keeps the `in_library`
split from collapsing into `held_out`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from casmi import data as D
from casmi import formula as FO
from casmi import fpnet as N
from casmi import metric as M
from casmi import spectra as SP

CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
CAFFEINE_MASS = 194.080376


# ----------------------------------------------------------------------------- metric

def test_stereochemistry_is_ignored():
    assert M.key14("C[C@@H](N)C(=O)O") == M.key14("C[C@H](N)C(=O)O")


def test_same_molecule_different_smiles_same_key():
    keys = {M.key14(s) for s in ("c1ccccc1O", "OC1=CC=CC=C1", "Oc1ccccc1")}
    assert len(keys) == 1


def test_unparseable_smiles_is_none():
    assert M.key14("this is not a smiles") is None


def test_duplicates_do_not_consume_a_slot():
    """'CCO' and 'OCC' are one ethanol, so the right answer lands at useful rank 2."""
    truth = {"m": M.key14(CAFFEINE)}
    preds = {"m": ["CCO", "OCC", CAFFEINE]}
    assert M.mrr(preds, truth)["m"] == pytest.approx(0.5)


def test_rank_one_and_miss():
    t = {"a": M.key14(CAFFEINE), "b": M.key14(CAFFEINE)}
    p = {"a": [CAFFEINE], "b": ["CCO"] * 30}
    per = M.mrr(p, t)
    assert per["a"] == pytest.approx(1.0)
    assert per["b"] == pytest.approx(0.0)


def test_missing_molecule_scores_zero_not_skipped():
    assert M.score({}, {"a": M.key14(CAFFEINE)}) == pytest.approx(0.0)


def test_submission_caps_at_25():
    row = M.to_submission_rows({"m": [CAFFEINE] * 40})[0]
    assert len(row["smiles"].split(";")) == M.N_SLOTS


def test_only_25_slots_are_scored():
    """25 *distinct* identities ahead of the answer push it out of the scored window.

    They have to be distinct: 25 copies of ethanol collapse to one identity and leave the
    answer at useful rank 2, which is the dedup rule doing its job, not a cap failure.
    """
    t = {"m": M.key14(CAFFEINE)}
    fillers = ["C" * n for n in range(1, 26)]            # methane, ethane, propane, ...
    assert len({M.key14(s) for s in fillers}) == 25
    assert M.mrr({"m": fillers + [CAFFEINE]}, t)["m"] == pytest.approx(0.0)
    # one filler fewer, and the answer is scored at rank 25
    assert M.mrr({"m": fillers[:24] + [CAFFEINE]}, t)["m"] == pytest.approx(1 / 25)


# ------------------------------------------------------------------------------ masses

@pytest.mark.parametrize("formula,expected", [
    ("C8H10N4O2", CAFFEINE_MASS),             # caffeine
    ("C9H8O4", 180.042259),                   # aspirin
    ("H2O", 18.010565),
])
def test_formula_mass(formula, expected):
    assert D.formula_mass(formula) == pytest.approx(expected, abs=1e-4)


def test_unknown_element_gives_nan():
    assert np.isnan(D.formula_mass("C6XzH6"))


@pytest.mark.parametrize("adduct,shift", [
    ("[M+H]+", 1.00728),
    ("[M-H]-", -1.00728),
    ("[M+Na]+", 22.98922),
    ("[M+K]+", 38.96316),
    ("[M+NH4]+", 18.03383),                   # the NH4+ ion mass, not NH4 minus 2 electrons
    ("[M+Cl]-", 34.96940),
    ("[M+CH2O2-H]-", 44.99820),
])
def test_adduct_shift_literals(adduct, shift):
    assert D.adduct_shift(adduct) == pytest.approx(shift, abs=1e-4)


@pytest.mark.parametrize("adduct", [
    "[M+H]+", "[M-H]-", "[M+Na]+", "[M+NH4]+", "[M+Cl]-", "[M+CH2O2-H]-",
    "[2M+H]+", "[2M+Na]+", "[2M-H]-", "[2M+CH2O2-H]-",
    "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Br]-", "[M+C2H4O2-H]-",
])
def test_neutral_mass_round_trips(adduct):
    """Forward from a known neutral mass, back through neutral_mass: must agree."""
    n, shift, z, sign = D.parse_adduct(adduct)
    mz = (n * CAFFEINE_MASS + shift - sign * z * D.ELECTRON) / z
    got = float(D.neutral_mass([mz], [adduct])[0])
    assert got == pytest.approx(CAFFEINE_MASS, rel=1e-9)


def test_dimers_are_parsed():
    """29% of timsTOF reference spectra are dimers; dropping them is not acceptable."""
    assert D.parse_adduct("[2M+Na]+")[0] == 2


def test_unparseable_adduct_is_nan_not_a_guess():
    assert D.parse_adduct("[M+Unobtainium]+") is None
    assert np.isnan(float(D.neutral_mass([300.0], ["nonsense"])[0]))


def test_mass_window_is_symmetric_and_inclusive():
    masses = np.array([99.0, 299.999, 300.0, 300.001, 701.0])
    lo, hi = D.mass_window(masses, 300.0, ppm=10.0)   # ±3 mDa at 300 Da
    assert list(masses[lo:hi]) == [299.999, 300.0, 300.001]


# ---------------------------------------------------------------------------- spectra

def _toy_spectra():
    return pd.DataFrame({
        "ms2_mzs": [np.array([100.0, 150.0]), np.array([100.0, 150.0]),
                    np.array([200.0, 250.0])],
        "ms2_normalized_intensities": [np.array([1.0, 0.5]), np.array([1.0, 0.5]),
                                       np.array([1.0, 1.0])],
        "precursor_mz": [300.0, 300.0, 300.0],
    })


def test_identical_spectra_have_cosine_one():
    X = SP.bin_spectra(_toy_spectra(), "fragment")
    s = SP.cosine(X, X)
    assert s[0, 1] == pytest.approx(1.0, abs=1e-5)


def test_disjoint_spectra_have_cosine_zero():
    X = SP.bin_spectra(_toy_spectra(), "fragment")
    assert SP.cosine(X, X)[0, 2] == pytest.approx(0.0, abs=1e-6)


def test_loss_view_differs_from_fragment_view():
    df = _toy_spectra()
    frag = SP.bin_spectra(df, "fragment")
    loss = SP.bin_spectra(df, "loss")
    assert (frag[0] != loss[0]).nnz > 0


def test_spectrum_uid_is_content_addressed():
    df = _toy_spectra()
    uid = D.spectrum_uid(df)
    assert uid.iloc[0] == uid.iloc[1]        # identical spectra, identical id
    assert uid.iloc[0] != uid.iloc[2]


# ---------------------------------------------------------------------------- formula

def test_a_formula_contains_itself_and_its_parts():
    """The grid implementation this replaced failed exactly here.

    Rounding each element's mass to a 1 mDa grid independently let the error accumulate
    along the sum: caffeine's own formula landed 2 mDa from where its exact mass puts it,
    so the formula did not contain itself. Exact float enumeration cannot drift.
    """
    reach = FO.reachable_masses("C8H10N4O2")
    for sub in ("C8H10N4O2", "C7H8N4O2", "C6H5N3O", "CH3", "C", "H"):
        m = D.formula_mass(sub)
        i = np.searchsorted(reach, m)
        nearest = min(abs(m - reach[max(i - 1, 0)]), abs(m - reach[min(i, len(reach) - 1)]))
        assert nearest < 1e-6, f"{sub} should be reachable inside caffeine"


def test_a_formula_does_not_contain_more_than_itself():
    reach = FO.reachable_masses("C8H10N4O2")
    for sub in ("C9H10N4O2", "C8H10N5O2", "C8H11N4O2"):
        m = D.formula_mass(sub)
        i = np.searchsorted(reach, m)
        nearest = min(abs(m - reach[max(i - 1, 0)]), abs(m - reach[min(i, len(reach) - 1)]))
        assert nearest > 1e-6, f"{sub} must not be reachable inside caffeine"


def test_enumeration_size_is_the_product_of_counts():
    assert len(FO.reachable_masses("C8H10N4O2")) == 9 * 11 * 5 * 3


def test_unknown_element_formula_is_none():
    assert FO.reachable_masses("C6XzH6") is None


def test_chance_rate_grows_with_formula_size():
    """The bias the adjustment exists to remove: bigger formulas match more by luck."""
    small = FO.chance_match_rate("C8H10N4O2", 200.0)
    big = FO.chance_match_rate("C25H40O10", 500.0)
    assert 0.0 < small < big <= 1.0


def test_adjusted_score_prefers_the_true_formula():
    """A caffeine spectrum must score caffeine's formula above a same-mass decoy.

    C7H14O6 sits in caffeine's own ±10 ppm mass window, so this is the real
    discrimination the gate has to make.
    """
    tr = D.load_train(
        columns=["molecular_formula", "adduct", "precursor_mz", "num_peaks",
                 "ms2_mzs", "ms2_normalized_intensities"],
        filters=[("molecular_formula", "==", "C8H10N4O2"), ("adduct", "==", "[M+H]+")])
    if tr.empty:                                   # no data on this machine
        pytest.skip("train.parquet not available")
    s = tr.sort_values("num_peaks", ascending=False).iloc[0]
    ranked = FO.rank_formulas(s.ms2_mzs, s.ms2_normalized_intensities, s.precursor_mz,
                              ["C8H10N4O2", "C7H14O6"])
    assert ranked[0][0] == "C8H10N4O2"


def test_logsumexp_pooling_is_a_soft_maximum_from_above():
    """log-sum-exp brackets the maximum from above, by at most log(n)/beta.

    Not from below: `lse = max + log(sum exp(beta*(s-max)))/beta >= max` always. Getting
    that backwards is what the first version of this test asserted.
    """
    s = np.array([[0.9, 0.1], [0.2, 0.1], [0.1, 0.1]], dtype=float)
    lse = SP.pool_logsumexp(s)
    mx, mean = s.max(axis=0), s.mean(axis=0)
    slack = np.log(s.shape[0]) / SP.LSE_BETA
    assert np.all(lse >= mx - 1e-9)
    assert np.all(lse <= mx + slack + 1e-9)
    assert np.all(lse > mean)
    # The agreeing column (all three equal) takes the full slack; the peaked one barely any.
    assert (lse[1] - mx[1]) > (lse[0] - mx[0])


def test_best_per_molecule_dedups_and_caps():
    sim = np.array([[0.9, 0.8, 0.7]], dtype=float)
    out = SP.best_per_molecule(sim, np.array(["m"], dtype=object),
                               np.array(["A", "A", "B"], dtype=object), n_keep=5)
    assert [k for k, _ in out["m"]] == ["A", "B"]


# ------------------------------------------------------------------------------ fpnet

def _toy_batch(n=3, p=6):
    import torch
    mz = torch.rand(n, p) * 400 + 50
    it = torch.rand(n, p)
    pad = torch.zeros(n, p, dtype=torch.bool)
    pad[-1, p // 2:] = True                       # last row partly padded
    return mz, it, pad


def test_collate_truncates_by_intensity_not_by_mz():
    """Keeping the first peaks by m/z would discard the high-mass fragments."""
    mzs = [np.array([100.0, 150.0, 200.0, 400.0])]
    its = [np.array([0.1, 1.0, 0.5, 0.2])]        # the 100 Da peak is the weakest
    mz, it, pad = N.collate_spectra(mzs, its, max_peaks=3, precursors=np.array([500.0]))
    kept = mz[0].numpy()[1:]                      # position 0 is the precursor
    assert 100.0 not in kept
    assert it[0, 0] == 2.0 and not pad[0, 0]      # precursor flagged out of range


def test_fpnet_is_permutation_invariant():
    """A spectrum is a set; peak storage order must not change the prediction."""
    import torch
    cfg = N.FPNetConfig(n_bits=64, d_model=64, n_layers=2, n_heads=4, d_ff=128)
    m = N.FPNet(cfg).eval()
    mz, it, pad = _toy_batch()
    perm = torch.randperm(mz.shape[1])
    with torch.no_grad():
        a = m(mz, it, pad)
        b = m(mz[:, perm], it[:, perm], pad[:, perm])
    assert float((a[:-1] - b[:-1]).abs().max()) < 1e-4


def test_fpnet_v2_conditioning_changes_the_output():
    """If the instrument embedding did nothing, the whole v2 experiment is void."""
    import torch
    cfg = N.FPNetV2Config(n_bits=64, d_model=64, n_layers=2, n_heads=4, d_ff=128)
    m = N.FPNetV2(cfg).eval()
    mz, it, pad = _toy_batch()
    prec = torch.full((3,), 400.0)
    ad, ce = torch.ones(3, dtype=torch.long), torch.full((3,), 40.0)
    pres, pol = torch.ones(3), torch.ones(3)
    with torch.no_grad():
        a = m(mz, it, pad, prec, ad, torch.ones(3, dtype=torch.long), ce, pres, pol)
        b = m(mz, it, pad, prec, ad, torch.full((3,), 5, dtype=torch.long), ce, pres, pol)
    assert float((a - b).abs().max()) > 1e-3


def test_vocab_reserves_zero_for_out_of_vocabulary():
    v = N.build_vocab(["[M+H]+"] * 10 + ["[M-H]-"] * 5 + ["[M+Na]+"], size=3)
    assert 0 not in v.values()
    assert list(N.encode_vocab(["[M+H]+", "[M-H]-", "[M+Na]+", "unseen"], v)) == [1, 2, 0, 0]


def test_zero_collision_energy_is_not_missing_collision_energy():
    """0 eV is a real reading meaning no fragmentation; absent is a different thing."""
    ce, present = N.mean_collision_energy([[20.0, 40.0], [], None, [0.0]])
    assert ce[0] == pytest.approx(30.0) and present[0] == 1.0
    assert present[1] == 0.0 and present[2] == 0.0
    assert ce[3] == 0.0 and present[3] == 1.0


def test_pos_weight_is_capped():
    """Uncapped, a bit in 0.5% of molecules gets weight 200 and drowns every other bit."""
    Y = np.zeros((1000, 3), dtype=np.uint8)
    Y[:500, 0] = 1          # 50%  -> weight 1
    Y[:1, 1] = 1            # 0.1% -> weight ~1000, must clamp
    Y[:, 2] = 1             # 100% -> weight 0
    w = N.pos_weight_from(Y, cap=50.0).numpy()
    assert w[0] == pytest.approx(1.0, abs=0.01)
    assert w[1] == pytest.approx(50.0)
    assert w[2] == pytest.approx(0.0, abs=1e-6)
