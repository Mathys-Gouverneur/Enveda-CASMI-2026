"""Molecular fingerprints: the pivot that lets us score a candidate with no spectrum.

The whole point. A spectral library can only rank structures it has a spectrum for, and
`measurement.md` shows the test is mostly the other case — structure in the bank, spectrum
absent. A fingerprint is computable from *either* side:

- from a **structure**, exactly, by RDKit, for all 275,810 candidates, free;
- from a **spectrum**, approximately, by a model (`casmi.fpnet`).

Rank candidates by the agreement between the predicted fingerprint and each candidate's
exact one, and the spectral library stops being the limit.

Which bits to predict is the design decision that matters. Predicting 4,096 Morgan bits
is mostly wasted: a bit set in 2 of 183,191 molecules cannot be learned and contributes
nothing to a Tanimoto. `select_bits` keeps bits with enough variance to be learnable and
enough information to discriminate — the same idea as the public `fp_bits.npy`, computed
ourselves because theirs is CC-BY-NC-SA and derived from FPNet checkpoints.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import MACCSkeys

RDLogger.DisableLog("rdApp.*")

#: Three complementary fingerprint families, concatenated. Morgan captures local
#: environments at two radii, the RDKit path fingerprint captures connectivity, MACCS is
#: 167 hand-curated substructure questions. They disagree usefully: a spectrum constrains
#: substructure presence (MACCS-like) better than it constrains topology.
_MORGAN2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_MORGAN3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
_RDKIT = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=6, fpSize=2048)

FP_BLOCKS = (("morgan2", 2048), ("morgan3", 2048), ("rdkit", 2048), ("maccs", 167))
FP_WIDTH = sum(n for _, n in FP_BLOCKS)       # 6,311


def fingerprint(mol_or_smiles) -> np.ndarray | None:
    """Full concatenated fingerprint as a uint8 vector of length FP_WIDTH."""
    mol = (Chem.MolFromSmiles(mol_or_smiles)
           if isinstance(mol_or_smiles, str) else mol_or_smiles)
    if mol is None:
        return None
    out = np.zeros(FP_WIDTH, dtype=np.uint8)
    at = 0
    for gen in (_MORGAN2, _MORGAN3, _RDKIT):
        arr = np.zeros(2048, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
        out[at:at + 2048] = arr
        at += 2048
    # MACCS keys are 1-indexed over a 168-bit vector whose bit 0 is unused, so the 167
    # real keys are shifted down by one to sit flush in the block.
    maccs = np.zeros(167, dtype=np.uint8)
    for b in MACCSkeys.GenMACCSKeys(mol).GetOnBits():
        if 0 < b <= 167:
            maccs[b - 1] = 1
    out[at:at + 167] = maccs
    return out


def fingerprint_matrix(smiles, n_jobs: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Fingerprints for many SMILES. Returns (matrix, ok mask).

    Rows where RDKit failed are zero and flagged false in the mask, rather than dropped,
    so the caller's indexing stays aligned with its own structure table.
    """
    smiles = list(smiles)
    X = np.zeros((len(smiles), FP_WIDTH), dtype=np.uint8)
    ok = np.zeros(len(smiles), dtype=bool)
    for i, s in enumerate(smiles):
        fp = fingerprint(s)
        if fp is not None:
            X[i] = fp
            ok[i] = True
    return X, ok


def select_bits(X: np.ndarray, min_freq: float = 0.005, max_freq: float = 0.995
                ) -> np.ndarray:
    """Indices of bits worth predicting.

    A bit set in almost no molecule is unlearnable; a bit set in almost all carries no
    discriminative information and inflates every Tanimoto towards 1. Both ends are
    dropped. Returned sorted, to be stored alongside a trained model — a model and a bit
    selection are one artefact, and mixing versions silently scrambles the vector.
    """
    freq = X.mean(axis=0)
    return np.flatnonzero((freq >= min_freq) & (freq <= max_freq))


def tanimoto_dense(pred: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Tanimoto between continuous predictions and binary bank fingerprints.

    `pred` is (n_queries, n_bits) in [0, 1] — a model outputs probabilities, not bits, and
    thresholding them throws away exactly the confidence that ranks candidates. The
    continuous generalisation keeps it:

        T(p, b) = <p, b> / (|p|_1 + |b|_1 - <p, b>)

    which reduces to the usual Tanimoto when p is binary.
    """
    pred = np.asarray(pred, dtype=np.float32)
    bank = np.asarray(bank, dtype=np.float32)
    inter = pred @ bank.T
    denom = pred.sum(axis=1, keepdims=True) + bank.sum(axis=1)[None, :] - inter
    return inter / np.maximum(denom, 1e-9)


def modified_tanimoto_logit(pred_logits: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Log-likelihood scoring of each candidate under independent per-bit predictions.

    Often ranks better than Tanimoto because it is calibrated: a bit the model is
    confidently wrong about is penalised in proportion to that confidence, where Tanimoto
    only ever counts agreement. Both are cheap, and which wins is an empirical question
    to settle on the validation split rather than by argument.
    """
    from scipy.special import log_expit

    pred_logits = np.asarray(pred_logits, dtype=np.float32)
    bank = np.asarray(bank, dtype=np.float32)
    # log p(b=1) = log_expit(z), log p(b=0) = log_expit(-z)
    lp1, lp0 = log_expit(pred_logits), log_expit(-pred_logits)
    return bank @ lp1.T.astype(np.float32) + (1.0 - bank) @ lp0.T.astype(np.float32)
