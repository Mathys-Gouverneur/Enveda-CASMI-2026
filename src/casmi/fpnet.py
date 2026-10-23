"""Spectrum -> molecular fingerprint: a transformer encoder over peaks.

The model the test regime calls for. `measurement.md` establishes that the test is mostly
"structure in the candidate bank, spectrum absent from the spectral library", where plain
retrieval scores exactly 0 and random ranking of the mass window scores 0.085. This is
what beats that floor.

Architecture, deliberately close to the official tutorial's encoder so the comparison is
about the training data rather than the design:

    (m/z, intensity) pairs -> sinusoidal peak embedding -> transformer encoder
                           -> masked mean pool -> MLP head -> fingerprint logits

Three choices worth justifying:

**Sinusoidal embedding over log-m/z.** A peak's position is a continuous mass, not a
sequence index, and the information is in exact position — 4 decimals, because that is
what separates two formulas. Wavelengths spaced logarithmically from 0.01 to 1000 Da give
the model both "this is around 150" and "this is 150.0344 not 150.0280".

**No positional encoding.** A spectrum is a *set* of peaks; their storage order is
meaningless. Self-attention without positions is permutation-equivariant, which is
exactly right here, and the masked mean pool makes the output permutation-invariant.

**Per-bit binary cross-entropy, pos_weight balanced.** Fingerprint bits are sparse and
very unequally frequent; unweighted BCE converges to predicting all zeros for the rare
bits, which are the discriminative ones.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class FPNetConfig:
    n_bits: int                       #: fingerprint bits to predict (after bit selection)
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_peaks: int = 256
    #: Sinusoidal wavelengths span 10^-2 to 10^3 Da, covering both coarse position and
    #: the sub-0.01 Da detail that distinguishes formulas.
    mz_log_lims: tuple[float, float] = (-2.0, 3.0)


class PeakEmbedding(nn.Module):
    """(m/z, intensity) -> d_model, by sinusoidal features of m/z scaled by intensity."""

    def __init__(self, cfg: FPNetConfig):
        super().__init__()
        half = cfg.d_model // 2
        wavelength = torch.pow(
            10.0, torch.linspace(cfg.mz_log_lims[0], cfg.mz_log_lims[1], half))
        self.register_buffer("frequency", 2 * math.pi / wavelength)
        self.proj = nn.Sequential(
            nn.Linear(cfg.d_model + 2, cfg.d_model), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

    def forward(self, mz: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
        angle = mz.unsqueeze(-1) * self.frequency
        feats = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
        # Intensity enters twice: raw and square-rooted. sqrt is the standard damping for
        # spectral similarity, and giving the model both lets it choose per bit.
        extra = torch.stack([intensity, torch.sqrt(intensity.clamp_min(0))], dim=-1)
        return self.proj(torch.cat([feats, extra], dim=-1))


class FPNet(nn.Module):
    """Predicts fingerprint logits from a spectrum."""

    def __init__(self, cfg: FPNetConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = PeakEmbedding(cfg)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True)
        # enable_nested_tensor would be ignored anyway with norm_first=True, and asking
        # for it only produces a warning on every construction.
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.n_bits),
        )

    def forward(self, mz: torch.Tensor, intensity: torch.Tensor,
                pad_mask: torch.Tensor) -> torch.Tensor:
        """`pad_mask` is True where a position is padding. Returns (batch, n_bits) logits."""
        h = self.encoder(self.embed(mz, intensity), src_key_padding_mask=pad_mask)
        keep = (~pad_mask).unsqueeze(-1).to(h.dtype)
        # Masked mean, not a CLS token: a spectrum has no distinguished element, and the
        # mean keeps the output invariant to peak order.
        pooled = (h * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)
        return self.head(self.norm(pooled))


# --------------------------------------------------------------------------- batching

def collate_spectra(mzs, intensities, max_peaks: int, precursors=None):
    """Pad a list of spectra to a batch, keeping the most intense peaks.

    Truncation is by intensity, not by m/z: keeping the 256 strongest peaks retains the
    informative part of a 1,000-peak timsTOF spectrum, while keeping the first 256 by m/z
    would throw away the high-mass fragments that constrain the structure most.

    When `precursors` is given it is prepended as a peak at intensity 2.0 — outside the
    normalised range, so the model can identify it — because the precursor mass is the
    single most informative number in the spectrum.
    """
    n = len(mzs)
    width = max_peaks + (1 if precursors is not None else 0)
    MZ = np.zeros((n, width), dtype=np.float32)
    IT = np.zeros((n, width), dtype=np.float32)
    PAD = np.ones((n, width), dtype=bool)

    for i, (mz, it) in enumerate(zip(mzs, intensities)):
        mz = np.asarray(mz, dtype=np.float32)
        it = np.asarray(it, dtype=np.float32)
        if it.size:
            order = np.argsort(it)[::-1][:max_peaks]
            order = order[np.argsort(mz[order])]        # back to m/z order for readability
            mz, it = mz[order], it[order]
            peak = it.max()
            it = it / peak if peak > 0 else it
        at = 0
        if precursors is not None:
            MZ[i, 0], IT[i, 0], PAD[i, 0] = precursors[i], 2.0, False
            at = 1
        k = len(mz)
        MZ[i, at:at + k] = mz
        IT[i, at:at + k] = it
        PAD[i, at:at + k] = False
    return (torch.from_numpy(MZ), torch.from_numpy(IT), torch.from_numpy(PAD))


def pos_weight_from(Y: np.ndarray, cap: float = 50.0) -> torch.Tensor:
    """Per-bit positive weight for BCE: (negatives / positives), capped.

    Uncapped, a bit present in 0.5% of molecules gets weight 200 and its gradient drowns
    out every other bit. The cap is a floor on how much any single rare bit can dominate.
    """
    pos = Y.mean(axis=0)
    w = (1.0 - pos) / np.maximum(pos, 1e-6)
    return torch.from_numpy(np.minimum(w, cap).astype(np.float32))
