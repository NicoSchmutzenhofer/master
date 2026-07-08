"""
material_decomposition.py -- core image-domain material-decomposition library.

Implements the professor's model (whiteboard / DECOMPOSITION_PLAN.md sec.1):

        B = M x + n            ->            x = argmin (Mx-B)^T W (Mx-B) [+ prior]

  B : per-voxel vector of the 4 energy-bin linear attenuations (1/cm)
  M : 4x3 material-signature matrix, columns = bin-averaged <mu/rho> (cm^2/g) from NIST
  x : per-voxel partial densities (g/cm^3) of the 3 basis materials  <-- solved for

Estimators (config.estimator), best -> simplest quality (Phase B):
  wls_joint  : penalised WLS -- data-fit + edge-preserving (TV) prior, solved iteratively
  wls_denoise: WLS solve, then an edge-preserving denoise pass on the maps
  wls        : weighted least squares only
  ols        : plain least squares (Phase A baseline; x = (M^T M)^-1 M^T B)

All weights/strengths are DATA-ADAPTIVE: the noise covariance is measured from the data each
run (global or spatial), so nothing assumes which bin/region is noisiest (memory:
adaptive-no-hardcoding). See noise_estimation.py and denoising.py.

Design (docs/SOFTWARE_ROADMAP.md): pure library layer -- compute takes numpy arrays, never
paths, never prints; decompose() is the single entry point driven by a serializable
DecompConfig with optional progress/cancel callbacks; all file I/O is the SimpleITK section
at the bottom (imported lazily) so the math runs without any imaging dependency.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import material_library as mlib
from . import decomposition_modes as modes
from . import noise_estimation as ne
from . import denoising as dn

logger = logging.getLogger(__name__)

ProgressCB = Optional[Callable[[float, str], None]]
CancelCB = Optional[Callable[[], bool]]


# ============================================================================
# Config & result (serializable; GUI-ready)
# ============================================================================
@dataclass
class DecompConfig:
    """All knobs for a decomposition run. Serializable so a GUI can populate it."""
    mode: str = "phantom_ca_i"
    threshold_option: int = 1
    bin_domain: str = "exclusive"        # 'exclusive' (A-B,B-C,C-D,D) | 'cumulative' (A,B,C,D)

    # --- estimator & noise model (best -> simplest; defaults = best quality) ---
    estimator: str = "wls_joint"         # 'wls_joint'|'wls_denoise'|'wls'|'ols'
    noise_model: str = "spatial"         # 'spatial'|'global'
    noise_window: int = 5                # box size (voxels) for residual/covariance estimation
    flat_percentile: float = 40.0        # flattest fraction used for noise measurement
    noise_sample_stride: int = 24        # z-stride for the global covariance sample (memory)

    # --- edge-preserving denoise (wls_denoise, and the prior inside wls_joint) ---
    denoise_method: str = "tv"           # 'tv'|'nlm'|'bilateral'|'guided'|'gaussian'
    denoise_scale: float = 1.0           # strength multiplier on the *measured* map noise
    denoise_guide: bool = False          # cross-channel guiding (guide = highest-SNR channel, runtime)
    joint_iters: int = 10                # iterations for wls_joint
    joint_beta: float = 1.0              # proximity weight scale (mu = beta * mean diag(M^T W M))

    # --- HU->attenuation & unit calibration ---
    hu_input: bool = True                # inputs are HU volumes (else already linear attenuation)
    water_calibration: bool = True       # per-bin gain so a water ROI matches NIST water (unit scaling)
    water_hu_tol: float = 60.0           # |HU| < tol on threshold A defines the water ROI
    compute_residual: bool = True        # also return the per-voxel LS fit-residual norm
    z_chunk: int = 16                    # slices solved per chunk (memory bound)

    # --- I/O (used by the driver; the library decompose() takes arrays) ----
    input_format: str = "nifti"          # 'nifti' (our own recon) | 'dicom' (Siemens series)
    input_dir: Optional[str] = None
    input_pattern: str = "reconstruction_thr_{label}_HU.nii.gz"
    threshold_labels: Tuple[str, ...] = ("A", "B", "C", "D")
    input_paths: Optional[List[str]] = None
    calibration_pattern: str = "calibration_thr_{label}.json"
    output_dir: Optional[str] = None
    z_slab_mm: Optional[Tuple[float, float]] = None

    # --- DICOM input (input_format='dicom'): Siemens series live flat in one folder ----
    dicom_dir: Optional[str] = None      # the flat folder of DICOMs (defaults to input_dir)
    # ordered SeriesNumber (int) OR SeriesInstanceUID (str), one per threshold_label (A..D).
    # Identify which series are the 4 thresholds via inspect_dicom.py --dump-series first.
    dicom_series: Optional[List] = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DecompConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        if kw.get("threshold_labels") is not None:
            kw["threshold_labels"] = tuple(kw["threshold_labels"])
        if kw.get("z_slab_mm") is not None:
            kw["z_slab_mm"] = tuple(kw["z_slab_mm"])
        return cls(**kw)


@dataclass
class DecompResult:
    materials: List[str]
    material_maps: Dict[str, np.ndarray]   # name -> density map (g/cm^3), shape (Z,Y,X)
    M: np.ndarray                          # (n_bins, n_materials)
    stability: dict
    metadata: dict
    config: DecompConfig
    residual: Optional[np.ndarray] = None  # per-voxel LS residual norm, shape (Z,Y,X)


# ============================================================================
# Pure math -- forward model, bins, estimators
# ============================================================================
def cumulative_to_exclusive(vols: np.ndarray) -> np.ndarray:
    """
    Cumulative thresholds A>=20, B>=40, C>=56, D>=75 (axis 0, order A..D)
    -> exclusive energy bins [20-40, 40-56, 56-75, 75-140] = [A-B, B-C, C-D, D].
    Image-domain differencing only (CLAUDE.md invariant #3).
    """
    vols = np.asarray(vols)
    if vols.shape[0] != 4:
        raise ValueError(f"Expected 4 cumulative volumes on axis 0, got shape {vols.shape}")
    a, b, c, d = vols
    return np.stack([a - b, b - c, c - d, d], axis=0)


def hu_to_linear_attenuation(hu: np.ndarray, mu_water: float) -> np.ndarray:
    """HU -> linear attenuation (1/cm): mu = mu_water * (1 + HU/1000)."""
    return mu_water * (1.0 + np.asarray(hu, dtype=np.float64) / 1000.0)


def form_bin_measurements(vols_mu: np.ndarray, bin_domain: str) -> np.ndarray:
    """Per-threshold linear attenuation (4, ...) -> measurement stack B (n_bins, ...)."""
    if bin_domain == "exclusive":
        return cumulative_to_exclusive(vols_mu)
    if bin_domain == "cumulative":
        return np.asarray(vols_mu)
    raise ValueError(f"bin_domain must be 'exclusive' or 'cumulative', got '{bin_domain}'")


def solve_ols(B: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Ordinary least squares x = (M^T M)^-1 M^T B for every column of B at once.
    B (n_bins, n_vox), M (n_bins, n_materials) -> X (n_materials, n_vox).
    Uses the pseudoinverse (== (M^T M)^-1 M^T for full-rank tall M) for numerical stability.
    """
    return np.linalg.pinv(M) @ B


def _safe_inv(A: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


def wls_operator(M: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Global-W WLS linear operator L = (M^T W M)^-1 M^T W, so x = L B."""
    MtW = M.T @ W
    return _safe_inv(MtW @ M) @ MtW


def solve_wls(B: np.ndarray, M: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Weighted least squares with a single (global) weight matrix W = Sigma^-1."""
    return wls_operator(M, W) @ B


def voxel_wls(Bc: np.ndarray, M: np.ndarray, Wfield: np.ndarray,
              mu: float = 0.0, Zprox: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Per-voxel (optionally proximity-regularised) WLS.
      Bc     : (n_bins, n_vox)
      Wfield : (n_vox, n_bins, n_bins)   per-voxel inverse-covariance weights
    Solves (M^T W M + mu I) x = M^T W b + mu z  per voxel; returns (n_mat, n_vox).
    """
    Mt = M.T
    MtW = np.einsum("ib,vbc->vic", Mt, Wfield)        # (v, n_mat, n_bins)
    A = np.einsum("vic,cj->vij", MtW, M)              # (v, n_mat, n_mat)
    rhs = np.einsum("vic,cv->vi", MtW, Bc)            # (v, n_mat)
    if mu > 0:
        A = A + mu * np.eye(M.shape[1])
        if Zprox is not None:
            rhs = rhs + mu * Zprox.T
    X = np.linalg.solve(A, rhs[..., None])[..., 0]    # (v, n_mat)
    return X.T


# ---- stability / conditioning (M-only) ------------------------------------
def column_cosines(M: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity of M's columns (1.0 => collinear => singular)."""
    norm = M / np.linalg.norm(M, axis=0, keepdims=True)
    return norm.T @ norm


def stability_verdict(kappa: float) -> str:
    if kappa < 10:
        return "excellent"
    if kappa < 50:
        return "good"
    if kappa < 200:
        return "usable"
    if kappa < 1000:
        return "poor"
    return "unstable"


def stability_report(M: np.ndarray, materials: Sequence[str]) -> dict:
    """Conditioning / noise diagnostics of the material matrix (depends only on M)."""
    materials = list(materials)
    s = np.linalg.svd(M, compute_uv=False)
    kappa = float(s[0] / s[-1])
    Mn = M / np.linalg.norm(M, axis=0, keepdims=True)
    sn = np.linalg.svd(Mn, compute_uv=False)
    inv = _safe_inv(M.T @ M)
    noise_gain = np.sqrt(np.clip(np.diag(inv), 0, None))
    return {
        "materials": materials,
        "condition_number": kappa,
        "condition_number_MtM": float(kappa ** 2),
        "condition_number_colnorm": float(sn[0] / sn[-1]),
        "singular_values": s.tolist(),
        "noise_amplification": {m: float(g) for m, g in zip(materials, noise_gain)},
        "column_cosines": column_cosines(M).tolist(),
        "verdict": stability_verdict(kappa),
    }


# ============================================================================
# Water calibration (unit scaling; not a stability remedy, so estimator-agnostic)
# ============================================================================
def _water_reference(bin_domain: str, option: int, mu_water: Optional[np.ndarray]) -> np.ndarray:
    """Expected water linear attenuation (1/cm) per bin, from NIST (density 1.0 g/cm^3)."""
    water_mass = mlib.material_signature("Water", option)
    rho = mlib.material_info("Water").density_g_cm3
    if bin_domain == "exclusive":
        return water_mass * rho
    if mu_water is not None:
        return np.asarray(mu_water, dtype=float)
    return water_mass * rho


def estimate_water_gains(B: np.ndarray, water_mask: np.ndarray,
                         reference: np.ndarray) -> np.ndarray:
    """Per-bin gain so median(B in water ROI) matches the NIST water reference."""
    gains = np.ones(B.shape[0], dtype=float)
    flat = B.reshape(B.shape[0], -1)
    m = water_mask.reshape(-1).astype(bool)
    if m.sum() < 10:
        warnings.warn("Water ROI too small for calibration; skipping (gains=1).")
        return gains
    for i in range(B.shape[0]):
        med = np.median(flat[i, m])
        if abs(med) > 1e-9:
            gains[i] = reference[i] / med
    return gains


# ============================================================================
# Estimator drivers (chunked over Z; data-adaptive weights)
# ============================================================================
def estimate_global_W(vols_mu: np.ndarray, gains: np.ndarray, config: DecompConfig) -> np.ndarray:
    """Sample strided slices, form the bins, estimate Sigma, return W = Sigma^-1 (one matrix)."""
    Z = vols_mu.shape[1]
    stride = max(1, int(config.noise_sample_stride))
    zc = np.arange(0, Z, stride)
    if zc.size < 3:
        zc = np.arange(Z)
    Bs = form_bin_measurements(vols_mu[:, zc], config.bin_domain) * gains[:, None, None, None]
    Sigma = ne.noise_covariance_global(Bs, config.noise_window, config.flat_percentile)
    return _safe_inv(Sigma)


def _shape(vols_mu):
    return vols_mu.shape[0], vols_mu.shape[1], vols_mu.shape[2], vols_mu.shape[3]


def _solve_linear_chunked(vols_mu, gains, M, L, config, n_mat, _p, cancel):
    """x = L @ B over Z-chunks (OLS or global-W WLS). Returns (maps, residual)."""
    n_bins, Z, Y, X = _shape(vols_mu)
    maps = np.zeros((n_mat, Z, Y, X), dtype=np.float32)
    residual = np.zeros((Z, Y, X), dtype=np.float32) if config.compute_residual else None
    chunk = max(1, int(config.z_chunk))
    for z0 in range(0, Z, chunk):
        if cancel and cancel():
            warnings.warn("decompose() cancelled; returning partial maps."); break
        z1 = min(z0 + chunk, Z)
        B = form_bin_measurements(vols_mu[:, z0:z1], config.bin_domain)
        B = (B * gains[:, None, None, None]).reshape(n_bins, -1)
        Xsol = L @ B
        maps[:, z0:z1] = Xsol.reshape(n_mat, z1 - z0, Y, X).astype(np.float32)
        if residual is not None:
            residual[z0:z1] = np.linalg.norm(M @ Xsol - B, axis=0).reshape(z1 - z0, Y, X)
        _p(0.10 + 0.80 * (z1 / Z), f"solved z {z1}/{Z}")
    return maps, residual


def _covariance_field_for_chunk(vols_mu, gains, config, z0, z1, n_bins):
    """Local inverse-covariance weights for a Z-chunk, using a halo to avoid boundary bias."""
    Z = vols_mu.shape[1]
    h = max(1, int(config.noise_window))
    a, b = max(0, z0 - h), min(Z, z1 + h)
    Bh = form_bin_measurements(vols_mu[:, a:b], config.bin_domain) * gains[:, None, None, None]
    field = ne.noise_covariance_field(Bh, config.noise_window)          # (nz_h,Y,X,nb,nb)
    sl = slice(z0 - a, z0 - a + (z1 - z0))
    Wf = np.linalg.inv(field[sl].reshape(-1, n_bins, n_bins))
    Bc = Bh[:, sl].reshape(n_bins, -1)
    return Bc, Wf


def _solve_spatial_chunked(vols_mu, gains, M, config, n_mat, _p, cancel):
    """Spatially-varying-W WLS: per-chunk local covariance field + per-voxel solve."""
    n_bins, Z, Y, X = _shape(vols_mu)
    maps = np.zeros((n_mat, Z, Y, X), dtype=np.float32)
    residual = np.zeros((Z, Y, X), dtype=np.float32) if config.compute_residual else None
    chunk = max(1, int(config.z_chunk))
    for z0 in range(0, Z, chunk):
        if cancel and cancel():
            warnings.warn("decompose() cancelled; returning partial maps."); break
        z1 = min(z0 + chunk, Z)
        Bc, Wf = _covariance_field_for_chunk(vols_mu, gains, config, z0, z1, n_bins)
        Xsol = voxel_wls(Bc, M, Wf)
        maps[:, z0:z1] = Xsol.reshape(n_mat, z1 - z0, Y, X).astype(np.float32)
        if residual is not None:
            residual[z0:z1] = np.linalg.norm(M @ Xsol - Bc, axis=0).reshape(z1 - z0, Y, X)
        _p(0.10 + 0.80 * (z1 / Z), f"solved z {z1}/{Z} (spatial noise)")
    return maps, residual


def _denoise_stack(maps: np.ndarray, config: DecompConfig) -> np.ndarray:
    """Edge-preserving denoise each material map, strength from its own measured noise."""
    sigmas = [ne.estimate_map_noise(maps[i], config.noise_window, config.flat_percentile)
              for i in range(maps.shape[0])]
    guide = dn.select_guide_index(list(maps), sigmas) if config.denoise_guide else None
    den = dn.denoise_maps(list(maps), sigmas, config.denoise_method, config.denoise_scale, guide)
    return np.stack(den).astype(np.float32)


def _solve_joint(vols_mu, gains, M, config, n_mat, _p, cancel):
    """
    Penalised WLS via half-quadratic splitting (plug-and-play):
      repeat:  z <- edge-preserving-denoise(x)   [TV prox, strength from measured noise]
               x <- argmin (Mx-B)^T W (Mx-B) + mu ||x - z||^2
    W is the data-adaptive noise weighting (global or spatial). Best quality; iterative.
    """
    n_bins, Z, Y, X = _shape(vols_mu)
    B = form_bin_measurements(vols_mu, config.bin_domain) * gains[:, None, None, None]
    Bflat = B.reshape(n_bins, -1)
    spatial = (config.noise_model == "spatial")
    tv = dn.DENOISERS["tv"]
    iters = max(1, int(config.joint_iters))

    if not spatial:
        W = estimate_global_W(vols_mu, gains, config)
        A0 = M.T @ W @ M
        mu = float(config.joint_beta) * float(np.mean(np.diag(A0)))
        invA = _safe_inv(A0 + mu * np.eye(n_mat))
        MtW = M.T @ W
        data_term = (invA @ MtW) @ Bflat                    # constant across iterations
        x = wls_operator(M, W) @ Bflat                      # WLS init
    else:
        maps0, _ = _solve_spatial_chunked(vols_mu, gains, M, config, n_mat,
                                          lambda *a: None, cancel)
        x = maps0.reshape(n_mat, -1)
        mu = float(config.joint_beta) * float(np.mean(np.diag(M.T @ M)))

    for it in range(iters):
        if cancel and cancel():
            warnings.warn("decompose() cancelled; returning partial maps."); break
        xvol = x.reshape(n_mat, Z, Y, X)
        z = np.empty_like(xvol)
        for c in range(n_mat):
            sig = ne.estimate_map_noise(xvol[c], config.noise_window, config.flat_percentile)
            z[c] = tv(xvol[c], sig, scale=config.denoise_scale)
        zflat = z.reshape(n_mat, -1)
        if not spatial:
            x = data_term + mu * (invA @ zflat)
        else:
            x = _joint_spatial_update(vols_mu, gains, M, config, n_mat, zflat, mu, cancel)
        _p(0.10 + 0.80 * ((it + 1) / iters), f"joint iter {it + 1}/{iters}")

    residual = None
    if config.compute_residual:
        residual = np.linalg.norm(M @ x - Bflat, axis=0).reshape(Z, Y, X).astype(np.float32)
    return x.reshape(n_mat, Z, Y, X).astype(np.float32), residual


def _joint_spatial_update(vols_mu, gains, M, config, n_mat, zflat, mu, cancel):
    """Data-update step of wls_joint under spatially-varying W (chunked, per-voxel)."""
    n_bins, Z, Y, X = _shape(vols_mu)
    out = np.zeros((n_mat, Z, Y, X), dtype=np.float64)
    zv = zflat.reshape(n_mat, Z, Y, X)
    chunk = max(1, int(config.z_chunk))
    for z0 in range(0, Z, chunk):
        if cancel and cancel():
            break
        z1 = min(z0 + chunk, Z)
        Bc, Wf = _covariance_field_for_chunk(vols_mu, gains, config, z0, z1, n_bins)
        Zc = zv[:, z0:z1].reshape(n_mat, -1)
        Xc = voxel_wls(Bc, M, Wf, mu=mu, Zprox=Zc)
        out[:, z0:z1] = Xc.reshape(n_mat, z1 - z0, Y, X)
    return out.reshape(n_mat, -1)


# ============================================================================
# Entry point
# ============================================================================
def decompose(volumes: np.ndarray, config: DecompConfig,
              mu_water: Optional[Sequence[float]] = None,
              water_mask: Optional[np.ndarray] = None,
              progress: ProgressCB = None, cancel: CancelCB = None) -> DecompResult:
    """
    Decompose 4 reconstructed threshold volumes into per-material density maps.

    volumes  : (4, Z, Y, X) cumulative threshold volumes in HU (order A..D = >=20..>=75 keV),
               or linear attenuation if config.hu_input is False.
    mu_water : (4,) per-threshold water attenuation for HU->mu; if None and hu_input, a nominal
               constant is used (absolute density scale then relies on water_calibration).
    water_mask : optional bool (Z,Y,X) water ROI; if None and water_calibration, auto-detected
               as |HU_A| < config.water_hu_tol.
    """
    def _p(frac, msg):
        if progress:
            progress(float(frac), msg)

    volumes = np.asarray(volumes, dtype=np.float32)   # float32 storage; per-chunk math upcasts
    if volumes.ndim != 4 or volumes.shape[0] != 4:
        raise ValueError(f"volumes must be (4,Z,Y,X); got {volumes.shape}")
    Z, Y, X = int(volumes.shape[1]), int(volumes.shape[2]), int(volumes.shape[3])

    spec = modes.get_mode(config.mode)
    materials = list(spec.materials)
    M = mlib.build_M(materials, config.threshold_option)
    n_bins, n_mat = M.shape
    if n_bins != volumes.shape[0]:
        raise ValueError(f"M has {n_bins} bins but {volumes.shape[0]} volumes supplied")

    _p(0.02, f"mode '{config.mode}' [{'/'.join(materials)}] -- stability audit")
    stability = stability_report(M, materials)
    logger.info("mode %s: kappa=%.3g (%s)", config.mode,
                stability["condition_number"], stability["verdict"])

    # HU -> linear attenuation (per cumulative threshold) --------------------
    if config.hu_input:
        if mu_water is None:
            mu_water = np.full(n_bins, 0.20)
            warnings.warn("mu_water not provided; using nominal 0.20 /cm. Absolute density "
                          "scale relies on water_calibration.")
        mu_water = np.asarray(mu_water, dtype=float)
        vols_mu = np.stack([hu_to_linear_attenuation(volumes[i], mu_water[i])
                            for i in range(n_bins)], axis=0).astype(np.float32)
    else:
        vols_mu = volumes
        mu_water = None if mu_water is None else np.asarray(mu_water, dtype=float)

    # Water ROI (needs the HU input), then free the input volumes to bound memory ----
    if config.water_calibration and water_mask is None and config.hu_input:
        water_mask = np.abs(volumes[0]) < config.water_hu_tol
    volumes = None  # HU input no longer needed (vols_mu holds the attenuation)

    gains = np.ones(n_bins)
    if config.water_calibration:
        if water_mask is None:
            warnings.warn("water_calibration on but no water ROI available; skipping.")
        else:
            B_probe = form_bin_measurements(vols_mu, config.bin_domain)
            reference = _water_reference(config.bin_domain, config.threshold_option, mu_water)
            gains = estimate_water_gains(B_probe, water_mask, reference)
            del B_probe

    # Estimator dispatch ------------------------------------------------------
    est = config.estimator
    _p(0.06, f"estimator '{est}', noise '{config.noise_model}'")
    if est == "ols":
        maps, residual = _solve_linear_chunked(vols_mu, gains, M, np.linalg.pinv(M),
                                               config, n_mat, _p, cancel)
    elif est in ("wls", "wls_denoise"):
        if config.noise_model == "global":
            W = estimate_global_W(vols_mu, gains, config)
            maps, residual = _solve_linear_chunked(vols_mu, gains, M, wls_operator(M, W),
                                                   config, n_mat, _p, cancel)
        else:
            maps, residual = _solve_spatial_chunked(vols_mu, gains, M, config, n_mat, _p, cancel)
    elif est == "wls_joint":
        maps, residual = _solve_joint(vols_mu, gains, M, config, n_mat, _p, cancel)
    else:
        raise ValueError(f"Unknown estimator '{est}' "
                         f"(use 'ols'|'wls'|'wls_denoise'|'wls_joint')")

    vols_mu = None  # free the attenuation volumes before any full-volume denoise
    if est == "wls_denoise":
        _p(0.92, f"edge-preserving denoise ({config.denoise_method})")
        maps = _denoise_stack(maps, config)

    material_maps = {m: maps[i] for i, m in enumerate(materials)}
    metadata = {
        "mode": config.mode,
        "display_name": spec.display_name,
        "clinical_question": spec.clinical_question,
        "materials": materials,
        "threshold_option": config.threshold_option,
        "bin_edges_keV": mlib.bin_edges(config.threshold_option),
        "bin_domain": config.bin_domain,
        "estimator": config.estimator,
        "noise_model": config.noise_model,
        "denoise": ({"method": config.denoise_method, "scale": config.denoise_scale,
                     "guide": config.denoise_guide}
                    if est in ("wls_denoise", "wls_joint") else None),
        "mu_water_per_threshold": None if mu_water is None else mu_water.tolist(),
        "water_calibration_gains": gains.tolist(),
        "volume_shape": [int(Z), int(Y), int(X)],
    }
    _p(1.0, "done")
    return DecompResult(materials=materials, material_maps=material_maps, M=M,
                        stability=stability, metadata=metadata, config=config,
                        residual=residual)


def mode_stability(mode_key: str, option: int = 1) -> dict:
    """Stability report for a mode without any volumes (for the research/figures path)."""
    spec = modes.get_mode(mode_key)
    rep = stability_report(mlib.build_M(spec.materials, option), spec.materials)
    rep["mode"] = mode_key
    rep["display_name"] = spec.display_name
    return rep


# ============================================================================
# I/O layer (SimpleITK, lazy import -- kept separate from all compute above)
# ============================================================================
def _sitk():
    import SimpleITK as sitk
    return sitk


def _resolve_input_paths(config: DecompConfig) -> List[Path]:
    if config.input_paths:
        return [Path(p) for p in config.input_paths]
    if not config.input_dir:
        raise ValueError("config.input_dir or config.input_paths must be set")
    d = Path(config.input_dir)
    return [d / config.input_pattern.format(label=lbl) for lbl in config.threshold_labels]


def read_recon_calibration(config: DecompConfig) -> Optional[np.ndarray]:
    """Read per-threshold mu_water from the reconstruction calibration JSONs, if present."""
    if not config.input_dir:
        return None
    d = Path(config.input_dir)
    vals = []
    for lbl in config.threshold_labels:
        p = d / config.calibration_pattern.format(label=lbl)
        if not p.exists():
            return None
        with open(p) as f:
            cal = json.load(f)
        mw = cal.get("mu_water") or cal.get("mu_water_per_cm") or cal.get("muWater")
        if mw is None:
            return None
        vals.append(float(mw))
    return np.array(vals, dtype=float)


def _slab_slice(ref, z_slab_mm: Tuple[float, float]) -> Tuple[int, int]:
    """Map a physical z-range (mm) to numpy slice indices along axis 0 (slowest)."""
    nz = ref.GetSize()[2]
    z_of = [ref.TransformIndexToPhysicalPoint((0, 0, k))[2] for k in range(nz)]
    lo, hi = sorted(z_slab_mm)
    ks = [k for k, z in enumerate(z_of) if lo <= z <= hi]
    if not ks:
        raise ValueError(f"z_slab_mm {z_slab_mm} selects no slices "
                         f"(z range {min(z_of):.1f}..{max(z_of):.1f})")
    return min(ks), max(ks) + 1


def _is_dicom(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def build_dicom_index(folder: str, cache_path: Optional[str] = None) -> dict:
    """
    Index a flat folder of DICOMs -> {series_uid: {number, desc, files: [[z, path], ...] sorted}}.

    The Siemens export drops every series' files into one folder (survey: 6 series, one dir), so
    series are grouped by SeriesInstanceUID from the headers, not by path. Reading 11 k headers
    takes a couple of minutes; the result is cached to cache_path so repeated dev runs are instant.
    """
    sitk = _sitk()
    folder = str(folder)
    if cache_path and Path(cache_path).exists():
        try:
            cached = json.loads(Path(cache_path).read_text(encoding="utf-8"))
            if cached.get("folder") == folder:
                logger.info("using cached DICOM index %s", cache_path)
                return cached["series"]
        except (OSError, ValueError):
            pass

    series: Dict[str, dict] = {}
    n_seen = 0
    for root, _dirs, names in os.walk(folder):
        for name in names:
            fp = os.path.join(root, name)
            if not (name.lower().endswith((".dcm", ".ima")) or _is_dicom(fp)):
                continue
            r = sitk.ImageFileReader()
            r.SetFileName(fp)
            try:
                r.ReadImageInformation()
            except RuntimeError:
                continue

            def g(t):
                return r.GetMetaData(t).strip() if r.HasMetaDataKey(t) else ""
            uid = g("0020|000e")
            if not uid:
                continue
            try:
                z = float(g("0020|0032").split("\\")[2])
            except (ValueError, IndexError):
                z = 0.0
            s = series.setdefault(uid, {"number": g("0020|0011"), "desc": g("0008|103e"),
                                        "files": []})
            s["files"].append([z, fp])
            n_seen += 1
            if n_seen % 2000 == 0:
                logger.info("  indexed %d DICOM headers ...", n_seen)
    for s in series.values():
        s["files"].sort(key=lambda zf: zf[0])
    logger.info("DICOM index: %d series, %d files under %s", len(series), n_seen, folder)
    if cache_path:
        try:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text(json.dumps({"folder": folder, "series": series}),
                                        encoding="utf-8")
        except OSError:
            pass
    return series


def _resolve_series(index: dict, selectors: Sequence) -> List[str]:
    """Map each selector (SeriesNumber int/str or SeriesInstanceUID) to a SeriesInstanceUID."""
    num_to_uid = {}
    for uid, s in index.items():
        num = s.get("number", "")
        if num:
            try:
                num_to_uid[str(int(float(num)))] = uid
            except ValueError:
                num_to_uid[num] = uid
    resolved = []
    for sel in selectors:
        key = str(sel).strip()
        if key in index:                     # already a UID
            resolved.append(key)
            continue
        try:
            key = str(int(float(key)))       # normalise "24"/"24.0"/24 -> "24"
        except ValueError:
            pass
        if key in num_to_uid:
            resolved.append(num_to_uid[key])
        else:
            raise ValueError(
                f"series selector {sel!r} not found. Available SeriesNumbers "
                f"{sorted(num_to_uid, key=lambda x: int(x) if x.isdigit() else 1 << 30)}")
    return resolved


def _slab_files(files: List[list], z_slab_mm: Optional[Tuple[float, float]]) -> List[str]:
    """Return the file paths whose z (mm) falls in z_slab_mm; all of them if no slab set."""
    paths = [f for _z, f in files]
    if not z_slab_mm:
        return paths
    lo, hi = sorted(z_slab_mm)
    keep = [f for z, f in files if lo <= z <= hi]
    if not keep:
        zs = [z for z, _f in files]
        raise ValueError(f"z_slab_mm {z_slab_mm} selects no slices "
                         f"(series z {min(zs):.1f}..{max(zs):.1f})")
    return keep


def load_threshold_volumes_dicom(config: DecompConfig):
    """
    Load the threshold volumes from a flat DICOM folder (Siemens export) -> the same
    (volumes (4,Z,Y,X) float32, reference sitk image, mu_water|None) contract as the NIfTI loader.

    config.dicom_series selects which series are the thresholds, in A..D order. RAM is bounded by
    reading only the z_slab_mm slices, converting to float32 per series, and freeing each series
    image before the next. mu_water is None (no recon calibration JSONs for Siemens data) -> the
    solver's water_calibration anchors the per-bin unit scale instead.
    """
    sitk = _sitk()
    folder = config.dicom_dir or config.input_dir
    if not folder or not config.dicom_series:
        raise ValueError("dicom input needs config.dicom_dir (or input_dir) and config.dicom_series "
                         "(ordered SeriesNumber/UID list, one per threshold label A..D)")
    if len(config.dicom_series) != len(config.threshold_labels):
        raise ValueError(f"dicom_series has {len(config.dicom_series)} entries but there are "
                         f"{len(config.threshold_labels)} threshold_labels")
    cache = None
    if config.output_dir:
        cache = str(Path(config.output_dir) / "dicom_index.json")
    index = build_dicom_index(folder, cache_path=cache)
    uids = _resolve_series(index, config.dicom_series)

    arrs, ref = [], None
    for lbl, uid in zip(config.threshold_labels, uids):
        paths = _slab_files(index[uid]["files"], config.z_slab_mm)
        r = sitk.ImageSeriesReader()
        r.SetFileNames(paths)
        img = r.Execute()                                  # 3D, HU (rescale applied by GDCM)
        if ref is None:
            ref = img
        arrs.append(sitk.GetArrayFromImage(img).astype(np.float32))
        del img
        logger.info("threshold %s <- series #%s (%s) : %d slices",
                    lbl, index[uid]["number"], uid[-10:], arrs[-1].shape[0])
    shapes = {a.shape for a in arrs}
    if len(shapes) != 1:
        raise ValueError(f"Threshold series have mismatched shapes {shapes} "
                         "-- check dicom_series selection / z_slab_mm")
    vols = np.stack(arrs, axis=0)
    del arrs
    return vols, ref, None


def load_threshold_volumes(config: DecompConfig):
    """
    Load the 4 threshold volumes -> (volumes (4,Z,Y,X) float32, reference sitk image, mu_water|None).
    Dispatches on config.input_format ('nifti' | 'dicom'). Applies config.z_slab_mm if set;
    the reference image carries the geometry used when saving results.
    """
    if getattr(config, "input_format", "nifti") == "dicom":
        return load_threshold_volumes_dicom(config)
    sitk = _sitk()
    paths = _resolve_input_paths(config)
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing threshold volume(s):\n  " + "\n  ".join(missing))

    imgs = [sitk.ReadImage(str(p)) for p in paths]
    ref = imgs[0]
    arrs = [sitk.GetArrayFromImage(im).astype(np.float32) for im in imgs]  # (Z,Y,X)
    shapes = {a.shape for a in arrs}
    if len(shapes) != 1:
        raise ValueError(f"Threshold volumes have mismatched shapes: {shapes}")

    if config.z_slab_mm is not None:
        z0, z1 = _slab_slice(ref, config.z_slab_mm)
        arrs = [a[z0:z1] for a in arrs]
        ref = ref[:, :, z0:z1]
        logger.info("z_slab_mm %s -> slices [%d:%d]", config.z_slab_mm, z0, z1)

    mu_water = read_recon_calibration(config)
    return np.stack(arrs, axis=0), ref, mu_water


def save_decomp_result(result: DecompResult, ref, out_dir) -> List[Path]:
    """Write one NIfTI per material (geometry copied from ref) + stability/config/metadata JSON."""
    sitk = _sitk()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    def _save(arr, name):
        img = sitk.GetImageFromArray(arr.astype(np.float32))
        img.CopyInformation(ref)
        p = out / name
        sitk.WriteImage(img, str(p))
        written.append(p)

    mode = result.config.mode
    for mat, arr in result.material_maps.items():
        _save(arr, f"decomp_{mode}_{mat}.nii.gz")
    if result.residual is not None:
        _save(result.residual, f"decomp_{mode}_residual.nii.gz")

    (out / f"decomp_{mode}_stability.json").write_text(json.dumps(result.stability, indent=2))
    (out / f"decomp_{mode}_metadata.json").write_text(json.dumps(result.metadata, indent=2))
    (out / f"decomp_{mode}_config.json").write_text(json.dumps(result.config.to_dict(), indent=2))
    written += [out / f"decomp_{mode}_stability.json",
                out / f"decomp_{mode}_metadata.json",
                out / f"decomp_{mode}_config.json"]
    return written
