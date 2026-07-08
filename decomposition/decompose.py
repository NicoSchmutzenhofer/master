"""
decompose.py -- production driver for image-domain material decomposition (Phase A).

Runs ONE clinical mode with the plain OLS estimator and writes the material maps plus the
full stability report. Config-object-first (GUI-ready): the constants below just build a
DecompConfig; a future GUI builds the same object from widgets and calls decompose().

Run from the repo root:
    python -m decomposition.decompose      # preferred (package form)
    python decomposition/decompose.py      # also works

Inputs: the reconstructed threshold volumes in output/reconstruction/ (see CONFIG). Reconstruction
code is not touched. Outputs: output/decomposition/decomp_<mode>_*.nii.gz + *.json.
"""
from __future__ import annotations

import logging
from pathlib import Path

# --- import shim: work both as `-m decomposition.decompose` and as a script path ---
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decomposition.material_decomposition import (
    DecompConfig, decompose, load_threshold_volumes, save_decomp_result)

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ============================== CONFIG ==============================
MODE = "phantom_ca_i"        # see decomposition_modes.list_modes()
THRESHOLD_OPTION = 1         # 1 == the actual scan thresholds (20/40/56/75 keV)
BIN_DOMAIN = "exclusive"     # 'exclusive' (A-B,B-C,C-D,D) | 'cumulative' (A,B,C,D)

# --- input source --------------------------------------------------------------
# 'nifti' = our own reconstruction (output/reconstruction/reconstruction_thr_*_HU.nii.gz) -- the
#           current default. 'dicom' = the Siemens clinical reconstructions (one flat folder of
#           series) -- kept fully wired for the later switch to the Siemens output.
INPUT_FORMAT = "nifti"       # 'nifti' (our recon, default) | 'dicom' (Siemens series, later)
DICOM_DIR = "/data/Data2/4_BIN_PCCT/"   # flat folder of all series (used when INPUT_FORMAT='dicom')
# Ordered SeriesNumber (or SeriesInstanceUID) for the 4 thresholds A,B,C,D.
#   !!! NOT YET AVAILABLE -- the --dump-series check (2026-07-08) showed the 6 series in
#   /data/Data2/4_BIN_PCCT/ are a DOSE SWEEP of the single lowest-threshold conventional image
#   (ImageType ...\THRESHOLD\...\T1\COUNT\CONVCT for all six; differing CTDIvol/mAs/tube current),
#   NOT the 4 energy bins. Material decomposition needs the T1..T4 threshold set from ONE
#   acquisition (or VMI / material maps). Re-run inspect_dicom.py on the folder that holds those,
#   then set the four series numbers here (A = lowest keV .. D = highest). Placeholder = no-op. !!!
DICOM_SERIES = [24, 25, 26, 27]

# Estimator & noise model. Full-volume default is wls_denoise (fast + memory-safe).
# wls_joint is the highest quality but iterative + memory-heavy -> run it on a SLAB only
# (set Z_SLAB_MM). Menu order (best->simplest): 'wls_joint'|'wls_denoise'|'wls'|'ols'.
ESTIMATOR = "wls_denoise"    # full-volume-safe; use 'wls_joint' + a Z_SLAB_MM for best quality
NOISE_MODEL = "spatial"      # 'spatial' | 'global'
DENOISE_METHOD = "tv"        # 'tv' | 'nlm' | 'bilateral' | 'guided' | 'gaussian'
DENOISE_SCALE = 1.0          # strength multiplier on the measured noise
DENOISE_GUIDE = False        # cross-channel guiding (guide = highest-SNR channel, chosen at runtime)
JOINT_ITERS = 10             # iterations for wls_joint

WATER_CALIBRATION = True     # per-bin unit scaling to NIST water (NOT a stability remedy)
# Full volume by default (None). The decomposition is memory-bounded (chunked solve + in-place
# HU->attenuation), so the full ~2000-slice x4-threshold volume fits the 64 GB SLURM budget.
# Set a (z_lo, z_hi) mm range to restrict to a slab -- useful for the DICOM/Siemens workflow
# (its z-axis is table position ImagePositionPatient, full extent ~ -2267.9..-1377.9 mm) or for
# quick dev runs / the iterative wls_joint estimator.
Z_SLAB_MM = None
INPUT_DIR = str(_REPO_ROOT / "output" / "reconstruction")
OUTPUT_DIR = str(_REPO_ROOT / "output" / "decomposition")
# ====================================================================


def _print_stability(result) -> None:
    s = result.stability
    mats = s["materials"]
    cos = s["column_cosines"]
    print("\n" + "=" * 66)
    print(f"STABILITY   mode '{result.config.mode}'   [{'/'.join(mats)}]")
    print("=" * 66)
    print(f"  condition number  kappa(M)     = {s['condition_number']:.1f}   -> {s['verdict'].upper()}")
    print(f"  kappa(M^T M) = kappa^2         = {s['condition_number_MtM']:.3g}")
    print(f"  column-normalised kappa        = {s['condition_number_colnorm']:.1f}")
    print(f"  singular values                = {['%.4g' % v for v in s['singular_values']]}")
    print("  per-material noise amplification  sqrt(diag((M^T M)^-1)):")
    for m, g in s["noise_amplification"].items():
        print(f"      {m:16s} {g:12.3f}")
    print("  column cosines (1.0 = collinear = singular):")
    print("           " + "".join(f"{m[:8]:>10s}" for m in mats))
    for i, m in enumerate(mats):
        print(f"    {m[:8]:>8s} " + "".join(f"{cos[i][j]:10.4f}" for j in range(len(mats))))
    print("=" * 66)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = DecompConfig(
        mode=MODE, threshold_option=THRESHOLD_OPTION, bin_domain=BIN_DOMAIN,
        estimator=ESTIMATOR, noise_model=NOISE_MODEL, denoise_method=DENOISE_METHOD,
        denoise_scale=DENOISE_SCALE, denoise_guide=DENOISE_GUIDE, joint_iters=JOINT_ITERS,
        water_calibration=WATER_CALIBRATION, input_dir=INPUT_DIR, output_dir=OUTPUT_DIR,
        z_slab_mm=Z_SLAB_MM, input_format=INPUT_FORMAT, dicom_dir=DICOM_DIR,
        dicom_series=DICOM_SERIES,
    )
    src = cfg.dicom_dir if cfg.input_format == "dicom" else cfg.input_dir
    print(f"Loading threshold volumes ({cfg.input_format}) from {src} ...")
    if cfg.input_format == "dicom":
        print(f"  series {cfg.dicom_series} -> thresholds {list(cfg.threshold_labels)}"
              f"   z_slab_mm={cfg.z_slab_mm}")
    volumes, ref, mu_water = load_threshold_volumes(cfg)
    print(f"  volumes {tuple(volumes.shape)}   mu_water(per threshold) = {mu_water}")
    print(f"  estimator={cfg.estimator}  noise_model={cfg.noise_model}  bin_domain={cfg.bin_domain}"
          + (f"  denoise={cfg.denoise_method}" if cfg.estimator in ("wls_denoise", "wls_joint") else ""))

    def prog(frac, msg):
        print(f"  [{frac * 100:5.1f}%] {msg}")

    result = decompose(volumes, cfg, mu_water=mu_water, progress=prog)
    _print_stability(result)

    written = save_decomp_result(result, ref, cfg.output_dir)
    print(f"\nWrote {len(written)} files to {cfg.output_dir}:")
    for p in written:
        print("   ", p.name)


if __name__ == "__main__":
    main()
