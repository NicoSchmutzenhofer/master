"""
decompose.py -- production driver for image-domain material decomposition.

Runs ONE clinical mode over ONE input source and writes the material maps + reliability +
stability. Config-object-first (GUI-ready): the constants below just build a DecompConfig; a GUI
would build the same object and call decompose().

THREE input sources share ONE pipeline -- only the loader differs (material_decomposition
.load_energy_stack), and the energy-channel COUNT is discovered from the data, never hardcoded:

    INPUT_SOURCE = 'own'  -> our reconstruction: output/reconstruction/reconstruction_thr_*_HU.nii.gz
                            (threshold channels; uses however many are present)
                   'wfbp' -> Siemens WFBP threshold DICOMs (same 20/40/56/75 keV scan thresholds)
                   'vmi'  -> Siemens monoenergetic (VMI) DICOMs (keV series auto-discovered)

Outputs go to output/decomposition/<source>/ so the three approaches never overwrite each other.
Estimator starts at OLS (rawest baseline); advance to wls / wls_denoise / wls_joint later. The
per-material RELIABILITY block (printed to stdout and stored in metadata) flags which material is
likely degenerate -- especially useful for VMI, where a 3-material solve is near-degenerate until
the low-keV (K-edge-straddling) VMIs are available.

Run from the repo root:
    python -m decomposition.decompose
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

# --- import shim: work both as `-m decomposition.decompose` and as a script path ---
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decomposition.material_decomposition import (
    DecompConfig, decompose, load_energy_stack, save_decomp_result)

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ============================== CONFIG ==============================
INPUT_SOURCE = os.environ.get("DECOMP_SOURCE", "own").lower()   # 'own'|'wfbp'|'vmi'; batch.sh sets it from $1
MODE = "phantom_ca_i"         # see decomposition_modes.list_modes()
THRESHOLD_OPTION = 1          # threshold windows for 'own'/'wfbp' (VMI ignores this)

# --- estimator: start simple (OLS = rawest baseline); advance later ---
ESTIMATOR = "ols"             # 'ols' -> 'wls' -> 'wls_denoise' -> 'wls_joint'
NOISE_MODEL = "global"        # ignored by OLS; used by wls*/joint
BIN_DOMAIN = "cumulative"     # thresholds: 'cumulative' (no subtraction) | 'exclusive'; VMI auto -> 'direct'
WATER_CALIBRATION = False     # baseline: physical per-channel mu_water, no per-bin gains

# --- input locations -------------------------------------------------------------
OWN_DIR = str(_REPO_ROOT / "output" / "reconstruction")     # our NIfTI recon (INPUT_SOURCE='own')
# Siemens DICOM export root (holds the Mono_* and WFBP_T* series folders); used for 'wfbp'/'vmi'.
SIEMENS_DIR = "/data/Data2/4_BIN_PCCT/Reconstructions/4-bin_Phantom-Scan/Thx.- Abdomen Staging_Standard - PNR_20260715_151401"
Z_SLAB_MM = None              # (z_lo, z_hi) mm to restrict to a slab; None = full volume

OUTPUT_ROOT = str(_REPO_ROOT / "output" / "decomposition")  # per-source subfolder added below
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
    src = INPUT_SOURCE.lower()
    if src not in ("own", "wfbp", "vmi"):
        raise ValueError(f"INPUT_SOURCE must be 'own'|'wfbp'|'vmi' (got {INPUT_SOURCE!r})")

    cfg = DecompConfig(
        mode=MODE, threshold_option=THRESHOLD_OPTION, bin_domain=BIN_DOMAIN,
        estimator=ESTIMATOR, noise_model=NOISE_MODEL, water_calibration=WATER_CALIBRATION,
        input_source=src,
        input_format=("nifti" if src == "own" else "dicom"),
        input_dir=OWN_DIR, dicom_dir=SIEMENS_DIR, z_slab_mm=Z_SLAB_MM,
        output_dir=str(Path(OUTPUT_ROOT) / src),
    )

    print(f"=== material decomposition   source='{src}'   mode='{cfg.mode}'   estimator='{cfg.estimator}' ===")
    volumes, channels, ref = load_energy_stack(cfg)
    print(f"  loaded {len(channels)} channels {[c.label for c in channels]}  volumes {tuple(volumes.shape)}")

    def prog(frac, msg):
        print(f"  [{frac * 100:5.1f}%] {msg}")

    result = decompose(volumes, cfg, channels=channels, progress=prog)
    _print_stability(result)

    written = save_decomp_result(result, ref, cfg.output_dir)
    print(f"\nWrote {len(written)} files to {cfg.output_dir}:")
    for p in written:
        print("   ", p.name)


if __name__ == "__main__":
    main()
