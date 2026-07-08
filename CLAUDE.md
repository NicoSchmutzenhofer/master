# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Master's thesis code for **4-bin photon-counting CT (PCCT) reconstruction** on the Siemens NAEOTOM Alpha scanner (P63 detector, M4 collimation mode). The pipeline performs **Single-Slice Rebinning (SSR) helical reconstruction** of each of the 4 energy threshold sinograms independently — the per-threshold volumes are then the input for downstream material decomposition.

A Git repository organised into folders: reconstruction code in `reconstruction/`, detector-geometry inputs in `geometry/`, design docs in `docs/`, and `decomposition/` holding the material-decomposition stage (Phases A+B: image-domain least-squares + data-adaptive WLS / edge-preserving denoise / joint estimators). `CLAUDE.md` and `batch.sh` stay at the repo root; `output/` and `logs/` are gitignored. No test suite or package config — a small set of standalone scripts.

## Running

The working directory on this Windows machine is a mirror of the code that actually runs on a SLURM-managed Linux GPU cluster. Raw data lives at `/data/Data2/4_BIN_PCCT/...` (Linux paths hardcoded in scripts) and reconstruction needs a CUDA GPU for ASTRA's `FBP_CUDA`.

```bash
# On the cluster — submit the job
sbatch batch.sh        # runs reconstruction/python_reconstruction.py on node 'sauron', rtx6000ada GPU
```

`batch.sh` activates the conda env `/home/nisc24/.conda/envs/MatDecomp` (Python + numpy, scipy, h5py, SimpleITK, astra-toolbox, pywavelets, scikit-image, matplotlib).

### Tuning loop

[python_reconstruction.py](reconstruction/python_reconstruction.py) is the main entry point (the driver) and is mode-switched at the top of the file:

- `FAST_MODE = True` — slab preview for all 4 thresholds (~seconds/threshold). Use this when tuning preprocessing, MAR, or geometry. Reconstructs an image-domain average over `PREVIEW_SLAB_MM` (so preview SNR ≈ the full volume), HU-calibrates each threshold, and shows all four on the same `PREVIEW_HU_WL` window so HU separation is judged fairly. Outputs: `output/preview_4thresholds_fast.png` and `output/defect_mask_diagnostic.png` (cross-threshold defect-mask comparison).
- `FAST_MODE = False` — full helical volumes for all 4 thresholds (~hours). Writes one raw-attenuation NIfTI and one HU-calibrated `_HU.nii.gz` per threshold, plus a 4D `reconstruction_4thr_multienergy.nii.gz`.

Config knobs at the top of the driver (all have inline comment blocks explaining failure modes):

| Knob | Default | What it does |
|---|---|---|
| `FAST_MODE` | `True` | Single-slice/slab preview vs full volume |
| `RECON_METHOD` | `'fbp'` | Reconstruction algorithm: `'fbp'` (default, established) / `'sirt'` / `'cgls'` (iterative, ASTRA). FBP behaviour is unchanged while `'fbp'`. |
| `N_ITER` | `150` | Iterations for `'sirt'`/`'cgls'` (ignored for FBP). Full-volume iterative is ~2·N_ITER× slower per slice — use deliberately |
| `PREVIEW_SLAB_MM` | `2.0` | FAST preview: image-domain average over this slab thickness so preview SNR ≈ full volume. `0` = single native slice |
| `SPIKE_MAD_K` / `IPR_MAD_K` | `5.0` / `6.0` | Defect-detection MAD multipliers; **raise** to mask fewer channels (see `output/defect_mask_diagnostic.png`) |
| `PREVIEW_HU_WL` | `(40, 400)` | HU window `(level, width)` for the FAST preview when HU calibration is on |
| `FILTER_NAME` | `'shepp-logan'` | FBP filter. Use `'ram-lak'` for MTF/geometry tests only |
| `GEOMETRY_MODEL` | `'curved'` | `'curved'` = correct equiangular→flat remap; `'flat'` = legacy |
| `Z_WEIGHTING` | `'balanced'` | Helical rebinning weight: `'balanced'` (angularly-balanced 360°LI — removes the rotating low-frequency "light cone") / `'hann'` (legacy raised-cosine z-window, the rotating lobe) / `'none'` (uniform). Sinogram-formation only, so FBP/SIRT/CGLS all benefit |
| `WAVELET_RING_THRESHOLD` | `2.0` | Gate for adaptive wavelet stripe removal; `0` = always on, `999` = always off |
| `ENABLE_HU_CALIBRATION` | `True` | Auto-detect µ_water/µ_air and write HU-scaled NIfTIs |
| `Z_SMOOTH_MM` | `0.0` | Post-recon Gaussian z-smoothing (FWHM mm). 1.5–3 for SNR boost on low-contrast inserts |
| `MAR_STRENGTH` | `0.0` | Metal artifact reduction blend (0 = off) |
| `HAMPEL_THRESHOLD` | `None` | Per-projection spike suppression (None = off) |

### Diagnostic scripts

Before touching reconstruction code, when the `.mat` file format is unfamiliar:
- [mat_info.py](reconstruction/mat_info.py) — detects MATLAB v5 vs v7.3/HDF5, prints top-level variable shapes.
- [mat_structure.py](reconstruction/mat_structure.py) — recursive struct/cell inspector. Used to map out `descriptor.Config`, `descriptor.ScanDescr`, etc.

## Architecture

### File layout

Folders (entry points `batch.sh` and `CLAUDE.md` stay at the root; `output/`, `logs/` are gitignored):
- `reconstruction/` — all reconstruction code (library, driver, invariants, `.mat` diagnostics, bin-separation).
- `geometry/` — detector-geometry inputs (`beta_*`/`zIso_*` text files + `Geo_P63.pdf`).
- `docs/` — design/roadmap docs.
- `decomposition/` — material-decomposition stage (Phases A+B: image-domain LS + stability audit; data-adaptive WLS / edge-preserving denoise / joint estimators, all selectable). A Python package (`python -m decomposition.decompose`) with its own `DECOMPOSITION_PLAN.md` / `README.md`.

- **[helical_reconstruction.py](reconstruction/helical_reconstruction.py)** — pure library, no I/O of raw data, no `__main__`. All algorithm code: geometry build, defect detection, rebinning, preprocessing, MAR, reconstruction (`_astra_reconstruct` dispatches FBP/SIRT/CGLS; `_astra_fbp` is a thin wrapper), `reconstruct_slab` (image-domain slab averaging), HU calibration.
- **[python_reconstruction.py](reconstruction/python_reconstruction.py)** — the driver: loads HDF5, loops over thresholds, calls library functions, writes outputs.
- **[recon_invariants.py](reconstruction/recon_invariants.py)** — invariant/assertion module. **Do not modify check logic** — only append new checks. Called by the driver at every key pipeline stage; hard-fails on geometry errors, soft-warns everything else into `output/invariant_log.json`.
- **[bin_separation_investigation.py](reconstruction/bin_separation_investigation.py)** — standalone, image-domain investigation of threshold separation for image quality (label-free; reads the reconstructed HU volumes). Concluded **negative** for image quality (cumulative bins share photons → correlated noise; see plan §4a); kept for the later material-decomposition work.
- **[IMAGE_QUALITY_PLAN.md](docs/IMAGE_QUALITY_PLAN.md)** — the design/roadmap doc behind the current revision. Explains *why* the noise/HU-separation knobs exist (WS1: `PREVIEW_SLAB_MM`, FAST-mode HU calibration, `SPIKE_MAD_K`/`IPR_MAD_K`; WS2: `RECON_METHOD`/`N_ITER` iterative recon) and records the project's non-breaking guarantees. **Deferred (out of scope, §4):** spectral-guided denoising (use high-SNR threshold A to guide the others), TV-regularized iterative, exclusive energy-bin images (A−B, B−C, C−D), and the material-decomposition stage itself. Read it before proposing image-quality changes so you don't re-litigate decided trade-offs or pull deferred work forward.

### Data pipeline (per threshold)

```
HDF5 file (4 thresholds, stored REVERSED: physical 0→D, 1→C, 2→B, 3→A)
    │  _load_threshold(f, logical_idx) handles the 3-logical_idx mapping
    │  and reverses channels: data[:, :, ::-1]
    ▼
sino_full : float32 [N_proj, n_rows, n_channels]  ≈ 16 GB
    │
    ├─► detect_defect_channels(sino_full)  ── ONCE per scan ──► geom['spike_mask']
    │   (runs on threshold A only; mask reused for B/C/D)
    │
    └─► for each z_target:
            rebin_helical_to_axial(...)        # SSR: linear interp between
                                               # detector rows bracketing z_offset
            preprocess_sinogram(...)           # defect interp, wavelet stripe
                                               # removal, air baseline subtract
            apply_cor_shift(...)               # sub-pixel COR correction
            [apply_mar(...) if enabled]        # first-pass FBP → segment metal
                                               # → forward-project → inpaint+blend
            _astra_fbp(...)                    # fan-beam FBP_CUDA, Ram-Lak
                                                 ▼
                                          slice [n_pixels, n_pixels]
```

### New pipeline steps (added in latest revision)

1. **Curved-detector remap** (`_remap_curved_to_flat` in `helical_reconstruction.py`) — inside `_astra_fbp` when `geometry_model='curved'` (default). Applies cosine pre-weight `cos(β_k)` then cubic interpolation from equiangular channel positions to equispaced flat-detector positions (`s = SDD·tan(β_k)`). Fixes radial position distortion and peripheral HU cupping. Only system A data (1376 channels, `beta_M4_A.txt`) — system B not available for researcher use.
2. **Shepp-Logan filter default** — replaces Ram-Lak. Reduces noise ~30–60% without MTF impact at the center. Set `FILTER_NAME='ram-lak'` for geometry/MTF verification runs.
3. **Adaptive wavelet stripe gating** — wavelet-FFT stripe removal (`remove_stripes_wavelet_fft`) now only runs when the sinogram's stripe-SNR proxy exceeds `WAVELET_RING_THRESHOLD`. Prevents it from washing out low-contrast inserts when no rings are present.
4. **Helical projection weighting** (`z_weighting` in `rebin_helical_to_axial`, knob `Z_WEIGHTING`, default `'balanced'`) — controls how the rays of a rebinned rotation are weighted. The legacy **raised-cosine z-window** (`'hann'`, also `z_window=True`) tapers the projection axis to suppress 10–25 HU z-shading bands, but over a one-rotation window it doubles as an **angular apodization whose peak rides the helix** → a low-frequency brightness lobe that **rotates as you scroll z** (the "light cone"). `'balanced'` is **angularly-balanced helical weighting** (360°LI / complementary rebinning, Crawford & King 1990): it spans ~2 rotations, tapers each ray by `|z_offset|` (kills the z-shading) **and normalises every view angle to equal total weight** (kills the rotating bias). It is purely a sinogram-formation change — collapses to the same `(n_proj, n_ch)` contract and one ray per canonical view angle — so FBP/SIRT/CGLS all benefit identically and the recon code is untouched. `'none'` = uniform (no rotation, z-bands return). `check_angular_balance` (recon_invariants, soft) verifies the per-angle weight sum is 1. The wider window trims one rotation (not half) at each scan end — the driver passes `end_margin_rotations` to `z_targets_for_full_scan` accordingly.
5. **Auto HU calibration** (`auto_hu_calibrate` / `apply_hu_calibration`) — Otsu body segmentation → 10 mm erosion → mode-based µ_water → median µ_air. Cached per-threshold to `output/calibration_thr_<label>.json`. Outputs both raw-attenuation and HU-scaled NIfTIs.

### Critical, non-obvious invariants

1. **Threshold storage order is reversed in the HDF5.** Physical index 0 holds threshold D (highest, fewest photons), physical index 3 holds A (lowest, most photons). `_load_threshold` does `3 - logical_idx` so code can use logical A=0..D=3 freely. Any new HDF5 access must do the same mapping.

2. **Channels are flipped at load time** (`data[:, :, ::-1]`), and `build_geom` is called with `channels_flipped=True` (it negates `det_alignment` accordingly). **This flip is required — do not remove it.** Reconstructing in *native* channel order makes the full volume develop a **z-direction helix** (the object spirals along the table axis), confirmed by a full-volume A/B test. In fan-beam a detector-channel flip is equivalent to the gantry **rotation sense**, so the flip sets the correct helical handedness together with `_VIEW_ANGLE_SIGN = +1`; un-flipping exposes the wrong sense as a spiral (it does *not* simply mirror — it 180°-rotates and spirals). If you ever change the flip, change the flag **and** re-verify there is no helix in coronal/sagittal. A residual left–right mirror vs Siemens, if any, must be fixed in the **output** (a 2-D flip / NIfTI direction matrix), never by un-flipping channels.

3. **Threshold sinograms are NOT in line-integral domain.** Object voxels → HIGH values, air → LOW values (opposite of Beer-Lambert). Reconstruct each threshold image directly; do **not** subtract adjacent thresholds to get energy-bin sinograms — the scanner's per-threshold gain calibration breaks monotone ordering A≥B≥C≥D in 44–54% of samples. Energy-bin images are an image-domain post-processing step on the reconstructed volumes.

4. **Spike detection uses `max_excess = col_max − p99`, not IPR.** Inter-module gap spikes fire at <1% of projections, so percentile-based detection (p99) reports their *background* level. See the docstring of [`detect_defect_channels`](reconstruction/helical_reconstruction.py) for the full reasoning. Don't switch this back to IPR-only. Dilation is **±1 channel** (not ±3): scans with metal produce many spike channels 4–6 apart; ±3 bridges them and masks ~21% of central channels, far worse than the spikes themselves.

5. **Hampel filter (`suppress_projection_spikes`) is OFF by default.** Inside a water phantom the local MAD is small enough that bone-equivalent inserts look like spikes and get replaced by the water median, compressing bone HU to near water. Hard-defect masking (step 1 of `preprocess_sinogram`) handles real spikes. Only enable with `hampel_threshold ≥ 20` if rings persist after reviewing the defect-mask diagnostics.

6. **`row_zIso` must straddle z=0** or SSR interpolation clips at row boundaries. `_check_row_mapping` raises with a hint to try `row_mapping='first' | 'last' | 'reversed'` instead of `'central'`.

7. **All scan-dependent values are derived from the descriptor / geometry text files** — never hardcode slice width, active rows, or air-channel count. `build_geom` reads `SliceWidthCollimated` / `NoOfSlicesCollimated` from `ModeParXML` when present, falls back to deriving from `zIso_M4.txt` and `NoOfSlices` otherwise.

8. **The reconstruction view angle is the physical tube angle, not the projection index.** `build_geom` derives `geom['view_angle_per_proj']` from `ScanDescr.Det.FirstTubeAngle` (detector A, millidegrees) + a uniform `360/FramesPerRotation` increment, and `rebin_helical_to_axial` uses it. Assuming reading 0 = angle 0 (the old behaviour) ignores the ~151° start angle and rotates the whole image (e.g. the patient table appears on the side instead of the bottom). The fixed Siemens-gantry→ASTRA-fanflat convention is captured by two module constants `_VIEW_ANGLE_SIGN` / `_VIEW_ANGLE_OFFSET_DEG` (NOT scan data — same for every M4/system-A scan). `SystemAngle`/`TubeBOffsetAngle` (~95.7°) is the **dual-source A↔B mounting offset** and is deliberately unused (we reconstruct system A only). `check_orientation` (recon_invariants, soft) logs the measured table angle each run so `_VIEW_ANGLE_OFFSET_DEG` can be locked once and is then permanent — never per-scan, never an image rotate. **`FirstTubeAngle` is mandatory:** if `ScanDescr.Det.FirstTubeAngle` cannot be read, `build_geom` now raises rather than silently falling back to reading-0 = angle 0 (which produced a rotated volume), and `rebin_helical_to_axial` raises if `view_angle_per_proj` is missing. **`PatientPosition` is not in the raw descriptor** and is supplied via the driver's `PATIENT_POSITION` knob (default `'HFS'`). It does not affect the gantry-frame reconstruction (table always at the bottom) — only the array→patient (LPS) axis labelling (which side is Left/Right, Anterior/Posterior, Head/Foot). Only `'HFS'` is implemented/validated; other positions derive from it by a fixed flip table (FFS: flip L-R & head-foot; HFP: flip L-R & A-P; FFP: flip A-P & head-foot) and the knob errors out until each is added.

### Geometry text files

Six text files live in [geometry/](geometry/) (alongside the `Geo_P63.pdf` datasheet), named by collimation mode and detector half:

- `beta_M4_A.txt`, `beta_M4_B.txt` — channel fan angles (radians) for M4 mode, halves A and B.
- `beta_S1_A.txt`, `beta_S1_B.txt` — same for S1 mode.
- `zIso_M4.txt`, `zIso_S1.txt` — detector row z-positions at isocentre (mm).

`geometry/Geo_P63.pdf` is the Siemens P63 detector geometry datasheet — the reference source behind the channel fan angles and row z-positions in the text files above. Consult it (not the code) when a geometry value looks wrong.

The driver passes `geo_dir = <repo>/geometry` (resolved from `__file__`), and `build_geom` reads `beta_M4_A.txt` and `zIso_M4.txt` from there. Switching collimation mode means swapping the filenames in `build_geom`.

### MAR strength semantics

`apply_mar` blends original and metal-interpolated sinograms **only** in metal-contaminated bins. `strength=1.0` is full projection completion (zeros out metal attenuation in the reconstruction) — bad for material decomposition. `strength=0.3–0.5` is the recommended range when the reconstructed metal HU must be preserved.
