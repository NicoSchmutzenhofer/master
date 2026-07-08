# 4-bin Photon-Counting CT Reconstruction

Master's-thesis code for **4-bin photon-counting CT (PCCT) reconstruction** on the Siemens
NAEOTOM Alpha scanner (P63 detector, M4 collimation). The pipeline performs **Single-Slice
Rebinning (SSR) helical reconstruction** of each of the four energy-threshold sinograms
independently — the per-threshold volumes are the input for a downstream material-decomposition
stage.

## Repository structure

```
reconstruction/   reconstruction code
    helical_reconstruction.py             pure library (geometry, rebinning, MAR, FBP/SIRT/CGLS, HU calib)
    python_reconstruction.py              the driver (entry point; mode-switched at the top)
    recon_invariants.py                   invariant/assertion checks (append-only)
    bin_separation_investigation.py       image-domain threshold-separation study (label-free)
    sinogram_separation_investigation.py  sinogram-domain threshold-separation study (SIRT; needs raw .mat + GPU)
    mat_info.py / mat_structure.py        .mat inspection diagnostics
geometry/         detector-geometry inputs: beta_*/zIso_* text files + Geo_P63.pdf datasheet
docs/             IMAGE_QUALITY_PLAN.md, BIN_SEPARATION_FINDINGS.md, SOFTWARE_ROADMAP.md
decomposition/    material-decomposition stage (Phase A+B; python -m decomposition.decompose)
output/           run outputs (gitignored): reconstruction/ · research/<name>/ · decomposition/
logs/             SLURM logs (gitignored)
CLAUDE.md         detailed developer guidance (architecture, invariants, knobs)
batch.sh          SLURM submit script (reconstruction + decomposition)
batch_sinogram.sh SLURM submit script (sinogram-separation investigation)
```

## Running

The reconstruction runs on a SLURM-managed Linux GPU cluster (raw data at `/data/Data2/4_BIN_PCCT/...`,
ASTRA needs a CUDA GPU). This repo is the source mirror.

```bash
sbatch batch.sh        # runs reconstruction/python_reconstruction.py
```

`batch.sh` activates the conda env `MatDecomp` (numpy, scipy, h5py, SimpleITK, astra-toolbox,
pywavelets, scikit-image, matplotlib). The driver resolves `geometry/` and `output/` from its own
location, so it runs from any working directory.

Tuning is mode-switched at the top of [reconstruction/python_reconstruction.py](reconstruction/python_reconstruction.py):
`FAST_MODE = True` for a quick slab preview of all four thresholds, `False` for the full helical
volumes. See the knob table and architecture notes in [CLAUDE.md](CLAUDE.md).

## Status

- **Reconstruction:** done. Includes a curved-detector remap, descriptor-derived orientation, HU
  calibration, iterative (SIRT/CGLS) options, and angularly-balanced helical weighting (`Z_WEIGHTING`)
  that removes the rotating low-frequency "light-cone" artifact.
- **Bin separation for image quality (image domain):** investigated and found **not** to help
  (cumulative thresholds share photons → correlated noise; see
  [docs/BIN_SEPARATION_FINDINGS.md](docs/BIN_SEPARATION_FINDINGS.md) and
  [docs/IMAGE_QUALITY_PLAN.md](docs/IMAGE_QUALITY_PLAN.md) §4a). Tooling kept for the decomposition step.
- **Sinogram-domain separation:** standalone investigation
  ([reconstruction/sinogram_separation_investigation.py](reconstruction/sinogram_separation_investigation.py))
  testing whether subtracting threshold sinograms is quantum noise or a gain bias (the premise of
  invariant #3) and whether it feeds decomposition. Does not change the production pipeline.
- **Material decomposition:** Phase A+B implemented (image-domain least-squares + data-adaptive
  WLS / edge-preserving denoise / joint estimators; see `decomposition/`). Reads our reconstruction
  (`output/reconstruction/`) by default; DICOM/Siemens input switchable.
