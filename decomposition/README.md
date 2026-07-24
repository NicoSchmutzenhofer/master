# Material decomposition (4-bin PCCT)

Image-domain material decomposition of the reconstructed energy-threshold volumes, following
the professor's least-squares matrix model (whiteboard / [DECOMPOSITION_PLAN.md](DECOMPOSITION_PLAN.md)):

```
B = M x + n      ->      x = (Mᵀ M)⁻¹ Mᵀ B      (ordinary least squares)
```

- **B** — the N energy-channel linear attenuations at a voxel (from the reconstructions). **N is
  discovered from the data, not fixed at 4:** threshold images (own recon / Siemens WFBP) or any
  number of monoenergetic VMIs.
- **M** — N×k material-signature matrix `<μ/ρ>` (cm²/g), **generated per channel** from NIST (the
  professor's Excel): bin-averaged over each threshold window (`data/mu_rho_binavg.csv`) or
  K-edge-aware monoenergetic interpolation at each VMI keV (`data/mu_rho_mono.csv`).
- **x** — the per-voxel partial densities (g/cm³) of the k basis materials (solved). Requires N ≥ k.

**Estimators** (config `estimator`, best→simplest): `wls_joint` · `wls_denoise` · `wls` · `ols`;
**noise model** (`noise_model`): `spatial` · `global`. All weights and denoise strengths are measured
from the data at runtime — nothing hard-coded (memory: adaptive-no-hardcoding).

**Status: Phase A + B implemented & validated (synthetic math).** Phase A = plain OLS + stability
audit; Phase B = data-adaptive WLS + edge-preserving denoising + joint solver. The stage is
**N-channel and multi-source**: one pipeline decomposes our threshold recon, Siemens WFBP
thresholds, or Siemens VMIs (channel count discovered from the data). The GUI
([../docs/SOFTWARE_ROADMAP.md](../docs/SOFTWARE_ROADMAP.md)) is later.

## Layout

```
data/
  mu_rho_binavg.csv        NIST bin-averaged <μ/ρ> over threshold windows, Options 1/2/3 (Opt 1 = scan)
  mu_rho_mono.csv          NIST monoenergetic <μ/ρ> vs keV per material, K-edge doublets (for VMI)
  materials.csv            density + K-edge + category per material
build_mono_table.py        one-off generator: Excel per-element sheets -> data/mu_rho_mono.csv
material_library.py        build_M(materials, option|channels); Channel; mono_mu_rho() K-edge-aware
decomposition_modes.py     mode registry: clinical question -> basis materials (extensible)
noise_estimation.py        data-adaptive noise covariance (global + spatial), no-reference
denoising.py               edge-preserving denoise registry (tv/nlm/bilateral/guided) + guide select
material_decomposition.py  DecompConfig, decompose() (N-channel; ols/wls/wls_denoise/wls_joint),
                           stability + per-material reliability flag, load_energy_stack (own/vmi/wfbp), I/O
decompose.py               driver: INPUT_SOURCE own|wfbp|vmi -> output/decomposition/<source>/
research_decomposition.py  ablation harness (stability, cosine, threshold scan, bin-domain,
                           estimator-ladder panels + no-reference metrics) -> the GUI Research mode
selftest_decomposition.py  synthetic self-test (numpy + scipy/scikit-image)
```

## Run (from the repo root)

```bash
python -m decomposition.selftest_decomposition      # math self-test (Phase A + B)
python -m decomposition.research_decomposition       # κ tables + figures + findings (no volumes needed)
python -m decomposition.decompose                    # one mode on the real volumes (needs SimpleITK + data)
```

**Input sources** (driver `INPUT_SOURCE` — one shared pipeline, only the loader differs):
- `own`  — our reconstruction `output/reconstruction/reconstruction_thr_{A,B,C,D}_HU.nii.gz`
  (threshold channels; uses however many are present — 3 works).
- `wfbp` — Siemens WFBP threshold DICOMs (same 20/40/56/75 keV scan thresholds).
- `vmi`  — Siemens monoenergetic (VMI) DICOMs; keV series auto-discovered from the folder.

Outputs go to `output/decomposition/<source>/` (own/wfbp/vmi never overwrite each other). Each run
prints and stores a per-material **reliability** flag naming the likely-degenerate material — for
K-edge-poor VMI 3-material solves this is expected, not a bug.

## Design notes

- **Config-object-first / GUI-ready:** `decompose(volumes, config, progress=…)` is the stable
  entry point; the drivers just build a serializable `DecompConfig`. Compute is array-in / array-out
  and I/O-free; all file reading/writing is the SimpleITK section at the bottom of the library.
- **Extensibility (open-source goal):** add a material = one column in `mu_rho_binavg.csv` + one row
  in `materials.csv`; add a mode = one `register_mode(...)`; add a research method = one `@experiment`
  function. No core edits needed.
- **bin_domain:** thresholds can be fed `cumulative` (as-is) or `exclusive` (`A−B,…`, image-domain
  per invariant #3); monoenergetic VMIs are never subtracted (`direct`, auto-selected for mono
  channels). Which works better is resolved empirically by the research harness.
- **Siemens-closed corrections:** charge sharing / pile-up / scatter are applied in-detector on the
  raw counts and are not reproducible here — this is an image-domain toolkit by necessity. The QRM
  Dual-Energy Phantom V5 (matched iodine/calcium inserts at equal 120 kV HU) is the validation target.

## Key Phase-A finding (stability)

Only bases containing a spectrally distinct (ideally K-edge) material are stable. With the scan
thresholds (Option 1): iodine bases κ≈160 (usable); tissue-only bases κ≈6,000–12,500 (unstable),
because low-Z tissue signatures are near-collinear (water vs soft-tissue cosine 0.99998). See
[DECOMPOSITION_PLAN.md](DECOMPOSITION_PLAN.md) §3 and `output/research/`.
