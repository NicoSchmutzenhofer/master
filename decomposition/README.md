# Material decomposition (4-bin PCCT)

Image-domain material decomposition of the reconstructed energy-threshold volumes, following
the professor's least-squares matrix model (whiteboard / [DECOMPOSITION_PLAN.md](DECOMPOSITION_PLAN.md)):

```
B = M x + n      ->      x = (Mᵀ M)⁻¹ Mᵀ B      (ordinary least squares)
```

- **B** — the 4 energy-bin linear attenuations at a voxel (from the reconstructions).
- **M** — 4×3 material-signature matrix, columns = bin-averaged `<μ/ρ>` (cm²/g) from NIST
  (the professor's Excel, ported to `data/mu_rho_binavg.csv`).
- **x** — the per-voxel partial densities (g/cm³) of the 3 basis materials (solved).

**Estimators** (config `estimator`, best→simplest): `wls_joint` · `wls_denoise` · `wls` · `ols`;
**noise model** (`noise_model`): `spatial` · `global`. All weights and denoise strengths are measured
from the data at runtime — nothing hard-coded (memory: adaptive-no-hardcoding).

**Status: Phase A + B implemented & validated (synthetic math).** Phase A = plain OLS + stability
audit; Phase B = data-adaptive WLS + edge-preserving denoising + joint solver. The GUI
([../docs/SOFTWARE_ROADMAP.md](../docs/SOFTWARE_ROADMAP.md)) is later.

## Layout

```
data/
  mu_rho_binavg.csv        NIST bin-averaged <μ/ρ>, Threshold Options 1/2/3 (Option 1 = the scan)
  materials.csv            density + K-edge + category per material
material_library.py        build_M(materials, option); material / bin / K-edge access
decomposition_modes.py     mode registry: clinical question -> 3 basis materials (extensible)
noise_estimation.py        data-adaptive noise covariance (global + spatial), no-reference
denoising.py               edge-preserving denoise registry (tv/nlm/bilateral/guided) + guide select
material_decomposition.py  DecompConfig, decompose() (ols/wls/wls_denoise/wls_joint), stability, I/O
decompose.py               production driver: run ONE mode -> material maps + stability report
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

Inputs: `output/reconstruction_thr_{A,B,C,D}_HU.nii.gz` (cumulative thresholds A≥20…D≥75 keV;
configurable in `DecompConfig` — **confirm the actual filenames on the cluster**).
Outputs: `output/decomposition/`.

## Design notes

- **Config-object-first / GUI-ready:** `decompose(volumes, config, progress=…)` is the stable
  entry point; the drivers just build a serializable `DecompConfig`. Compute is array-in / array-out
  and I/O-free; all file reading/writing is the SimpleITK section at the bottom of the library.
- **Extensibility (open-source goal):** add a material = one column in `mu_rho_binavg.csv` + one row
  in `materials.csv`; add a mode = one `register_mode(...)`; add a research method = one `@experiment`
  function. No core edits needed.
- **Cumulative → exclusive:** exclusive energy bins (`A−B, B−C, C−D, D`) are formed in the image
  domain (CLAUDE.md invariant #3), selectable via `bin_domain`. Which bin domain works better is an
  open question resolved empirically by the research harness.
- **Siemens-closed corrections:** charge sharing / pile-up / scatter are applied in-detector on the
  raw counts and are not reproducible here — this is an image-domain toolkit by necessity. The QRM
  Dual-Energy Phantom V5 (matched iodine/calcium inserts at equal 120 kV HU) is the validation target.

## Key Phase-A finding (stability)

Only bases containing a spectrally distinct (ideally K-edge) material are stable. With the scan
thresholds (Option 1): iodine bases κ≈160 (usable); tissue-only bases κ≈6,000–12,500 (unstable),
because low-Z tissue signatures are near-collinear (water vs soft-tissue cosine 0.99998). See
[DECOMPOSITION_PLAN.md](DECOMPOSITION_PLAN.md) §3 and `output/decomposition/research/`.
