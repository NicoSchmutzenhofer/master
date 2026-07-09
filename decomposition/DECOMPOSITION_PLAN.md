# Material Decomposition — Plan & Stability Analysis (4-bin PCCT)

Status: **Phase A + Phase B implemented & validated (synthetic math).** Phase A = plain-OLS
`decompose()` + full stability audit (real-data run done: iodine separates, HA noise-limited — the
motivation for Phase B). Phase B = data-adaptive noise estimation (`noise_estimation.py`, global +
spatial), the estimator ladder `ols`/`wls`/`wls_denoise`/`wls_joint`, an edge-preserving denoise
registry (`denoising.py`), a research/ablation harness (estimator-ladder panels + no-reference
metrics), and driver wiring — all self-tested (noise estimator finds the noisiest bin, WLS beats OLS,
denoise preserves edges, joint cuts flat-region noise). The real-data Phase-B run on the cluster is
next. This document is the design + the pre-computed
stability answer to the professor's two questions ("are the transformations applicable?"
and "is the matrix stable?"). It is the decomposition-stage counterpart to
[../docs/IMAGE_QUALITY_PLAN.md](../docs/IMAGE_QUALITY_PLAN.md).

Inputs are the finished reconstruction outputs (`output/reconstruction/reconstruction_thr_{A,B,C,D}_HU.nii.gz`);
the reconstruction pipeline is **not** touched.

---

## 1. Method — what this stage does

Each voxel has **four numbers** (one reconstructed attenuation per energy bin). Material
decomposition models those four numbers as a weighted sum of a chosen set of **3 materials**,
where the weights are the materials' known bin-averaged X-ray attenuations (from NIST). This is
the professor's whiteboard model:

```
b_i = (μ/ρ)_{i,mat1}·x1 + (μ/ρ)_{i,mat2}·x2 + (μ/ρ)_{i,mat3}·x3 + n_i        (per bin i = 1..4)

  →   B = M · x + n           B: 4×1 bin values (per voxel)
                              M: 4×3 material-signature matrix (KNOWN, from the Excel)
                              x: 3×1 material amounts = partial density g/cm³ (UNKNOWN)
                              n: noise

  →   x = (Mᵀ M)⁻¹ Mᵀ B       ordinary least squares (the pseudoinverse)
```

- This is the standard **image-domain** decomposition (Niu 2014; Xue 2017), chosen because we
  only have the reconstructed bin volumes, not the raw projection counts (the projection-domain
  Alvarez–Macovski 1976 approach needs the counts and models beam hardening — not available here).
- With **4 bins and 3 materials the system is overdetermined** (4 > 3). Consequence: unlike
  classic dual-energy 3-material decomposition, we do **not** need the volume-conservation /
  sum-to-one assumption to get a solution — least squares suffices. (Sum-to-one becomes an
  *optional* stabiliser later, not a requirement.)
- **Domain / units:** M columns are `<μ/ρ>` (cm²/g), so B must be **linear attenuation (1/cm)** and
  the solved `x` is **partial density (g/cm³)**. We convert the reconstructed HU volumes to 1/cm
  using the NIST water value per bin as the reference (`μ = μ_water·(1 + HU/1000)`), so B and M are
  tied to the *same* NIST source and are self-consistent.

---

## 2. What the professor's Excel provides (`NIST_XrayMassCoef_Elements_BinSignature.xlsx`)

- **Sheet `Bin-Averaged μ÷ρ`** — the M-matrix library: bin-averaged `<μ/ρ>` (cm²/g) for **12
  materials** (Water, Fat, Soft Tissue, HA, Iodine, Iron, Titanium, Stainless Steel, CoCr,
  Tantalum, Platinum, Gold) × **4 energy bins**, from K-edge-aware numerical integration of a
  140 kV spectrum.
- **Three Threshold Options.** **Option 1 = 20‑40 / 40‑56 / 56‑75 / 75‑140 keV = the actual scan
  thresholds (20/40/56/75).** Options 2 & 3 are alternative bin placements (for a "would other
  thresholds be more stable?" study). *Option 1 is the production matrix.*
- Documented **material compositions + densities** (ICRU adipose & soft tissue, HA formula, pure
  elements) and the **18 per-element NIST sheets** (H…Au) as provenance for the composites.
- **In-range K-edges** flagged: **Iodine 33.17 keV → bin 1**, Tantalum 67.4 → bin 3, Platinum 78.4
  & Gold 80.7 → bin 4. Iron (7.1) and Ti (5.0) K-edges are below range. This is the key to §3.

The Excel bins are **exclusive** windows; the scanner reconstructs **cumulative** thresholds — see
the bin-domain experiment in §6.

---

## 3. Stability analysis — the answer, pre-computed

κ(M) computed for every mock-up mode from the Option-1 coefficients. **Rule of thumb: κ < ~30 good,
~100 marginal, > ~1000 unusable without regularisation.**

| Clinical mode | Materials | κ(M) | Verdict |
|---|---|---:|---|
| Anaemia *(whiteboard)* | Soft tissue / **Iodine** / Iron | **166** | usable |
| **Test mode (phantom layer B)** | **Soft tissue / HA / Iodine** | *(≈ same class as above — iodine-driven)* | usable |
| Gout | Soft tissue / HA / Iron | 990 | poor |
| Peri-implant (CoCr) | Soft tissue / HA / CoCr | 950 | poor |
| Peri-implant (Ti) | Soft tissue / HA / Titanium | 1,660 | unstable |
| Kidney stone | Water / Fat / HA | 6,300 | unstable |
| Bone / Vessels / Soft-tis / Marrow | Soft tissue / Fat / HA | 7,700 | unstable |
| Liver (iron+steatosis) | Water / Fat / Iron | 12,600 | unstable |
| *(pathological check)* | Water / Fat / Soft tissue | 8,400 | unstable |

**Why (collinearity of the signatures — cosine 1.0 = identical columns = singular):**

| Pair | cosine |
|---|---:|
| Water vs Soft tissue | **0.99998** |
| Water vs Fat | 0.9955 |
| Fat vs Soft tissue | 0.9948 |
| HA vs Iron | 0.9982 |
| Soft tissue vs HA | 0.9191 |
| Soft tissue vs Iodine | 0.9561 |
| HA vs Iodine | 0.9121 |
| Iron vs Iodine | **0.8943** |

**Physical conclusion (this is the direct answer to the professor):** any material with **no K-edge
in 20–140 keV** (water, fat, soft tissue, HA, iron, Ti, CoCr) has a smooth, monotonically-falling
attenuation curve, so their 4-bin signatures are near-parallel → the matrix is **intrinsically
ill-conditioned for any tissue-only basis.** The `Water/Fat/Soft-tissue` case proves this is real
collinearity, not a units artefact: all three columns are ~0.16–0.44 (no magnitude disparity) yet
κ = 8,400. **Only a K-edge material (iodine here) — or a very dense/high-Z one — injects a linearly
independent column** and pulls κ into the usable range. That is exactly why the phantom test mode
(with iodine) is well-conditioned and the non-contrast tissue modes are not.

**Two real-data caveats** (make the practical case harder than this ideal table, and are what the
research pipeline quantifies):
1. Forming exclusive bins requires **subtracting cumulative reconstructions**, which adds correlated
   noise (the same nested-bin correlation documented in the bin-separation study, §4a of the image
   plan). Part of the raw κ for metal/iodine modes is also column-magnitude disparity, partly
   removable by **column scaling** — the tool reports both raw and scaled κ.
2. HU→attenuation conversion must be pinned to a reference; per-threshold gain calibration can break
   monotone bin ordering in some voxels (CLAUDE.md invariant #3), so **non-negativity helps**.

**Verdict:** the transformation is *applicable* (the model is exactly correct and well-established),
but *stable* only for bases containing a spectrally distinct material. Non-contrast tissue bases need
the Phase-B stabilisers to be usable.

---

## 4. Architecture

New `decomposition/` module, mirroring `reconstruction/` (pure library + driver + registries),
reusing reconstruction outputs. Two entry points: a **clean production driver** and a separate
**research/ablation harness**.

```
decomposition/
  data/
    mu_rho_binavg.csv          # Option 1/2/3 <μ/ρ> ported from the Excel (checked in; no Excel on cluster)
  material_library.py          # load table + densities + K-edge meta; build_M(materials, option) -> 4×3
  decomposition_modes.py       # mode registry: mode name -> [3 materials]  (all 8 mock-up modes + custom)
  material_decomposition.py    # LIBRARY: HU->μ, cumulative<->exclusive, LS solve, stability metrics
  decompose.py                 # PRODUCTION driver: pick MODE, run, write material maps + stability report
  research_decomposition.py    # RESEARCH harness: compare approaches, emit thesis figures + findings.md
  DECOMPOSITION_PLAN.md        # this file
  README.md                    # existing placeholder (update when implementation starts)
```

Production and research **share the same library** (`material_decomposition.py`,
`material_library.py`), exactly as `image_subtraction_investigation.py` imports the reconstruction
library. `decompose.py` runs one mode cleanly; `research_decomposition.py` loops over approaches.

**Data flow (per mode):**
```
output/reconstruction/reconstruction_thr_{A,B,C,D}_HU.nii.gz
   → HU → linear attenuation (1/cm), NIST water reference per bin
   → form B:  exclusive [A−B, B−C, C−D, D]   OR   cumulative [A, B, C, D]   (research-selectable)
   → build M (4×3) from selected mode's 3 materials, Option 1
   → per-voxel solve  x = (MᵀM)⁻¹ Mᵀ B     (Phase A: plain OLS)
   → 3 material maps (g/cm³) + stability report (κ, scaled κ, SVD, per-material noise gain)
```

---

## 5. Phases (ordered to answer "is it stable?" honestly)

- **Phase A — plain matrix approach + stability audit (core deliverable).** Implement exactly
  `x = (MᵀM)⁻¹MᵀB`, **no regularisation**, and *report* κ / SVD / per-material noise on real data.
  You can only demonstrate stability if you have not already regularised it away. Validate the test
  mode against the phantom middle layer. **[Decision locked: plain OLS first.]**
- **Phase B — stabilisers (the fix).** Add as documented, comparable options: column scaling,
  non-negativity, optional volume-conservation (sum-to-one), and weighted-LS / Tikhonov using the
  measured noise covariance. Show before/after on the ill-conditioned tissue modes.
- **Phase C — full mode registry.** All 8 mock-up modes selectable, each with its own stability
  sheet; the professor's mode-dropdown vision.

### Phase B — detailed design (data-adaptive; finalized)

Goal: best **qualitative** image quality for a radiologist (no manual ground truth yet), with every
parameter measured from the data at runtime — **nothing hard-coded**, so the filtering self-corrects if
a different channel/region than expected turns out noisiest (see memory `adaptive-no-hardcoding`).
Quality > speed. All of the following are **selectable options** in the config (and the future GUI
menus), listed **best-quality first** (the default is the top of each list):

**Estimator (`estimator`), best → simplest:**
1. `wls_joint` *(default)* — penalised least squares: `min_X (MX−B)ᵀW(MX−B) + λ·TV(X)`, an
   edge-preserving total-variation prior solved iteratively over the volume (denoise fused into the
   inversion). Best quality; global iterative → minutes/volume; `λ` auto-set from the measured noise.
2. `wls_denoise` — per-voxel WLS solve, then a separate edge-preserving denoise pass on the maps.
3. `wls` — per-voxel WLS solve only.
4. `ols` — plain least squares (Phase A baseline; unchanged).

**Noise model (`noise_model`), best → simplest:**
1. `spatial` *(default)* — spatially-varying noise: a local covariance map (sliding window / low-res grid
   on the per-bin high-pass residual) so `W` adapts region-to-region (handles photon-starved dense areas).
2. `global` — one 4×4 covariance `Σ` per scan from auto-selected flat regions.

**Foundation — adaptive noise estimation** (drives everything): estimate per-bin noise + the full
inter-bin covariance from the data via robust high-pass / wavelet-MAD residuals in automatically-selected
low-gradient regions (reuse `noise_from_highpass`). `W = Σ⁻¹`. No ROI, no assumed ordering → if channel C
is worst, `Σ` shows it and the weights/strengths follow automatically.

**WLS:** `x = (MᵀWM)⁻¹MᵀWB` — optimally down-weights the noisiest bin(s) (attacks the bin-1 `A−B`
amplification seen on real data). Statistically optimal linear estimator; HA stays harder than iodine
(its distinguishing signal lives in the noisy low-energy bin) → hence the denoise/joint options.

**Edge-preserving denoise** (`wls_denoise`, and the prior inside `wls_joint`): strength set from the
measured noise (no hard-coded `λ`); 3-D, per-slice-noise-aware. Method is a swappable registry
(TV / NLM / bilateral / guided), compared in the research harness. Optional cross-channel guiding with
the guide = the **highest-SNR channel measured at runtime** (the adaptive fix for the earlier
"A-guided backfired" finding).

**Judging without ground truth (qualitative):** the research harness emits before/after panels
(`ols → wls → wls_denoise → wls_joint`) at chosen slices for radiologist review (the acceptance
criterion), plus no-reference metrics — noise SD in auto-detected flat regions, edge sharpness at strong
edges, and a noise-vs-sharpness trade-off curve. No manual ROI needed.

**Non-breaking:** `estimator='ols'` reproduces Phase A exactly; the WLS/denoise/joint estimators and the
two noise models are additive, every parameter is derived, no per-scan constants.

---

## 6. Research / ablation pipeline (`research_decomposition.py`)

A dedicated, **extensible** experimentation harness — separate from the clean production path — that
tries competing approaches, compares them quantitatively **and visually**, and produces the
justification figures for the thesis. Written with a small "approach registry" so a future
researcher adds a method as one function + one registry entry (built for reuse if published).

**Experiments (each = a registry entry producing numbers + a figure):**
1. **Bin domain — exclusive-via-subtraction vs cumulative-M vs cumulative-only.** The open question:
   which is better-conditioned and less noisy? (cumulative-M requires recomputing `<μ/ρ>` over the
   cumulative windows [T_i,140] via spectrum-weighted combination of the exclusive bins — a small
   helper using the same 140 kV weights the Excel used.)
2. **Threshold option — Option 1 vs 2 vs 3.** Does different threshold placement improve κ?
3. **Estimator — OLS vs WLS vs non-negative vs Tikhonov.** The Phase-A→B story, quantified.
4. **Column scaling — raw κ vs equilibrated κ.** Separates "fixable by scaling" from "fundamental
   collinearity".

**Figures / outputs (thesis-ready):**
- Condition-number bar chart per mode (the §3 table, generated from data).
- Cosine/correlation heatmap of the material signatures (the §3 "why").
- Exclusive-vs-cumulative side-by-side material maps + noise maps.
- **Insert-recovery scatter:** measured vs known density for the phantom Ca/I inserts (the money plot).
- A `image_subtraction`-style `decomposition_research_findings.md` write-up documenting which approach
  wins and *why* — the visual/quantitative justification of the design choices.

---

**Open-science role.** This harness is the backend for the future GUI's **Advanced / Research mode**
(see [../docs/SOFTWARE_ROADMAP.md](../docs/SOFTWARE_ROADMAP.md)). Its approach registry (bin-domain,
estimator, experiment) is the extension surface external researchers use to add methods without
touching core — an openly adaptable toolkit, deliberately the opposite of the closed Siemens
processing.

## 7. First test mode + phantom validation

**Mode:** `{Soft tissue, HA (calcium), Iodine}` — the QRM Dual-Energy Phantom V5 **middle layer**
(soft-tissue background with calcium and iodine inserts). Well-conditioned (iodine's K-edge).

**Why it's the right first test:** the phantom's Ca and I inserts are **matched to identical HU on a
conventional scan** — indistinguishable on a normal image. A successful decomposition separates them
into a clean HA map and a clean iodine map, with the soft-tissue background in the third map. That is
a direct, quantitative, publishable proof the method works, with known ground-truth densities to
check against (insert-recovery scatter). The non-contrast clinical modes (mock-up) follow in Phase C.

---

## 8. Locked decisions (from the planning meeting)

- **Test mode:** Soft tissue / HA / Iodine (phantom middle layer B). ✔
- **Estimator sequencing:** plain OLS first + full stability report, remedies as Phase B. ✔
- **Bin domain:** not guessed — resolved empirically in the research pipeline (exclusive vs
  cumulative), with visual + quantitative justification for the thesis. ✔
- **Scope:** decomposition only; reconstruction pipeline untouched; image-domain only (invariant #3).
- **Phase B (finalized):** ship all estimators as selectable options ordered by quality
  (`wls_joint` > `wls_denoise` > `wls` > `ols`, default `wls_joint`) and both noise models
  (`spatial` > `global`, default `spatial`); everything data-adaptive, nothing hard-coded; judged
  qualitatively (no ground truth). See "Phase B — detailed design" above. ✔

---

## 9. Non-breaking guarantees & conventions

- Reconstruction code (`reconstruction/`, `recon_invariants.py`) is **not modified**.
- All coefficients come from the checked-in `data/mu_rho_binavg.csv` (ported from the Excel) — **no
  Excel dependency on the cluster**, no hardcoded per-scan constants.
- Production (`decompose.py`) and research (`research_decomposition.py`) share one library.
- Every design choice ships with a before/after figure in the research findings (thesis evidence).

---

## 10. Open questions (revisit during Phase A)

- **Which "Implant metal"** backs the peri-implant mode (Ti / CoCr / Ta)? Affects κ; decide per the
  actual implant when that mode is built.
- **Absolute HU→μ calibration vs empirical per-bin gain fit** from known ROIs — start with the NIST
  water reference; the research pipeline can test an empirical gain if residuals are large.
- **4-material option** (4 bins → up to 4 materials, square M) — out of scope; the design is 3.

---

## 11. Future: publishable software / GUI (design-ahead only)

The whole pipeline (reconstruction + decomposition) is intended to eventually become a **publishable
GUI application**. Not now — but so the retrofit stays cheap, this module is built **GUI-ready from
the start**: a pure-library core + a small entry-point function
(`decompose(volumes, mode, config, progress=None) -> result`) driven by a **serializable config
object** (not module-level constants), returning **structured results** with file-writing as a thin
separate layer, and **progress/cancel callbacks** on the per-voxel solve. The mode registry (§4) maps
1:1 to a future GUI dropdown (**clinical mode**), and the research harness (§6) becomes the GUI's
**Advanced / Research mode** — open and registry-extensible, so researchers can add materials, modes,
and methods without editing core (the open-science counter to Siemens' closed processing). Scope now:
this design applies to **decomposition only**; the reconstruction retrofit is parked until
decomposition is finished. Full cross-project plan (entry points, packaging, napari GUI candidate,
open-source/extensibility, reconstruction retrofit): [../docs/SOFTWARE_ROADMAP.md](../docs/SOFTWARE_ROADMAP.md).

## 12. References (to verify/complete during write-up)

1. Alvarez & Macovski (1976), "Energy-selective reconstructions in X-ray computerised tomography,"
   *Phys. Med. Biol.* 21(5):733–744. doi:10.1088/0031-9155/21/5/002. *(projection-domain seminal;
   the contrast to image-domain)* — verified.
2. Niu, Dong, Petrongolo, Zhu (2014), "Iterative image-domain decomposition for dual-energy CT,"
   *Med. Phys.* 41(4):041901. doi:10.1118/1.4866386. *(canonical image-domain WLS with
   variance-covariance weighting)* — verified.
3. Xue et al. (2017), "Statistical image-domain multimaterial decomposition for dual-energy CT,"
   *Med. Phys.* doi:10.1002/mp.12096 (PMC5515554). *(explicit B = M·x + OLS/WLS equations; open
   access)* — verified.
4. Mendonça, Lamb, Sahani (2014), "A flexible method for multi-material decomposition of dual-energy
   CT images," *IEEE Trans. Med. Imaging* 33(1):99–116. doi:10.1109/TMI.2013.2281719. *(image-based
   multi-material framework)* — **verify DOI before citing.**
5. Liu et al. (2009), three-material decomposition with volume conservation — *(verify full citation
   when Phase B adds the sum-to-one constraint).*
6. NIST XCOM / X-Ray Mass Attenuation Coefficients — provenance of the Excel element tables.
