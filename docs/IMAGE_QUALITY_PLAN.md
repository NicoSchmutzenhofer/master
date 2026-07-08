# Image-Quality Improvement Plan — 4-bin PCCT Reconstruction

Status: **WS1 + WS2 implemented** (FBP remains the default; iterative is opt-in via
`RECON_METHOD`). Deferred items in §4 not started. Script left in `FAST_MODE`.

**Addendum — rotating "light cone" fix (`Z_WEIGHTING`, default `'balanced'`).** The
raised-cosine z-window, over a one-rotation rebinning window, is an angular apodization whose
peak rides the helix → a low-frequency brightness lobe that rotates as you scroll z (confirmed
from the user's volume). Fixed by **angularly-balanced helical weighting** (`z_weighting=
'balanced'` in [rebin_helical_to_axial](../reconstruction/helical_reconstruction.py)): span ~2 rotations, taper by
`|z_offset|` (keeps z-shading suppressed) **and** normalise every view angle to equal total
weight (removes the rotating bias). Purely a sinogram-formation change, so SIRT/FBP/CGLS all
benefit with no recon refactor; legacy `'hann'` and uniform `'none'` stay one knob away. New
soft invariant `check_angular_balance`. This is the complete fix for the *rotating* artifact;
ASSR (tilted-plane rebinning) is only needed if a *non-rotating* cone term that grows with
radius remains.

Target pipeline: [helical_reconstruction.py](../reconstruction/helical_reconstruction.py) (library) +
[python_reconstruction.py](../reconstruction/python_reconstruction.py) (driver) +
[recon_invariants.py](../reconstruction/recon_invariants.py) (checks).

---

## 1. Goal & scope

Two problems observed in the current 4-threshold preview:

1. **Noise** — inserts are hard to distinguish (the example given was threshold D).
2. **HU too similar** across thresholds — poor separation of the phantom body.

**Goal:** a solid, *defensible* baseline to build on. Strongest/most-complex methods come later.

**Agreed constraints**
- Improvements apply to **all four thresholds** (A, B, C, D) — D was only an example.
- The pipeline must stay **parameter-driven, nothing hardcoded per scan**.
- The **current reconstruction method (FBP, `shepp-logan`, `curved`) remains the default** and stays available. New methods are *additive options*, not replacements.
- Favor **well-established, citable methods** first.
- **Material decomposition is a separate, post-reconstruction stage** — not part of this work.

---

## 2. Diagnosis (root causes)

| Symptom | Root cause |
|---|---|
| "Even D is too noisy to see inserts" | Thresholds are **cumulative** (A = all photons ≥ T1 … D = hard photons ≥ T4). SNR falls monotonically A→D (log: `max(A)=7796 → max(D)=7206`). **D is by construction the noisiest, lowest-soft-tissue-contrast image.** For *seeing* inserts, A is the best single image. |
| "HU too similar across thresholds" | Cumulative thresholds are **highly correlated**. Material-separating signal lives in the **spectral differences** (A−B, B−C, C−D) and is extracted by **material decomposition** — a later, separate stage. Single-threshold HU will always look similar. |
| Tuning view looks very noisy | `FAST_MODE` reconstructs **one 0.4 mm slice, no z-averaging** — the noisiest possible view. The real product (full volume + `Z_SMOOTH_MM=1.5–3`) is 2–3× better SNR. **HU calibration is also OFF in FAST mode**, so the preview can't show the HU axis at all. |

**Secondary contributors found in code**
- **14% of channels masked** (192/1376); [detect_defect_channels](../reconstruction/helical_reconstruction.py:127) itself warns "too aggressive." Masked runs cluster in the central channels (where the phantom projects) and are linearly interpolated in [preprocess_sinogram](../reconstruction/helical_reconstruction.py:498) → may **blur small inserts**. May also be genuine inter-module-gap spikes amplified by the dense crescent object — needs a diagnostic before changing.
- **Raised-cosine z-window** ([rebin_helical_to_axial](../reconstruction/helical_reconstruction.py)) reduces effective angular statistics ≈1.3× (trades SNR for less z-shading). **Superseded** by `Z_WEIGHTING='balanced'` (see the addendum at the top): balanced weighting suppresses the oblique rays *without* the angular bias, so it removes the rotating-lobe side-effect of the Hann while keeping z-shading down.
- **FBP** amplifies noise; literature shows iterative recon cuts PCCT noise 50–70%.
- Minor: `ModeParXML` parse fails to unwrap (`a bytes-like object is required, not 'tuple'`) → slice-width/active-rows fall back to zIso-derived (correct here, 0.4013 mm), but the XML is never validated. Cosmetic.

---

## 3. Workstreams — IN SCOPE

### WS1 — Quick wins + diagnostics
*Purpose: make the preview honest and decide whether masking hurts inserts. Cheap, do first.*

**1a. Representative FAST preview (multi-slice slab).**
- *What:* reconstruct N adjacent z-targets around `z_centre`, average **in image domain**; N from a new `PREVIEW_SLAB_MM` knob (default **2.0 mm**) ÷ `geom['z_spacing_mm']` → odd count.
- *Why:* image-domain averaging of independently-rebinned slices mirrors exactly what [z_average](../reconstruction/helical_reconstruction.py:1226) does to the full volume, so preview SNR ≈ full-volume SNR. Decisions then transfer.
- *Non-hardcoded:* slab in mm → slices via `z_spacing`.

**1b. HU calibration in FAST mode.**
- *What:* run [auto_hu_calibrate](../reconstruction/helical_reconstruction.py:1053) on the preview slab; show HU window in titles; print per-threshold `mu_water`.
- *Why:* lets you actually judge HU separation and cross-threshold consistency in the preview. Reuses the existing per-threshold JSON cache.

**1c. Channel-masking diagnostic (the 14% question).**
- *What:* run [detect_defect_channels](../reconstruction/helical_reconstruction.py:127) on **all 4 thresholds**, report overlap (true defects are threshold-independent; object-induced ones are not). Save a figure (mean profile + flagged channels + run-length histogram). **Expose the hardcoded MAD multipliers** (`5.0` at [line 183](../reconstruction/helical_reconstruction.py:183), `6.0` at line 190) as parameters with current defaults. Preview A/B: current mask vs relaxed.
- *Why:* evidence-based keep/relax decision; removes hardcoded constants.
- *Constraint:* do **not** change detection logic or the ±1 dilation (per CLAUDE.md); only parameterize the thresholds.

**1d. (Optional, cheap) z-window audit.** Quantify the angular-weight/SNR loss from the Hann taper; offer `z_window=False` for comparison. Already parameterized.

### WS2 — Iterative reconstruction (additive, all thresholds)
*Purpose: the substantive, defensible noise lever. FBP stays the default.*

**2a. Generalize the ASTRA path.**
- *What:* refactor [_astra_fbp](../reconstruction/helical_reconstruction.py:702) into a shared geometry builder + algorithm dispatch (`FBP_CUDA | SIRT_CUDA | CGLS_CUDA`), keeping the **same curved→flat remap** and fan-flat geometry. `_astra_fbp` stays as a thin wrapper → **existing behavior unchanged**. MAR/COR paths untouched.

**2b. Config knobs (driver), threaded through the existing `method` param of [reconstruct_helical_slice](../reconstruction/helical_reconstruction.py:949)/`_stack`.**
- `RECON_METHOD = 'fbp'` (**default — current method, nothing changes unless opted in**) `| 'sirt' | 'cgls'`
- `N_ITER = 150` (configurable)

**2c. Baseline choice.**
- **SIRT_CUDA + non-negativity** (`MinConstraint=0`, valid for attenuation data), fixed iteration count. Most defensible: monotonic, well-understood noise/resolution trade-off, **no regularization weight to hand-tune** (no hidden hardcoded λ).
- CGLS available as an alternative (faster, but semi-convergent → iteration count touchier).
- **TV regularization deferred** to the later "best-quality" phase.

**2d. Runtime — measure before any full run.**
- SIRT ≈ 2 projector ops × `N_ITER` per slice → a full 2262-slice × 4-threshold volume is very long.
- Plan: iterative is free in FAST mode (tuning); for full volume it is **explicit opt-in** with a runtime warning. Report measured FAST per-slice timing on the rtx6000ada before choosing `N_ITER` or running full.

---

## 4. Deferred (future phases — not now)

- **Spectral-guided denoising** — use the high-SNR threshold A to guide denoising of all bins (guided/joint filter, low-rank/NLM). **[INVESTIGATED June 2026 → negative for image quality — see §4a.]**
- **TV-regularized iterative** (adds a λ hyperparameter + custom solver).
- **Exclusive energy-bin images** (A−B, B−C, C−D, image-domain only) — feeds decomposition. **[INVESTIGATED June 2026 → decorrelation works but no image-quality gain; reserved for the decomposition step — see §4a.]**
- **Material decomposition stage** — separate, post-reconstruction.
- **Targeted/zoom FOV** for the small phantom (finer insert sampling).

### 4a. Investigated June 2026 — bin separation for image quality: NEGATIVE result

Tooling: [bin_separation_investigation.py](../reconstruction/bin_separation_investigation.py) (standalone, image-domain,
label-free; reads the 4 reconstructed HU volumes from `output/reconstruction/`; durable write-up in
[BIN_SEPARATION_FINDINGS.md](BIN_SEPARATION_FINDINGS.md)). Goal was to use threshold separation to
*improve image quality* (a pre-step before the later, separate material decomposition).

**Outcome:** separation/decorrelation works perfectly (inter-bin off-diagonal correlation
0.985 → 0.393 → 0.000 for cumulative → exclusive → noise-whitened-PCA), but it yields **no image-quality
gain**. The four thresholds are **cumulative/nested** (every photon in D is also counted in C, B, A), so
their quantum noise is strongly correlated and the data is ~rank-1 (only ~1 % of variance sits in an
independent-noise subspace). Consequently low-rank spectral denoising removes ≤ 8 % noise (negligible;
−2 % for bin D) and threshold-A-guided denoising *injects* noise (+57–88 % on C/D — A is the noisiest bin
in HU). There is no way to reduce noise across these bins without collapsing them toward the shared
structural image, which erases the spectral content.

**Decision:** do **not** add a bin-separation stage for image quality. Image-quality gains come from the
reconstruction domain (iterative SIRT, FBP filter, z-smoothing — already implemented). Threshold
separation is reserved for the later material-decomposition stage, where the small spectral component is
exactly the material signal. (Quantitative caveat: noise was measured on an auto water ROI at ~+34 HU
soft tissue; pinning to the Ø25 mm 0-HU calibration cylinder would give a textbook-clean confirmation but
does not change the structural conclusion.)

---

## 5. Sequencing & milestones

1. **M1 — WS1:** honest, HU-calibrated slab preview + the masking keep/relax decision. *(Re-evaluate after this — may already close much of the gap.)*
2. **M2 — WS2 (FAST):** SIRT vs FBP comparison figures + per-slice timing.
3. **M3:** decide whether/when to run iterative on the full volume.
4. **Later:** deferred items in §4, then material decomposition (separate work).

---

## 6. Non-breaking & non-hardcoded guarantees

- FBP (`shepp-logan`/`curved`) remains the **default**; iterative is opt-in via `RECON_METHOD`.
- [recon_invariants.py](../reconstruction/recon_invariants.py) is only **appended** to (per CLAUDE.md), never edited.
- New parameters: slab thickness (mm), `RECON_METHOD`, `N_ITER`, MAD multipliers — all config/derived; **no new per-scan constants**.
- Every change ships with a before/after preview figure.

---

## 7. Open questions to resolve before implementing

- `PREVIEW_SLAB_MM` default (proposed 2.0 mm).
- `N_ITER` default for SIRT (proposed ~150, pending M2 timing).
- Is full-volume iterative needed soon, or FAST-mode comparison sufficient for now?
