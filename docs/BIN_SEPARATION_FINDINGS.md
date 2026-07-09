# Threshold-separation for image quality — findings (image domain)

Status: **investigation complete (June 2026), negative result.** These are durable,
version-controlled notes for the thesis. The runtime-generated report lives at
`output/research/image_subtraction/image_subtraction_findings.md` (gitignored); this document is the
curated summary and the reasoning behind the decision.

Tooling: [image_subtraction_investigation.py](../reconstruction/image_subtraction_investigation.py)
(standalone, image-domain, label-free; reads the four reconstructed HU volumes). Summary also
recorded in [IMAGE_QUALITY_PLAN.md §4a](IMAGE_QUALITY_PLAN.md).

---

## 1. Question and scope

Can we exploit the redundancy between the four cumulative energy-threshold volumes (A ⊇ B ⊇ C ⊇ D)
to produce **cleaner threshold / energy-window images** — i.e. an *image-quality* gain, purely in
the **image domain**, with **no material information** (label-free)?

Framing that guided the study: *separating* the thresholds does not improve quality by itself — the
four bins are ~98 % redundant. Any gain must come from **exploiting that correlation to denoise**
(low-rank across bins, or guiding the noisy bins with the high-SNR bin A). Decorrelation/separation
is the **vehicle**; denoising is the **quality lever**. The study therefore measured both: how well
the bins decorrelate, *and* whether that buys a measurable noise/CNR improvement at fixed resolution.

Scope boundaries:
- **Image domain only.** The `.mat` data are Siemens-processed, gain-calibrated sinograms; forming
  energy-window images by subtracting *sinograms* is separately forbidden by
  [CLAUDE.md](../CLAUDE.md) invariant #3 (per-threshold gain breaks the A≥B≥C≥D ordering). Exclusive
  windows here are formed as **image** differences of the reconstructed volumes.
- **Not material decomposition.** That is a separate, later stage; this study is only about whether
  bin separation helps *image quality*.
- **Energy bins vs phantom layers.** Bins A–D are the four detector thresholds
  (≥20/40/56/75 keV; exclusive windows 20–40 / 40–56 / 56–75 / 75–140). The QRM phantom's "layers"
  are an unrelated geometric feature.

---

## 2. Method

Three stages, on the reconstructed HU volumes over the phantom slab, with an auto-detected
homogeneous water region for the noise estimate and auto-detected round high-contrast features for
the label-free contrast/resolution metrics:

1. **Redundancy** — inter-bin correlation of the cumulative volumes vs the exclusive windows
   (`[A−B, B−C, C−D, D]`), plus a robust noise covariance from the in-plane high-pass residual in
   the flat region (slice-difference estimate kept only as a cross-check, since it is biased low when
   `Z_SMOOTH_MM > 0` correlates adjacent slices).
2. **Separation** — noise-whitened spectral PCA (whiten by Σ^−1/2, then PCA). The leading component
   is the shared high-SNR structure; the trailing components are the decorrelated spectral signal.
   This quantifies how much *independent* information survives whitening.
3. **Quality (the lever)** — two denoisers evaluated against the cumulative baseline:
   - **Low-rank spectral denoise** — project each voxel's 4-vector onto the top-*k* PCA basis along
     the **spectral axis only** (no spatial neighbourhood → spatial resolution preserved exactly by
     construction).
   - **Threshold-A-guided denoise** — use the high-SNR bin A as a structural guide for the noisier
     bins.

Metrics are **label-free**: noise SD from the high-pass residual in the water region; per-feature
**CNR** = |mean_insert − water_mean| / noise_SD; and an **edge-sharpness ratio** (before/after) as a
resolution proxy (~1 = resolution preserved, < 1 = blurred).

---

## 3. Results

| Quantity | Cumulative | Exclusive | Noise-whitened PCA |
|---|---:|---:|---:|
| Mean \|off-diagonal\| inter-bin correlation | **0.985** | 0.393 | **0.000** |

- **Decorrelation works perfectly** — off-diagonal correlation collapses 0.985 → 0.393 → 0.000. The
  separation machinery does exactly what it should.
- **But the data are ≈ rank-1.** After noise-whitening, only **~1 %** of the variance sits in an
  independent-noise subspace; the rest is the single shared structural image.
- **Low-rank spectral denoise → negligible.** Noise reduction **≤ 8 %**, and **−2 % for bin D** (it
  made the hardest bin slightly *worse*). Edge-sharpness ratio ≈ 1 (resolution preserved, as
  expected — but there is nothing to gain).
- **Threshold-A-guided denoise → actively harmful.** It **injects +57–88 % noise on C/D**, because
  bin A, although highest-count, is the **noisiest in HU** after calibration, so guiding with it
  pushes A's noise into the harder bins.

(Exact per-bin figures are in `output/research/image_subtraction/image_subtraction_metrics.json` from the
run; the numbers above are the run-level summary.)

---

## 4. Physical interpretation

The four thresholds are **cumulative / nested**: every photon counted in D is also counted in C, B,
and A. Their quantum noise is therefore **strongly correlated**, not independent. There is no
linear combination across the bins that reduces noise without simply **collapsing them toward the
shared structural image** — which erases the very spectral content that distinguishes the bins.
Redundancy this high (a rank-1 signal) leaves almost no independent-noise subspace for a
cross-bin denoiser to work in, which is exactly what the whitened-PCA variance fractions show.

---

## 5. Decision and implications

- **Do not add a bin-separation stage for image quality.** It is mathematically clean but yields no
  measurable quality gain, and the guided variant is counter-productive.
- **Image-quality gains come from the reconstruction domain instead** — iterative reconstruction
  (SIRT), FBP filter choice, helical weighting, and z-smoothing — all already implemented and
  controlled by the driver knobs.
- **Threshold separation is reserved for material decomposition**, where the small spectral component
  that survives whitening *is* the material signal (a weak component is fatal for denoising but can
  still be informative for decomposition, subject to the conditioning analysis in
  [../decomposition/DECOMPOSITION_PLAN.md](../decomposition/DECOMPOSITION_PLAN.md)).

---

## 6. Caveats & reproducibility

- **Water ROI.** Noise was measured on an auto-detected water region at ~+34 HU (soft tissue).
  Pinning `WATER_ROI` to the Ø25 mm 0-HU calibration cylinder would give a textbook-clean
  confirmation, but does **not** change the structural conclusion (the rank-1 argument is
  independent of the ROI).
- **Regenerate:** `python reconstruction/image_subtraction_investigation.py` (reads
  `output/reconstruction/reconstruction_thr_{A,B,C,D}_HU.nii.gz`; writes figures + metrics +
  `image_subtraction_findings.md` to `output/research/image_subtraction/`). No GPU needed when the volumes
  already exist.

---

## 7. Relation to the sinogram-domain study

This study answers the **image-quality** question in the **image domain**. A separate investigation
([sinogram_subtraction_investigation.py](../reconstruction/sinogram_subtraction_investigation.py))
asks a **different** question: does forming the exclusive energy windows by **subtracting the
sinograms** (before reconstruction) — treating the Siemens pile-up/scatter corrections as given —
behave differently from subtracting the reconstructed images, and does it feed material
decomposition? Because reconstruction is **nonlinear** under the production SIRT + non-negativity
setup, the two are not guaranteed identical, and the sinogram study also measures directly whether
the A≥B≥C≥D ordering violations that motivate invariant #3 are quantum-noise-like (unbiased, average
out) or a systematic gain bias. That study does **not** revisit the image-quality conclusion here.
