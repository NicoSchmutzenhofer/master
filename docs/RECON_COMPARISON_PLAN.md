# Reconstruction comparison plan — ours vs Siemens WFBP vs Siemens VMI

Design document for the **reconstruction-stage** image-quality comparison
([reconstruction/recon_comparison.py](../reconstruction/recon_comparison.py),
[reconstruction/image_quality_metrics.py](../reconstruction/image_quality_metrics.py),
[batch_compare.sh](../batch_compare.sh)). Material-map accuracy is a **separate, later**
study — it needs the iodine maps and a different metric set, and nothing here depends on it.

Read this before changing the metric set, so the trade-offs below are not re-litigated.

---

## 1. The question, and why it is not "who wins"

Siemens does not disclose the reconstruction parameters behind their exports — kernel and
slice thickness are readable from standard DICOM tags, but the QIR (quantum iterative
reconstruction) strength is not. A single head-to-head number could therefore never be
defended: any difference could always be attributed to unmatched processing.

The defensible question is different, and it is the one this study answers:

> Where does a simple, fully inspectable pipeline **land** relative to a mature commercial
> one, and by what **mechanism** does the gap arise?

That is answerable without knowing their settings. Sweep our own reconstruction across its
knobs to trace the noise/resolution trade-off curve it can reach, then plot where the vendor
falls relative to that curve:

| Vendor lands | Interpretation |
|---|---|
| **below** the curve | genuinely ahead — and the margin is now measured, not asserted |
| **on** the curve | same operating characteristic, merely a different operating point; nothing unavailable to us |
| **above** the curve | we are ahead; say so |

All three are publishable and none requires matching their processing. This is the reason the
trade-off curve, not a table of noise values, is the headline result.

The expected outcome is that Siemens is ahead. That supports rather than undermines the thesis
argument: the contribution is a transparent, reconfigurable pipeline (see
[SOFTWARE_ROADMAP.md](SOFTWARE_ROADMAP.md) and the thesis §`sec:bg-software`), and quantifying
the distance to a mature product is a result, not a defeat.

## 2. What pairs with what

**Ours vs WFBP is a true 1:1 comparison.** Same photons, same four cumulative thresholds,
same scan; only the reconstruction differs. A↔T1, B↔T2, C↔T3, D↔T4.

**VMI has no channel correspondence at all.** A monoenergetic image at 70 keV and a
cumulative ≥40 keV threshold image are different physical quantities on different HU scales.
Forcing a pairing (by mean energy, or by best HU fit) would be the weakest step in the
chapter. VMI is therefore characterised as **its own family** — a noise-and-texture curve
versus keV — and is excluded from the paired difference analysis.

> State the reason as the **HU scale**, not image quality. Excluding VMI because it ranks
> lower would be circular; excluding it because an RMSE against it is undefined is correct.

**One genuine advantage worth stating in the thesis:** all three families come from the *same
scan*, so dose, geometry, phantom and positioning match by construction. Most vendor
comparisons cannot claim that. Only the processing differs.

## 3. Fairness: the two confounds that are removed, and the one that is not

| Confound | Handling |
|---|---|
| **Slice thickness** | `Z_SMOOTH` off; native 0.4 mm slices are **averaged** to the vendor's `SliceThickness`. Averaging is physically what a thicker slice *is*, so it reproduces their slice profile instead of approximating it with a Gaussian. Both `SliceThickness (0018,0050)` and `SpacingBetweenSlices (0018,0088)` are matched — Siemens often exports overlapping slices, and matching only thickness leaves the z-correlation different. |
| **Pixel grid** | We reconstruct on the vendor's matrix and FOV. NPS is a function of spatial frequency; curves computed on different voxel sizes have frequency axes that do not align, and noise per voxel differs for reasons unrelated to the reconstruction. |
| **Kernel / QIR** | **Cannot** be matched, and is not attempted. This is exactly what the trade-off curve exists to absorb: a kernel difference moves you *along* a curve, better data handling moves you *off* it. |

Two rules that follow, and are enforced in the code:

- The √T noise shortcut is **wrong here.** SSR slices at neighbouring z interpolate between
  overlapping detector rows and are correlated, so noise does not fall as √N. Average, never
  rescale.
- **Never resample before measuring noise or resolution.** Interpolation alters exactly what
  is being measured. Each family is measured on its own native grid; resampling is confined to
  the bias analysis, where only smooth structure is examined.

## 4. Metric set

Each metric alone can be gamed by blurring, so they are reported together.

| Metric | What it captures | Why it is needed |
|---|---|---|
| **Noise SD** in a uniform ROI | how much noise | the headline number, but insufficient alone |
| **NPS** (radially averaged) | magnitude **and texture** | two images can have identical SD and look nothing alike. Fine-grained analytic noise vs coarse blotchy iterative/denoised noise lives entirely here — this is the metric that reveals *mechanism* |
| **TTF** (circular-edge), TTF50/TTF10, 10–90 % edge width | resolution at stated contrast | the guard against every noise metric. MTF is ill-defined for SIRT because resolution becomes contrast-dependent |
| **NEQ** = contrast²·TTF²/NPS | both combined, per frequency | the principled combination; assumes no task |
| **d′** | both combined, one number | task-dependent, so the task (5 mm, 50 HU by default) is quoted with the value |
| **CNR** per insert | contrast normalised by noise | cancels the scale problem: threshold D and 190 keV VMI are both low-contrast *and* low-noise, so raw SD would rank them best |

Conventions are fixed in the module docstring and must stay fixed so numbers remain
comparable: NPS two-sided in HU²mm² normalised so ∫∫NPS = variance; radial NPS is the ring
mean; `f_av` uses the 1-D convention on the radial curve; TTF normalised to 1 at DC (which is
what lets one routine serve both threshold and monoenergetic images).

**Rejected: reference-based similarity (RMSE / PSNR / SSIM against WFBP as "truth").**
For unbiased estimates with noise σ₁ and σ₂, RMSE → √(σ₁²+σ₂²) — it is dominated by the fact
that the two noise fields are independent and would be large even if our reconstruction were
perfect. It measures difference, not error, and it assumes the vendor image is truth when it
is another processed estimate. SSIM additionally carries no physical meaning in HU. Worth one
paragraph in the thesis stating why it was declined.

**Accepted instead: scale-separated difference analysis** (`difference_analysis`). Low-pass
the difference image → the systematic part (HU offsets, cupping, shading, geometric
distortion), which is real and interpretable; the high-pass remainder is just two noise
realisations and is already characterised by the NPS. Paired ROI means feed a Bland–Altman
plot (bias + 95 % limits of agreement), the standard method-comparison tool.

## 5. The sweep: five settings

`FILTER_NAME` is **inert under SIRT** — `_astra_reconstruct` only sets `FilterType` in the FBP
branch — so the two families trade off along different axes: filter for FBP, iteration count
for SIRT.

| # | Setting | Role |
|---|---|---|
| 1 | `fbp` + `ram-lak` | sharp/noisy anchor (not a shippable setting; it pins the end of the curve) |
| 2 | `fbp` + `shepp-logan` | documented quantitative default, ~Qr40 equivalent |
| 3 | `fbp` + `hann` | smooth/quiet anchor |
| 4 | `sirt` ×25 | under-converged: smooth, quiet |
| 5 | `sirt` ×100 | the production setting |

Three FBP points define the analytic trade-off curve; the two SIRT points then test themselves
against it. **If SIRT lands on the FBP curve, iterative reconstruction is apodisation by
another name and buys nothing** — a real result. If it lands below, the gain is genuine and
measured. Same question as the vendor comparison, answered by the same figure.

SIRT converges smooth→sharp and decelerates, so ×25 vs ×100 gives a visible segment where
×100 vs ×200 would land almost on top of itself. If ×25 proves too blurry to be interesting,
×40 is the fallback.

## 6. Figures

**Headline — `tradeoff_per_threshold.png`: one panel per threshold.** Thresholds are never
mixed on one set of axes. A has all the photons and D very few, so a combined plot would
mostly display the bins' photon statistics and hide the reconstruction differences it is meant
to show. Each panel carries our five-point curve plus the WFBP star, so the claim becomes
per-channel and specific — *"in A–C we reach the vendor's noise at comparable resolution; in D
we do not, by this margin."*

Supporting: `nps_curves.png` (magnitude and texture per threshold), `vmi_vs_kev.png` (the VMI
family on its own terms), `bland_altman_own_vs_wfbp.png`.

## 7. Pipeline: three resumable stages

| Stage | Where | Does |
|---|---|---|
| **0 probe** | CPU | read the Siemens geometry, auto-locate the insert slab → `match_config.json` + QC figure |
| **1 sweep** | **GPU** | reconstruct the five variants on that slab and grid → `sweep/<variant>/` |
| **2 metrics** | CPU | NPS/TTF/NEQ/d′/bias over all three families → `metrics/`, `figures/`, `qc/` |

Stage 0 determines stage 1 entirely — this is what "match Siemens first, then reconstruct"
means in practice, and it removes the need to know the slab in advance.

**Slab auto-detection** runs on the vendor volume (already exists, no GPU). The insert layer
is the run of slices with strong in-plane structure inside the body. Structure is scored as
the **fraction of body voxels far from the body median**, not as a spread statistic: inserts
occupy only a few percent of the body area, and MAD is designed precisely to discard a small
minority of deviant voxels, so it stays flat straight through the insert layer. The body edge
is eroded before scoring, because smoothing drags −1000 HU air across the boundary and would
otherwise score a "structure" rim in *every* slice.

**`--phantom-z` is effectively required on a clinical scan range.** The first run searched the
whole Thx–Abdomen acquisition and selected a slab at z ≈ −2125 mm — roughly **665 mm from the
phantom**, which sits at −1515.5…−1408.3 mm. A long acquisition contains the table,
positioning aids and scan-end artefacts, any of which can out-score the phantom. Passing the
phantom's axial extent (read off the vendor volume in Slicer) confines the search and also
limits how much of a multi-thousand-slice series is read; `--no-phantom-z` restores the
unrestricted search. Note the `=` form — `--phantom-z=-1515.5,-1408.3` — because a value
beginning with `-` is otherwise parsed as a flag (the same applies to `--slab`).

Detection now returns **every** candidate layer, not just the winner; all are logged and drawn
on the QC figure, so "picked the wrong layer" is distinguishable from "only one layer exists",
and `--slab` can force a different one. `--slab-select peak` (default) takes the layer holding
the most insert-covered slice — the score is essentially insert area fraction, so this prefers
the layer carrying the most inserts, **including one sitting hard against the end of the
phantom**, which is the layer most at risk of being missed. `--slab-select longest` restores
the previous longest-run behaviour. `--expect-inserts N` warns when the detected count
disagrees; that matters beyond TTF, because an undetected insert stays *inside* the background
region and inflates the noise estimate.

**Skip logic.** A variant counts as done only if all four volumes exist **and** its recorded
`sweep_config.json` signature matches (slab, grid, thickness, weighting, defect thresholds,
data path). File existence alone would silently reuse volumes reconstructed from a different
slab — the failure mode that quietly corrupts a comparison. `--force` overrides.

### Output layout

Entirely inside the documented `output/research/<name>/` convention, so it adds one name and
touches nothing existing. **`output/reconstruction/` is deliberately untouched** — that is the
production output the decomposition stage reads by hard filename contract. Each sweep variant
reuses the *production* filename pattern, so the existing loaders read a sweep folder unchanged
and any variant could be fed to the decomposition stage later without new code.

```
output/research/recon_comparison/
  match_config.json  compare_config.json
  sweep/<variant>/   reconstruction_thr_{A,B,C,D}_HU.nii.gz  calibration_thr_*.json  sweep_config.json
  metrics/           metrics.json
  figures/           tradeoff_per_threshold.png  nps_curves.png  vmi_vs_kev.png  bland_altman_own_vs_wfbp.png
  qc/                stage0_slab_detection.png  roi_{own,wfbp,vmi}.png
```

## 8. Validation

[selftest_image_quality.py](../reconstruction/selftest_image_quality.py) checks every metric
against an analytically known answer (`python -m reconstruction.selftest_image_quality`, no
data, no GPU, seconds). It exists because writing these metrics surfaced three defects that
would each have silently biased the thesis numbers:

1. **The first radial NPS bin held only the DC term**, which detrending forces to zero — so
   `nps[0]` was ~0 and corrupted `f_av` and any normalisation. DC is now excluded.
2. **The slab score used MAD**, which by construction ignores the inserts (see §7).
3. **Polynomial detrending attenuates the lowest NPS bins** — 48 % at bin 0 for order 2 — and
   the low-frequency end is exactly where iterative and denoised noise concentrates its power.
   The attenuation is an exact function of `(patch size, order)` and is now divided back out;
   the corrected NPS integrates to the true variance to ~0.02 %.

A fourth was caught by the end-to-end smoke test: `noise_power_spectrum` originally *inferred*
its patch size, so patch origins validated for one size were read at another, reaching over the
inserts and inflating noise several-fold. The size is now passed explicitly and mismatches raise.

**Known limitation — spectral leakage.** A finite patch convolves the true NPS with the Fejér
kernel of its window, so power leaks from strong bins into weak ones. Measured error is ~4 %
while the spectrum falls ≲1000× across the band (the realistic CT case) and grows as it
steepens. Compare curves within ~2 decades of the peak; do not read the extreme tail
quantitatively.

**NPS patch size is set by the phantom, not by preference.** The vendor exports this
phantom at a whole-body FOV (500 mm on 512 px → 0.977 mm pixels), so a 64 px patch is
62 mm wide and **no such square of clear background exists** inside a ~200 mm phantom
between its inserts. `auto_background_patches` halves the request until enough fit
(32 px ≈ 31 mm here) and the chosen size is then **shared by every family**, because the
NPS frequency grid depends on it and curves measured at different sizes are not
comparable. The size actually used is logged and stored with the metrics.

**A hand-drawn insert prior overrides automatic detection.** `--segmentation` takes a
Slicer `.nrrd` / `.seg.nrrd` (auto-discovered as `<wfbp-dir>/Segmentation.nrrd`). It is a
*seed*, not the answer: each label is treated as a GROUP — one label per row of inserts is
the natural way to draw it — split into its connected components, and then **every edge is
refined from the image**. The seed only has to overlap its insert; the refinement estimates
the inside and outside levels, thresholds halfway between them, takes the component
containing the seed for a robust centre, and reads the radius off the 50 % crossing of the
radial profile. On deliberately sloppy seeds (centres off by up to 2.5 px, radii wrong by
−35 % to +55 %) this recovers centres to <0.5 mm and cuts the radius error ~10×. That matters
because the TTF needs the true centre to build a clean edge profile and the background
exclusion needs the true radius. A **partial** annotation is fine — inserts that are not
visible need not be drawn.

Resampling is done in **physical space**, never by array index: a Slicer segmentation is
normally cropped to the segment bounding box, so its extent and origin differ from the volume
it was drawn on. This also carries the segmentation onto our own reconstruction, whose slab
covers a different z range. If the resampled label map is **empty** for a family, that is
reported as a finding rather than a nuisance: for `own` it means our helical z and the
vendor's `ImagePositionPatient` z are not the same frame (see §9), and the family falls back
to automatic detection.

**The phantom is anthropomorphic, which changes where noise may be measured.** It is a
thorax phantom with an insert module, not a uniform cylinder: lung, bone and the patient
table all lie inside the body outline. Background patches placed on the body outline
therefore measured anatomy — the first full run reported 54–250 HU "noise" with an f_av of
0.078 cyc/mm, a noise grain over a centimetre wide, which is structure. Three consequences,
all now enforced in code:

0. The background segment may be drawn as a filled region **or as the module's outline**
   — the interior is filled (`binary_fill_holes`), which matters because Slicer's
   fill-between-slices needs a clear path and tracing the boundary avoids the inserts
   entirely. It may also overlap the inserts: the background label is applied with
   explicit low priority, so an insert is never overwritten by it (an overwritten insert
   would go undetected and then be counted as noise). Outline, solid-over-inserts and
   solid-avoiding-inserts were verified to give identical inserts, region and noise SD.
1. The noise region is the drawn `--background-label` segment when supplied, else an
   automatically detected homogeneous region (`homogeneous_mask`: within an HU band of the
   modal tissue value AND locally flat), and only as a last resort the body outline, which
   is logged as UNRELIABLE.
2. Inserts and the region edge are removed from the noise region **always**, whatever its
   source. An insert left inside contributes its full contrast: measured 102 HU against a
   true 40 HU, and because the contrast is identical in every channel while the noise is
   not, it compresses and can invert the very ranking the study exists to establish.
3. The noise **SD** is measured over the whole region and needs no square blocks; only the
   NPS *curve* does. Tying the headline number to patch geometry would report nothing at
   all on a phantom whose uniform material is a thin ring of matrix between inserts.

`homogeneous_mask` thresholds local SD **relative to its median**, not at a fixed
percentile: a percentile keeps exactly that fraction by construction, and in uniform
material — where the ranking is pure chance — the survivors are speckle, which the
following erosion destroys (measured: 1.4 % of a completely uniform body surviving a "keep
the flattest 50 %" rule, which then silently forced the fallback to the body outline).

**ROI placement is automatic and must still be eyeballed.** Detection runs once per family on
the **z-average** of the slab (√Z better SNR, and the inserts are z-invariant within the layer
stage 0 selected), then the same ROIs are reused for every channel and variant of that family
— so a metric difference can never come from the ROIs having moved. Every run writes a `qc/roi_*.nrrd` LABEL VOLUME (plus a `.json` sidecar naming each
label), loadable straight into Slicer on top of the data — a single mid-slice picture can
only ever show one plane, and whether a patch touches an insert is a question you answer by
scrolling; automatic detection that nobody looks at is how a results chapter
quietly goes wrong. The overlays double as an appendix figure.

## 9. Open risk

**The z-origin assumption.** The slab is detected in the vendor's `ImagePositionPatient` frame
and handed to our reconstruction. Both should derive from the scanner table coordinate, but
this is unverified. Stage 1 raises a clear error if the requested slab falls outside our
reconstructable range; a subtler offset would show up as inserts missing from
`qc/roi_own.png`. Check that overlay before trusting stage 2, and pass `--slab` explicitly if
the origins turn out to differ.

## 10. Deferred

- **Material-map accuracy** (iodine/HA concentration vs nominal, crosstalk, predicted vs
  realised noise amplification) — a separate study once the iodine maps exist.
- **Numerical rank of the channel stack**, which would test the thesis claim that a VMI series
  is "close to rank-deficient by construction" (§`sec:bg-gap`) empirically rather than
  theoretically. Cheap, needs no ground truth, and belongs with the decomposition work.
- **Full task-based `d′` treatment** across a range of lesion sizes; currently one stated task.
