"""
recon_comparison.py -- reconstruction-stage image-quality comparison:
our reconstruction vs Siemens WFBP vs Siemens VMI.

Standalone investigation (like image_subtraction_investigation.py); it does NOT touch
the production driver or output/reconstruction/.  Everything lands under
output/research/recon_comparison/.

    THE QUESTION.  Not "is mine better" -- it very probably is not, and the vendor's
    reconstruction parameters (QIR strength, kernel) are not disclosed, so a single
    head-to-head number could never be defended.  The question is where a simple,
    fully inspectable pipeline LANDS relative to a mature commercial one, and by what
    mechanism the gap arises.  That is answerable without knowing their settings:
    sweep our own reconstruction across its knobs to trace the noise/resolution
    trade-off curve we can reach, then plot where the vendor falls relative to it.
      - vendor BELOW the curve -> genuinely ahead, and by a measured margin
      - vendor ON the curve    -> same operating characteristic, different operating
                                  point; nothing unavailable to us
    Both outcomes are publishable; neither depends on matching their processing.

    WHAT PAIRS WITH WHAT.  own vs WFBP is a true 1:1 comparison -- same photons, same
    four cumulative thresholds, same scan, so only the reconstruction differs.  VMI has
    no channel correspondence with a threshold image at all (a monoenergetic image at
    70 keV and a cumulative >=40 keV image are different physical quantities on
    different HU scales), so it is characterised as its own noise-vs-keV family and is
    deliberately excluded from the paired difference analysis.  The reason is the HU
    scale, not image quality.

    FAIRNESS.  All three come from the SAME scan, so dose, geometry, phantom and
    positioning match by construction -- most vendor comparisons cannot claim that.
    Only the processing differs.  Two confounds are removed explicitly:
      * slice thickness -- Z_SMOOTH is off and native 0.4 mm slices are AVERAGED to the
        vendor's SliceThickness.  Averaging is physically what a thicker slice is, so
        this reproduces their slice profile.  The sqrt(N) shortcut is wrong here: SSR
        slices at neighbouring z share detector rows and are correlated.
      * pixel grid -- we reconstruct on the vendor's matrix and FOV, because NPS is a
        function of spatial frequency and curves computed on different voxel sizes have
        frequency axes that do not align.
    The kernel cannot be matched, which is precisely what the trade-off curve is for.

Three stages, each resumable:

    0  probe    CPU  read the Siemens geometry, auto-locate the insert slab   -> match_config.json
    1  sweep    GPU  reconstruct the 5 variants on that slab and grid         -> sweep/<variant>/
    2  metrics  CPU  NPS / TTF / NEQ / d' / bias over all three families      -> metrics/ figures/ qc/

Run (from the repo root):

    python -m reconstruction.recon_comparison --stage 0 --wfbp-dir ... --vmi-dir ...
    python -m reconstruction.recon_comparison --stage 1
    python -m reconstruction.recon_comparison --stage 2
    python -m reconstruction.recon_comparison --stage all --wfbp-dir ... --vmi-dir ...
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reconstruction import image_quality_metrics as iq

logger = logging.getLogger("recon_comparison")
_REPO_ROOT = Path(__file__).resolve().parent.parent


# ═════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════
# Five reconstruction settings.  Three FBP filters define the analytic
# noise/resolution trade-off curve; the two SIRT points then test themselves
# AGAINST that curve.  If SIRT lands on it, iterative reconstruction is apodisation
# by another name and buys nothing -- a real result.  If it lands below, the gain is
# genuine and measurable.  Same question the vendor comparison asks, one figure.
#
# NOTE  filter_name is inert for SIRT: _astra_reconstruct only sets FilterType in the
# FBP branch, so SIRT's noise/resolution knob is n_iter.  SIRT converges smooth->sharp
# and decelerates, so 25 vs 100 gives a visible segment where 100 vs 200 would not.
DEFAULT_VARIANTS = [
    {"name": "fbp_ramlak",     "algorithm": "fbp",  "filter_name": "ram-lak"},
    {"name": "fbp_shepplogan", "algorithm": "fbp",  "filter_name": "shepp-logan"},
    {"name": "fbp_hann",       "algorithm": "fbp",  "filter_name": "hann"},
    {"name": "sirt_25",        "algorithm": "sirt", "n_iter": 25},
    {"name": "sirt_100",       "algorithm": "sirt", "n_iter": 100},
]

THRESHOLD_LABELS = ("A", "B", "C", "D")

_DEFAULT_DATA = ("/data/Data2/4_BIN_PCCT/4 bin Phantom raw data/Descriptor/"
                 "full_sinogram_4-bin_Phantom-Scan..CT.Thx.-_Abdomen_Staging.Körper.601.RAW."
                 "20260326.074506.20260401.175846.e00e8253-e854-4963-bed1-1ac627e653d7.raw.mat")
_DEFAULT_DESC = ("/data/Data2/4_BIN_PCCT/4 bin Phantom raw data/Descriptor/"
                 "descriptor_4-bin_Phantom-Scan..CT.Thx.-_Abdomen_Staging.Körper.601.RAW."
                 "20260326.074506.20260401.175846.e00e8253-e854-4963-bed1-1ac627e653d7.raw.mat")


@dataclass
class CompareConfig:
    """Serialisable configuration (SOFTWARE_ROADMAP: config object, not constants)."""
    out_root: str = str(_REPO_ROOT / "output" / "research" / "recon_comparison")
    wfbp_dir: Optional[str] = None
    vmi_dir: Optional[str] = None

    data_path: str = _DEFAULT_DATA
    desc_path: str = _DEFAULT_DESC
    geo_dir: str = str(_REPO_ROOT / "geometry")

    # Grid/slab: None = take it from the Siemens export (stage 0).
    n_pixels: Optional[int] = None
    fov_mm: Optional[float] = None
    slice_thickness_mm: Optional[float] = None
    slice_spacing_mm: Optional[float] = None
    slab_mm: Optional[tuple] = None
    slab_pad_mm: float = 2.0
    max_slab_mm: float = 40.0
    # Axial extent of the phantom itself, read off the vendor volume (e.g. in Slicer).
    # Strongly recommended for a clinical scan range: without it the search covers the
    # whole acquisition, where the table or a scan-end artefact can out-score the
    # phantom and place the slab hundreds of mm away from it.
    phantom_z_mm: Optional[tuple] = None
    use_phantom_z: bool = True      # False ignores a configured phantom_z_mm
    slab_select: str = "peak"       # "peak" (most insert-covered) | "longest"
    expect_inserts: Optional[int] = None   # warn if the detected count differs

    # Reconstruction settings held fixed across the sweep (only the swept knobs vary).
    geometry_model: str = "curved"
    z_weighting: str = "balanced"
    wavelet_ring_threshold: float = 2.0
    spike_mad_k: float = 5.0
    ipr_mad_k: float = 6.0
    patient_position: str = "HFS"

    variants: list = field(default_factory=lambda: [dict(v) for v in DEFAULT_VARIANTS])

    # Metric parameters
    nps_patch_px: int = 64          # requested; reduced automatically if it does not fit
    insert_mad_k: float = 4.0       # lower = detect fainter inserts (more false positives)
    # Hand-drawn insert prior (Slicer .nrrd / .seg.nrrd).  Positions only need to be
    # roughly right: every edge is refined from the image, and a partial annotation is
    # fine.  Auto-discovered as <wfbp_dir>/Segmentation.nrrd when not given.
    segmentation: Optional[str] = None
    # Label value inside the segmentation that marks the UNIFORM region the noise is
    # measured in.  Strongly recommended on an anthropomorphic phantom, where the body
    # outline contains lung, bone and the table and is not a noise region at all.
    background_label: Optional[int] = None
    task_diameter_mm: float = 5.0
    task_contrast_hu: float = 50.0
    bias_smooth_mm: float = 5.0

    force: bool = False

    def to_dict(self):
        return asdict(self)


# ═════════════════════════════════════════════════════════════════════
# Small helpers
# ═════════════════════════════════════════════════════════════════════
def _sitk():
    import SimpleITK as sitk
    return sitk


def load_segmentation(path, ref_img, low_priority_label=None):
    """
    Read a Slicer segmentation and resample it onto `ref_img`'s grid.

    Resampling is done in PHYSICAL space (nearest-neighbour), never by array index: a
    Slicer .seg.nrrd is normally cropped to the segment bounding box, so its extent and
    origin differ from the volume it was drawn on, and index-space alignment would be
    silently wrong.  This also maps the segmentation onto our own reconstruction, whose
    slab covers a different z range from the vendor series it was drawn on.

    A .seg.nrrd may store each segment as a separate binary layer in a 4-D array; those
    are merged into one label map (layer i -> label i+1).

    Returns an integer label array shaped like ref_img, or None if nothing overlaps.
    """
    sitk = _sitk()
    seg = sitk.ReadImage(str(path))
    layers = ([seg[:, :, :, i] for i in range(seg.GetSize()[3])]
              if seg.GetDimension() == 4 else [seg])

    merged, deferred = None, None
    for i, layer in enumerate(layers):
        r = sitk.Resample(layer, ref_img, sitk.Transform(), sitk.sitkNearestNeighbor,
                          0, sitk.sitkUInt16)
        a = sitk.GetArrayFromImage(r).astype(np.int32)
        if merged is None:
            merged = np.zeros_like(a)
        if len(layers) == 1:
            merged = a                       # already a label map with distinct values
        elif low_priority_label is not None and i + 1 == int(low_priority_label):
            deferred = a > 0                 # applied last, and only where nothing else is
        else:
            merged[a > 0] = i + 1
    if deferred is not None:
        # The background segment is allowed to overlap the inserts -- painting the whole
        # module is the natural way to draw it -- but an insert must never be overwritten
        # by it, or it would go undetected and then be counted as noise.  Priority is
        # explicit rather than depending on the order the segments happen to be stored in.
        merged[(merged == 0) & deferred] = int(low_priority_label)
    if merged is None or not np.any(merged > 0):
        return None
    return merged


def _load_match(out_root):
    """Read stage 0's output, with an actionable message if it was never run."""
    p = Path(out_root) / "match_config.json"
    if not p.exists():
        raise SystemExit(
            f"{p} not found -- run stage 0 first:\n"
            f"  python -m reconstruction.recon_comparison --stage 0 "
            f"--wfbp-dir ... --vmi-dir ... --out-root {out_root}")
    return json.loads(p.read_text(encoding="utf-8"))


def _jsonable(o):
    """Recursively convert numpy types/arrays so json.dump accepts them."""
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items() if not k.endswith("_map")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return o


def _write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_jsonable(obj), indent=2), encoding="utf-8")


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        logger.warning("matplotlib unavailable -- skipping figures")
        return None


# ═════════════════════════════════════════════════════════════════════
# Stage 0 -- probe the Siemens export
# ═════════════════════════════════════════════════════════════════════
def _classify_text(text, loose=False):
    """
    Classify one string as a monoenergetic or threshold series.

    Real NAEOTOM SeriesDescriptions look like
        'MonoEnergeticPlus 70 keV'
        'ProtocolModel WFBP_T1 Qr40f(3) 0.4 (0.4) [A,1]_0'
    so the keV number is NOT adjacent to the word 'Mono' -- the product name sits in
    between.  Match 'mono' and the keV figure independently rather than assuming they
    are neighbours.  'VNC' and other spectral products match neither and are ignored.
    """
    if re.search(r"mono", text, re.IGNORECASE):
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*ke?V", text, re.IGNORECASE)
        if m:
            return ("mono", float(m.group(1)))
    m = re.search(r"WFBP[_ ]*T([0-9]+)", text, re.IGNORECASE)
    if m:
        return ("threshold", int(m.group(1)))
    if loose:
        m = re.search(r"[_ /\\]T([0-9]+)[_ ]", text)
        if m:
            return ("threshold", int(m.group(1)))
    return None


def _classify_series(desc, path):
    """
    Series -> ('mono', keV) | ('threshold', n) | None.

    The SeriesDescription is tried ALONE first, and only then description+path.  A
    Siemens export puts several products in one folder -- the VMI export here also
    carries WFBP_T1/T2 series -- so a folder name containing 'Mono' must never be
    allowed to re-label a series whose own description says WFBP.

    Kept in step with decomposition/material_decomposition.py, but local, to avoid a
    reconstruction -> decomposition package dependency.
    """
    return (_classify_text(desc or "")
            or _classify_text(f"{desc or ''} {path or ''}", loose=True))


def _is_dicom(path):
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def index_dicom_series(folder, cache_dir=None):
    """
    Index DICOM series under `folder`, keeping the acquisition metadata the comparison
    needs.  SliceThickness and PixelSpacing are what make the comparison fair, and
    ConvolutionKernel records what we are comparing against.

    Reading 20-30 k headers takes minutes, and stages 0 and 2 both need the index, so
    the result is cached per folder under cache_dir and reused.  Delete the cache file
    (or pass --force) if the export itself changes.

    Returns {uid: {number, desc, kernel, thickness_mm, spacing_between_mm, pixel_mm,
                   rows, cols, recon_diameter_mm, files: [[z, path], ...]}}
    """
    cache = None
    if cache_dir:
        import hashlib
        h = hashlib.sha1(str(folder).encode("utf-8")).hexdigest()[:12]
        cache = Path(cache_dir) / f"dicom_index_{h}.json"
        if cache.exists():
            try:
                blob = json.loads(cache.read_text(encoding="utf-8"))
                if blob.get("folder") == str(folder):
                    logger.info("using cached DICOM index (%d series) %s",
                                len(blob["series"]), cache.name)
                    return blob["series"]
            except (OSError, ValueError, KeyError):
                pass

    sitk = _sitk()
    series = {}
    n_seen = 0
    for root, _dirs, names in os.walk(str(folder)):
        for name in names:
            fp = os.path.join(root, name)
            if not (name.lower().endswith((".dcm", ".ima")) or _is_dicom(fp)):
                continue
            r = sitk.ImageFileReader()
            r.SetFileName(fp)
            try:
                r.ReadImageInformation()
            except RuntimeError:
                continue

            def g(t):
                return r.GetMetaData(t).strip() if r.HasMetaDataKey(t) else ""

            uid = g("0020|000e")
            if not uid:
                continue
            try:
                z = float(g("0020|0032").split("\\")[2])
            except (ValueError, IndexError):
                z = 0.0
            if uid not in series:
                def fnum(tag, idx=None):
                    v = g(tag)
                    if not v:
                        return None
                    try:
                        return float(v.split("\\")[idx]) if idx is not None else float(v)
                    except (ValueError, IndexError):
                        return None
                series[uid] = {
                    "number": g("0020|0011"), "desc": g("0008|103e"),
                    "kernel": g("0018|1210"),
                    "thickness_mm": fnum("0018|0050"),
                    "spacing_between_mm": fnum("0018|0088"),
                    "pixel_mm": fnum("0028|0030", 0),
                    "recon_diameter_mm": fnum("0018|1100"),
                    "rows": int(float(g("0028|0010") or 0)) or None,
                    "cols": int(float(g("0028|0011") or 0)) or None,
                    "files": [],
                }
            series[uid]["files"].append([z, fp])
            n_seen += 1
            if n_seen % 2000 == 0:
                logger.info("  indexed %d DICOM headers ...", n_seen)
    for s in series.values():
        s["files"].sort(key=lambda zf: zf[0])
        if s["spacing_between_mm"] is None and len(s["files"]) > 1:
            dz = np.diff([z for z, _ in s["files"]])
            dz = dz[np.abs(dz) > 1e-6]
            s["spacing_between_mm"] = float(np.median(np.abs(dz))) if dz.size else None
    logger.info("indexed %d series / %d files under %s", len(series), n_seen, folder)
    if cache is not None:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"folder": str(folder), "series": series}),
                             encoding="utf-8")
            logger.info("cached index -> %s", cache.name)
        except OSError:
            pass
    return series


def select_family(index, kind):
    """Pick the channels of one product ('wfbp' or 'vmi') out of a series index."""
    want = "mono" if kind == "vmi" else "threshold"
    found = []
    for uid, s in index.items():
        spec = _classify_series(s.get("desc", ""), s["files"][0][1] if s["files"] else "")
        if not spec or spec[0] != want:
            continue
        label = f"{spec[1]:g}keV" if want == "mono" else f"T{spec[1]}"
        found.append({"uid": uid, "key": spec[1], "label": label, **s})
    found.sort(key=lambda d: d["key"])

    dup = [d["label"] for d in found]
    if len(set(dup)) != len(dup):
        raise ValueError(
            f"{kind}: duplicate channel labels {sorted(dup)} under one folder -- the "
            f"directory holds more than one {kind} reconstruction. Point --{kind}-dir at "
            f"the specific series folder, otherwise they are silently concatenated into "
            f"one oversized channel stack.")
    return found


def read_series_volume(entry, z_range=None):
    """Read one DICOM series as (array (Z,Y,X) float32, sitk image, z positions mm)."""
    sitk = _sitk()
    files = entry["files"]
    if z_range is not None:
        lo, hi = sorted(z_range)
        files = [f for f in files if lo <= f[0] <= hi] or files
    r = sitk.ImageSeriesReader()
    r.SetFileNames([p for _z, p in files])
    img = r.Execute()
    return (sitk.GetArrayFromImage(img).astype(np.float32), img,
            np.array([z for z, _ in files], dtype=float))


def stage0_probe(cfg: CompareConfig):
    """Read the vendor geometry and locate the insert slab -> match_config.json."""
    out = Path(cfg.out_root)
    (out / "qc").mkdir(parents=True, exist_ok=True)
    if not cfg.wfbp_dir and not cfg.vmi_dir:
        raise ValueError("stage 0 needs --wfbp-dir and/or --vmi-dir")

    families = {}
    for kind, folder in (("wfbp", cfg.wfbp_dir), ("vmi", cfg.vmi_dir)):
        if not folder:
            continue
        idx = index_dicom_series(folder, cache_dir=out)
        chans = select_family(idx, kind)
        if not chans:
            disc = [(s.get("number"), s.get("desc")) for s in idx.values()]
            raise ValueError(f"no '{kind}' series under {folder}. Discovered: {disc}")
        families[kind] = chans
        logger.info("%s: %d channels %s", kind, len(chans), [c["label"] for c in chans])

    # Geometry is taken from WFBP when available: it is the family our thresholds are
    # actually compared against 1:1, so matching its grid is what removes the confound.
    ref_kind = "wfbp" if "wfbp" in families else "vmi"
    ref = families[ref_kind][0]
    thickness = ref["thickness_mm"] or 1.0
    spacing = ref["spacing_between_mm"] or thickness
    pixel_mm = ref["pixel_mm"] or 1.0
    cols = ref["cols"] or 512
    fov = ref["recon_diameter_mm"] or (pixel_mm * cols)

    logger.info("reference geometry from %s/%s: %dx%d, pixel %.4f mm, FOV %.1f mm, "
                "thickness %.2f mm, spacing %.2f mm, kernel %r",
                ref_kind, ref["label"], ref["rows"], cols, pixel_mm, fov,
                thickness, spacing, ref["kernel"])
    if spacing < thickness - 1e-6:
        logger.info("  NOTE overlapping slices (spacing %.2f < thickness %.2f mm): the "
                    "sweep reproduces both, so the z-correlation matches too",
                    spacing, thickness)

    # Locate the insert layer on the reference volume (CPU, no GPU needed).
    # Reading is limited to the phantom when known: a Thx-Abdomen series is thousands of
    # slices, and the whole point of the restriction is that the rest of the scan range
    # (table, positioning aids, scan-end artefacts) can out-score the phantom entirely.
    phantom_z = cfg.phantom_z_mm if cfg.use_phantom_z else None
    if phantom_z:
        pad = max(cfg.slab_pad_mm, 5.0)
        read_range = (min(phantom_z) - pad, max(phantom_z) + pad)
        logger.info("restricting to the phantom: z = %.2f .. %.2f mm", *sorted(phantom_z))
    else:
        read_range = None
        logger.info("no --phantom-z given: searching the WHOLE scan range for the insert "
                    "layer. On a long clinical range the table or a scan-end artefact can "
                    "out-score the phantom -- check the QC figure carefully.")
    vol, img, zpos = read_series_volume(ref, z_range=read_range)
    logger.info("reference volume: %d slices, z = %.1f .. %.1f mm",
                vol.shape[0], zpos.min(), zpos.max())

    slab = iq.find_insert_slab(vol, pixel_mm, zpos, search_z_mm=phantom_z,
                               select=cfg.slab_select)
    logger.info("candidate insert layers (ranked by %s):", cfg.slab_select)
    for i, c in enumerate(slab["candidates"][:8]):
        mark = " <- selected" if (c["k_lo"], c["k_hi"]) == (slab["k_lo"], slab["k_hi"]) else ""
        logger.info("   %d. z = %+9.2f .. %+9.2f mm  %3d slices  peak %.4f  mean %.4f%s",
                    i + 1, c["z_lo_mm"], c["z_hi_mm"], c["n_slices"],
                    c["peak_score"], c["mean_score"], mark)
    if len(slab["candidates"]) > 1:
        logger.info("   (pass --slab z_lo,z_hi to force a different layer)")

    z_lo = slab["z_lo_mm"] - cfg.slab_pad_mm
    z_hi = slab["z_hi_mm"] + cfg.slab_pad_mm
    slab_source = f"structure score ({cfg.slab_select})"

    # A segmentation knows where the inserts are far better than any structure score can
    # infer it, so when one is supplied it defines the slab.  This matters when the score
    # is flat -- inserts that run as rods along z produce no distinguishable "layer", and
    # the score then spans the whole phantom and gets blindly trimmed to its middle.
    if cfg.segmentation and cfg.use_phantom_z is not False:
        labels = load_segmentation(cfg.segmentation, img,
                                   low_priority_label=cfg.background_label)
        if labels is None:
            logger.warning("segmentation %s does not overlap the reference volume -- "
                           "keeping the detected slab", cfg.segmentation)
        else:
            # The slab is defined by where the INSERTS are, so the background segment is
            # excluded here: it may be drawn over a different z range (or over the whole
            # module), and letting it stretch the slab would move the measurement away
            # from the inserts the TTF depends on.
            insert_labels = labels
            if cfg.background_label:
                insert_labels = np.where(labels == int(cfg.background_label), 0, labels)
            kz = np.where(insert_labels.reshape(insert_labels.shape[0], -1).any(axis=1))[0]
            if kz.size == 0:
                kz = np.where(labels.reshape(labels.shape[0], -1).any(axis=1))[0]
            z_lo = float(zpos[kz].min()) - cfg.slab_pad_mm
            z_hi = float(zpos[kz].max()) + cfg.slab_pad_mm
            slab_source = f"segmentation ({Path(cfg.segmentation).name})"
            logger.info("slab taken from the segmentation: z = %.2f .. %.2f mm "
                        "(%d annotated slices, labels %s)", z_lo, z_hi, kz.size,
                        sorted(int(v) for v in np.unique(labels) if v > 0))

    if cfg.slab_mm is not None:
        z_lo, z_hi = sorted(cfg.slab_mm)
        slab_source = "--slab"
        logger.info("slab overridden by --slab: %.2f .. %.2f mm", z_lo, z_hi)
    elif (z_hi - z_lo) > cfg.max_slab_mm:                 # keep the GPU sweep affordable
        mid = 0.5 * (z_lo + z_hi)
        z_lo, z_hi = mid - cfg.max_slab_mm / 2, mid + cfg.max_slab_mm / 2
        logger.info("slab trimmed to %.1f mm about its centre", cfg.max_slab_mm)
    if phantom_z:                       # never let padding leave the phantom
        z_lo = max(z_lo, min(phantom_z))
        z_hi = min(z_hi, max(phantom_z))
    logger.info("insert slab: z = %.2f .. %.2f mm (%d slices in the vendor volume)",
                z_lo, z_hi, int(np.sum((zpos >= z_lo) & (zpos <= z_hi))))

    _qc_slab_figure(out / "qc" / "stage0_slab_detection.png", vol, zpos, slab, z_lo, z_hi,
                    pixel_mm, ref_kind, ref["label"], phantom_z)

    match = {
        "reference_family": ref_kind,
        "reference_label": ref["label"],
        "kernel": ref["kernel"],
        "n_pixels": int(cfg.n_pixels or cols),
        "fov_mm": float(cfg.fov_mm or fov),
        "pixel_mm": float((cfg.fov_mm or fov) / (cfg.n_pixels or cols)),
        "slice_thickness_mm": float(cfg.slice_thickness_mm or thickness),
        "slice_spacing_mm": float(cfg.slice_spacing_mm or spacing),
        "z_lo_mm": float(z_lo), "z_hi_mm": float(z_hi),
        "families": {k: [{"label": c["label"], "uid": c["uid"], "desc": c["desc"],
                          "kernel": c["kernel"], "thickness_mm": c["thickness_mm"],
                          "spacing_between_mm": c["spacing_between_mm"],
                          "pixel_mm": c["pixel_mm"], "rows": c["rows"], "cols": c["cols"],
                          "n_files": len(c["files"])}
                         for c in v] for k, v in families.items()},
        "wfbp_dir": cfg.wfbp_dir, "vmi_dir": cfg.vmi_dir,
        # Persisted because each stage is a SEPARATE process: stage 2 is invoked
        # without --wfbp-dir, so auto-discovery cannot run there and the segmentation
        # would silently be dropped between stages.
        "segmentation": cfg.segmentation,
        "background_label": cfg.background_label,
    }
    _write_json(out / "match_config.json", match)
    logger.info("wrote %s", out / "match_config.json")
    logger.info("CHECK output/research/recon_comparison/qc/stage0_slab_detection.png "
                "before running stage 1 -- the marked band must contain the inserts.")
    del vol
    gc.collect()
    return match


def _qc_slab_figure(path, vol, zpos, slab, z_lo, z_hi, pixel_mm, kind, label,
                    phantom_z=None):
    plt = _mpl()
    if plt is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].plot(zpos, slab["score"], lw=1.2)
    ax[0].axhline(slab["threshold"], color="grey", ls=":", label="threshold")
    # Every candidate layer is drawn, not just the winner: a phantom with several insert
    # layers should show several bands, and seeing the ones that were NOT chosen is how
    # you tell "picked the wrong layer" from "only found one".
    for i, c in enumerate(slab.get("candidates", [])):
        ax[0].axvspan(c["z_lo_mm"], c["z_hi_mm"], color="tab:blue", alpha=0.12,
                      label="candidate layers" if i == 0 else None)
    if phantom_z:
        for zv in sorted(phantom_z):
            ax[0].axvline(zv, color="tab:green", ls="--", lw=1.0)
        ax[0].axvline(np.nan, color="tab:green", ls="--", lw=1.0, label="--phantom-z")
    ax[0].axvspan(z_lo, z_hi, color="tab:orange", alpha=0.35, label="selected slab")
    ax[0].set_xlabel("z (mm)")
    ax[0].set_ylabel("fraction of body voxels far from median")
    ax[0].set_title(f"insert-layer score ({len(slab.get('candidates', []))} candidates)")
    ax[0].legend(fontsize=8)

    k_mid = int(np.clip((slab["k_lo"] + slab["k_hi"]) // 2, 0, vol.shape[0] - 1))
    ax[1].imshow(vol[k_mid], cmap="gray", vmin=-200, vmax=300)
    ax[1].set_title(f"selected slab, mid slice (k={k_mid}, z={zpos[k_mid]:+.1f} mm)")
    ax[1].axis("off")

    k_out = 0 if slab["k_lo"] > vol.shape[0] // 2 else vol.shape[0] - 1
    ax[2].imshow(vol[k_out], cmap="gray", vmin=-200, vmax=300)
    ax[2].set_title(f"outside the slab (k={k_out}, z={zpos[k_out]:+.1f} mm)")
    ax[2].axis("off")
    fig.suptitle(f"Stage 0 slab detection -- reference {kind} {label}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════
# Stage 1 -- the reconstruction sweep (GPU)
# ═════════════════════════════════════════════════════════════════════
def _variant_signature(variant, match, cfg: CompareConfig):
    """Everything that would invalidate a cached variant if it changed."""
    return {
        "variant": variant,
        "n_pixels": match["n_pixels"], "fov_mm": match["fov_mm"],
        "slice_thickness_mm": match["slice_thickness_mm"],
        "slice_spacing_mm": match["slice_spacing_mm"],
        "z_lo_mm": match["z_lo_mm"], "z_hi_mm": match["z_hi_mm"],
        "geometry_model": cfg.geometry_model, "z_weighting": cfg.z_weighting,
        "wavelet_ring_threshold": cfg.wavelet_ring_threshold,
        "spike_mad_k": cfg.spike_mad_k, "ipr_mad_k": cfg.ipr_mad_k,
        "data_path": cfg.data_path,
    }


def _variant_is_current(vdir: Path, signature):
    """
    A variant counts as done only if every volume exists AND its recorded signature
    matches.  Checking file existence alone would silently reuse volumes reconstructed
    from a different slab or grid -- the failure mode that quietly corrupts a comparison.
    """
    cfg_path = vdir / "sweep_config.json"
    if not cfg_path.exists():
        return False
    if not all((vdir / f"reconstruction_thr_{l}_HU.nii.gz").exists() for l in THRESHOLD_LABELS):
        return False
    try:
        stored = json.loads(cfg_path.read_text(encoding="utf-8")).get("signature")
    except (OSError, ValueError):
        return False
    return _jsonable(stored) == _jsonable(signature)


def stage1_sweep(cfg: CompareConfig):
    """Reconstruct every variant over the matched slab and grid."""
    import h5py
    import scipy.io as sio
    from reconstruction.helical_reconstruction import (
        build_geom, detect_defect_channels, reconstruct_helical_stack,
        z_targets_for_full_scan, auto_hu_calibrate, apply_hu_calibration)

    out = Path(cfg.out_root)
    match = _load_match(out)

    todo = []
    for v in cfg.variants:
        vdir = out / "sweep" / v["name"]
        sig = _variant_signature(v, match, cfg)
        if not cfg.force and _variant_is_current(vdir, sig):
            logger.info("variant %-16s up to date -- skipping", v["name"])
            continue
        todo.append((v, vdir, sig))
    if not todo:
        logger.info("all %d variants up to date (use --force to redo)", len(cfg.variants))
        return
    logger.info("reconstructing %d/%d variants", len(todo), len(cfg.variants))

    descriptor = sio.loadmat(cfg.desc_path, struct_as_record=True,
                             squeeze_me=False)["descriptor"].flat[0]
    geom = build_geom(descriptor, geo_dir=cfg.geo_dir, channels_flipped=True)

    end_margin = 1.0 if cfg.z_weighting == "balanced" else 0.5
    z_all, z_spacing = z_targets_for_full_scan(geom, oversample=1,
                                               end_margin_rotations=end_margin)
    # Native slices spanning the slab, padded by half the target thickness so the
    # outermost matched slice still has a full averaging window.
    pad = 0.5 * match["slice_thickness_mm"] + abs(z_spacing)
    # Report both frames: the slab comes from the vendor DICOM z (ImagePositionPatient)
    # while we reconstruct in our own helical z.  They should share the scanner table
    # coordinate, but that is an assumption -- printing both makes a mismatch obvious
    # instead of it showing up later as a phantom that is missing from qc/roi_own.png.
    logger.info("z frames: ours %.1f .. %.1f mm | requested slab (vendor frame) "
                "%.1f .. %.1f mm", z_all.min(), z_all.max(),
                match["z_lo_mm"], match["z_hi_mm"])
    sel = (z_all >= match["z_lo_mm"] - pad) & (z_all <= match["z_hi_mm"] + pad)
    z_native = z_all[sel]
    if z_native.size == 0:
        raise ValueError(
            f"slab z={match['z_lo_mm']:.1f}..{match['z_hi_mm']:.1f} mm lies outside the "
            f"reconstructable range z={z_all.min():.1f}..{z_all.max():.1f} mm. The vendor "
            f"z-origin and ours may differ -- check qc/stage0_slab_detection.png and pass "
            f"--slab explicitly.")
    z_target = np.arange(match["z_lo_mm"], match["z_hi_mm"] + 1e-6,
                         match["slice_spacing_mm"])
    logger.info("slab: %d native slices (dz=%.4f mm) -> %d matched slices "
                "(thickness %.2f mm, spacing %.2f mm)",
                z_native.size, abs(z_spacing), z_target.size,
                match["slice_thickness_mm"], match["slice_spacing_mm"])

    with h5py.File(cfg.data_path, "r") as f:
        sino_A = f[f["data_full"]["A"][3, 0]][:][:, :, ::-1].astype(np.float32)
    geom["spike_mask"] = detect_defect_channels(sino_A, spike_mad_k=cfg.spike_mad_k,
                                                ipr_mad_k=cfg.ipr_mad_k)
    logger.info("defect mask: %d channels (from threshold A, reused for B/C/D)",
                int(geom["spike_mask"].sum()))
    del sino_A
    gc.collect()

    sitk = _sitk()
    n_pixels = match["n_pixels"]
    xy = match["fov_mm"] / n_pixels
    origin = (-match["fov_mm"] / 2 + xy / 2, -match["fov_mm"] / 2 + xy / 2,
              float(z_target[0]))

    for li, label in enumerate(THRESHOLD_LABELS):
        with h5py.File(cfg.data_path, "r") as f:
            sino = f[f["data_full"]["A"][3 - li, 0]][:][:, :, ::-1].astype(np.float32)
        logger.info("threshold %s loaded %s", label, sino.shape)
        for v, vdir, sig in todo:
            vdir.mkdir(parents=True, exist_ok=True)
            logger.info("  variant %-16s threshold %s ...", v["name"], label)
            vol = reconstruct_helical_stack(
                sino, geom, z_native, method="astra", n_pixels=n_pixels,
                filter_name=v.get("filter_name", "shepp-logan"),
                geometry_model=cfg.geometry_model,
                z_weighting=cfg.z_weighting,
                wavelet_ring_threshold=cfg.wavelet_ring_threshold,
                algorithm=v["algorithm"], n_iter=v.get("n_iter", 100),
            )
            # Z_SMOOTH stays off; thickness is matched by averaging instead.
            matched, n_used = iq.match_slice_thickness(
                vol, z_native, match["slice_thickness_mm"], z_target)
            del vol
            gc.collect()

            calib = auto_hu_calibrate(matched, fov_mm=match["fov_mm"])
            _write_json(vdir / f"calibration_thr_{label}.json", calib)
            vol_hu = apply_hu_calibration(matched, calib)
            del matched

            img = sitk.GetImageFromArray(vol_hu.astype(np.float32))
            img.SetSpacing((xy, xy, abs(match["slice_spacing_mm"])))
            img.SetOrigin(origin)
            img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
            sitk.WriteImage(img, str(vdir / f"reconstruction_thr_{label}_HU.nii.gz"))
            logger.info("    -> %s  HU [%.0f, %.0f]  (%.1f native slices averaged)",
                        f"reconstruction_thr_{label}_HU.nii.gz",
                        float(vol_hu.min()), float(vol_hu.max()), float(np.mean(n_used)))
            del vol_hu, img
            gc.collect()
        del sino
        gc.collect()

    for v, vdir, sig in todo:
        _write_json(vdir / "sweep_config.json",
                    {"signature": sig, "z_target_mm": z_target.tolist(),
                     "n_native_slices": int(z_native.size)})
    logger.info("sweep complete: %d variants", len(todo))


# ═════════════════════════════════════════════════════════════════════
# Stage 2 -- metrics
# ═════════════════════════════════════════════════════════════════════
def _measure_volume(vol, pixel_mm, rois, cfg: CompareConfig, body, patches, patch_px,
                    region):
    """
    Full metric set for one volume, given ROIs fixed by the family's reference.

    NPS is measured slice by slice -- never on a z-average, which would divide the
    noise by sqrt(Z) and report a quantity no single image has.  The TTF is measured on
    the z-average instead: averaging slices leaves in-plane resolution untouched (the
    slab lies inside one insert layer, so the cylinders are z-invariant across it) while
    cutting the noise that would otherwise corrupt the differentiated edge profile.
    """
    res = {}
    # Noise MAGNITUDE from the whole region: needs no square blocks, so it survives a
    # phantom whose uniform material is only a thin ring between inserts.
    sd, n_vox = iq.region_noise_sd(vol, region)
    res["noise_sd_hu"] = sd
    res["noise_region_px"] = n_vox
    # Noise TEXTURE needs a Fourier estimate, hence square patches; it degrades to
    # unavailable rather than dragging the magnitude down with it.
    if patches:
        nps = iq.noise_power_spectrum(vol, patches, pixel_mm, patch_px=patch_px)
        res["f_av"] = nps["f_av"]
        res["f_peak"] = nps["f_peak"]
        res["nps_f"] = nps["f"]
        res["nps"] = nps["nps"]
        res["n_patches"] = nps["n_patches"]
        res["patch_heterogeneity"] = nps["patch_heterogeneity"]
        res["nps_sd_hu"] = nps["noise_sd_hu"]
    else:
        res["f_av"] = float("nan")
        res["f_peak"] = float("nan")
        res["n_patches"] = 0
        nps = None

    flat = vol.mean(axis=0)
    best = None
    for ins in rois:
        try:
            t = iq.task_transfer_function(flat, (ins["cy"], ins["cx"]),
                                          ins["radius_mm"], pixel_mm)
        except ValueError:
            continue
        if best is None or abs(t["contrast_hu"]) > abs(best["contrast_hu"]):
            best = t
    if best is not None:
        res.update({"ttf50": best["ttf50"], "ttf10": best["ttf10"],
                    "edge_width_1090_mm": best["edge_width_1090_mm"],
                    "ttf_contrast_hu": best["contrast_hu"],
                    "ttf_f": best["f"], "ttf": best["ttf"]})
        if nps is not None:
            f_neq, val = iq.neq(best["f"], best["ttf"], nps["f"], nps["nps"],
                                contrast_hu=abs(best["contrast_hu"]) or 1.0)
            res["neq_f"], res["neq"] = f_neq, val
            res.update(iq.detectability_index(best["f"], best["ttf"], nps["f"],
                                              nps["nps"], cfg.task_diameter_mm,
                                              cfg.task_contrast_hu))
    else:
        logger.warning("    no usable edge -- TTF/NEQ/d' unavailable for this volume")

    res["roi"] = iq.roi_statistics(vol, rois, body, pixel_mm)
    return res


def _family_rois(vol_ref, pixel_mm, cfg: CompareConfig, tag, qc_dir, force_patch_px=None,
                 ref_img=None):
    """
    Detect body/inserts/background patches ONCE per family, then reuse for every channel
    and variant of that family, so a metric difference can never come from the ROIs
    having moved.

    Detection runs on the z-AVERAGE of the slab, not on one slice.  The first volume the
    driver happens to load is whichever variant comes first (ram-lak -- deliberately the
    noisiest of the five), and on a single noisy slice a low-contrast insert can fail the
    contrast test, so it is never excluded from the background and the NPS patches then
    sit on top of it.  Averaging the slab lifts detection SNR by sqrt(Z) without moving
    anything, because the inserts are z-invariant within the layer stage 0 selected.
    """
    flat = vol_ref.mean(axis=0)
    body = iq.detect_body_mask(flat, pixel_mm)

    inserts, source, bg_region = [], "auto", None
    seg_path = cfg.segmentation
    if seg_path and ref_img is not None:
        labels = load_segmentation(seg_path, ref_img,
                                   low_priority_label=cfg.background_label)
        if labels is None:
            # Empty overlap is itself a finding: the segmentation was drawn on the vendor
            # volume, so if it maps nowhere onto ours the two z frames disagree.
            logger.warning("  %s: the segmentation does not overlap this volume at all. "
                           "For 'own' that means our helical z and the vendor's "
                           "ImagePositionPatient z are NOT the same frame. Falling back "
                           "to automatic detection for this family.", tag)
        else:
            if cfg.background_label:
                # One drawn segment marks the uniform region the noise is measured in.
                # It must NOT also be treated as an insert.
                drawn = np.any(labels == int(cfg.background_label), axis=0)
                # Filling means an OUTLINE of the module works as well as a solid
                # paint -- which matters because Slicer's fill-between-slices needs
                # a clear path, and tracing the module boundary avoids the inserts
                # entirely.  A solid region is unchanged by this.
                bg_region = ndimage.binary_fill_holes(drawn)
                labels = np.where(labels == int(cfg.background_label), 0, labels)
                logger.info("  %s: background from label %s -- %d px drawn, %d px "
                            "after filling the interior", tag, cfg.background_label,
                            int(drawn.sum()), int(bg_region.sum()))
            inserts = iq.inserts_from_labelmap(labels, flat, pixel_mm, refine=True)
            source = f"segmentation ({Path(seg_path).name})"
            n_ref = sum(1 for i in inserts if i["refined"])
            logger.info("  %s: %d seeded ROIs from %d labels, %d edges refined from the "
                        "image", tag, len(inserts),
                        len({i["label"] for i in inserts}), n_ref)
            for i in inserts:
                if not i["refined"]:
                    logger.warning("      label %s ROI at (%.0f,%.0f): too little contrast "
                                   "to refine, kept the drawn outline (r=%.1f mm)",
                                   i["label"], i["cy"], i["cx"], i["radius_mm"])
    if not inserts:
        inserts = iq.detect_inserts(flat, body, pixel_mm,
                                    contrast_mad_k=cfg.insert_mad_k)

    # ---- where the noise is measured -------------------------------------------
    # The body outline is NOT a noise region on an anthropomorphic phantom: lung, bone,
    # mediastinum and the table all lie inside it, and a patch on any of them measures
    # anatomy.  Preference order: a drawn background segment, else the automatically
    # detected homogeneous region, and only as a last resort the whole body.
    if bg_region is not None:
        region, region_src, edge_mm = bg_region, "drawn background label", 2.0
    else:
        region = iq.homogeneous_mask(flat, body, pixel_mm)
        lab, nlab = ndimage.label(region)
        if nlab > 1:                       # a fragmented mask cannot hold square patches
            sizes = ndimage.sum(region, lab, range(1, nlab + 1))
            region = lab == (1 + int(np.argmax(sizes)))
        region_src, edge_mm = "auto homogeneous region", 2.0
        if region.sum() < 0.02 * max(body.sum(), 1):
            logger.warning("  %s: the homogeneous region is only %d px; falling back to "
                           "the whole body, which on a non-uniform phantom measures "
                           "anatomy rather than noise", tag, int(region.sum()))
            region, region_src, edge_mm = body, "body outline (UNRELIABLE)", 10.0

    if force_patch_px:
        # Every family must share one patch size: the NPS frequency grid depends on it,
        # so curves measured at different sizes are not comparable.
        patches = iq.background_patches(region, inserts, pixel_mm, patch_px=force_patch_px,
                                        edge_margin_mm=edge_mm)
        patch_px, tried = force_patch_px, [(force_patch_px, len(patches))]
    else:
        patches, patch_px, tried = iq.auto_background_patches(
            region, inserts, pixel_mm, patch_px=cfg.nps_patch_px, edge_margin_mm=edge_mm)
        if patch_px != cfg.nps_patch_px:
            logger.info("  %s: %d px patches did not fit; using %d px (%.1f mm). Tried %s",
                        tag, cfg.nps_patch_px, patch_px, patch_px * pixel_mm,
                        ", ".join(f"{s}px->{n}" for s, n in tried))
    # Whatever the region came from, the inserts and the edge must come OUT of it before
    # anything is measured -- otherwise insert contrast is counted as noise.
    region = iq.noise_region(region, inserts, pixel_mm, edge_margin_mm=edge_mm)
    logger.info("  %s: noise region = %s (%d px, %.0f mm2 after removing inserts+edge)",
                tag, region_src, int(region.sum()), region.sum() * pixel_mm ** 2)
    if region.sum() < 200:
        logger.warning("  %s: the noise region is only %d px -- the noise SD will be very "
                       "uncertain. Draw a background segment and pass --background-label.",
                       tag, int(region.sum()))
    if len(patches) < 8 or patch_px < 24:
        logger.warning("  %s: only %d patches of %d px (%.1f mm) -- the noise SD is still "
                       "usable but the NPS CURVE is coarse (%d frequency bins to Nyquist) "
                       "and f_av is unreliable. Draw a background segment in a uniform "
                       "part of the phantom and pass --background-label to fix this.",
                       tag, len(patches), patch_px, patch_px * pixel_mm, patch_px // 2)

    body_px = int(body.sum())
    logger.info("  %s: body %d px (%.0f mm2), %d inserts [%s], %d NPS patches of "
                "%d px (%.1f mm)", tag, body_px, body_px * pixel_mm ** 2, len(inserts),
                source, len(patches), patch_px, patch_px * pixel_mm)
    for i, ins in enumerate(inserts):
        logger.info("      insert %d: r=%.1f mm  %+.0f HU vs background  circularity %.2f",
                    i, ins["radius_mm"], ins["contrast_hu"], ins["circularity"])
    if not inserts:
        logger.warning("  %s: no inserts detected -- TTF/NEQ/d' unavailable, and the "
                       "background is not insert-excluded. Check qc/roi_%s.png", tag, tag)
    elif cfg.expect_inserts and len(inserts) != cfg.expect_inserts:
        logger.warning("  %s: found %d inserts but %d were expected. THE NOISE NUMBERS "
                       "FOR THIS FAMILY ARE NOT VALID: every missing insert stays inside "
                       "the background region, inflating the noise SD and dragging f_av "
                       "down. Supply --segmentation, or lower --insert-mad-k, then "
                       "confirm on qc/roi_%s.png before using any of it.",
                       tag, len(inserts), cfg.expect_inserts, tag)

    # Written BEFORE any failure: when detection goes wrong the overlay is the only way
    # to see why, so it must exist even on the unhappy path.
    _qc_roi_volume(Path(qc_dir) / f"roi_{tag}.nrrd", vol_ref.shape[0], flat.shape, ref_img,
                   body, inserts, patches, patch_px, tag, source)

    if not patches:
        logger.warning(
            "  %s: no square patch fits the noise region, so the NPS curve, f_av, NEQ and "
            "d' are unavailable for this family. The noise SD is still measured over the "
            "whole region and remains valid. Tried %s", tag,
            ", ".join(f"{s}px({s * pixel_mm:.0f}mm)->{n}" for s, n in tried))
    return body, inserts, patches, patch_px, region


def _qc_roi_volume(path, n_slices, shape_yx, ref_img, body, inserts, patches, patch_px,
                   tag, source):
    """
    Write the ROI placement as a 3-D LABEL VOLUME, loadable straight into Slicer on top
    of the data, instead of a single-slice picture.

    A mid-slice PNG can only ever show one plane, and the thing that has to be checked --
    do the circles sit on the inserts, does any NPS patch touch one -- is easier to judge
    while scrolling through the actual volume.  The labels are replicated through z
    because the ROIs really are applied to every slice of the slab (the NPS is measured
    on all of them), so the volume is a faithful picture of what was measured, not a
    stylised one.

    Label values (also written to the .json sidecar):
        1              NPS background patches -- the noise was measured HERE
        2 + i          insert i, at its measured radius
    The body is not painted; it is plainly visible in the underlying image.
    """
    sitk = _sitk()
    ny, nx = shape_yx
    plane = np.zeros((ny, nx), dtype=np.uint16)
    for (y0, x0) in patches:
        plane[y0:y0 + patch_px, x0:x0 + patch_px] = 1
    yy, xx = np.ogrid[:ny, :nx]
    legend = {"1": "NPS background patches"}
    for i, ins in enumerate(inserts):
        disc = ((yy - ins["cy"]) ** 2 + (xx - ins["cx"]) ** 2) <= ins["radius_px"] ** 2
        plane[disc] = 2 + i          # inserts win over patches so overlap is visible
        legend[str(2 + i)] = (f"insert {i}  r={ins['radius_mm']:.2f} mm  "
                              f"{ins.get('contrast_hu', float('nan')):+.0f} HU"
                              + ("  [refined]" if ins.get("refined") else ""))

    vol = np.repeat(plane[None], max(1, int(n_slices)), axis=0)
    img = sitk.GetImageFromArray(vol)
    if ref_img is not None:          # carry the geometry so it overlays correctly
        img.SetSpacing(ref_img.GetSpacing())
        img.SetOrigin(ref_img.GetOrigin())
        img.SetDirection(ref_img.GetDirection())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))

    _write_json(Path(path).with_suffix(".json"), {
        "tag": tag, "roi_source": source, "patch_px": int(patch_px),
        "n_inserts": len(inserts), "n_patches": len(patches),
        "labels": legend,
        "inserts": [{k: v for k, v in ins.items() if k != "area_px"} for ins in inserts],
    })


def stage2_metrics(cfg: CompareConfig):
    """Measure every family and write metrics + figures."""
    sitk = _sitk()
    out = Path(cfg.out_root)
    match = _load_match(out)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    qc = out / "qc"
    qc.mkdir(parents=True, exist_ok=True)

    if not cfg.segmentation and match.get("segmentation"):
        cfg.segmentation = match["segmentation"]
        logger.info("segmentation carried over from stage 0: %s", cfg.segmentation)
    if cfg.background_label is None and match.get("background_label"):
        cfg.background_label = match["background_label"]

    zr = (match["z_lo_mm"], match["z_hi_mm"])
    results = {"match_config": match, "own": {}, "wfbp": {}, "vmi": {}}

    # ---- own: every swept variant ------------------------------------
    own_ctx = None
    shared_patch_px = None      # one patch size for ALL families (comparable NPS grids)
    for v in cfg.variants:
        vdir = out / "sweep" / v["name"]
        if not (vdir / "reconstruction_thr_A_HU.nii.gz").exists():
            logger.warning("variant %s missing -- run stage 1 first", v["name"])
            continue
        results["own"][v["name"]] = {}
        for label in THRESHOLD_LABELS:
            p = vdir / f"reconstruction_thr_{label}_HU.nii.gz"
            if not p.exists():
                continue
            img_own = sitk.ReadImage(str(p))
            vol = sitk.GetArrayFromImage(img_own).astype(np.float32)
            pixel_mm = match["pixel_mm"]
            if own_ctx is None:      # detect once, reuse for every variant/threshold
                own_ctx = _family_rois(vol, pixel_mm, cfg, "own", qc, ref_img=img_own)
                shared_patch_px = own_ctx[3]
            body, inserts, patches, patch_px, region = own_ctx
            logger.info("own/%s thr %s", v["name"], label)
            results["own"][v["name"]][label] = _measure_volume(
                vol, pixel_mm, inserts, cfg, body, patches, patch_px, region)
            del vol
            gc.collect()

    # ---- Siemens families --------------------------------------------
    siemens_vols = {}
    for kind, folder in (("wfbp", match.get("wfbp_dir")), ("vmi", match.get("vmi_dir"))):
        if not folder or kind not in match["families"]:
            continue
        chans = select_family(index_dicom_series(folder, cache_dir=out), kind)
        ctx = None
        for c in chans:
            vol, img, _z = read_series_volume(c, z_range=zr)
            pixel_mm = float(img.GetSpacing()[0])
            if ctx is None:
                ctx = _family_rois(vol, pixel_mm, cfg, kind, qc,
                                   force_patch_px=shared_patch_px, ref_img=img)
                if shared_patch_px is None:
                    shared_patch_px = ctx[3]
            body, inserts, patches, patch_px, region = ctx
            logger.info("%s %s", kind, c["label"])
            results[kind][c["label"]] = _measure_volume(vol, pixel_mm, inserts, cfg,
                                                        body, patches, patch_px,
                                                        region)
            results[kind][c["label"]]["pixel_mm"] = pixel_mm
            results[kind][c["label"]]["kernel"] = c["kernel"]
            if kind == "wfbp":
                siemens_vols[c["label"]] = vol
            else:
                del vol
            gc.collect()

    # ---- paired difference: own vs WFBP, 1:1 by threshold ------------
    # VMI is excluded here because of the HU-scale mismatch, not its quality.
    if siemens_vols and own_ctx is not None:
        results["bias"] = _bias_analysis(cfg, match, siemens_vols, own_ctx)
    siemens_vols.clear()
    gc.collect()

    _write_json(out / "metrics" / "metrics.json", results)
    _figures(cfg, out, results, match)
    _print_summary(results)
    logger.info("wrote %s", out / "metrics" / "metrics.json")
    return results


def _bias_analysis(cfg, match, wfbp_vols, own_ctx):
    """Systematic (low-frequency) difference between our recon and WFBP, per threshold."""
    sitk = _sitk()
    out = Path(cfg.out_root)
    body, inserts, _patches, _patch_px, _region = own_ctx
    pixel_mm = match["pixel_mm"]
    bias = {}
    # T1..T4 are the vendor's names for the same cumulative thresholds as our A..D.
    pairs = list(zip(THRESHOLD_LABELS, [f"T{i}" for i in (1, 2, 3, 4)]))
    for v in cfg.variants:
        vdir = out / "sweep" / v["name"]
        if not (vdir / "reconstruction_thr_A_HU.nii.gz").exists():
            continue
        bias[v["name"]] = {}
        for own_lbl, wf_lbl in pairs:
            p = vdir / f"reconstruction_thr_{own_lbl}_HU.nii.gz"
            if not p.exists() or wf_lbl not in wfbp_vols:
                continue
            a = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)
            b = wfbp_vols[wf_lbl]
            n = min(a.shape[0], b.shape[0])
            if a.shape[1:] != b.shape[1:]:
                logger.warning("bias %s %s: in-plane shape %s vs %s -- skipped "
                               "(stage 1 grid did not match the vendor's)",
                               v["name"], own_lbl, a.shape[1:], b.shape[1:])
                del a
                continue
            d = iq.difference_analysis(a[:n], b[:n], pixel_mm, body,
                                       smooth_mm=cfg.bias_smooth_mm)
            d.pop("systematic_map", None)
            # Bland-Altman inputs: paired ROI means
            sa = iq.roi_statistics(a[:n], inserts, body, pixel_mm)
            sb = iq.roi_statistics(b[:n], inserts, body, pixel_mm)
            d["roi_pairs"] = [
                {"roi": i, "own_hu": x["mean_hu"], "wfbp_hu": y["mean_hu"],
                 "diff_hu": x["mean_hu"] - y["mean_hu"],
                 "mean_hu": 0.5 * (x["mean_hu"] + y["mean_hu"])}
                for i, (x, y) in enumerate(zip(sa["inserts"], sb["inserts"]))]
            d["roi_pairs"].append(
                {"roi": "background", "own_hu": sa["background"]["mean_hu"],
                 "wfbp_hu": sb["background"]["mean_hu"],
                 "diff_hu": sa["background"]["mean_hu"] - sb["background"]["mean_hu"],
                 "mean_hu": 0.5 * (sa["background"]["mean_hu"] + sb["background"]["mean_hu"])})
            bias[v["name"]][own_lbl] = d
            del a
            gc.collect()
    return bias


# ═════════════════════════════════════════════════════════════════════
# Figures
# ═════════════════════════════════════════════════════════════════════
def _figures(cfg, out, res, match):
    plt = _mpl()
    if plt is None:
        return
    fig_dir = Path(out) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- the headline: noise vs resolution, ONE PANEL PER THRESHOLD ---
    # Thresholds are never mixed on one axes: A has all the photons and D very few, so a
    # combined plot would mostly display the bins' photon statistics and hide the
    # reconstruction differences it is meant to show.
    fig, axes = plt.subplots(1, len(THRESHOLD_LABELS),
                             figsize=(4.6 * len(THRESHOLD_LABELS), 4.4), squeeze=False)
    for j, label in enumerate(THRESHOLD_LABELS):
        ax = axes[0][j]
        xs, ys, names = [], [], []
        for vname in [v["name"] for v in cfg.variants]:
            m = res["own"].get(vname, {}).get(label)
            if m and np.isfinite(m.get("ttf50", np.nan)):
                xs.append(m["ttf50"])
                ys.append(m["noise_sd_hu"])
                names.append(vname)
        if xs:
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order], "-o", color="tab:blue",
                    label="our reconstruction (sweep)")
            for x, y, nm in zip(xs, ys, names):
                ax.annotate(nm.replace("fbp_", "").replace("sirt_", "SIRT×"),
                            (x, y), fontsize=7, xytext=(3, 3),
                            textcoords="offset points")
        wf = res["wfbp"].get(f"T{j + 1}")
        if wf and np.isfinite(wf.get("ttf50", np.nan)):
            ax.plot(wf["ttf50"], wf["noise_sd_hu"], "*", ms=16, color="tab:red",
                    label="Siemens WFBP")
        ax.set_xlabel("TTF$_{50}$ (cyc/mm)   →  sharper")
        if j == 0:
            ax.set_ylabel("noise SD in background (HU)   →  noisier")
        ax.set_title(f"threshold {label}  /  WFBP T{j + 1}")
        ax.grid(alpha=0.3)
        if j == 0 and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        elif j == 0:
            ax.text(0.5, 0.5, "no TTF available\n(no insert edge measured)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9,
                    color="tab:red")
    fig.suptitle("Noise-resolution trade-off, per energy channel  "
                 "(below/left = better; the vendor's position relative to our curve "
                 "is the result)")
    fig.tight_layout()
    fig.savefig(fig_dir / "tradeoff_per_threshold.png", dpi=130)
    plt.close(fig)

    # --- NPS curves per threshold -------------------------------------
    fig, axes = plt.subplots(1, len(THRESHOLD_LABELS),
                             figsize=(4.6 * len(THRESHOLD_LABELS), 4.0), squeeze=False)
    for j, label in enumerate(THRESHOLD_LABELS):
        ax = axes[0][j]
        for vname in [v["name"] for v in cfg.variants]:
            m = res["own"].get(vname, {}).get(label)
            if m and "nps_f" in m:
                ax.plot(m["nps_f"], m["nps"], lw=1.1, label=vname)
        wf = res["wfbp"].get(f"T{j + 1}")
        if wf and "nps_f" in wf:
            ax.plot(wf["nps_f"], wf["nps"], "k--", lw=1.8, label="WFBP")
        ax.set_xlabel("spatial frequency (cyc/mm)")
        if j == 0:
            ax.set_ylabel("NPS (HU$^2$ mm$^2$)")
        ax.set_title(f"threshold {label}")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        if j == 0 and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    fig.suptitle("Noise power spectra -- magnitude AND texture "
                 "(a leftward shift means blotchier noise, i.e. denoising)")
    fig.tight_layout()
    fig.savefig(fig_dir / "nps_curves.png", dpi=130)
    plt.close(fig)

    # --- VMI as its own family: noise and texture vs keV --------------
    if res["vmi"]:
        kev, sd, fav, dp = [], [], [], []
        for lbl, m in res["vmi"].items():
            try:
                kev.append(float(lbl.replace("keV", "")))
            except ValueError:
                continue
            sd.append(m["noise_sd_hu"])
            fav.append(m["f_av"])
            dp.append(m.get("d_prime", np.nan))
        o = np.argsort(kev)
        kev = np.array(kev)[o]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ax[0].plot(kev, np.array(sd)[o], "-o")
        ax[0].set_xlabel("keV")
        ax[0].set_ylabel("noise SD (HU)")
        ax[0].set_title("VMI noise vs keV")
        ax[1].plot(kev, np.array(fav)[o], "-o", color="tab:green")
        ax[1].set_xlabel("keV")
        ax[1].set_ylabel("$f_{av}$ (cyc/mm)")
        ax[1].set_title("VMI noise texture vs keV")
        ax[2].plot(kev, np.array(dp)[o], "-o", color="tab:purple")
        ax[2].set_xlabel("keV")
        ax[2].set_ylabel("d'")
        ax[2].set_title(f"VMI d' ({cfg.task_diameter_mm:g} mm, "
                        f"{cfg.task_contrast_hu:g} HU task)")
        for a in ax:
            a.grid(alpha=0.3)
        fig.suptitle("Siemens VMI family -- reported separately: no channel of it "
                     "corresponds to a cumulative threshold image")
        fig.tight_layout()
        fig.savefig(fig_dir / "vmi_vs_kev.png", dpi=130)
        plt.close(fig)

    # --- Bland-Altman, own vs WFBP ------------------------------------
    bias = res.get("bias") or {}
    if bias:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for vname, per_thr in bias.items():
            xs = [p["mean_hu"] for d in per_thr.values() for p in d["roi_pairs"]]
            ys = [p["diff_hu"] for d in per_thr.values() for p in d["roi_pairs"]]
            ax.scatter(xs, ys, s=22, alpha=0.75, label=vname)
        allv = [p["diff_hu"] for per in bias.values() for d in per.values()
                for p in d["roi_pairs"]]
        if allv:
            mu, sg = float(np.mean(allv)), float(np.std(allv))
            ax.axhline(mu, color="k", lw=1.2, label=f"bias {mu:+.1f} HU")
            ax.axhline(mu + 1.96 * sg, color="k", ls="--", lw=1.0,
                       label=f"95 % LoA ±{1.96 * sg:.1f} HU")
            ax.axhline(mu - 1.96 * sg, color="k", ls="--", lw=1.0)
        ax.set_xlabel("mean of the two measurements (HU)")
        ax.set_ylabel("ours − WFBP (HU)")
        ax.set_title("Bland–Altman: ROI-mean HU agreement, our reconstruction vs WFBP")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "bland_altman_own_vs_wfbp.png", dpi=130)
        plt.close(fig)


def _print_summary(res):
    print("\n" + "=" * 96)
    print("RECONSTRUCTION COMPARISON -- summary")
    print("=" * 96)
    hdr = f"{'family / variant':24s} {'ch':>6s} {'noise SD':>10s} {'f_av':>8s} " \
          f"{'TTF50':>8s} {'edge90':>8s} {'d prime':>9s}"
    print(hdr)
    print("-" * 96)
    for vname, per in res.get("own", {}).items():
        for label, m in per.items():
            print(f"{'own/' + vname:24s} {label:>6s} {m['noise_sd_hu']:10.2f} "
                  f"{m['f_av']:8.3f} {m.get('ttf50', float('nan')):8.3f} "
                  f"{m.get('edge_width_1090_mm', float('nan')):8.3f} "
                  f"{m.get('d_prime', float('nan')):9.2f}")
    for kind in ("wfbp", "vmi"):
        for label, m in res.get(kind, {}).items():
            print(f"{'siemens/' + kind:24s} {label:>6s} {m['noise_sd_hu']:10.2f} "
                  f"{m['f_av']:8.3f} {m.get('ttf50', float('nan')):8.3f} "
                  f"{m.get('edge_width_1090_mm', float('nan')):8.3f} "
                  f"{m.get('d_prime', float('nan')):9.2f}")
    print("=" * 96)
    print("Read the trade-off figure, not this table alone: lower noise at lower TTF50 "
          "is\njust a smoother kernel, which is a move ALONG the curve, not a better "
          "reconstruction.")


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════
def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m reconstruction.recon_comparison",
        description="Compare our reconstruction against Siemens WFBP and VMI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", default="all", choices=["0", "1", "2", "all", "probe",
                                                      "sweep", "metrics"],
                   help="0/probe = read Siemens geometry + find slab (CPU); "
                        "1/sweep = reconstruct the variants (GPU); "
                        "2/metrics = measure everything (CPU)")
    p.add_argument("--wfbp-dir", help="folder of ONE Siemens WFBP threshold series set")
    p.add_argument("--vmi-dir", help="folder of ONE Siemens monoenergetic (VMI) series set")
    p.add_argument("--out-root", default=str(_REPO_ROOT / "output" / "research"
                                             / "recon_comparison"))
    p.add_argument("--data-path", default=_DEFAULT_DATA, help="raw sinogram .mat (HDF5)")
    p.add_argument("--desc-path", default=_DEFAULT_DESC, help="descriptor .mat (v5 struct)")
    p.add_argument("--geo-dir", default=str(_REPO_ROOT / "geometry"))
    p.add_argument("--slab",
                   help="override the auto-detected slab, 'z_lo,z_hi' in mm. "
                        "Write it as --slab=-1500,-1470 -- a value starting with "
                        "'-' is otherwise parsed as a flag")
    p.add_argument("--phantom-z", metavar="LO,HI",
                   help="axial extent of the phantom in mm (read off the vendor volume, "
                        "e.g. in Slicer). Restricts the insert-layer search to the "
                        "phantom. Strongly recommended for a clinical scan range. "
                        "Write it as --phantom-z=-1515.5,-1408.3 -- a value "
                        "starting with '-' is otherwise parsed as a flag")
    p.add_argument("--no-phantom-z", action="store_true",
                   help="ignore --phantom-z and search the whole scan range")
    p.add_argument("--slab-select", choices=["peak", "longest"], default="peak",
                   help="which candidate layer to take: peak = most insert-covered "
                        "slice (best when layers carry different insert counts)")
    p.add_argument("--expect-inserts", type=int,
                   help="expected number of inserts; warns when detection disagrees")
    p.add_argument("--max-slab-mm", type=float, default=40.0,
                   help="cap on the swept slab so the GPU stage stays affordable")
    p.add_argument("--n-pixels", type=int, help="override the matched matrix size")
    p.add_argument("--fov-mm", type=float, help="override the matched FOV")
    p.add_argument("--nps-patch-px", type=int, default=64,
                   help="requested NPS patch size; halved automatically until it fits "
                        "the phantom, and the chosen size is shared by all families")
    p.add_argument("--segmentation", metavar="FILE",
                   help="hand-drawn insert prior (Slicer .nrrd/.seg.nrrd). Positions "
                        "only need to be roughly right - every edge is refined from "
                        "the image, and a partial annotation is fine. Defaults to "
                        "<wfbp-dir>/Segmentation.nrrd if that exists")
    p.add_argument("--background-label", type=int, metavar="N",
                   help="label value in the segmentation marking a UNIFORM region for "
                        "the noise measurement. On an anthropomorphic phantom this is "
                        "close to required: the body outline contains lung, bone and "
                        "the table, none of which are noise")
    p.add_argument("--insert-mad-k", type=float, default=4.0,
                   help="insert-detection sensitivity in robust SDs; lower finds "
                        "fainter inserts. Check qc/roi_*.png after changing it")
    p.add_argument("--task-diameter-mm", type=float, default=5.0)
    p.add_argument("--task-contrast-hu", type=float, default=50.0)
    p.add_argument("--variants", help="comma-separated subset of variant names to run")
    p.add_argument("--list-series", metavar="DIR",
                   help="print every DICOM series under DIR with its geometry and exit "
                        "(use this to find the right --wfbp-dir/--vmi-dir)")
    p.add_argument("--force", action="store_true",
                   help="re-run sweep variants even if they are up to date")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _list_series(folder, cache_dir=None):
    idx = index_dicom_series(folder, cache_dir=cache_dir)
    print(f"\n{len(idx)} series under {folder}\n")
    print(f"{'#':>5s} {'files':>6s} {'thick':>7s} {'space':>7s} {'pixel':>7s} "
          f"{'matrix':>10s} {'kernel':<12s} {'class':<12s} description")
    print("-" * 118)
    for uid, s in sorted(idx.items(), key=lambda kv: str(kv[1].get("number"))):
        spec = _classify_series(s.get("desc", ""), s["files"][0][1] if s["files"] else "")
        cls = f"{spec[0]}:{spec[1]:g}" if spec else "-"
        print(f"{str(s['number']):>5s} {len(s['files']):>6d} "
              f"{(s['thickness_mm'] or 0):7.2f} {(s['spacing_between_mm'] or 0):7.2f} "
              f"{(s['pixel_mm'] or 0):7.4f} "
              f"{str(s['rows']) + 'x' + str(s['cols']):>10s} "
              f"{(s['kernel'] or '-'):<12s} {cls:<12s} {s['desc']}")
    print("\nPoint --wfbp-dir / --vmi-dir at a folder holding exactly ONE set: duplicate "
          "\nchannel labels across sets would otherwise be concatenated into one stack.")


def main(argv=None):
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    if a.list_series:
        Path(a.out_root).mkdir(parents=True, exist_ok=True)
        _list_series(a.list_series, cache_dir=a.out_root)
        return 0

    cfg = CompareConfig(
        out_root=a.out_root, wfbp_dir=a.wfbp_dir, vmi_dir=a.vmi_dir,
        data_path=a.data_path, desc_path=a.desc_path, geo_dir=a.geo_dir,
        n_pixels=a.n_pixels, fov_mm=a.fov_mm, max_slab_mm=a.max_slab_mm,
        nps_patch_px=a.nps_patch_px, insert_mad_k=a.insert_mad_k,
        segmentation=a.segmentation, background_label=a.background_label,
        task_diameter_mm=a.task_diameter_mm,
        task_contrast_hu=a.task_contrast_hu, force=a.force,
        slab_mm=tuple(float(x) for x in a.slab.split(",")) if a.slab else None,
        phantom_z_mm=(tuple(float(x) for x in a.phantom_z.split(","))
                      if a.phantom_z else None),
        use_phantom_z=not a.no_phantom_z,
        slab_select=a.slab_select, expect_inserts=a.expect_inserts,
    )
    if a.variants:
        want = {s.strip() for s in a.variants.split(",")}
        cfg.variants = [v for v in cfg.variants if v["name"] in want]
        if not cfg.variants:
            raise SystemExit(f"no variants matched {sorted(want)}; available: "
                             f"{[v['name'] for v in DEFAULT_VARIANTS]}")

    if not cfg.segmentation and cfg.wfbp_dir:
        for name in ("Segmentation.nrrd", "Segmentation.seg.nrrd"):
            cand = Path(cfg.wfbp_dir) / name
            if cand.exists():
                cfg.segmentation = str(cand)
                logger.info("using the segmentation found next to the WFBP data: %s",
                            cand)
                break
    if cfg.segmentation and not Path(cfg.segmentation).exists():
        raise SystemExit(f"--segmentation {cfg.segmentation} does not exist")

    Path(cfg.out_root).mkdir(parents=True, exist_ok=True)
    _write_json(Path(cfg.out_root) / "compare_config.json", cfg.to_dict())

    stage = {"probe": "0", "sweep": "1", "metrics": "2"}.get(a.stage, a.stage)
    if stage in ("0", "all"):
        stage0_probe(cfg)
    if stage in ("1", "all"):
        stage1_sweep(cfg)
    if stage in ("2", "all"):
        stage2_metrics(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
