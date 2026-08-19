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

    # Reconstruction settings held fixed across the sweep (only the swept knobs vary).
    geometry_model: str = "curved"
    z_weighting: str = "balanced"
    wavelet_ring_threshold: float = 2.0
    spike_mad_k: float = 5.0
    ipr_mad_k: float = 6.0
    patient_position: str = "HFS"

    variants: list = field(default_factory=lambda: [dict(v) for v in DEFAULT_VARIANTS])

    # Metric parameters
    nps_patch_px: int = 64
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
def _classify_series(desc, path):
    """
    Series -> ('mono', keV) | ('threshold', n) | None.

    Deliberately the same patterns as decomposition/material_decomposition.py so the
    two stages agree on what a 'WFBP T2' or a 'Mono 70 keV' series is.  Kept local
    rather than imported to avoid a reconstruction -> decomposition package dependency.
    """
    text = f"{desc or ''} {path or ''}"
    m = re.search(r"Mono[_ ]*([0-9]+(?:\.[0-9]+)?)\s*keV", text, re.IGNORECASE)
    if m:
        return ("mono", float(m.group(1)))
    m = (re.search(r"WFBP[_ ]*T([0-9]+)", text, re.IGNORECASE)
         or re.search(r"[_ /\\]T([0-9]+)[_ ]", text))
    if m:
        return ("threshold", int(m.group(1)))
    return None


def _is_dicom(path):
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def index_dicom_series(folder):
    """
    Index DICOM series under `folder`, keeping the acquisition metadata the comparison
    needs.  SliceThickness and PixelSpacing are what make the comparison fair, and
    ConvolutionKernel records what we are comparing against -- all three are standard
    tags, present even though the QIR strength is not exposed.

    Returns {uid: {number, desc, kernel, thickness_mm, spacing_between_mm, pixel_mm,
                   rows, cols, recon_diameter_mm, files: [[z, path], ...]}}
    """
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
        idx = index_dicom_series(folder)
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
    vol, img, zpos = read_series_volume(ref)
    slab = iq.find_insert_slab(vol, pixel_mm, zpos)
    z_lo = slab["z_lo_mm"] - cfg.slab_pad_mm
    z_hi = slab["z_hi_mm"] + cfg.slab_pad_mm
    if cfg.slab_mm is not None:
        z_lo, z_hi = sorted(cfg.slab_mm)
        logger.info("slab overridden by --slab: %.2f .. %.2f mm", z_lo, z_hi)
    elif (z_hi - z_lo) > cfg.max_slab_mm:                 # keep the GPU sweep affordable
        mid = 0.5 * (z_lo + z_hi)
        z_lo, z_hi = mid - cfg.max_slab_mm / 2, mid + cfg.max_slab_mm / 2
        logger.info("slab trimmed to %.1f mm about its centre", cfg.max_slab_mm)
    logger.info("insert slab: z = %.2f .. %.2f mm (%d slices in the vendor volume)",
                z_lo, z_hi, int(np.sum((zpos >= z_lo) & (zpos <= z_hi))))

    _qc_slab_figure(out / "qc" / "stage0_slab_detection.png", vol, zpos, slab, z_lo, z_hi,
                    pixel_mm, ref_kind, ref["label"])

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
    }
    _write_json(out / "match_config.json", match)
    logger.info("wrote %s", out / "match_config.json")
    logger.info("CHECK output/research/recon_comparison/qc/stage0_slab_detection.png "
                "before running stage 1 -- the marked band must contain the inserts.")
    del vol
    gc.collect()
    return match


def _qc_slab_figure(path, vol, zpos, slab, z_lo, z_hi, pixel_mm, kind, label):
    plt = _mpl()
    if plt is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].plot(zpos, slab["score"], lw=1.2)
    ax[0].axhline(slab["threshold"], color="grey", ls=":", label="threshold")
    ax[0].axvspan(z_lo, z_hi, color="tab:orange", alpha=0.25, label="selected slab")
    ax[0].set_xlabel("z (mm)")
    ax[0].set_ylabel("fraction of body voxels far from median")
    ax[0].set_title("insert-layer score")
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
def _measure_volume(vol, pixel_mm, rois, cfg: CompareConfig, body, patches):
    """
    Full metric set for one volume, given ROIs fixed by the family's reference.

    NPS is measured slice by slice -- never on a z-average, which would divide the
    noise by sqrt(Z) and report a quantity no single image has.  The TTF is measured on
    the z-average instead: averaging slices leaves in-plane resolution untouched (the
    slab lies inside one insert layer, so the cylinders are z-invariant across it) while
    cutting the noise that would otherwise corrupt the differentiated edge profile.
    """
    res = {}
    nps = iq.noise_power_spectrum(vol, patches, pixel_mm, patch_px=cfg.nps_patch_px)
    res["noise_sd_hu"] = nps["noise_sd_hu"]
    res["f_av"] = nps["f_av"]
    res["f_peak"] = nps["f_peak"]
    res["nps_f"] = nps["f"]
    res["nps"] = nps["nps"]
    res["n_patches"] = nps["n_patches"]

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
        f_neq, val = iq.neq(best["f"], best["ttf"], nps["f"], nps["nps"],
                            contrast_hu=abs(best["contrast_hu"]) or 1.0)
        res["neq_f"], res["neq"] = f_neq, val
        res.update(iq.detectability_index(best["f"], best["ttf"], nps["f"], nps["nps"],
                                          cfg.task_diameter_mm, cfg.task_contrast_hu))
    else:
        logger.warning("    no usable edge -- TTF/NEQ/d' unavailable for this volume")

    res["roi"] = iq.roi_statistics(vol, rois, body, pixel_mm)
    return res


def _family_rois(vol_ref, pixel_mm, cfg: CompareConfig, tag, qc_dir):
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
    inserts = iq.detect_inserts(flat, body, pixel_mm)
    patches = iq.background_patches(body, inserts, pixel_mm, patch_px=cfg.nps_patch_px)
    logger.info("  %s: %d inserts, %d NPS patches", tag, len(inserts), len(patches))
    if not inserts:
        logger.warning("  %s: no inserts detected -- TTF/NEQ/d' will be unavailable and "
                       "background patches are not insert-excluded. Check qc/roi_%s.png",
                       tag, tag)
    if not patches:
        raise ValueError(f"{tag}: no uniform background patches found -- try a smaller "
                         f"--nps-patch-px than {cfg.nps_patch_px}")
    _qc_roi_figure(qc_dir / f"roi_{tag}.png", flat, body, inserts, patches,
                   cfg.nps_patch_px, tag)
    return body, inserts, patches


def _qc_roi_figure(path, img, body, inserts, patches, patch_px, tag):
    plt = _mpl()
    if plt is None:
        return
    from matplotlib.patches import Circle, Rectangle
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap="gray", vmin=-200, vmax=300)
    ax.contour(body, levels=[0.5], colors="tab:blue", linewidths=0.8)
    for y0, x0 in patches:
        ax.add_patch(Rectangle((x0, y0), patch_px, patch_px, fill=False,
                               ec="tab:green", lw=0.6))
    for i, ins in enumerate(inserts):
        ax.add_patch(Circle((ins["cx"], ins["cy"]), ins["radius_px"], fill=False,
                            ec="tab:orange", lw=1.4))
        ax.text(ins["cx"], ins["cy"], str(i), color="tab:orange", ha="center",
                va="center", fontsize=9)
    ax.set_title(f"ROI placement -- {tag}\n"
                 f"blue = body, orange = inserts, green = NPS patches")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def stage2_metrics(cfg: CompareConfig):
    """Measure every family and write metrics + figures."""
    sitk = _sitk()
    out = Path(cfg.out_root)
    match = _load_match(out)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    qc = out / "qc"
    qc.mkdir(parents=True, exist_ok=True)

    zr = (match["z_lo_mm"], match["z_hi_mm"])
    results = {"match_config": match, "own": {}, "wfbp": {}, "vmi": {}}

    # ---- own: every swept variant ------------------------------------
    own_ctx = None
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
            vol = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)
            pixel_mm = match["pixel_mm"]
            if own_ctx is None:      # detect once on the first (highest-SNR) volume
                own_ctx = _family_rois(vol, pixel_mm, cfg, "own", qc)
            body, inserts, patches = own_ctx
            logger.info("own/%s thr %s", v["name"], label)
            results["own"][v["name"]][label] = _measure_volume(
                vol, pixel_mm, inserts, cfg, body, patches)
            del vol
            gc.collect()

    # ---- Siemens families --------------------------------------------
    siemens_vols = {}
    for kind, folder in (("wfbp", match.get("wfbp_dir")), ("vmi", match.get("vmi_dir"))):
        if not folder or kind not in match["families"]:
            continue
        chans = select_family(index_dicom_series(folder), kind)
        ctx = None
        for c in chans:
            vol, img, _z = read_series_volume(c, z_range=zr)
            pixel_mm = float(img.GetSpacing()[0])
            if ctx is None:
                ctx = _family_rois(vol, pixel_mm, cfg, kind, qc)
            body, inserts, patches = ctx
            logger.info("%s %s", kind, c["label"])
            results[kind][c["label"]] = _measure_volume(vol, pixel_mm, inserts, cfg,
                                                        body, patches)
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
    body, inserts, _patches = own_ctx
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
        if j == 0:
            ax.legend(fontsize=8)
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
            if m:
                ax.plot(m["nps_f"], m["nps"], lw=1.1, label=vname)
        wf = res["wfbp"].get(f"T{j + 1}")
        if wf:
            ax.plot(wf["nps_f"], wf["nps"], "k--", lw=1.8, label="WFBP")
        ax.set_xlabel("spatial frequency (cyc/mm)")
        if j == 0:
            ax.set_ylabel("NPS (HU$^2$ mm$^2$)")
        ax.set_title(f"threshold {label}")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        if j == 0:
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
    p.add_argument("--slab", help="override the auto-detected slab, 'z_lo,z_hi' in mm")
    p.add_argument("--max-slab-mm", type=float, default=40.0,
                   help="cap on the swept slab so the GPU stage stays affordable")
    p.add_argument("--n-pixels", type=int, help="override the matched matrix size")
    p.add_argument("--fov-mm", type=float, help="override the matched FOV")
    p.add_argument("--nps-patch-px", type=int, default=64)
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


def _list_series(folder):
    idx = index_dicom_series(folder)
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
        _list_series(a.list_series)
        return 0

    cfg = CompareConfig(
        out_root=a.out_root, wfbp_dir=a.wfbp_dir, vmi_dir=a.vmi_dir,
        data_path=a.data_path, desc_path=a.desc_path, geo_dir=a.geo_dir,
        n_pixels=a.n_pixels, fov_mm=a.fov_mm, max_slab_mm=a.max_slab_mm,
        nps_patch_px=a.nps_patch_px, task_diameter_mm=a.task_diameter_mm,
        task_contrast_hu=a.task_contrast_hu, force=a.force,
        slab_mm=tuple(float(x) for x in a.slab.split(",")) if a.slab else None,
    )
    if a.variants:
        want = {s.strip() for s in a.variants.split(",")}
        cfg.variants = [v for v in cfg.variants if v["name"] in want]
        if not cfg.variants:
            raise SystemExit(f"no variants matched {sorted(want)}; available: "
                             f"{[v['name'] for v in DEFAULT_VARIANTS]}")

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
