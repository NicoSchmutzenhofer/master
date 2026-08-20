"""
image_quality_metrics.py -- pure, no-reference image-quality metrics for CT volumes.

Library only: takes arrays, returns numbers/arrays.  No file I/O, no plotting, no
hardcoded paths, no __main__ (mirrors helical_reconstruction.py's role).  The driver
that feeds it volumes and writes results is recon_comparison.py.

The metric set answers three questions that must be answered TOGETHER, because each
one alone can be gamed by blurring:

    how much noise ?   -> noise_power_spectrum()   (sigma, and the NPS curve)
    how much detail ?  -> task_transfer_function() (TTF50/TTF10, 10-90% edge width)
    both at once ?     -> neq() / detectability_index()

Conventions (stated because they differ between papers -- keep them fixed so the
numbers stay comparable across the three image families):

  * NPS is two-sided, in HU^2 mm^2, normalised so that  int int NPS du dv = variance.
  * The radial NPS is the ring MEAN of the 2-D NPS, so the variance identity above
    holds on the 2-D array, not on a naive sum of the 1-D curve.
  * f_av (noise texture / "grain size") uses the common 1-D convention
        f_av = int f*NPS(f) df / int NPS(f) df
    on the radially averaged curve.  Low f_av = coarse, blotchy noise (iterative /
    denoised); high f_av = fine-grained noise (sharp analytic reconstruction).
  * TTF is normalised to 1 at f = 0, so it is a relative-resolution measure and is
    meaningful across images with different contrast scales.

All spatial frequencies are in cycles/mm (lp/mm).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal


# ═════════════════════════════════════════════════════════════════════
# Region detection -- automatic, so no ROI has to be drawn per scan
# ═════════════════════════════════════════════════════════════════════
def detect_body_mask(img_hu, pixel_mm, air_hu=-300.0, erode_mm=0.0):
    """
    Segment the phantom/patient body from air.

    Uses a fixed HU threshold rather than Otsu: these volumes are HU-calibrated, so
    -300 HU is a physically meaningful air/body boundary and is stable across
    thresholds and keV levels (Otsu drifts when the histogram shape changes between
    channels, which would silently move the ROIs between the images being compared).

    Returns a bool mask of the largest connected component, holes filled.
    """
    img = np.asarray(img_hu, dtype=np.float32)
    mask = img > air_hu
    if not mask.any():
        return mask
    lab, n = ndimage.label(mask)
    if n > 1:                                  # keep the largest blob (drop table, noise)
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        mask = lab == (int(np.argmax(sizes)) + 1)
    mask = ndimage.binary_fill_holes(mask)
    if erode_mm > 0:
        r = max(1, int(round(erode_mm / pixel_mm)))
        mask = ndimage.binary_erosion(mask, _disk(r))
    return mask


def _disk(radius):
    """Boolean disk structuring element of the given pixel radius."""
    r = int(max(1, radius))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (y * y + x * x) <= r * r


def detect_inserts(img_hu, body_mask, pixel_mm,
                   min_diameter_mm=4.0, max_diameter_mm=40.0,
                   contrast_mad_k=4.0, min_circularity=0.65):
    """
    Find the phantom's cylindrical inserts inside `body_mask`.

    An insert is a compact, roughly circular region whose HU differs from the modal
    body HU by more than contrast_mad_k robust standard deviations.  Detection is
    deliberately run on ONE high-contrast reference channel by the caller and the
    resulting geometry reused for every other channel of that family, so that any
    metric difference between channels comes from the images and not from the ROIs
    having moved.

    Returns a list of dicts: {cy, cx, radius_px, radius_mm, mean_hu, contrast_hu,
    circularity, area_px}, sorted by descending area.
    """
    img = np.asarray(img_hu, dtype=np.float32)
    if not body_mask.any():
        return []
    inside = img[body_mask]
    bg = float(np.median(inside))
    mad = float(np.median(np.abs(inside - bg))) * 1.4826
    if mad <= 0:
        mad = float(np.std(inside)) or 1.0

    # Smooth before thresholding so noise does not fragment the inserts.
    sm = ndimage.gaussian_filter(img, sigma=max(1.0, 0.6 / pixel_mm))
    cand = body_mask & (np.abs(sm - bg) > contrast_mad_k * mad)
    cand = ndimage.binary_opening(cand, _disk(max(1, int(round(0.6 / pixel_mm)))))

    lab, n = ndimage.label(cand)
    out = []
    for i in range(1, n + 1):
        comp = lab == i
        area = int(comp.sum())
        r_eq = np.sqrt(area / np.pi) * pixel_mm
        if not (min_diameter_mm / 2 <= r_eq <= max_diameter_mm / 2):
            continue
        # circularity 4*pi*A/P^2 -> 1.0 for a perfect disc
        per = float(np.sum(comp ^ ndimage.binary_erosion(comp)))
        circ = 4 * np.pi * area / (per * per) if per > 0 else 0.0
        if circ < min_circularity:
            continue
        cy, cx = ndimage.center_of_mass(comp)
        out.append({
            "cy": float(cy), "cx": float(cx),
            "radius_px": float(np.sqrt(area / np.pi)),
            "radius_mm": float(r_eq),
            "mean_hu": float(img[comp].mean()),
            "contrast_hu": float(img[comp].mean() - bg),
            "circularity": float(circ),
            "area_px": area,
        })
    out.sort(key=lambda d: -d["area_px"])
    return out


def background_patches(body_mask, inserts, pixel_mm, patch_px=64,
                       insert_margin_mm=6.0, edge_margin_mm=10.0, max_patches=64):
    """
    Tile the uniform background with square patches for the NPS.

    Patches must lie entirely inside the body, clear of every insert by
    insert_margin_mm and clear of the body edge by edge_margin_mm (beam hardening and
    the body boundary would otherwise leak structure into the noise estimate).

    Returns a list of (y0, x0) top-left origins.
    """
    ny, nx = body_mask.shape
    keep = ndimage.binary_erosion(body_mask, _disk(max(1, int(round(edge_margin_mm / pixel_mm)))))
    for ins in inserts:
        r = ins["radius_px"] + insert_margin_mm / pixel_mm
        y, x = np.ogrid[:ny, :nx]
        keep &= ((y - ins["cy"]) ** 2 + (x - ins["cx"]) ** 2) > r * r

    # Integral image -> a patch is valid iff every pixel under it is valid.
    ii = np.cumsum(np.cumsum(keep.astype(np.int32), axis=0), axis=1)
    p = int(patch_px)
    out = []
    step = max(p // 2, 8)                       # 50 % overlap: more patches, same area
    for y0 in range(0, ny - p + 1, step):
        for x0 in range(0, nx - p + 1, step):
            y1, x1 = y0 + p - 1, x0 + p - 1
            tot = ii[y1, x1]
            if y0 > 0:
                tot -= ii[y0 - 1, x1]
            if x0 > 0:
                tot -= ii[y1, x0 - 1]
            if y0 > 0 and x0 > 0:
                tot += ii[y0 - 1, x0 - 1]
            if tot == p * p:
                out.append((y0, x0))
    if len(out) > max_patches:                  # spread the sample over the whole region
        idx = np.linspace(0, len(out) - 1, max_patches).astype(int)
        out = [out[i] for i in idx]
    return out


def auto_background_patches(body_mask, inserts, pixel_mm, patch_px=64, min_patches=8,
                            min_patch_px=16, **kw):
    """
    background_patches() with automatic size reduction.

    The usable patch size is set by the phantom's geometry, not by preference: a vendor
    export reconstructed at a whole-body FOV puts a ~200 mm phantom on ~1 mm pixels, so
    a 64 px patch is 62 mm wide and no such square of clear background exists between
    the inserts.  Halve until enough patches fit rather than failing.

    Returns (patches, patch_px_used, tried) where `tried` is [(size, count), ...] for
    diagnostics.  Use the SAME patch_px_used for every image being compared -- the NPS
    frequency grid depends on it.
    """
    p = int(patch_px)
    tried = []
    while p >= int(min_patch_px):
        pts = background_patches(body_mask, inserts, pixel_mm, patch_px=p, **kw)
        tried.append((p, len(pts)))
        if len(pts) >= int(min_patches):
            return pts, p, tried
        p //= 2
    best = max(tried, key=lambda t: t[1]) if tried else (int(patch_px), 0)
    if best[1] > 0:
        return (background_patches(body_mask, inserts, pixel_mm, patch_px=best[0], **kw),
                best[0], tried)
    return [], int(patch_px), tried


# ═════════════════════════════════════════════════════════════════════
# Noise power spectrum
# ═════════════════════════════════════════════════════════════════════
def _poly_basis(ny, nx, order):
    """Design matrix of the 2-D polynomial basis used for patch detrending."""
    y, x = np.mgrid[0:ny, 0:nx]
    y = (y - ny / 2) / ny
    x = (x - nx / 2) / nx
    cols = [np.ones(ny * nx)]
    for i in range(1, order + 1):
        for j in range(i + 1):
            cols.append((y ** (i - j) * x ** j).ravel())
    return np.stack(cols, axis=1)


def _detrend_patch(patch, order=2):
    """Remove a low-order 2-D polynomial (residual cupping/shading) from a patch."""
    if order <= 0:
        return patch - patch.mean()
    ny, nx = patch.shape
    A = _poly_basis(ny, nx, order)
    coef, *_ = np.linalg.lstsq(A, patch.ravel(), rcond=None)
    return patch - (A @ coef).reshape(ny, nx)


def _detrend_nps_response(p, order):
    """
    Multiplicative attenuation that polynomial detrending imposes on the NPS, per bin.

    Detrending is the linear projection R = I - A(A^T A)^-1 A^T onto the complement of
    the polynomial basis A, so for a white input the expected periodogram at frequency
    u is  N - g(u)^T (A^T A)^-1 conj(g(u)),  where g(u) collects the DFT of each basis
    image at u.  The result depends only on (p, order) -- never on the data -- so it can
    be divided straight back out.

    This correction matters here rather than being a nicety: detrending removes ~48 % of
    the lowest-frequency bin at order 2, and the low-frequency end is exactly where
    iteratively reconstructed and denoised noise concentrates its power.  Leaving it in
    would bias NEQ and d' (whose integrands weight low frequencies heavily) and would
    distort f_av for precisely the images the comparison is about.
    """
    if order <= 0:
        att = np.ones((p, p))
        att[0, 0] = 0.0                       # mean subtraction kills DC exactly
        return att
    N = p * p
    A = _poly_basis(p, p, order)
    M_inv = np.linalg.inv(A.T @ A)
    G = np.stack([np.fft.fft2(A[:, j].reshape(p, p)) for j in range(A.shape[1])], axis=0)
    # quadratic form g^T M^-1 conj(g) evaluated at every frequency bin at once
    quad = np.einsum("iyx,ij,jyx->yx", G, M_inv, np.conj(G)).real
    return np.clip(1.0 - quad / N, 0.0, 1.0)


def noise_power_spectrum(volume, patches, pixel_mm, patch_px=64, slices=None,
                         detrend_order=2):
    """
    2-D NPS from uniform patches, averaged over patches and slices.

    volume   : (Z, Y, X) HU
    patches  : [(y0, x0), ...] from background_patches()
    patch_px : MUST be the same patch_px passed to background_patches().  The origins
               were validated for that size; reading a larger block from them reaches
               back over the inserts and the body edge and inflates the "noise" by a
               factor of several -- so a mismatch is rejected rather than inferred.
    slices   : iterable of z indices (default: all)

    Returns dict with the 2-D NPS (HU^2 mm^2), its frequency axes, the radially
    averaged curve, and summary numbers.  After the detrending correction the
    normalisation satisfies
        int int NPS du dv == variance of the underlying noise field
    (verified to ~0.02 % by selftest_image_quality.py), which is slightly ABOVE the
    variance actually measured on the detrended patches -- the difference is the
    low-frequency power detrending removed and the correction puts back.

    Known limitation -- spectral leakage: a finite patch convolves the true NPS with
    the Fejer kernel of the 64x64 window, so power leaks from strong bins into weak
    ones.  This is negligible while the NPS varies by one or two decades across the
    band (measured error ~5 %, the realistic CT case) but inflates the far tail of a
    very steeply falling spectrum.  Compare curves over the range within ~2 decades of
    the peak and do not read the extreme tail quantitatively.
    """
    vol = np.asarray(volume, dtype=np.float64)
    if vol.ndim == 2:
        vol = vol[None]
    if slices is None:
        slices = range(vol.shape[0])
    if not patches:
        raise ValueError("noise_power_spectrum: no background patches supplied")

    p = int(patch_px)
    ny, nx = vol.shape[1:]
    bad = [(y0, x0) for y0, x0 in patches if y0 + p > ny or x0 + p > nx]
    if bad:
        raise ValueError(
            f"noise_power_spectrum: {len(bad)} patch origins do not fit a {p}x{p} block "
            f"inside a {ny}x{nx} image -- patch_px must match the value given to "
            f"background_patches()")
    acc = np.zeros((p, p))
    n = 0
    var_acc = 0.0
    for z in slices:
        sl = vol[z]
        for (y0, x0) in patches:
            sub = sl[y0:y0 + p, x0:x0 + p]
            d = _detrend_patch(sub, detrend_order)
            var_acc += float(np.mean(d * d))
            acc += np.abs(np.fft.fft2(d)) ** 2
            n += 1
    if n == 0:
        raise ValueError("noise_power_spectrum: no usable patches")

    acc /= n
    dx = float(pixel_mm)
    nps2 = acc * (dx * dx) / (p * p)             # -> HU^2 mm^2, Parseval-consistent

    # Undo the (exactly known) low-frequency suppression caused by detrending.
    att = _detrend_nps_response(p, detrend_order)
    ok_att = att > 1e-3
    nps2 = np.where(ok_att, nps2 / np.where(ok_att, att, 1.0), 0.0)

    fx = np.fft.fftfreq(p, d=dx)
    f, nps_r = radial_average(nps2, fx, fx)

    df = 1.0 / (p * dx)
    variance_from_nps = float(nps2.sum() * df * df)
    return {
        "nps_2d": np.fft.fftshift(nps2),
        "f_axis": np.fft.fftshift(fx),
        "f": f,
        "nps": nps_r,
        "n_patches": n,
        "patch_px": p,
        "pixel_mm": dx,
        "variance_measured": var_acc / n,
        "variance_from_nps": variance_from_nps,
        "noise_sd_hu": float(np.sqrt(var_acc / n)),
        **nps_summary(f, nps_r),
    }


def radial_average(img2d, fx, fy, drop_dc=True):
    """
    Ring-mean of a 2-D spectrum -> (f, radial_mean).  f in the same units as fx.

    drop_dc excludes the exact zero-frequency term.  It must stay on for an NPS:
    detrending forces the DC bin to zero, and since no other FFT bin falls in the
    first radial ring, leaving it in produces a spurious ~0 as the first point of the
    curve -- which then corrupts f_av and anything normalised to nps[0].
    """
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    r = np.sqrt(FX ** 2 + FY ** 2).ravel()
    v = np.asarray(img2d, dtype=np.float64).ravel()
    if drop_dc:
        keep = r > 0
        r, v = r[keep], v[keep]
    d = np.abs(np.diff(np.sort(fx)))
    df = float(np.min(d[d > 0]))
    nb = int(np.ceil(r.max() / df)) + 1
    idx = np.clip((r / df).astype(int), 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb)
    tot = np.bincount(idx, weights=v, minlength=nb)
    ok = cnt > 0
    f = (np.arange(nb) + 0.5) * df
    return f[ok], tot[ok] / cnt[ok]


def nps_summary(f, nps):
    """sigma-free texture descriptors of an NPS curve: mean and peak frequency."""
    f = np.asarray(f, dtype=float)
    nps = np.asarray(nps, dtype=float)
    tot = np.trapezoid(nps, f) if hasattr(np, "trapezoid") else np.trapz(nps, f)
    if tot <= 0:
        return {"f_av": float("nan"), "f_peak": float("nan")}
    num = np.trapezoid(f * nps, f) if hasattr(np, "trapezoid") else np.trapz(f * nps, f)
    return {"f_av": float(num / tot), "f_peak": float(f[int(np.argmax(nps))])}


# ═════════════════════════════════════════════════════════════════════
# Task transfer function (circular-edge method)
# ═════════════════════════════════════════════════════════════════════
def radial_edge_profile(img_hu, center_yx, pixel_mm, r_max_mm, bins_per_pixel=4):
    """
    Oversampled edge-spread function around a cylindrical insert.

    Every pixel within r_max_mm contributes its own radial distance, so binning finely
    in r yields an ESF sampled far below the pixel pitch -- this is what makes the
    circular-edge method able to measure resolution beyond the Nyquist of the grid.

    Returns (r_mm, esf) with r_mm ascending.
    """
    img = np.asarray(img_hu, dtype=np.float64)
    cy, cx = center_yx
    ny, nx = img.shape
    r_max_px = r_max_mm / pixel_mm
    y0, y1 = max(0, int(cy - r_max_px - 2)), min(ny, int(cy + r_max_px + 3))
    x0, x1 = max(0, int(cx - r_max_px - 2)), min(nx, int(cx + r_max_px + 3))
    sub = img[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).ravel() * pixel_mm
    v = sub.ravel()
    keep = r <= r_max_mm
    r, v = r[keep], v[keep]

    dr = pixel_mm / float(bins_per_pixel)
    nb = int(np.ceil(r_max_mm / dr))
    idx = np.clip((r / dr).astype(int), 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb)
    tot = np.bincount(idx, weights=v, minlength=nb)
    ok = cnt > 0
    return (np.arange(nb)[ok] + 0.5) * dr, tot[ok] / cnt[ok]


def task_transfer_function(img_hu, center_yx, radius_mm, pixel_mm,
                           bins_per_pixel=4, smooth_bins=5):
    """
    TTF from one cylindrical insert, via the circular-edge method (AAPM TG-233).

    ESF -> (smooth) -> differentiate -> LSF -> |FFT| -> normalise to 1 at DC.

    Because TTF is normalised at DC it is a *relative* resolution measure, so it can
    be compared between images with completely different contrast scales -- which is
    what lets the same routine serve threshold images and monoenergetic images.

    Returns dict: f (cyc/mm), ttf, ttf50, ttf10, edge_width_1090_mm, contrast_hu.
    """
    r_max_mm = radius_mm * 2.2
    r, esf = radial_edge_profile(img_hu, center_yx, pixel_mm, r_max_mm, bins_per_pixel)
    if r.size < 16:
        raise ValueError("task_transfer_function: edge profile too short")

    inner = esf[r < radius_mm * 0.6]
    outer = esf[r > radius_mm * 1.5]
    if inner.size < 3 or outer.size < 3:
        raise ValueError("task_transfer_function: insert too small or too close to another")
    lo, hi = float(np.mean(outer)), float(np.mean(inner))
    contrast = hi - lo

    if smooth_bins and smooth_bins >= 3 and r.size > smooth_bins:
        w = int(smooth_bins) | 1
        esf_s = signal.savgol_filter(esf, w, 2)
    else:
        esf_s = esf

    dr = float(np.mean(np.diff(r)))
    lsf = np.gradient(esf_s, dr)
    lsf = np.abs(lsf)
    if lsf.max() <= 0:
        raise ValueError("task_transfer_function: flat edge (no contrast)")

    # Hann window centred on the LSF peak: suppresses the noisy tails without
    # touching the transition itself (untapered tails alias into the TTF).
    k = int(np.argmax(lsf))
    half = max(8, int(round((radius_mm * 0.8) / dr)))
    a, b = max(0, k - half), min(len(lsf), k + half + 1)
    seg = lsf[a:b] * np.hanning(b - a)

    n = int(2 ** np.ceil(np.log2(max(64, seg.size * 4))))     # zero-pad for a smooth curve
    F = np.abs(np.fft.rfft(seg, n))
    if F[0] <= 0:
        raise ValueError("task_transfer_function: zero DC")
    ttf = F / F[0]
    f = np.fft.rfftfreq(n, d=dr)

    # 10-90 % edge width straight off the normalised ESF: the intuitive companion to
    # TTF50, and it needs no Fourier transform to defend in the text.
    prof = (esf_s - lo) / contrast if abs(contrast) > 1e-9 else esf_s * 0
    if contrast < 0:
        prof = 1.0 - prof
    width = _crossing(r, prof, 0.10) - _crossing(r, prof, 0.90)

    return {
        "f": f, "ttf": ttf,
        "ttf50": _ttf_at(f, ttf, 0.50),
        "ttf10": _ttf_at(f, ttf, 0.10),
        "edge_width_1090_mm": float(abs(width)) if np.isfinite(width) else float("nan"),
        "contrast_hu": float(contrast),
        "r_mm": r, "esf": esf_s,
    }


def _crossing(x, y, level):
    """First x where y crosses `level`, linearly interpolated (y assumed monotone-ish)."""
    y = np.asarray(y, dtype=float)
    for i in range(len(y) - 1):
        if (y[i] - level) * (y[i + 1] - level) <= 0 and y[i] != y[i + 1]:
            t = (level - y[i]) / (y[i + 1] - y[i])
            return float(x[i] + t * (x[i + 1] - x[i]))
    return float("nan")


def _ttf_at(f, ttf, level):
    """Frequency (cyc/mm) at which the TTF first falls to `level`."""
    for i in range(len(ttf) - 1):
        if ttf[i] >= level >= ttf[i + 1]:
            d = ttf[i] - ttf[i + 1]
            t = 0.0 if d == 0 else (ttf[i] - level) / d
            return float(f[i] + t * (f[i + 1] - f[i]))
    return float("nan")


# ═════════════════════════════════════════════════════════════════════
# Combined: NEQ and detectability index
# ═════════════════════════════════════════════════════════════════════
def neq(f_ttf, ttf, f_nps, nps, contrast_hu=1.0):
    """
    Noise-equivalent quanta: NEQ(f) = contrast^2 * TTF(f)^2 / NPS(f).

    Literally "resolution squared over noise", frequency by frequency -- the
    principled way to combine the two without assuming what is being looked for.
    Higher = more usable information at that spatial scale.  Units: mm^-2 when NPS
    is in HU^2 mm^2 and contrast in HU.

    Returns (f, neq) on the NPS frequency grid (the coarser of the two).
    """
    f = np.asarray(f_nps, dtype=float)
    t = np.interp(f, np.asarray(f_ttf, dtype=float), np.asarray(ttf, dtype=float))
    n = np.asarray(nps, dtype=float)
    out = np.full(f.shape, np.nan)
    ok = n > 0
    out[ok] = (contrast_hu ** 2) * (t[ok] ** 2) / n[ok]
    return f, out


def detectability_index(f_ttf, ttf, f_nps, nps,
                        task_diameter_mm=5.0, task_contrast_hu=50.0):
    """
    Ideal-observer detectability index d' for a flat circular lesion.

        d'^2 = 2*pi * int f * |W(f)|^2 * TTF(f)^2 / NPS(f) df

    with W(f) the 2-D Fourier transform of a disc of the stated diameter and contrast
    (a jinc).  One number per image, but it depends on the assumed task, so the task
    parameters are returned alongside and must be quoted with the value.
    """
    from scipy.special import j1

    f = np.asarray(f_nps, dtype=float)
    t = np.interp(f, np.asarray(f_ttf, dtype=float), np.asarray(ttf, dtype=float))
    n = np.asarray(nps, dtype=float)

    R = 0.5 * float(task_diameter_mm)
    a = 2 * np.pi * R * f
    W = np.where(a > 1e-9, task_contrast_hu * np.pi * R * R * 2 * j1(np.maximum(a, 1e-9)) / np.maximum(a, 1e-9),
                 task_contrast_hu * np.pi * R * R)
    integ = np.zeros_like(f)
    ok = n > 0
    integ[ok] = f[ok] * (W[ok] ** 2) * (t[ok] ** 2) / n[ok]
    d2 = 2 * np.pi * (np.trapezoid(integ, f) if hasattr(np, "trapezoid") else np.trapz(integ, f))
    return {
        "d_prime": float(np.sqrt(max(d2, 0.0))),
        "task_diameter_mm": float(task_diameter_mm),
        "task_contrast_hu": float(task_contrast_hu),
    }


# ═════════════════════════════════════════════════════════════════════
# Paired comparison (own vs WFBP) -- separate bias from independent noise
# ═════════════════════════════════════════════════════════════════════
def difference_analysis(vol_a, vol_b, pixel_mm, body_mask=None, smooth_mm=5.0):
    """
    Split (a - b) into a systematic and a noise part.

    A plain RMSE between two independent reconstructions of the same object is
    dominated by the fact that their noise is independent: for unbiased a and b it
    tends to sqrt(sigma_a^2 + sigma_b^2) even when both are perfect.  Low-pass
    filtering the difference averages that independent noise away and leaves the part
    that actually matters -- HU offsets, cupping, shading, geometric distortion.

    Returns dict with the systematic RMS/mean/max (HU), the total RMSE for reference,
    and the smoothed difference map for plotting.
    """
    a = np.asarray(vol_a, dtype=np.float32)
    b = np.asarray(vol_b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"difference_analysis: shape mismatch {a.shape} vs {b.shape}")
    d = a - b
    sig = ndimage.gaussian_filter(d, sigma=(0, smooth_mm / pixel_mm, smooth_mm / pixel_mm))
    m = np.ones(a.shape, dtype=bool) if body_mask is None else np.broadcast_to(body_mask, a.shape)
    return {
        "rmse_total_hu": float(np.sqrt(np.mean(d[m] ** 2))),
        "systematic_rms_hu": float(np.sqrt(np.mean(sig[m] ** 2))),
        "systematic_mean_hu": float(np.mean(sig[m])),
        "systematic_max_abs_hu": float(np.max(np.abs(sig[m]))),
        "noise_rms_hu": float(np.sqrt(np.mean((d - sig)[m] ** 2))),
        "systematic_map": sig,
    }


def roi_statistics(volume, inserts, body_mask, pixel_mm, shrink=0.7):
    """
    Mean/SD in each insert (shrunk to avoid the partial-volume rim) and in the
    background -- the inputs to a Bland-Altman comparison and to CNR.
    """
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim == 2:
        vol = vol[None]
    ny, nx = vol.shape[1:]
    y, x = np.ogrid[:ny, :nx]
    bg = body_mask.copy()
    out = {"inserts": []}
    for i, ins in enumerate(inserts):
        rr = ins["radius_px"] * shrink
        m = ((y - ins["cy"]) ** 2 + (x - ins["cx"]) ** 2) <= rr * rr
        bg &= ((y - ins["cy"]) ** 2 + (x - ins["cx"]) ** 2) > (ins["radius_px"] + 6.0 / pixel_mm) ** 2
        v = vol[:, m]
        out["inserts"].append({
            "index": i, "cy": ins["cy"], "cx": ins["cx"], "radius_mm": ins["radius_mm"],
            "mean_hu": float(v.mean()), "sd_hu": float(v.std()), "n_vox": int(v.size),
        })
    vb = vol[:, bg]
    out["background"] = {"mean_hu": float(vb.mean()), "sd_hu": float(vb.std()),
                         "n_vox": int(vb.size)}
    for ins in out["inserts"]:
        ins["cnr"] = ((ins["mean_hu"] - out["background"]["mean_hu"])
                      / out["background"]["sd_hu"]) if out["background"]["sd_hu"] > 0 else float("nan")
    return out


# ═════════════════════════════════════════════════════════════════════
# Slab / structure detection
# ═════════════════════════════════════════════════════════════════════
def find_insert_slab(volume, pixel_mm, z_positions_mm, air_hu=-300.0, smooth_mm=1.0,
                     search_z_mm=None, min_run=3, select="peak"):
    """
    Locate the z-range containing a phantom insert layer.

    The insert layer is a run of slices with strong in-plane structure inside the body;
    homogeneous layers have none.  Structure is scored as the FRACTION of body voxels
    lying far from the body median, after light smoothing so that quantum noise -- the
    very thing that differs between the channels -- does not drive the detection.  A
    spread statistic such as the MAD does not work here: inserts occupy only a few
    percent of the body area, and MAD is designed precisely to discard a small minority
    of deviant voxels, so it stays flat straight through the insert layer.

    search_z_mm : (lo, hi) restricting where a slab may be chosen.  **Supply this for a
        long clinical scan range.**  A Thx-Abdomen acquisition contains the table, the
        positioning aids and scan-end artefacts, any of which can out-score the phantom
        and put the slab hundreds of millimetres away from it.  Slices outside the range
        are not even scored, which is also much faster.

    select : 'peak'    -- the run containing the single most insert-covered slice.
                          Best when layers differ in how many inserts they carry, since
                          the score is essentially the insert area fraction.
             'longest' -- the longest run above threshold (the previous behaviour).

    Returns dict with the chosen k_lo/k_hi/z_lo_mm/z_hi_mm, the full per-slice `score`,
    and **`candidates`**: every run found, so a layer that was not selected is still
    visible in the log and the QC figure rather than being silently discarded.
    """
    vol = np.asarray(volume, dtype=np.float32)
    nz = vol.shape[0]
    z = np.asarray(z_positions_mm, dtype=float)
    score = np.zeros(nz)

    if search_z_mm is not None:
        lo, hi = sorted(float(v) for v in search_z_mm)
        in_range = (z >= lo) & (z <= hi)
        if not in_range.any():
            raise ValueError(
                f"find_insert_slab: search range {lo:.1f}..{hi:.1f} mm contains no "
                f"slices (volume spans {z.min():.1f}..{z.max():.1f} mm)")
    else:
        in_range = np.ones(nz, dtype=bool)

    sigma_px = max(0.5, smooth_mm / pixel_mm)
    # The body edge must be eroded away before scoring: smoothing drags the -1000 HU
    # air across the boundary, producing a rim of extreme values in EVERY slice, which
    # would otherwise score as "structure" and flatten the curve completely.
    erode_px = int(np.ceil(3 * sigma_px))
    for k in range(nz):
        if not in_range[k]:
            continue
        sl = vol[k]
        body = sl > air_hu
        if body.sum() < 100:
            continue
        body = ndimage.binary_erosion(body, _disk(erode_px))
        if body.sum() < 100:
            continue
        sm = ndimage.gaussian_filter(sl, sigma=sigma_px)
        v = sm[body]
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med))) * 1.4826
        if mad <= 0:
            continue
        score[k] = float(np.mean(np.abs(v - med) > 5.0 * mad))

    if not np.any(score > 0):
        raise ValueError("find_insert_slab: no structure found in the searched range")

    # Bridge single-slice dropouts before thresholding at a fraction of the peak.
    score_s = ndimage.median_filter(score, size=3, mode="nearest")
    score_s[~in_range] = 0.0
    thr = 0.25 * float(score_s.max())
    above = score_s >= thr

    runs = []
    cur = None
    for k in range(nz + 1):
        hot = bool(above[k]) if k < nz else False
        if hot and cur is None:
            cur = k
        elif not hot and cur is not None:
            a, b = cur, k - 1
            if b - a + 1 >= int(min_run):
                runs.append({
                    "k_lo": int(a), "k_hi": int(b), "n_slices": int(b - a + 1),
                    "z_lo_mm": float(min(z[a], z[b])), "z_hi_mm": float(max(z[a], z[b])),
                    "peak_score": float(score_s[a:b + 1].max()),
                    "mean_score": float(score_s[a:b + 1].mean()),
                })
            cur = None
    if not runs:                       # nothing survived min_run -- fall back to the peak
        k = int(np.argmax(score_s))
        runs = [{"k_lo": k, "k_hi": k, "n_slices": 1,
                 "z_lo_mm": float(z[k]), "z_hi_mm": float(z[k]),
                 "peak_score": float(score_s[k]), "mean_score": float(score_s[k])}]

    key = (lambda r: r["n_slices"]) if select == "longest" else (lambda r: r["peak_score"])
    best = max(runs, key=key)
    return {"k_lo": best["k_lo"], "k_hi": best["k_hi"],
            "z_lo_mm": best["z_lo_mm"], "z_hi_mm": best["z_hi_mm"],
            "score": score, "threshold": float(thr),
            "candidates": sorted(runs, key=key, reverse=True),
            "select": select}


def match_slice_thickness(vol_native, z_native_mm, target_thickness_mm, target_z_mm):
    """
    Average native slices into thicker ones, reproducing a target slice profile.

    Averaging adjacent slices is physically what a thicker slice IS, so this matches
    the vendor's slice sensitivity profile rather than approximating it with a
    Gaussian.  Doing it this way also avoids the sqrt(N) noise shortcut, which is
    wrong here because SSR slices at neighbouring z share detector rows and are
    therefore correlated.

    Returns (vol_matched, n_used_per_slice).
    """
    zn = np.asarray(z_native_mm, dtype=float)
    out = np.zeros((len(target_z_mm), *vol_native.shape[1:]), dtype=np.float32)
    used = []
    half = 0.5 * float(target_thickness_mm)
    for i, zc in enumerate(target_z_mm):
        sel = np.where(np.abs(zn - zc) <= half)[0]
        if sel.size == 0:                      # thinner target than native spacing
            sel = np.array([int(np.argmin(np.abs(zn - zc)))])
        out[i] = vol_native[sel].mean(axis=0)
        used.append(int(sel.size))
    return out, used
