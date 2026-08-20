"""
selftest_image_quality.py -- synthetic validation of image_quality_metrics.py.

No scan data and no GPU needed: every case has an analytically known answer, so this
runs anywhere in seconds and is the check to run before trusting a comparison run.

    python -m reconstruction.selftest_image_quality      (from the repo root)

What is actually pinned down here:
  * NPS normalisation      -- recovers the TRUE variance of the noise field to ~0.02 %,
                              including undoing the low-frequency loss from detrending
  * NPS shape              -- matches the exact discrete filter response of a known
                              blur (see the leakage caveat in noise_power_spectrum)
  * NPS texture ordering   -- smoothed noise has a lower f_av than white noise
  * TTF                    -- a disc blurred by a Gaussian of sigma has
                              TTF50 = sqrt(ln2 / (2 pi^2 sigma^2)) and a 10-90 % edge
                              width of 2.563*sigma; both are reproduced to <8 %
  * d'                     -- scales linearly with contrast and as 1/sqrt(NPS)
  * ROI / slab detection   -- finds the right number of inserts and the right z-range
  * thickness matching     -- averaging n native slices gives sd/sqrt(n)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reconstruction import image_quality_metrics as iq

PIX = 0.5           # mm/pixel for the synthetic images
_FAILS: list[str] = []


def _chk(name, got, want, rtol):
    ok = abs(got - want) <= rtol * max(abs(want), 1e-12)
    print(f"  {'OK  ' if ok else 'FAIL'} {name:44s} got={got:12.5f} want={want:12.5f}")
    if not ok:
        _FAILS.append(name)


def _chk_true(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name:44s} {detail}")
    if not cond:
        _FAILS.append(name)


def test_nps_normalisation(rng):
    print("\n[1] NPS normalisation (white noise)")
    sd = 20.0
    vol = rng.normal(0, sd, (40, 256, 256))
    patches = [(y, x) for y in range(32, 176, 48) for x in range(32, 176, 48)]
    r = iq.noise_power_spectrum(vol, patches, PIX)
    _chk("sigma recovered", r["noise_sd_hu"], sd, 0.02)
    _chk("int NPS == TRUE variance", r["variance_from_nps"], sd ** 2, 0.02)
    _chk("white NPS level = sd^2 dx^2", float(np.median(r["nps"])), sd ** 2 * PIX ** 2, 0.05)
    _chk_true("detrending loss is restored, not ignored",
              r["variance_measured"] < r["variance_from_nps"],
              f"detrended={r['variance_measured']:.2f} < corrected={r['variance_from_nps']:.2f}")
    return r


def test_nps_shape(rng, white):
    """
    Blur chosen so the spectrum falls ~76x across the band -- the realistic CT regime.

    The estimator's accuracy degrades as the spectrum steepens (spectral leakage from
    the finite patch): measured full-band error is ~4 % up to a 1000x fall and ~6 % at
    19000x.  Validating at a physically representative steepness is the point; a
    contrived 1e7x fall would only measure the leakage floor.
    """
    print("\n[2] NPS shape vs exact discrete blur")
    sd, sig_px = 20.0, 0.6
    # 60 slices x 25 patches = 1500 patches: enough that the estimate has converged onto
    # its systematic floor (~4 %) instead of fluctuating (a 360-patch sample scatters up
    # to 9 % between seeds and would make this check flaky).
    sm = ndimage.gaussian_filter(rng.normal(0, sd, (60, 256, 256)),
                                 sigma=(0, sig_px, sig_px), mode="wrap")
    patches = [(y, x) for y in range(32, 176, 32) for x in range(32, 176, 32)]
    r = iq.noise_power_spectrum(sm, patches, PIX)

    p = r["patch_px"]
    imp = np.zeros((p, p))
    imp[0, 0] = 1.0
    H = np.abs(np.fft.fft2(ndimage.gaussian_filter(imp, sig_px, mode="wrap")))
    fx = np.fft.fftfreq(p, d=PIX)
    _f, pred = iq.radial_average((sd ** 2 * PIX ** 2) * H ** 2, fx, fx)

    rel = float(np.max(np.abs(r["nps"] - pred)) / pred.max())
    _chk_true("NPS shape over the full band", rel < 0.06,
              f"max rel dev={rel:.4f}, spectrum falls {pred.max() / pred.min():.0f}x")
    _chk_true("smoothed noise is coarser than white",
              r["f_av"] < white["f_av"],
              f"f_av {r['f_av']:.4f} < {white['f_av']:.4f}")


def test_ttf():
    print("\n[3] TTF of a Gaussian-blurred disc")
    n = 512
    yy, xx = np.mgrid[:n, :n]
    cy = cx = n / 2 - 0.5
    r_mm = 12.0
    disc = ((((yy - cy) ** 2 + (xx - cx) ** 2) <= (r_mm / PIX) ** 2)
            .astype(np.float64) * 300.0)
    for sig_px in (1.5, 2.5):
        img = ndimage.gaussian_filter(disc, sigma=sig_px)
        t = iq.task_transfer_function(img, (cy, cx), r_mm, PIX)
        s = sig_px * PIX
        _chk(f"TTF50 (sigma={s:.2f} mm)", t["ttf50"],
             np.sqrt(np.log(2) / (2 * np.pi ** 2 * s ** 2)), 0.08)
        _chk(f"TTF10 (sigma={s:.2f} mm)", t["ttf10"],
             np.sqrt(np.log(10) / (2 * np.pi ** 2 * s ** 2)), 0.12)
        _chk(f"10-90 % edge width (sigma={s:.2f} mm)",
             t["edge_width_1090_mm"], 2.563 * s, 0.15)
        _chk(f"contrast (sigma={s:.2f} mm)", t["contrast_hu"], 300.0, 0.02)
    return disc, (cy, cx), r_mm


def test_detectability(white, disc, centre, r_mm):
    print("\n[4] NEQ and detectability index")
    t = iq.task_transfer_function(ndimage.gaussian_filter(disc, 2.0), centre, r_mm, PIX)
    f, nq = iq.neq(t["f"], t["ttf"], white["f"], white["nps"], contrast_hu=100.0)
    _chk_true("NEQ finite across the band", np.all(np.isfinite(nq)),
              f"{int(np.isfinite(nq).sum())}/{len(nq)} bins")
    d1 = iq.detectability_index(t["f"], t["ttf"], white["f"], white["nps"], 5.0, 25.0)["d_prime"]
    d2 = iq.detectability_index(t["f"], t["ttf"], white["f"], white["nps"], 5.0, 50.0)["d_prime"]
    d4 = iq.detectability_index(t["f"], t["ttf"], white["f"], white["nps"] * 4, 5.0, 50.0)["d_prime"]
    _chk("d' doubles with doubled contrast", d2 / d1, 2.0, 0.02)
    _chk("d' halves with 4x noise power", d4 / d2, 0.5, 0.02)


def test_roi_detection(rng):
    print("\n[5] automatic ROI detection")
    n = 512
    yy, xx = np.mgrid[:n, :n]
    cy = cx = n / 2 - 0.5
    phan = np.full((n, n), -1000.0)
    phan[((yy - cy) ** 2 + (xx - cx) ** 2) <= (100 / PIX) ** 2] = 0.0
    truth = [(-60, -60, 8, 300.0), (60, -60, 8, 150.0), (0, 70, 10, -80.0)]
    for dy, dx, rmm, hu in truth:
        m = ((yy - (cy + dy / PIX)) ** 2 + (xx - (cx + dx / PIX)) ** 2) <= (rmm / PIX) ** 2
        phan[m] = hu
    noisy = phan + rng.normal(0, 8.0, phan.shape)

    body = iq.detect_body_mask(noisy, PIX)
    ins = iq.detect_inserts(noisy, body, PIX)
    _chk_true("insert count", len(ins) == len(truth), f"found {len(ins)}, expected {len(truth)}")
    for d in ins:
        print(f"        r={d['radius_mm']:5.2f} mm  HU={d['mean_hu']:8.1f}  "
              f"circularity={d['circularity']:.2f}")
    pts = iq.background_patches(body, ins, PIX, patch_px=32)
    _chk_true("background patches found", len(pts) >= 4, f"{len(pts)} patches")
    stats = iq.roi_statistics(noisy[None], ins, body, PIX)
    _chk_true("CNR computed per insert",
              all(np.isfinite(i["cnr"]) for i in stats["inserts"]),
              " ".join(f"{i['cnr']:.1f}" for i in stats["inserts"]))


def test_slab_and_thickness(rng):
    print("\n[6] slab detection and slice-thickness matching")
    n = 256
    yy, xx = np.mgrid[:n, :n]
    body = ((yy - n / 2) ** 2 + (xx - n / 2) ** 2) <= (80 / PIX) ** 2
    nz, k0, k1 = 60, 20, 39
    vol = np.empty((nz, n, n), dtype=np.float32)
    for k in range(nz):
        sl = np.full((n, n), -1000.0, dtype=np.float32)
        sl[body] = 0.0
        if k0 <= k <= k1:
            m = ((yy - n / 2 - 30) ** 2 + (xx - n / 2) ** 2) <= (10 / PIX) ** 2
            sl[m] = 300.0
        vol[k] = sl + rng.normal(0, 5.0, sl.shape)
    sd = iq.find_insert_slab(vol, PIX, np.arange(nz) * 0.4)
    _chk_true("insert slab located",
              abs(sd["k_lo"] - k0) <= 2 and abs(sd["k_hi"] - k1) <= 2,
              f"detected k=[{sd['k_lo']},{sd['k_hi']}], truth [{k0},{k1}]")

    test_slab_search_restriction(rng)

    native = rng.normal(0, 30.0, (100, 32, 32)).astype(np.float32)
    z_nat = np.arange(100) * 0.4
    matched, used = iq.match_slice_thickness(native, z_nat, 2.0, np.arange(10, 30) * 0.4)
    _chk("thickness-matched noise = sd/sqrt(n)", float(matched.std()),
         30.0 / np.sqrt(np.mean(used)), 0.10)
    print(f"        averaged {np.mean(used):.1f} native slices per 2.0 mm slice")


def test_slab_search_restriction(rng):
    """
    Regression test for a real failure: on a long clinical scan range an off-phantom
    structure out-scored the phantom and put the slab ~665 mm away from it.

    Also checks that with several insert layers, select='peak' takes the one carrying
    the MOST inserts, including when that layer sits hard against the end of the
    phantom (the layer most at risk of being missed).
    """
    print("\n[7] slab search restricted to the phantom")
    pix = 500.0 / 512
    n = 256
    yy, xx = np.mgrid[:n, :n]
    cy = cx = n / 2 - 0.5
    body = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (100 / pix) ** 2
    z = np.arange(-2200, -1300, 4.0)
    vol = np.empty((len(z), n, n), dtype=np.float32)

    def ring(sl, count, r_ring, r_ins, hu):
        for j in range(count):
            a = 2 * np.pi * j / count
            m = (((yy - (cy + r_ring * np.sin(a) / pix)) ** 2
                  + (xx - (cx + r_ring * np.cos(a) / pix)) ** 2) <= (r_ins / pix) ** 2)
            sl[m] = hu

    phantom_z = (-1515.5, -1408.3)
    for k, zz in enumerate(z):
        sl = np.full((n, n), -1000.0, dtype=np.float32)
        if phantom_z[0] <= zz <= phantom_z[1]:
            sl[body] = 0.0
            if -1500 <= zz <= -1470:
                ring(sl, 10, 70, 7, 250.0)          # fewer inserts
            if -1425 <= zz <= -1410:
                ring(sl, 18, 70, 7, 250.0)          # most inserts, at the phantom end
        if -2140 <= zz <= -2110:                    # off-phantom decoy
            sl[body] = 0.0
            ring(sl, 24, 60, 11, 900.0)
        vol[k] = sl + rng.normal(0, 12.0, sl.shape)

    unrestricted = iq.find_insert_slab(vol, pix, z, select="peak")
    _chk_true("unrestricted search is captured by the decoy (the bug)",
              unrestricted["z_lo_mm"] < -2000,
              f"slab at {unrestricted['z_lo_mm']:+.0f}..{unrestricted['z_hi_mm']:+.0f} mm")

    restricted = iq.find_insert_slab(vol, pix, z, search_z_mm=phantom_z, select="peak")
    inside = (phantom_z[0] - 1 <= restricted["z_lo_mm"]
              and restricted["z_hi_mm"] <= phantom_z[1] + 1)
    _chk_true("search_z_mm confines the slab to the phantom", inside,
              f"slab at {restricted['z_lo_mm']:+.0f}..{restricted['z_hi_mm']:+.0f} mm")
    _chk_true("select='peak' takes the layer with the most inserts",
              restricted["z_lo_mm"] >= -1430,
              f"chose {restricted['z_lo_mm']:+.0f}..{restricted['z_hi_mm']:+.0f} mm "
              f"(18-insert layer at -1425..-1410, not the 10-insert one at -1500..-1470)")
    _chk_true("every candidate layer is reported",
              len(restricted["candidates"]) >= 2,
              f"{len(restricted['candidates'])} candidates")


def main() -> int:
    rng = np.random.default_rng(0)
    print("=" * 68)
    print("image_quality_metrics -- synthetic self-test")
    print("=" * 68)
    white = test_nps_normalisation(rng)
    test_nps_shape(rng, white)
    disc, centre, r_mm = test_ttf()
    test_detectability(white, disc, centre, r_mm)
    test_roi_detection(rng)
    test_slab_and_thickness(rng)
    print("\n" + "=" * 68)
    if _FAILS:
        print(f"FAILED ({len(_FAILS)}): " + ", ".join(_FAILS))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
