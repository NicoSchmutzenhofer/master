"""
material_library.py -- material X-ray signatures for 4-bin PCCT decomposition.

Pure data-access library (no volume I/O, no global mutable state beyond a read cache).
Loads the bin-averaged mass attenuation coefficients <mu/rho> (cm^2/g) the professor
computed from NIST (K-edge-aware 140 kV integration) plus per-material density / K-edge
metadata. These are the columns of the material-signature matrix M in the forward model

        B = M x            (B: bin attenuations 1/cm, M: <mu/rho> cm^2/g, x: density g/cm^3)

Data lives in data/ as checked-in CSVs (no Excel dependency on the cluster):
  - mu_rho_binavg.csv : option, bin, e_lo_keV, e_hi_keV, <one column per material>
  - materials.csv     : material, density_g_cm3, k_edge_keV, category, description

Extensibility (open-source goal): add a material = one column in mu_rho_binavg.csv + one
row in materials.csv; it then appears in available_materials() and works in any mode.
See DECOMPOSITION_PLAN.md and docs/SOFTWARE_ROADMAP.md.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_DATA_DIR = Path(__file__).resolve().parent / "data"
_COEF_CSV = _DATA_DIR / "mu_rho_binavg.csv"
_MAT_CSV = _DATA_DIR / "materials.csv"

# Detector spectral range (keV) -- used to flag whether a K-edge is exploitable.
DETECTOR_RANGE_KEV: Tuple[float, float] = (20.0, 140.0)

_DESCRIPTOR_COLS = {"option", "bin", "e_lo_keV", "e_hi_keV"}


@dataclass(frozen=True)
class MaterialInfo:
    name: str
    density_g_cm3: float
    k_edge_keV: Optional[float]
    category: str
    description: str = ""

    @property
    def k_edge_in_range(self) -> bool:
        lo, hi = DETECTOR_RANGE_KEV
        return self.k_edge_keV is not None and lo <= self.k_edge_keV <= hi


@lru_cache(maxsize=1)
def _coef_rows() -> Tuple[dict, ...]:
    with open(_COEF_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No coefficient rows in {_COEF_CSV}")
    return tuple(rows)


@lru_cache(maxsize=1)
def load_materials() -> Dict[str, MaterialInfo]:
    mats: Dict[str, MaterialInfo] = {}
    with open(_MAT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            name = row["material"].strip()
            k = (row.get("k_edge_keV") or "").strip()
            mats[name] = MaterialInfo(
                name=name,
                density_g_cm3=float(row["density_g_cm3"]),
                k_edge_keV=float(k) if k else None,
                category=(row.get("category") or "").strip(),
                description=(row.get("description") or "").strip(),
            )
    return mats


def material_info(material: str) -> MaterialInfo:
    mats = load_materials()
    if material not in mats:
        raise KeyError(f"Unknown material '{material}'. Known: {sorted(mats)}")
    return mats[material]


def available_materials() -> List[str]:
    """Material names that have a signature column in mu_rho_binavg.csv."""
    return [c for c in _coef_rows()[0].keys() if c not in _DESCRIPTOR_COLS]


def available_options() -> List[int]:
    return sorted({int(r["option"]) for r in _coef_rows()})


def _option_rows(option: int) -> List[dict]:
    rows = [r for r in _coef_rows() if int(r["option"]) == option]
    if not rows:
        raise KeyError(f"Threshold option {option} not in {_COEF_CSV}. "
                       f"Available: {available_options()}")
    rows.sort(key=lambda r: int(r["bin"]))
    return rows


def bin_edges(option: int = 1) -> List[Tuple[float, float]]:
    """[(e_lo, e_hi), ...] keV for each energy bin of `option`, ordered by bin index."""
    return [(float(r["e_lo_keV"]), float(r["e_hi_keV"])) for r in _option_rows(option)]


def n_bins(option: int = 1) -> int:
    return len(_option_rows(option))


def material_signature(material: str, option: int = 1) -> np.ndarray:
    """<mu/rho> (cm^2/g) across the bins of `option`, ordered by bin index."""
    rows = _option_rows(option)
    if material not in rows[0]:
        raise KeyError(f"Unknown material '{material}'. Available: {available_materials()}")
    return np.array([float(r[material]) for r in rows], dtype=float)


def build_M(materials: Sequence[str], spec=1) -> np.ndarray:
    """
    Material-signature matrix M, shape (n_channels, n_materials).

    Columns are the materials' <mu/rho> (cm^2/g) in the given order -- the M of the forward
    model B = M x, with x = partial density (g/cm^3). `spec` selects the energy structure:
      - int  -> a threshold Option (back-compat): the bin-averaged windows of mu_rho_binavg.csv.
      - list of Channel -> arbitrary channels (threshold windows and/or monoenergetic VMI
        energies); each channel's signature is generated on demand (bin-avg lookup or mono
        interpolation), so the channel COUNT is whatever the data has -- not fixed at 4.
    """
    if len(materials) == 0:
        raise ValueError("Need at least one material to build M")
    if isinstance(spec, (int, np.integer)):
        return np.column_stack([material_signature(m, int(spec)) for m in materials])
    channels = list(spec)
    if not channels:
        raise ValueError("Need at least one channel to build M")
    return np.array([[signature_for_channel(m, ch) for m in materials] for ch in channels],
                    dtype=float)


def densities(materials: Sequence[str]) -> np.ndarray:
    """Bulk densities (g/cm^3) of the given materials, in order."""
    mats = load_materials()
    return np.array([mats[m].density_g_cm3 for m in materials], dtype=float)


# ============================================================================
# Channels -- arbitrary energy structure (threshold windows and/or VMI monoenergies)
# ============================================================================
# A channel describes one volume of an input stack.  The channel COUNT and kind come from the
# data (the loader), never a hardcoded 4: own recon / Siemens WFBP -> 'threshold' windows;
# Siemens VMI -> 'mono' energies.  build_M(materials, channels) generates the matching signature
# for each channel, or errors clearly if it cannot (defensive).
_MONO_CSV = _DATA_DIR / "mu_rho_mono.csv"


@dataclass(frozen=True)
class Channel:
    kind: str                                          # 'threshold' | 'mono'
    label: str = ""
    energy_keV: Optional[float] = None                 # mono
    window_keV: Optional[Tuple[float, float]] = None   # threshold (e_lo, e_hi)
    option: Optional[int] = None                       # threshold: source option in the bin-avg CSV
    bin_index: Optional[int] = None                    # threshold: 0-based bin index in that option


@lru_cache(maxsize=1)
def _mono_table() -> Dict[str, tuple]:
    """{material: ((energy_keV, mu/rho), ...) sorted by energy} from mu_rho_mono.csv."""
    tab: Dict[str, list] = {}
    with open(_MONO_CSV, newline="") as f:
        for row in csv.DictReader(f):
            tab.setdefault(row["material"].strip(), []).append(
                (float(row["energy_keV"]), float(row["mu_rho_cm2_g"])))
    for m in tab:
        tab[m].sort(key=lambda t: t[0])
    return {m: tuple(v) for m, v in tab.items()}


def has_mono(material: str) -> bool:
    return material in _mono_table()


def available_mono_materials() -> List[str]:
    return sorted(_mono_table())


def _mono_segments(rows: Sequence[Tuple[float, float]]) -> List[list]:
    """Split (E, mu/rho) rows into monotone segments at K-edges (duplicate energies)."""
    segs, start = [], 0
    for i in range(1, len(rows)):
        if abs(rows[i][0] - rows[i - 1][0]) < 1e-9:   # edge: rows[i-1]=below end, rows[i]=above start
            segs.append(list(rows[start:i]))
            start = i
    segs.append(list(rows[start:]))
    return segs


def mono_mu_rho(material: str, energy_keV: float) -> float:
    """
    Monoenergetic <mu/rho> (cm^2/g) at `energy_keV`, log-log interpolated and K-edge-AWARE:
    interpolation never crosses an absorption edge (the table keeps below/above doublets), and at
    an exact edge energy the post-edge value is used.  Raises if the material has no mono data
    (e.g. StainlessSteel/CoCr -- missing element) or the energy is outside the tabulated range.
    """
    tab = _mono_table()
    if material not in tab:
        raise KeyError(f"No monoenergetic mu/rho for '{material}' -- cannot generate a VMI "
                       f"signature. Available: {available_mono_materials()}")
    rows = tab[material]
    E = float(energy_keV)
    chosen = None
    for seg in _mono_segments(rows):                  # later (upper) segment wins at an edge boundary
        if seg[0][0] - 1e-9 <= E <= seg[-1][0] + 1e-9:
            chosen = seg
    if chosen is None:
        raise ValueError(f"energy {E} keV outside tabulated range "
                         f"[{rows[0][0]}, {rows[-1][0]}] keV for '{material}'")
    xs = np.log(np.array([t[0] for t in chosen]))
    ys = np.log(np.array([t[1] for t in chosen]))
    return float(np.exp(np.interp(np.log(E), xs, ys)))


def threshold_channels(option: int = 1) -> List[Channel]:
    """The threshold channels (cumulative-window bins) of a mu_rho_binavg.csv option."""
    return [Channel(kind="threshold", label=f"T{i+1}", window_keV=e, option=int(option), bin_index=i)
            for i, e in enumerate(bin_edges(option))]


def mono_channels(energies_keV: Sequence[float], labels: Optional[Sequence[str]] = None) -> List[Channel]:
    """Monoenergetic (VMI) channels at the given keV energies."""
    ekv = list(energies_keV)
    return [Channel(kind="mono", label=(labels[i] if labels else f"{e:g}keV"), energy_keV=float(e))
            for i, e in enumerate(ekv)]


def signature_for_channel(material: str, ch: Channel) -> float:
    """<mu/rho> (cm^2/g) of `material` for one channel, generated per the channel's kind."""
    if ch.kind == "threshold":
        return float(material_signature(material, ch.option)[ch.bin_index])
    if ch.kind == "mono":
        return mono_mu_rho(material, ch.energy_keV)
    raise ValueError(f"Unknown channel kind '{ch.kind}' (expected 'threshold' or 'mono')")


def channel_water_mu(ch: Channel) -> float:
    """Physical water linear attenuation (1/cm) for HU->mu of this channel (Water rho = 1)."""
    return float(signature_for_channel("Water", ch))
