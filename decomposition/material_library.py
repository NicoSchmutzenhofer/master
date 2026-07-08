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


def build_M(materials: Sequence[str], option: int = 1) -> np.ndarray:
    """
    Material-signature matrix M, shape (n_bins, n_materials).

    Columns are the materials' bin-averaged <mu/rho> (cm^2/g) in the given order --
    the M of the forward model B = M x, with x = partial density (g/cm^3).
    """
    if len(materials) == 0:
        raise ValueError("Need at least one material to build M")
    return np.column_stack([material_signature(m, option) for m in materials])


def densities(materials: Sequence[str]) -> np.ndarray:
    """Bulk densities (g/cm^3) of the given materials, in order."""
    mats = load_materials()
    return np.array([mats[m].density_g_cm3 for m in materials], dtype=float)
