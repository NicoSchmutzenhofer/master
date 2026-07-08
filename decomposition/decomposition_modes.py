"""
decomposition_modes.py -- the clinical-mode registry.

Each clinical question maps to a set of (3) basis materials -> a specific M matrix.
Selecting a mode is how the professor's mock-up "dropdown" chooses what to decompose
into. Adding a mode = one register_mode() call (extensibility for the open-source goal).

Material names must exist in material_library (see available_materials()).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModeSpec:
    key: str
    display_name: str
    materials: List[str]                 # basis materials (columns of M)
    clinical_question: str = ""
    tissue_base: Optional[str] = None    # the ~soft-tissue reference column (calib/notes)
    notes: str = ""


MODES: Dict[str, ModeSpec] = {}


def register_mode(spec: ModeSpec) -> None:
    MODES[spec.key] = spec


def get_mode(key: str) -> ModeSpec:
    if key not in MODES:
        raise KeyError(f"Unknown mode '{key}'. Available: {list(MODES)}")
    return MODES[key]


def list_modes() -> List[str]:
    return list(MODES)


# --- Test / validation mode --------------------------------------------------
# QRM Dual-Energy Phantom V5, middle layer: soft-tissue background with matched
# calcium + iodine inserts. Best-conditioned (iodine K-edge in bin 1) and directly
# verifiable against known inserts -> the Phase-A test mode.
register_mode(ModeSpec(
    "phantom_ca_i", "Phantom Ca/I (test)",
    ["SoftTissue", "HA", "Iodine"],
    clinical_question="Validation: separate matched calcium & iodine inserts",
    tissue_base="SoftTissue",
    notes="QRM Dual-Energy Phantom V5 middle layer. Phase-A test mode (kappa~166).",
))

# --- The 8 clinical modes from the professor's mock-up (non-contrast) ---------
register_mode(ModeSpec(
    "liver", "Liver (iron + steatosis)", ["Water", "Fat", "Iron"],
    "Hepatic iron overload + fat fraction", tissue_base="Water",
    notes="Ill-conditioned (kappa~1.3e4): Water/Fat near-collinear -> needs Phase-B remedies."))
register_mode(ModeSpec(
    "bone", "Bone (mineral density)", ["SoftTissue", "Fat", "HA"],
    "Bone mineral density", tissue_base="SoftTissue",
    notes="kappa~7.7e3 (SoftTissue/Fat collinear)."))
register_mode(ModeSpec(
    "vessels", "Vessels (calcified plaque)", ["SoftTissue", "Fat", "HA"],
    "Calcified atherosclerotic plaque", tissue_base="SoftTissue"))
register_mode(ModeSpec(
    "soft_tissue_tumour", "Soft tissue (tumours)", ["SoftTissue", "Fat", "HA"],
    "Soft-tissue tumour characterisation", tissue_base="SoftTissue"))
register_mode(ModeSpec(
    "bone_marrow", "Bone marrow (metastases)", ["Fat", "SoftTissue", "HA"],
    "Marrow metastasis / oedema", tissue_base="SoftTissue"))
register_mode(ModeSpec(
    "kidney_stone", "Kidney (stone composition)", ["Water", "Fat", "HA"],
    "Renal stone composition", tissue_base="Water", notes="kappa~6.3e3."))
register_mode(ModeSpec(
    "gout", "Joints (gout)", ["SoftTissue", "HA", "Iron"],
    "Gout vs calcification", tissue_base="SoftTissue",
    notes="Iron per mock-up (placeholder for a dense/urate-distinct 3rd basis); revisit."))
register_mode(ModeSpec(
    "peri_implant", "Peri-implant (metal artefact)", ["SoftTissue", "HA", "Titanium"],
    "Peri-implant / metal-artefact reduction", tissue_base="SoftTissue",
    notes="Implant metal configurable (Titanium default; CoCr / Tantalum available)."))

# --- Whiteboard worked example (with contrast) -------------------------------
register_mode(ModeSpec(
    "anaemia", "Anaemia (whiteboard example)", ["SoftTissue", "Iodine", "Iron"],
    "Blood/iron with iodine contrast", tissue_base="SoftTissue",
    notes="Best-conditioned tissue+contrast basis (kappa~166)."))
