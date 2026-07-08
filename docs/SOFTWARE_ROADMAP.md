# Software Roadmap — toward a publishable GUI application

Status: **future goal, not started.** Captured now so entry points and module boundaries are
designed GUI-ready from the start and the eventual retrofit stays cheap. This is design-ahead only —
no existing code is refactored for it yet.

## Goal

Turn the reconstruction + material-decomposition pipeline into a **publishable, open-source software
application with a GUI** (spectral PCCT reconstruction + material decomposition). "Publishable" =
packaged, documented, citable, reproducible, and usable by others **without editing source**.

**Open and adaptable is a core requirement, not a nice-to-have.** Siemens keeps the raw photon counts
and the in-detector correction algorithms closed (we already established those stochastic corrections
are upstream and not reproducible from the processed sinograms). The contribution here is therefore an
**open, image-domain, easily adaptable** decomposition toolkit others can modify, extend, and build on:
permissive licence, everything data-driven, extension via registries/plugins (see *Openness &
extensibility* below).

**Two application modes:**
- **Clinical / standard mode** — pick a clinical question (a mode from the registry) → get the three
  material maps + a simple stability indicator. Guided and locked-down for routine use.
- **Advanced / research mode** — exposes the research pipeline directly: choose or compose experiments
  (bin-domain, threshold option, estimator, scaling), define custom material bases from the full
  material library, run side-by-side comparisons, and view/export the justification figures. This is
  the vehicle for the open-science goal — researchers drive the ablation machinery interactively, not
  just via batch scripts.

**Scope now:** apply this GUI-ready design to the **decomposition** module only. The reconstruction
retrofit is **parked until the decomposition stage is finished** and there is time to return to it.

## Design principles to hold now (cheap now, expensive later)

1. **Three clean layers.** *Core library* (pure algorithms — no I/O, no prints, no global state) →
   *entry-point API* (a few high-level functions) → *drivers* (CLI/batch now, GUI later). A GUI is
   just another driver over the same API. Reconstruction already splits library
   (`helical_reconstruction.py`) from driver (`python_reconstruction.py`) — keep it, and build
   decomposition the same way.
2. **Config as data, not module-level constants.** The reconstruction knobs currently live as
   top-of-file constants (`FAST_MODE`, `RECON_METHOD`, …). A GUI needs them as a **serializable config
   object** (dataclass ↔ JSON) that widgets populate. → Build the *decomposition* module
   **config-object-first** (it's new code, so free); retrofit reconstruction's knobs into a
   `ReconConfig` dataclass later (mechanical, low-risk).
3. **Stable entry-point functions** that both CLI and GUI call, e.g.
   - `reconstruct(config, progress=None, cancel=None) -> ReconResult`
   - `decompose(volumes, mode, config, progress=None) -> DecompResult`
4. **Return structured results; file-writing is a thin separate layer.** The GUI holds results in
   memory (display, re-decompose a different mode) without recomputing; the batch driver serializes
   them. Don't bury `save`/`print` inside compute functions.
5. **Progress + cancellation via callbacks.** Long ops (helical volume, per-voxel solve) take an
   optional `progress_callback(frac, msg)` and a cancel check → GUI progress bar + stop button, CLI
   logging. Purely additive to current signatures.
6. **No hardcoded paths.** Data path, geometry dir, output dir all from config. (geometry/output are
   already resolved via `__file__`; the `/data/Data2/...` raw-data path is still hardcoded in the
   driver → move to config during the retrofit.)
7. **Separate compute from visualization.** Return arrays + metadata (spacing, HU window,
   provenance); rendering lives in the driver/GUI/report layer, not the library.
8. **Registries as data** (already planned): the decomposition mode registry maps 1:1 to a GUI
   dropdown; the reconstruction method list similarly.

## Openness & extensibility (core requirement)

Design so an external researcher extends the tool **without editing core algorithms**:
- **Materials** — add one by adding a row to `decomposition/data/mu_rho_binavg.csv` (or dropping in
  element data); it appears in the picker automatically.
- **Clinical modes** — add one registry entry (`mode -> [3 materials]`).
- **Research approaches** — add a bin-domain method, estimator, or experiment as one function + one
  registry entry (the research harness is built around this registry, and is what the GUI's Advanced /
  Research mode surfaces).
- **Plugins (later)** — entry-point-based discovery so a pip-installable package can register new
  materials/modes/estimators without forking.
- Ship a permissive `LICENSE`, a `CONTRIBUTING` guide, and an extension how-to. Keep everything
  data-driven (coefficients in CSV, modes/approaches in registries) so adaptation never touches the math.

## Entry-point sketch (package name TBD, e.g. `pcct/`)

```
pcct.recon.reconstruct(config, progress, cancel) -> ReconResult
pcct.decomp.decompose(volumes, mode, config, progress) -> DecompResult
pcct.io      # load/save NIfTI & HDF5; config (de)serialize
pcct.viz     # slice / overlay / material-map rendering (shared by GUI and report figures)
CLI          # console_scripts entry points: pcct-recon, pcct-decompose
GUI          # thin layer over the API (see below)
```

## GUI technology (decision deferred)

Leading candidate: **napari** — an n-dimensional scientific image viewer (Qt-based), native for CT
volumes and multi-layer overlays (HU volume + material-map layers + ROI labels), with a plugin
architecture, and itself citable. Alternatives: **PyQt/PySide** directly (max control, more work) or a
**web app** (Streamlit/Dash) for a lighter, browser-based tool. Recommendation: prototype a thin
control panel over napari once the compute API is stable. Decide when the pipeline runs end-to-end.

## Packaging / publishability checklist (later)

- `pyproject.toml` + pinned environment; semantic version.
- Public API surfaced in `__init__`; docstrings; usage docs.
- `LICENSE` + `CITATION.cff` (citable software).
- Regression tests on the pure library (the reconstruction invariants + synthetic checks already
  model this style).
- Example/phantom walk-through dataset.

## Not now

No refactor of existing reconstruction code — **the reconstruction retrofit is parked until the
decomposition stage is finished** and there is time to return to it. These principles guide **new
decomposition code only** for now; the reconstruction retrofit (config object + entry-point wrapper +
callback progress) is a later, mechanical pass.
