#!/bin/bash
#SBATCH --nodelist=sauron
#SBATCH --job-name=ReconComparison
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ============================================================================
# Reconstruction-stage image-quality comparison:
#   our reconstruction  vs  Siemens WFBP  vs  Siemens VMI
#
# Runs all three stages in order.  Stage 1 (the GPU sweep) SKIPS any variant
# that already exists with a matching configuration, so re-running this script
# after a crash, or to redo only the metrics, costs nothing.
#
#   sbatch batch_compare.sh                       # all stages, paths from below
#   sbatch batch_compare.sh --stage 2             # metrics only (CPU, seconds)
#   sbatch batch_compare.sh --force               # redo the sweep from scratch
#   sbatch batch_compare.sh --slab 12,44          # override the detected slab
#
# Any extra argument is passed straight through to recon_comparison.py, so
# `--help` there is the authoritative list.
#
# IMPORTANT -- point WFBP_DIR / VMI_DIR at the folder holding exactly ONE
# reconstruction each.  A parent directory containing several WFBP sets would
# be walked recursively and the duplicate channels concatenated into one
# oversized stack; the loader raises rather than doing that silently, but the
# fix is to name the specific folder.  To see what is where:
#
#   python -m reconstruction.recon_comparison --list-series /path/to/export
# ============================================================================

# --- the three input paths (edit these) -------------------------------------
WFBP_DIR="/data/Data2/4_BIN_PCCT/Reconstructions/4-bin_Phantom-Scan/WFBP"
VMI_DIR="/data/Data2/4_BIN_PCCT/Reconstructions/4-bin_Phantom-Scan/Mono"
DATA_PATH="/data/Data2/4_BIN_PCCT/4 bin Phantom raw data/Descriptor/full_sinogram_4-bin_Phantom-Scan..CT.Thx.-_Abdomen_Staging.Körper.601.RAW.20260326.074506.20260401.175846.e00e8253-e854-4963-bed1-1ac627e653d7.raw.mat"
DESC_PATH="/data/Data2/4_BIN_PCCT/4 bin Phantom raw data/Descriptor/descriptor_4-bin_Phantom-Scan..CT.Thx.-_Abdomen_Staging.Körper.601.RAW.20260326.074506.20260401.175846.e00e8253-e854-4963-bed1-1ac627e653d7.raw.mat"

OUT_ROOT="output/research/recon_comparison"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /home/nisc24/.conda/envs/MatDecomp
cd "$SLURM_SUBMIT_DIR"

set -e   # stop on the first failure: a bad stage 0 makes stages 1-2 meaningless

echo "=== 0/3  metric self-test (no data needed) ==="
python -m reconstruction.selftest_image_quality

echo
echo "=== 1/3  probe Siemens geometry + locate the insert slab (CPU) ==="
echo "    WFBP: $WFBP_DIR"
echo "    VMI : $VMI_DIR"
python -m reconstruction.recon_comparison --stage 0 \
    --wfbp-dir "$WFBP_DIR" \
    --vmi-dir  "$VMI_DIR" \
    --out-root "$OUT_ROOT" \
    "$@"

echo
echo "=== 2/3  reconstruction sweep, 5 settings x 4 thresholds (GPU) ==="
echo "    existing variants with a matching config are skipped"
python -m reconstruction.recon_comparison --stage 1 \
    --out-root  "$OUT_ROOT" \
    --data-path "$DATA_PATH" \
    --desc-path "$DESC_PATH" \
    "$@"

echo
echo "=== 3/3  metrics: NPS / TTF / NEQ / d' / bias (CPU) ==="
python -m reconstruction.recon_comparison --stage 2 \
    --out-root "$OUT_ROOT" \
    "$@"

echo
echo "=== done ==="
echo "  QC (check these FIRST):"
echo "    $OUT_ROOT/qc/stage0_slab_detection.png   slab must contain the inserts"
echo "    $OUT_ROOT/qc/roi_own.png  roi_wfbp.png  roi_vmi.png"
echo "  Results:"
echo "    $OUT_ROOT/figures/tradeoff_per_threshold.png   <- the headline figure"
echo "    $OUT_ROOT/figures/nps_curves.png  vmi_vs_kev.png  bland_altman_own_vs_wfbp.png"
echo "    $OUT_ROOT/metrics/metrics.json"
