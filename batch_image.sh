#!/bin/bash
#SBATCH --nodelist=sauron
#SBATCH --job-name=BinSepImage
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Image-domain threshold-separation INVESTIGATION (standalone).
#
# Uses the redundancy between the four reconstructed cumulative-threshold volumes
# to test whether image-domain bin separation (exclusive windows A−B, B−C, C−D, D
# formed as IMAGE differences) plus cross-bin denoising improves IMAGE QUALITY.
# Concluded negative (see docs/BIN_SEPARATION_FINDINGS.md); tooling kept for the
# material-decomposition step. Does not modify the production pipeline.
#
# Input : output/reconstruction/reconstruction_thr_{A,B,C,D}_HU.nii.gz
#         (run the reconstruction first; this reads the finished NIfTIs).
# Compute: CPU-only in the default mode — it works on the already-reconstructed
#         volumes and does NOT use the GPU. The --gres line above is kept only so
#         the RECON_FALLBACK=True path (reconstruct a slab from the raw .mat when
#         the NIfTIs are absent, ASTRA SIRT) can run; drop it for a pure CPU slot
#         if your scheduler allows CPU jobs on this partition.
# RAM   : loads the four volumes then slabs them (~20 GB transient); 64G is safe.
# Output: output/research/image_subtraction/
#           image_subtraction_findings.md, image_subtraction_metrics.json,
#           image_subtraction_correlation.png, image_subtraction_panels.png
#
# Submit with:  sbatch batch_image.sh
# ─────────────────────────────────────────────────────────────────────────────

# 1. Initialize Conda using the system path (same for all users)
source /opt/miniconda3/etc/profile.d/conda.sh

# 2. Activate the environment (user-dependent path)
conda activate /home/nisc24/.conda/envs/MatDecomp

# 3. Run from the repo root (the directory sbatch was submitted from)
cd "$SLURM_SUBMIT_DIR"

# 4. Run the investigation (standalone script; knobs are at the top of it)
python reconstruction/image_subtraction_investigation.py
