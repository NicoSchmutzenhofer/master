#!/bin/bash
#SBATCH --nodelist=sauron
#SBATCH --job-name=SinoThresholdSep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Sinogram-domain threshold-separation INVESTIGATION (standalone).
#
# Forms the exclusive energy windows by SUBTRACTING the cumulative threshold
# sinograms (E1=A−B, E2=B−C, E3=C−D, E4=D) BEFORE reconstruction, reconstructs
# each with SIRT on the GPU, and (1) measures whether the A≥B≥C≥D ordering
# violation is zero-mean noise or a systematic gain bias (the empirical test of
# CLAUDE.md invariant #3), and (2) compares subtract-then-SIRT vs SIRT-then-
# subtract. It does NOT modify the production reconstruction or invariant #3.
#
# Needs : the raw .mat sinograms (paths hardcoded in the script) + a CUDA GPU.
# RAM   : peak ~32 GB (two 16 GB cumulative sinograms held at once) — 64G is safe.
# Time  : scales with the study slab and N_ITER, both set at the top of the
#         script (SLICE_IDX / N_SLAB_SLICES / N_ITER; defaults 41-slice slab,
#         SIRT ×100). Shrink N_SLAB_SLICES for a quick first pass.
# Output: output/research/sinogram_separation/
#           sinogram_separation_findings.md, sinogram_separation_metrics.json,
#           sinsep_negativity.png, sinsep_panels.png, sinsep_sino_vs_image.png
#
# Submit with:  sbatch batch_sinogram.sh
# ─────────────────────────────────────────────────────────────────────────────

# 1. Initialize Conda using the system path (same for all users)
source /opt/miniconda3/etc/profile.d/conda.sh

# 2. Activate the environment (user-dependent path)
conda activate /home/nisc24/.conda/envs/MatDecomp

# 3. Run from the repo root (the directory sbatch was submitted from)
cd "$SLURM_SUBMIT_DIR"

# 4. Run the investigation (standalone script; SIRT settings are at the top of it)
python reconstruction/sinogram_separation_investigation.py
