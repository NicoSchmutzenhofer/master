#!/bin/bash
#SBATCH --nodelist=sauron
#SBATCH --job-name=MaterialDecomposition
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --cpus-per-task=8          
#SBATCH --mem=64G                  
#SBATCH --output=logs/slurm_%j.out 
#SBATCH --error=logs/slurm_%j.err  


# mkdir -p /tmp/logs

# 1. Initialize Conda using the system path (should be same for all users) 
source /opt/miniconda3/etc/profile.d/conda.sh 

# 2. Activate your specific environment path (user dependent) 
conda activate /home/nisc24/.conda/envs/MatDecomp

# Run from the repo root (the directory sbatch was submitted from)
cd "$SLURM_SUBMIT_DIR"

# 3. Run the command (specify your command)
#
# --- Reconstruction (produces the per-threshold volumes the decomposition reads) ---
# Uncomment to (re)build the volumes first; this is the step that needs the GPU
# (--gres above). Skip it if output/reconstruction/reconstruction_thr_{A,B,C,D}_HU.nii.gz already exist.
# python reconstruction/python_reconstruction.py
#
# --- Material decomposition, Phase A (CPU only; runs in order, stops on first error) ---
#   1) self-test (math sanity)   2) decompose (material maps + stability)   3) research (figures)
python -m decomposition.selftest_decomposition \
  && python -m decomposition.decompose \
  && python -m decomposition.research_decomposition

# 4. Move logs to permanent location  
# mv /tmp/logs/${SLURM_JOB_ID}_result.out ./logs/  
# mv /tmp/logs/${SLURM_JOB_ID}_error.err ./logs/ 