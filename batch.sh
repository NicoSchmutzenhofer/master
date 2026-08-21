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

# 3. Material decomposition -- pick the INPUT SOURCE as the first sbatch argument:
#      sbatch batch.sh own     # our reconstruction  (output/reconstruction/*_HU.nii.gz)
#      sbatch batch.sh vmi     # Siemens monoenergetic (VMI) DICOMs
#      sbatch batch.sh wfbp    # Siemens WFBP threshold DICOMs
#   Default (no arg) = own. Output -> output/decomposition/<source>/.
#   Decomposition is CPU-only; the --gres GPU line above is only needed for the reconstruction
#   stanza below (drop --gres for a pure CPU slot if your scheduler allows).
SOURCE="${1:-own}"
export DECOMP_SOURCE="$SOURCE"
shift 2>/dev/null || true      # remaining args are forwarded to decompose.py

# --- Siemens export folders (edit these) ------------------------------------
# Each Siemens reconstruction lives in its OWN folder, so wfbp and vmi need separate
# paths -- one path cannot serve both.  Point each at a folder holding exactly ONE
# reconstruction: a parent holding several sets yields duplicate channel labels
# (T1,T1,T2,T2,...) and is rejected.  To see what is where:
#   python -m decomposition.decompose --list-series '/data/.../export'
WFBP_DIR="/data/Data2/4_BIN_PCCT/Reconstructions/4-bin_Phantom-Scan/Thx.- Abdomen Staging_Standard - PNR_20260729_124048/"
VMI_DIR="/data/Data2/4_BIN_PCCT/Reconstructions/4-bin_Phantom-Scan/Thx.- Abdomen Staging_Standard - PNR_20260729_092611/"
MODE="phantom_ca_i"            # python -m decomposition.decompose --list-modes

echo "=== material decomposition: source=$SOURCE  mode=$MODE ==="

# --- (optional) (re)build our per-threshold volumes first; needs the GPU. Skip for vmi/wfbp,
#     or if output/reconstruction/reconstruction_thr_{A,B,C,D}_HU.nii.gz already exist. ---
# python reconstruction/python_reconstruction.py

# self-test (math sanity) -> decompose the chosen source (stops on first error)
python -m decomposition.selftest_decomposition \
  && python -m decomposition.decompose \
        --source "$SOURCE" \
        --mode "$MODE" \
        --wfbp-dir "$WFBP_DIR" \
        --vmi-dir  "$VMI_DIR" \
        "$@"

# --- (optional) research ablations on our own recon (source-independent; needs the NIfTI volumes) ---
# python -m decomposition.research_decomposition

# 4. Move logs to permanent location  
# mv /tmp/logs/${SLURM_JOB_ID}_result.out ./logs/  
# mv /tmp/logs/${SLURM_JOB_ID}_error.err ./logs/ 