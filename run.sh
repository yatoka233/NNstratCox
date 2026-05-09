#!/bin/bash
# Submit with:
#   sbatch run.sh
#
# Optional overrides:
#   sbatch --export=ALL,CONDA_ENV=syn,N=600,EPOCHS=100,BATCH_SIZE=16,DEVICE=cpu run.sh

#SBATCH --job-name=nnstratcox
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --account=kevinhe1
#SBATCH --mail-user=2928980064@qq.com
#SBATCH --mail-type=BEGIN,END

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${PROJECT_DIR}"

mkdir -p logs

CONDA_ENV="${CONDA_ENV:-syn}"
N="${N:-600}"
N_CENTERS="${N_CENTERS:-8}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-123}"

echo "Job started on $(hostname) at $(date)"
echo "Project directory: ${PROJECT_DIR}"
echo "Conda environment: ${CONDA_ENV}"
echo "Arguments: --n ${N} --n-centers ${N_CENTERS} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --device ${DEVICE} --seed ${SEED}"

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
    echo "Could not find conda.sh. Load conda before submitting or edit run.sh."
    exit 1
fi

conda activate "${CONDA_ENV}"

python -u tutorial_deep_stratified_cox.py \
    --n "${N}" \
    --n-centers "${N_CENTERS}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --seed "${SEED}"

echo "Job finished at $(date)"
