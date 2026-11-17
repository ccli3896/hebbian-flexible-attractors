#!/bin/bash
# Sweep job: small networks, tau-having rules (COV, BCM, LC, HZ).
#
# Array job: one task per tau value. The array index selects the tau
# from the sweep range (0..3 for the default 4-value tau grid).
# Each task runs the full alpha x beta x eta sweep at one fixed tau.
#
# Submitted with:
#   sbatch --array=0-3 slurm/sweep_small_tau.sh
#
# Required environment variables:
#   SWEEP_ENV       : env name
#   SWEEP_RULE      : rule key (must be in {COV, BCM, LC, HZ})
#   SWEEP_TIMESTEPS : per-prob timesteps
#   SWEEP_OUTDIR    : where results go

#SBATCH --job-name=ssb_small_tau
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=8:00:00

#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL

# --- Optional: SLURM account/QoS -------------------------------------------
# #SBATCH --account=your_swc_account
# #SBATCH --qos=normal
# ---------------------------------------------------------------------------

set -euo pipefail

# CPU-partition job with the jax[cuda12] wheels installed: force CPU so JAX
# doesn't auto-load its CUDA plugin and fail cuInit(0) on a GPU-less node with
# the noisy "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED (303)" traceback (non-fatal,
# JAX falls back to CPU, but confusing + wastes startup). Mirrors local_test.sh.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export CUDA_VISIBLE_DEVICES=""

: "${SWEEP_ENV:?SWEEP_ENV must be set}"
: "${SWEEP_RULE:?SWEEP_RULE must be set}"
: "${SWEEP_TIMESTEPS:?SWEEP_TIMESTEPS must be set}"
: "${SWEEP_OUTDIR:?SWEEP_OUTDIR must be set}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set (this is an array job)}"

: "${SWEEP_ARCH:?SWEEP_ARCH must be set}"
TAU_IDX="${SLURM_ARRAY_TASK_ID}"

echo "=== ssb sweep: small/tau ==="
echo "node       : $(hostname)"
echo "start      : $(date)"
echo "env        : ${SWEEP_ENV}"
echo "rule       : ${SWEEP_RULE}"
echo "size       : ${SWEEP_ARCH}"
echo "tau_idx    : ${TAU_IDX}"
echo "timesteps  : ${SWEEP_TIMESTEPS}"
echo "outdir     : ${SWEEP_OUTDIR}"
echo "============================"

module load miniconda || true
source ~/venvs/ssb/bin/activate

python -c "import ssb; from ssb.scripts.run_sweep import run_sweep; print('ssb import OK')"

mkdir -p "${SWEEP_OUTDIR}/configs" "${SWEEP_OUTDIR}/results" logs

# Config file includes tau index so configs don't overwrite each other when many jobs are running
CONFIG_FILE="${SWEEP_OUTDIR}/configs/${SWEEP_ENV}_${SWEEP_RULE}_${SWEEP_ARCH}_tau${TAU_IDX}.json"
# Output filename includes tau index so per-slice files don't clobber each other
OUT_FILE="${SWEEP_OUTDIR}/results/${SWEEP_ENV}_${SWEEP_RULE}_${SWEEP_ARCH}_tau${TAU_IDX}.npy"

python -c "
import json
config = {
    'env': '${SWEEP_ENV}',
    'rule': '${SWEEP_RULE}',
    'arch': '${SWEEP_ARCH}',
    'timesteps': ${SWEEP_TIMESTEPS},
}
with open('${CONFIG_FILE}', 'w') as f:
    json.dump(config, f, indent=2)
print('wrote config:', '${CONFIG_FILE}')
"

cd "${SLURM_SUBMIT_DIR}"
python -u -m ssb.scripts.run_sweep \
    --config "${CONFIG_FILE}" \
    --out    "${OUT_FILE}" \
    --tau-idx "${TAU_IDX}"

echo "end : $(date)"
echo "done"
