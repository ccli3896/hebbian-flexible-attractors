#!/bin/bash
# Sweep job: small networks, non-tau rules (PC, OJA, HR).
#
# Runs the full alpha x beta x eta sweep in one job. No array
# parallelization needed for these — they're the fastest jobs.
#
# Required environment variables (set by launch_array.sh or manually):
#   SWEEP_ENV       : env name (det, bandit_unbait, matchingpennies, slowfeature, selectivity)
#   SWEEP_RULE      : rule key (PC, OJA, HR)
#   SWEEP_TIMESTEPS : per-prob timesteps
#   SWEEP_OUTDIR    : where results go

#SBATCH --job-name=ssb_small_notau
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=8:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
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

: "${SWEEP_ARCH:?SWEEP_ARCH must be set}"

echo "=== ssb sweep: small/non-tau ==="
echo "node       : $(hostname)"
echo "start      : $(date)"
echo "env        : ${SWEEP_ENV}"
echo "rule       : ${SWEEP_RULE}"
echo "size       : ${SWEEP_ARCH}"
echo "timesteps  : ${SWEEP_TIMESTEPS}"
echo "outdir     : ${SWEEP_OUTDIR}"
echo "================================"

module load miniconda || true
source ~/venvs/ssb/bin/activate

python -c "import ssb; from ssb.scripts.run_sweep import run_sweep; print('ssb import OK')"

mkdir -p "${SWEEP_OUTDIR}/configs" "${SWEEP_OUTDIR}/results" logs

CONFIG_FILE="${SWEEP_OUTDIR}/configs/${SWEEP_ENV}_${SWEEP_RULE}_${SWEEP_ARCH}.json"
OUT_FILE="${SWEEP_OUTDIR}/results/${SWEEP_ENV}_${SWEEP_RULE}_${SWEEP_ARCH}.npy"

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
    --out    "${OUT_FILE}"

echo "end : $(date)"
echo "done"
