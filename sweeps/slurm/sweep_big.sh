#!/bin/bash
# Sweep job: BIG networks, all rules. Array job parallelized over eta
# (and tau for tau-having rules).
#
# The launcher sets N_TAUS to indicate how the array index is encoded:
#   N_TAUS=1   : array index = eta_idx           (non-tau rules)
#   N_TAUS=4   : array index = eta_idx * 4 + tau_idx   (tau rules)
#
# Default tau sweep is 4 values; default eta sweep is 6 values. So:
#   Non-tau rules: --array=0-5  (6 jobs)
#   Tau rules    : --array=0-23 (24 jobs)
#
# Required environment variables:
#   SWEEP_ENV, SWEEP_RULE, SWEEP_TIMESTEPS, SWEEP_OUTDIR, N_TAUS
# Note: the launcher (launch_array.sh) overrides --time per job: 16 h for
# big + bandit_unbait (the slowest combination), 8 h for everything else.
# The 8 h #SBATCH default below applies only if this script is submitted
# directly without the launcher.

#SBATCH --job-name=ssb_big
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00

#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL

# --- Optional: SLURM account/QoS -------------------------------------------
# #SBATCH --account=your_swc_account
# #SBATCH --qos=normal
# ---------------------------------------------------------------------------

set -euo pipefail

# This is a CPU-partition job, but the venv ships the jax[cuda12] wheels. Left
# to itself JAX auto-discovers its bundled CUDA plugin and calls cuInit(0),
# which on a GPU-less node fails with the noisy
#   "Jax plugin configuration error ... CUDA_ERROR_SHARED_OBJECT_INIT_FAILED (303)"
# traceback. JAX then falls back to CPU and the run continues, but the message
# is alarming and the probe wastes startup time. Force CPU up front so JAX
# never touches CUDA. Mirrors local_test.sh / convergence_scan.sbatch. Override
# only if you deliberately point this template at a GPU node.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export CUDA_VISIBLE_DEVICES=""

: "${SWEEP_ENV:?SWEEP_ENV must be set}"
: "${SWEEP_RULE:?SWEEP_RULE must be set}"
: "${SWEEP_TIMESTEPS:?SWEEP_TIMESTEPS must be set}"
: "${SWEEP_OUTDIR:?SWEEP_OUTDIR must be set}"
: "${N_TAUS:?N_TAUS must be set (1 for non-tau rules, 4 for tau rules)}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set (this is an array job)}"

SWEEP_ARCH="${SWEEP_ARCH:-big}"
TASK_ID="${SLURM_ARRAY_TASK_ID}"
N_ALPHA_SHARDS="${N_ALPHA_SHARDS:-1}"

# Decode (eta_idx, tau_idx, alpha_shard) from the flat array index using the
# mixed radix TASK_ID = (eta*N_TAUS + tau)*N_ALPHA_SHARDS + alpha_shard.
# With N_ALPHA_SHARDS=1 (default) alpha_shard is always 0 and this reduces
# EXACTLY to the previous ETA_IDX=TASK_ID/N_TAUS, TAU_IDX=TASK_ID%N_TAUS.
ALPHA_IDX=$(( TASK_ID % N_ALPHA_SHARDS ))
REST=$(( TASK_ID / N_ALPHA_SHARDS ))
ETA_IDX=$(( REST / N_TAUS ))
TAU_IDX=$(( REST % N_TAUS ))

# Alpha-shard tag/arg — empty unless sharding is active, so unsharded runs
# produce byte-identical filenames and commands as before.
if [[ "${N_ALPHA_SHARDS}" -gt 1 ]]; then
    ALPHA_TAG="_a${ALPHA_IDX}"
    ALPHA_ARG="--alpha-idx ${ALPHA_IDX}"
else
    ALPHA_TAG=""
    ALPHA_ARG=""
fi

echo "=== ssb sweep: big ==="
echo "node       : $(hostname)"
echo "start      : $(date)"
echo "env        : ${SWEEP_ENV}"
echo "rule       : ${SWEEP_RULE}"
echo "size       : ${SWEEP_ARCH}"
echo "task_id    : ${TASK_ID}"
echo "eta_idx    : ${ETA_IDX}"
echo "tau_idx    : ${TAU_IDX} (used only if N_TAUS>1)"
echo "n_taus     : ${N_TAUS}"
echo "alpha_idx  : ${ALPHA_IDX} (used only if N_ALPHA_SHARDS>1)"
echo "n_alpha_sh : ${N_ALPHA_SHARDS}"
echo "timesteps  : ${SWEEP_TIMESTEPS}"
echo "outdir     : ${SWEEP_OUTDIR}"
echo "======================"

module load miniconda || true
source ~/venvs/ssb/bin/activate

python -c "import ssb; from ssb.scripts.run_sweep import run_sweep; print('ssb import OK')"

mkdir -p "${SWEEP_OUTDIR}/configs" "${SWEEP_OUTDIR}/results" logs


# Configs are labeled with unique indices so they don't overwrite each other
# when many jobs are running. Output filename encodes the slice: eta, then tau
# (tau rules only), then the alpha shard (only when N_ALPHA_SHARDS>1). Build the
# stem once so the ordering matches the aggregator's filename parser.
STEM="${SWEEP_ENV}_${SWEEP_RULE}_${SWEEP_ARCH}_eta${ETA_IDX}"
if [[ "${N_TAUS}" -eq 1 ]]; then
    TAU_ARG=""
else
    STEM="${STEM}_tau${TAU_IDX}"
    TAU_ARG="--tau-idx ${TAU_IDX}"
fi
STEM="${STEM}${ALPHA_TAG}"
OUT_FILE="${SWEEP_OUTDIR}/results/${STEM}.npy"
CONFIG_FILE="${SWEEP_OUTDIR}/configs/${STEM}.json"

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
    --eta-idx "${ETA_IDX}" \
    ${TAU_ARG} \
    ${ALPHA_ARG}

echo "end : $(date)"
echo "done"
