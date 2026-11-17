#!/bin/bash
# Supplementary sweeps (reward magnitude / continual learning) — one array task
# per CELL SHARD.
#
# Unlike the master parameter sweep, which slices the (eta, tau) axes, these
# sweeps slice the *cell list*: the array index selects a contiguous block of
# CELLS_PER_TASK cells from either the manifest (mag/cl) or the full grid (the
# architecture sweep). Every cell is independent, so the split is arbitrary and
# the shards recombine into exactly the unsharded tensor.
#
# Required environment variables (set by launch_supp.sh):
#   SUPP_KIND        mag | cl | forget
#   SUPP_RULE        PC | HR | COV | OJA | BCM | LC | HZ
#   SUPP_ARCH        small | big | N2_L2 | ...
#   SUPP_OUTDIR      results root
#   SUPP_TIMESTEPS   per-run timestep budget
#   CELLS_PER_TASK   cells handled by one array task
# Optional:
#   SUPP_MANIFEST    manifest CSV (omit => --full-grid, the architecture sweep)
#   SUPP_REWARDS     colon-separated levels, e.g. "0.2:0.4:0.6:0.8:1.0"
#                    (colons, not commas: sbatch --export splits on commas)
#   SUPP_SEEDS       seeds per parameter point (default 20)
#   SUPP_ATTRACTOR   1|0 — run the attractor probe (default 1, mag only)
#   SUPP_STAGE       tag written into filenames, e.g. s1 / s2 (default s1)
#   SUPP_NUM_REVS    reversal blocks (cl only, default 10)
#   SUPP_N_ALPHA /   grid resolution for the arch sweep; must match what the
#   SUPP_N_BETA      launcher used to size the array (default 15, 9 for N200_L2)

#SBATCH --job-name=ssb_supp
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --requeue

# --- Optional: SLURM account/QoS -------------------------------------------
# #SBATCH --account=your_swc_account
# #SBATCH --qos=normal
# ---------------------------------------------------------------------------

set -euo pipefail

# The venv ships jax[cuda12]; on a CPU node JAX would otherwise call cuInit(0),
# fail noisily with CUDA_ERROR_SHARED_OBJECT_INIT_FAILED, and fall back to CPU
# anyway. Force CPU up front. Mirrors sweep_big.sh.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export CUDA_VISIBLE_DEVICES=""

: "${SUPP_KIND:?SUPP_KIND must be set (mag|cl)}"
: "${SUPP_RULE:?SUPP_RULE must be set}"
: "${SUPP_ARCH:?SUPP_ARCH must be set}"
: "${SUPP_OUTDIR:?SUPP_OUTDIR must be set}"
: "${SUPP_TIMESTEPS:?SUPP_TIMESTEPS must be set}"
: "${CELLS_PER_TASK:?CELLS_PER_TASK must be set}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set (this is an array job)}"

STAGE="${SUPP_STAGE:-s1}"
SEEDS="${SUPP_SEEDS:-20}"
CELL_START=$(( SLURM_ARRAY_TASK_ID * CELLS_PER_TASK ))

echo "=== ssb supplementary sweep ==="
echo "node        : $(hostname)"
echo "start       : $(date)"
echo "kind        : ${SUPP_KIND}"
echo "rule        : ${SUPP_RULE}"
echo "arch        : ${SUPP_ARCH}"
echo "stage       : ${STAGE}"
echo "task_id     : ${SLURM_ARRAY_TASK_ID}"
echo "cells       : [${CELL_START}, ${CELL_START}+${CELLS_PER_TASK})"
echo "timesteps   : ${SUPP_TIMESTEPS}"
echo "seeds       : ${SEEDS}"
echo "manifest    : ${SUPP_MANIFEST:-<full grid>}"
echo "outdir      : ${SUPP_OUTDIR}"
echo "==============================="

module load miniconda || true
source ~/venvs/ssb/bin/activate

python -c "import ssb; print('ssb import OK')"

mkdir -p "${SUPP_OUTDIR}/configs" "${SUPP_OUTDIR}/results" logs

STEM="${SUPP_KIND}_${SUPP_RULE}_${SUPP_ARCH}_${STAGE}_c${CELL_START}"
OUT_FILE="${SUPP_OUTDIR}/results/${STEM}.npz"
CONFIG_FILE="${SUPP_OUTDIR}/configs/${STEM}.json"

python - "${CONFIG_FILE}" <<PYEOF
import json, sys
config = {
    'kind': '${SUPP_KIND}',
    'rule': '${SUPP_RULE}',
    'arch': '${SUPP_ARCH}',
    'timesteps': ${SUPP_TIMESTEPS},
    'n_seeds': ${SEEDS},
    'stage': '${STAGE}',
}
rewards = '''${SUPP_REWARDS:-}'''.strip()
if rewards:
    # Colon-separated, because sbatch --export would split a comma-separated
    # value into separate environment assignments.
    config['rewards'] = [float(x) for x in rewards.split(':') if x]
attr = '''${SUPP_ATTRACTOR:-}'''.strip()
if attr:
    config['attractor'] = bool(int(attr))
nrev = '''${SUPP_NUM_REVS:-}'''.strip()
if nrev:
    config['num_revs'] = int(nrev)
# Grid resolution for the architecture sweep. Must match what the launcher used
# to size this job's array, or the shards will not tile the cell list.
for _key, _val in (('n_alpha', '''${SUPP_N_ALPHA:-}'''.strip()),
                   ('n_beta', '''${SUPP_N_BETA:-}'''.strip())):
    if _val:
        config[_key] = int(_val)
with open(sys.argv[1], 'w') as f:
    json.dump(config, f, indent=2)
print('wrote config:', sys.argv[1])
PYEOF

cd "${SLURM_SUBMIT_DIR}"

if [[ "${SUPP_KIND}" == "mag" ]]; then
    if [[ -n "${SUPP_MANIFEST:-}" ]]; then
        SRC_ARG="--manifest ${SUPP_MANIFEST}"
    else
        SRC_ARG="--full-grid"
    fi
    python -u -m ssb.scripts.run_mag_sweep \
        --config "${CONFIG_FILE}" \
        --out    "${OUT_FILE}" \
        ${SRC_ARG} \
        --cell-start "${CELL_START}" \
        --cell-count "${CELLS_PER_TASK}"
elif [[ "${SUPP_KIND}" == "forget" ]]; then
    : "${SUPP_MANIFEST:?forgetting sweep requires SUPP_MANIFEST}"
    python -u -m ssb.scripts.run_forget_sweep \
        --config "${CONFIG_FILE}" \
        --out    "${OUT_FILE}" \
        --manifest "${SUPP_MANIFEST}" \
        --cell-start "${CELL_START}" \
        --cell-count "${CELLS_PER_TASK}"
else
    : "${SUPP_MANIFEST:?continual-learning sweep requires SUPP_MANIFEST}"
    python -u -m ssb.scripts.run_cl_sweep \
        --config "${CONFIG_FILE}" \
        --out    "${OUT_FILE}" \
        --manifest "${SUPP_MANIFEST}" \
        --cell-start "${CELL_START}" \
        --cell-count "${CELLS_PER_TASK}"
fi

echo "end : $(date)"
echo "done"
