#!/bin/bash
# One array task of the steady-state matching-pennies sweep (ssb/scripts/run_mp_steady.py).
#
# Array index encodes the slice:  TASK_ID = (eta_idx * N_TAUS + tau_idx) * N_ALPHA_SHARDS + alpha_idx
#   N_TAUS         : 1 for PC/HR/OJA, 4 for COV/BCM/LC/HZ
#   N_ALPHA_SHARDS : 1 = whole alpha x beta grid per task; 15 = one alpha value per task (big nets)
# Submitted by launch_mp_steady.sh. Tasks whose output already exists exit immediately, so the
# same launch command can be re-run to fill in failed slices.
#
# Required env vars: MP_RULE, MP_ARCH, MP_TIMESTEPS, MP_CYCLES, MP_SEEDS, N_TAUS, N_ALPHA_SHARDS, MP_OUTDIR

#SBATCH --job-name=ssb_mp_steady
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
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"   # CPU partition: keep JAX away from its CUDA plugin
export CUDA_VISIBLE_DEVICES=""

: "${MP_RULE:?}"; : "${MP_ARCH:?}"; : "${MP_TIMESTEPS:?}"; : "${MP_CYCLES:?}"; : "${MP_SEEDS:?}"
: "${N_TAUS:?}"; : "${N_ALPHA_SHARDS:?}"; : "${MP_OUTDIR:?}"; : "${SLURM_ARRAY_TASK_ID:?}"

TASK_ID="${SLURM_ARRAY_TASK_ID}"
ALPHA_IDX=$(( TASK_ID % N_ALPHA_SHARDS ))
REST=$(( TASK_ID / N_ALPHA_SHARDS ))
ETA_IDX=$(( REST / N_TAUS ))
TAU_IDX=$(( REST % N_TAUS ))

STEM="mp_steady_${MP_RULE}_${MP_ARCH}_eta${ETA_IDX}"
TAU_ARG=""; ALPHA_ARG=""
if [[ "${N_TAUS}" -gt 1 ]]; then STEM="${STEM}_tau${TAU_IDX}"; TAU_ARG="--tau-idx ${TAU_IDX}"; fi
if [[ "${N_ALPHA_SHARDS}" -gt 1 ]]; then STEM="${STEM}_a${ALPHA_IDX}"; ALPHA_ARG="--alpha-idx ${ALPHA_IDX}"; fi

echo "=== mp_steady: ${STEM} | node $(hostname) | start $(date) ==="
module load miniconda || true
source ~/venvs/ssb/bin/activate
mkdir -p "${MP_OUTDIR}/configs" "${MP_OUTDIR}/results" logs

CONFIG_FILE="${MP_OUTDIR}/configs/${STEM}.json"
cat > "${CONFIG_FILE}" <<JSON
{"rule": "${MP_RULE}", "arch": "${MP_ARCH}", "timesteps": ${MP_TIMESTEPS}, "n_cycles": ${MP_CYCLES}}
JSON

cd "${SLURM_SUBMIT_DIR}"
python -u -m ssb.scripts.run_mp_steady \
    --config "${CONFIG_FILE}" \
    --out    "${MP_OUTDIR}/results/${STEM}.npz" \
    --seeds  "${MP_SEEDS}" \
    --eta-idx "${ETA_IDX}" ${TAU_ARG} ${ALPHA_ARG}

echo "end: $(date)"
