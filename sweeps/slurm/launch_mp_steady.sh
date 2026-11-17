#!/bin/bash
# Launch the steady-state matching-pennies sweep (per-step task) for every rule x architecture.
# Run from the repository root (the folder that contains the ssb/ package), e.g. ~/local-learning-2025:
#
#   bash slurm/launch_mp_steady.sh --dry-run     # print the sbatch commands
#   bash slurm/launch_mp_steady.sh               # submit
#   RULES=PC ARCHS=small bash slurm/launch_mp_steady.sh      # one combination (try this first)
#
# Re-running the same command is safe: slices whose result file exists exit in seconds.
#
# Defaults: 7 phases (computer, then 3 x [coin flip, computer]) of 20,000 steps, 20 seeds,
# small and big networks, all 7 rules. Array tasks per (rule, arch):
#   small: 6 (eta) or 24 (eta x tau)          -> whole 15 x 15 alpha x beta grid per task
#   big  : x 15 alpha shards (90 or 360 tasks) -> one alpha value x 15 betas per task
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # folder holding this launcher and sweep_mp_steady.sh

RULES="${RULES:-PC HR OJA COV BCM LC HZ}"
ARCHS="${ARCHS:-small big}"
MP_TIMESTEPS="${MP_TIMESTEPS:-20000}"     # steps per phase (same as the original sweep)
MP_CYCLES="${MP_CYCLES:-3}"               # coin-flip/computer cycles after the first computer phase
MP_SEEDS="${MP_SEEDS:-20}"
MP_OUTDIR="${MP_OUTDIR:-./sweep_mp_steady}"
MAX_RUNNING="${MAX_RUNNING:-50}"          # per-array cap on simultaneously running tasks (%N)
SMALL_TIME="${SMALL_TIME:-08:00:00}"
BIG_TIME="${BIG_TIME:-24:00:00}"
BIG_ALPHA_SHARDS="${BIG_ALPHA_SHARDS:-15}"
PARTITION="${PARTITION:-}"               # override the template's #SBATCH --partition=cpu, e.g. PARTITION=fast
PART_ARG=(); [[ -n "${PARTITION}" ]] && PART_ARG=(--partition="${PARTITION}")
EXCLUDE="${EXCLUDE:-}"                   # nodes to avoid, comma-separated, e.g. EXCLUDE=gpu-350-04
[[ -n "${EXCLUDE}" ]] && PART_ARG+=(--exclude="${EXCLUDE}")
DRY_RUN=0; [[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

mkdir -p "${MP_OUTDIR}/results" logs
for rule in ${RULES}; do
  case "${rule}" in COV|BCM|LC|HZ) n_taus=4 ;; *) n_taus=1 ;; esac
  for arch in ${ARCHS}; do
    if [[ "${arch}" == "small" ]]; then shards=1; tlim="${SMALL_TIME}"; else shards="${BIG_ALPHA_SHARDS}"; tlim="${BIG_TIME}"; fi
    n=$(( 6 * n_taus * shards ))
    cmd=(sbatch "${PART_ARG[@]}" --job-name="mp_${rule}_${arch}" --time="${tlim}" --array="0-$((n-1))%${MAX_RUNNING}"
         --export="ALL,MP_RULE=${rule},MP_ARCH=${arch},MP_TIMESTEPS=${MP_TIMESTEPS},MP_CYCLES=${MP_CYCLES},MP_SEEDS=${MP_SEEDS},N_TAUS=${n_taus},N_ALPHA_SHARDS=${shards},MP_OUTDIR=${MP_OUTDIR}"
         "${HERE}/sweep_mp_steady.sh")
    if [[ "${DRY_RUN}" -eq 1 ]]; then echo "${cmd[@]}"; else "${cmd[@]}"; fi
  done
done
echo "When done:  python -m ssb.analysis.aggregate_mp_steady --indir ${MP_OUTDIR}/results --outdir data/sweeps"
