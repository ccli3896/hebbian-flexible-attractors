#!/bin/bash
# Check (and optionally fill) the steady-state matching-pennies sweep.
# Run from the repository root (the folder with ssb/ and slurm/), e.g. ~/local-learning-2025:
#
#   bash slurm/check_mp_steady.sh                 # report: done/missing slices, queue, why slices failed
#   bash slurm/check_mp_steady.sh --submit        # also submit arrays containing ONLY the missing slices
#
# Same env vars as launch_mp_steady.sh: RULES, ARCHS (default: big), PARTITION, EXCLUDE, BIG_TIME,
# SMALL_TIME, MAX_RUNNING, MP_OUTDIR. Slice numbering matches sweep_mp_steady.sh:
#   TASK_ID = (eta_idx * N_TAUS + tau_idx) * N_ALPHA_SHARDS + alpha_idx
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RULES="${RULES:-PC HR OJA COV BCM LC HZ}"
ARCHS="${ARCHS:-big}"
MP_OUTDIR="${MP_OUTDIR:-./sweep_mp_steady}"
MP_TIMESTEPS="${MP_TIMESTEPS:-20000}"; MP_CYCLES="${MP_CYCLES:-3}"; MP_SEEDS="${MP_SEEDS:-20}"
MAX_RUNNING="${MAX_RUNNING:-50}"; SMALL_TIME="${SMALL_TIME:-08:00:00}"; BIG_TIME="${BIG_TIME:-24:00:00}"
PART_ARG=(); [[ -n "${PARTITION:-}" ]] && PART_ARG=(--partition="${PARTITION}")
[[ -n "${EXCLUDE:-}" ]] && PART_ARG+=(--exclude="${EXCLUDE}")
SUBMIT=0; [[ "${1:-}" == "--submit" ]] && SUBMIT=1
R="${MP_OUTDIR}/results"

ranges() {  # 0 1 2 5 7 8 -> 0-2,5,7-8
  local out="" s="" p=""
  for i in "$@"; do
    if [[ -z "$s" ]]; then s=$i; p=$i
    elif (( i == p + 1 )); then p=$i
    else out+="${s}$([[ $s != $p ]] && echo "-$p"),"; s=$i; p=$i; fi
  done
  [[ -n "$s" ]] && out+="${s}$([[ $s != $p ]] && echo "-$p")"
  echo "${out%,}"
}

echo "== Jobs still queued/running =="
squeue -u "$USER" -h -o "%j %T" | grep '^mp_' | sort | uniq -c || echo "   none"
echo
echo "== Slices on disk =="
TOTAL_MISSING=0
for rule in ${RULES}; do
  case "${rule}" in COV|BCM|LC|HZ) nt=4 ;; *) nt=1 ;; esac
  for arch in ${ARCHS}; do
    if [[ "$arch" == big ]]; then sh=15; tl="$BIG_TIME"; else sh=1; tl="$SMALL_TIME"; fi
    n=$(( 6 * nt * sh )); miss=()
    for (( id=0; id<n; id++ )); do
      a=$(( id % sh )); rest=$(( id / sh )); e=$(( rest / nt )); t=$(( rest % nt ))
      stem="mp_steady_${rule}_${arch}_eta${e}"; (( nt > 1 )) && stem+="_tau${t}"; (( sh > 1 )) && stem+="_a${a}"
      [[ -s "$R/$stem.npz" ]] || miss+=("$id")
    done
    nm=${#miss[@]}; TOTAL_MISSING=$(( TOTAL_MISSING + nm ))
    printf "%-4s %-5s %4d/%-4d done" "$rule" "$arch" $(( n - nm )) "$n"
    if (( nm == 0 )); then echo "   COMPLETE"; continue; fi
    echo "   missing array ids: $(ranges "${miss[@]}")"
    # why did they fail? look at the newest log of each missing slice
    declare -A why=()
    for id in "${miss[@]}"; do
      f=$(ls -t logs/mp_${rule}_${arch}_*_${id}.err 2>/dev/null | head -1)
      if [[ -z "$f" ]]; then r="no log (never ran)"
      elif grep -q "DUE TO TIME LIMIT" "$f"; then r="time limit"
      elif grep -qi "oom\|out of memory\|MemoryError" "$f"; then r="out of memory"
      elif grep -q "CANCELLED" "$f"; then r="cancelled"
      elif grep -q "NODE FAIL\|launch failed" "$f"; then r="node failure"
      elif grep -q "Traceback" "$f"; then r="python error: $(grep -v '^\s' "$f" | tail -1 | cut -c1-70)"
      else r="other (see $f)"; fi
      why[$r]=$(( ${why[$r]:-0} + 1 ))
    done
    for r in "${!why[@]}"; do printf "       %4d x %s\n" "${why[$r]}" "$r"; done
    unset why
    if (( SUBMIT )); then
      sbatch "${PART_ARG[@]}" --job-name="mp_${rule}_${arch}" --time="${tl}" \
        --array="$(ranges "${miss[@]}")%${MAX_RUNNING}" \
        --export="ALL,MP_RULE=${rule},MP_ARCH=${arch},MP_TIMESTEPS=${MP_TIMESTEPS},MP_CYCLES=${MP_CYCLES},MP_SEEDS=${MP_SEEDS},N_TAUS=${nt},N_ALPHA_SHARDS=${sh},MP_OUTDIR=${MP_OUTDIR}" \
        "${HERE}/sweep_mp_steady.sh"
    fi
  done
done
echo
echo "Total missing slices: ${TOTAL_MISSING}"
(( SUBMIT == 0 && TOTAL_MISSING > 0 )) && echo "To fill them:  PARTITION=cpu_backfill EXCLUDE=gpu-350-04 bash slurm/check_mp_steady.sh --submit"
(( TOTAL_MISSING == 0 )) && echo "All done. Next:  python -m ssb.analysis.aggregate_mp_steady --indir ${MP_OUTDIR}/results --outdir data/sweeps"
exit 0
