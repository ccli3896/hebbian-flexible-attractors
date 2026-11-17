#!/bin/bash
# Master launcher for the F2/F4 master sweep on SLURM.
#
# Comments on running:
# bandit_unbait requires the most time: 12 h is safe for the big networks. (small can use 8)
#
# Dispatches each (env, rule, size) combination to the appropriate
# sbatch template based on the parallelization strategy:
#
#   small, no-tau (PC, OJA, HR)  -> sweep_small_notau.sh  (1 job)
#   small, tau   (COV, BCM, LC, HZ) -> sweep_small_tau.sh   (4-element array, one per tau)
#   big, no-tau                  -> sweep_big.sh           (6-element array, one per eta)
#   big, tau                     -> sweep_big.sh           (24-element array, one per (eta, tau))
#
# Total job count for full sweep (5 default envs; bandit_unbait replaces
# bandit_bait, which is opt-in via ENVS=bandit_bait):
#   small no-tau : 3 rules x 5 envs x 1            = 15 jobs
#   small tau    : 4 rules x 5 envs x 4 (array)    = 80 jobs
#   big no-tau   : 3 rules x 5 envs x 6 (array)    = 90 jobs
#   big tau      : 4 rules x 5 envs x 24 (array)   = 480 jobs
#   total                                          = 665 jobs
#
# Usage:
#   bash slurm/launch_array.sh                # submit full F2/F4 master sweep
#   bash slurm/launch_array.sh --dry-run      # print sbatch commands only
#   SWEEP_OUTDIR=./big_run bash slurm/launch_array.sh
#
# Filtering (run only a subset):
#   RULES=PC bash slurm/launch_array.sh                  # PC only
#   SIZES=small bash slurm/launch_array.sh               # small networks only
#   ENVS="det bandit_unbait" bash slurm/launch_array.sh  # two envs only
#   ENVS=bandit_bait bash slurm/launch_array.sh          # opt-in: old baited env
#
# Resume (idempotent re-run after failures / admin holds):
#   RESUME=1 bash slurm/launch_array.sh                  # submit only slices
#      whose result .npy is missing. Safe to re-run the SAME command as many
#      times as needed: completed slices are skipped (no duplicates), only the
#      ones that failed / never ran are resubmitted. Assumes some tasks fail —
#      "done" is judged per-slice from disk, not from job exit codes. Pair with
#      --dry-run first to see exactly what would be resubmitted.
#
# Grid-sharding (make the expensive big-network slices cheap on CPU):
#   ALPHA_SHARDS=15 ARCH="big N50_L3" ENVS=bandit_unbait bash slurm/launch_array.sh
#   -> splits every big-style (eta[,tau]) slice into 15 alpha shards, each at a
#      15x-smaller batch (15x less CPU wall / GPU compile per task); the shards
#      aggregate back to an identical tensor. ALPHA_SHARDS must equal the alpha
#      grid width (n_alpha; default 15). Default ALPHA_SHARDS=1 = OFF.

set -euo pipefail

# ---------------------------------------------------------------------------
# What to sweep — environment-variable overrides supported
# ---------------------------------------------------------------------------

# Rules and their tau classification
NO_TAU_RULES=(PC HR OJA)
TAU_RULES=(COV BCM LC HZ)
ALL_RULES=("${NO_TAU_RULES[@]}" "${TAU_RULES[@]}")

# Allow user to override via RULES env var (space-separated)
if [[ -n "${RULES:-}" ]]; then
    read -ra ALL_RULES <<< "${RULES}"
fi

# Envs: "name:timesteps" pairs.
#
# KNOWN_ENVS is the registry of every env we know how to sweep, with its 1x
# timestep baseline. It is used ONLY to resolve timesteps for names passed via
# the ENVS= override. bandit_bait stays in the registry so it remains runnable
# on demand (ENVS=bandit_bait ...), but it is NOT part of the default sweep:
# the sweeps now use bandit_unbait instead (same task, rewards redrawn every
# trial rather than baited). DEFAULT_ENVS is what runs when ENVS is unset.
KNOWN_ENVS=(
    "det:100000"
    "bandit_bait:50000"
    "bandit_unbait:50000"
    "matchingpennies:20000"
    "slowfeature:50000"
    "selectivity:10000"
)
DEFAULT_ENVS=(
    "det:100000"
    "bandit_unbait:50000"
    "matchingpennies:20000"
    "slowfeature:50000"
    "selectivity:10000"
)
ALL_ENVS=("${DEFAULT_ENVS[@]}")
if [[ -n "${ENVS:-}" ]]; then
    # ENVS env var format: space-separated env names (no timesteps).
    # Look up timesteps from KNOWN_ENVS for each requested name (so opt-in
    # envs like bandit_bait, absent from the default sweep, still resolve).
    declare -a FILTERED_ENVS
    read -ra ENV_NAMES <<< "${ENVS}"
    for name in "${ENV_NAMES[@]}"; do
        for entry in "${KNOWN_ENVS[@]}"; do
            if [[ "${entry%%:*}" == "${name}" ]]; then
                FILTERED_ENVS+=("${entry}")
                break
            fi
        done
    done
    ALL_ENVS=("${FILTERED_ENVS[@]}")
fi

# Per-env timestep budgets above are the 1x baselines. TIMESTEPS_SCALE
# multiplies every env's budget at submit time, so the converged (right-
# sized) timesteps are set per run WITHOUT editing the master list. The
# convergence work sized this by network capacity:
#   small (0 hidden)            -> 1x   (baseline)
#   in-between (N2_L2, N10_L2)  -> 2x
#   big / >=50 hidden           -> 4x   (big, N50_L2, N50_L3, N200_L2)
# Run each arch tier as its own invocation with the matching scale, e.g.:
#   TIMESTEPS_SCALE=1 SIZES=small                       bash slurm/launch_array.sh
#   TIMESTEPS_SCALE=2 ARCH="N2_L2 N10_L2" ENVS=bandit_unbait ...
#   TIMESTEPS_SCALE=4 ARCH="big N50_L2 N50_L3 N200_L2"  ...
# (Mixing tiers in one call is unsupported: timesteps are per-env, not
# per-arch, so one call can carry only one scale.) Scaled timesteps are
# floored to an integer.
TIMESTEPS_SCALE="${TIMESTEPS_SCALE:-1}"

# Architectures to sweep. By default these come from the SIZES variable
# (legacy: small, big) but can be overridden with ARCH for F5-style sweeps.
ALL_ARCHS=(small big)
if [[ -n "${ARCH:-}" ]]; then
    read -ra ALL_ARCHS <<< "${ARCH}"
elif [[ -n "${SIZES:-}" ]]; then
    read -ra ALL_ARCHS <<< "${SIZES}"
fi

# Sweep grid sizes (must match the ranges in ssb/scripts/run_sweep.py)
N_ETAS=6
N_TAUS=4

# Alpha grid-sharding (opt-in; default 1 = OFF -> identical to before).
# When ALPHA_SHARDS=K>1, each big-style (env,rule,arch) job's array is grown
# K-fold: every (eta[,tau]) slice is split into K alpha shards that run the
# same beta column at a K-times-smaller batch. This is the lever that makes
# the expensive big-network slices cheap: batch (hence CPU wall AND GPU XLA
# compile) scales down ~K-fold, at identical total compute, and the shards
# aggregate back to an identical tensor. K MUST equal the alpha grid width the
# run uses (n_alpha; default 15) so every alpha index is covered exactly once.
# Only big-style templates (sweep_big.sh; big nets + det) are sharded — small
# networks are already cheap. See slurm/GPU_CPU_PLAN.md.
ALPHA_SHARDS="${ALPHA_SHARDS:-1}"

# Global cap on the number of array tasks/jobs IN THE QUEUE (pending +
# running) at once across the entire sweep. Enforced by a polling gate in
# submit_with_gate, which counts submitted tasks via squeue and blocks
# before each submission until there is room. We count pending tasks too
# (not just running) because freshly-submitted array tasks sit PENDING
# before SLURM schedules them; counting only running ones lets the whole
# sweep flood the queue in a burst and blow past the cap.
#
# SLURM enqueues EVERY task of an --array as PENDING the instant the array is
# submitted (the `%cap` throttle only limits how many RUN), so a single
# oversized array — post-sharding these reach N_ETAS*N_TAUS*ALPHA_SHARDS =
# 24*15 = 360 tasks — would put far more than MAX_CONCURRENT tasks in the
# queue at once even though only ~MAX_CONCURRENT ever run. To keep the QUEUE
# itself bounded (be a polite neighbour / respect cluster MaxSubmitJobs and
# array limits), submit_with_gate submits each oversized array in contiguous
# index windows of at most MAX_CONCURRENT tasks, gating on committed+window
# between windows. Bounding the queue serialises windows (a window can't
# enqueue until earlier tasks drain), so throughput is roughly MAX_CONCURRENT-
# wide — raise MAX_CONCURRENT if the cluster allows it. Override with
# MAX_CONCURRENT to relax or tighten the limit.
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"

# ---------------------------------------------------------------------------
# GPU routing (opt-in; default OFF -> identical to the original CPU behavior)
# ---------------------------------------------------------------------------
# Archs listed in GPU_ARCHS are dispatched to the GPU template
# (slurm/sweep_big_gpu.sh) with --partition=${PARTITION} --gres=${GRES},
# instead of the CPU sweep_big.sh. Everything NOT in GPU_ARCHS keeps running
# on CPU exactly as before. Use this to send the GPU-only arch (N200_L2) to
# a GPU partition while big/N50_L3 stay on CPU, all from one invocation.
#
#   # N200_L2 on the dedicated A100 partition; be a polite neighbour with a
#   # small concurrency cap (only 4 A100s cluster-wide):
#   GPU_ARCHS=N200_L2 PARTITION=a100 MAX_CONCURRENT=4 \
#   ARCH="big N50_L3 N200_L2" ENVS=bandit_unbait \
#   bash slurm/launch_array.sh --dry-run
#
#   # ... or on the fast preemptible partition (needs --requeue, which the
#   # GPU template already sets; keep slices short so preemption is cheap):
#   GPU_ARCHS=N200_L2 PARTITION=gpu_lowp MAX_CONCURRENT=8 ...
#
# Per-arch alpha x beta grid resolution: N200_L2 defaults to the coarser
# 9x9 (see the paper plan); everything else stays 15x15. Override globally
# with N_ALPHA / N_BETA if needed.
GPU_ARCHS="${GPU_ARCHS:-}"
GPU_SCRIPT="${GPU_SCRIPT:-slurm/sweep_big_gpu.sh}"
PARTITION="${PARTITION:-}"           # e.g. a100 | gpu_lowp | gpu (GPU archs only)
GRES="${GRES:-gpu:1}"                # per-task GPU request for GPU archs
GPU_ARCH_SET=" ${GPU_ARCHS} "
is_gpu_arch() { [[ "${GPU_ARCH_SET}" == *" $1 "* ]]; }

# ---------------------------------------------------------------------------
# Per-job --time from measured timings (optional)
# ---------------------------------------------------------------------------
# If TIMES_FILE points at a time_limits.tsv produced by
# `python -m ssb.scripts.aggregate_timing`, each (env,rule,arch) job gets a
# --time sized from its measured slice cost instead of the coarse 24h/8h
# heuristic below. The measured seconds are PADDED by TIME_PAD to absorb
# node-to-node speed variance (slower nodes than the timing host), floored
# at MIN_TIME_S, and optionally capped at MAX_TIME. Combos with no
# measurement (or whose timing errored, so they're absent from the file)
# fall back to the heuristic with a warning. With TIMES_FILE unset the
# behavior is exactly the original heuristic — this feature is opt-in.
TIMES_FILE="${TIMES_FILE:-}"
TIME_PAD="${TIME_PAD:-2.0}"        # multiply measured seconds (2.0 = +100% headroom)
MIN_TIME_S="${MIN_TIME_S:-1800}"   # floor: never request less than 30 min
# Cap on any single job's requested --time. This MUST be <= your SLURM QoS
# wall-clock limit, or the padded request gets rejected (over-cap) — and it
# must be >= your slowest measured slice, or that job is clipped too short
# and killed at the wall. Our QoS cap here is 10-00:00:00 (240 h) and the
# slowest measured slice is ~208 h, so 10-00:00:00 both fits under the QoS
# limit and leaves the 208 h job the full 240 h wall. Override per-cluster.
MAX_TIME="${MAX_TIME:-10-00:00:00}"

# Flat --time limits, used whenever a measured TIMES_FILE estimate is NOT
# available for a job. These are the original proposed limits: small networks
# get SMALL_TIME (8 h), every bigger arch gets BIG_TIME (24 h). Deliberately
# generous — run_sweep writes each slice atomically, so a job killed at the
# wall loses the whole slice; over-requesting wall time is cheaper than that.
SMALL_TIME="${SMALL_TIME:-08:00:00}"
BIG_TIME="${BIG_TIME:-24:00:00}"

declare -A TIME_TABLE
if [[ -n "${TIMES_FILE}" ]]; then
    [[ -f "${TIMES_FILE}" ]] || { echo "TIMES_FILE not found: ${TIMES_FILE}" >&2; exit 1; }
    while IFS=$'\t' read -r t_env t_rule t_arch t_est _t_rec; do
        [[ "${t_env}" == "env" || -z "${t_env}" ]] && continue   # header / blanks
        TIME_TABLE["${t_env}|${t_rule}|${t_arch}"]="${t_est}"
    done < "${TIMES_FILE}"
fi

# Seconds -> SLURM time string (D-HH:MM:SS once >= 24h, else HH:MM:SS).
_secs_to_slurm() {
    local s="${1%.*}" d h m sec r
    d=$(( s / 86400 )); r=$(( s % 86400 ))
    h=$(( r / 3600 )); m=$(( (r % 3600) / 60 )); sec=$(( r % 60 ))
    if (( d > 0 )); then printf '%d-%02d:%02d:%02d' "$d" "$h" "$m" "$sec"
    else printf '%02d:%02d:%02d' "$h" "$m" "$sec"; fi
}

# SLURM time string -> seconds (accepts HH:MM:SS or D-HH:MM:SS).
_slurm_to_secs() {
    local t="$1" d=0 h m s
    [[ "$t" == *-* ]] && { d="${t%%-*}"; t="${t#*-}"; }
    IFS=: read -r h m s <<< "$t"
    echo $(( 10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}

# Resolve the --time for one (env,rule,arch,shards). Echoes the time string on
# stdout; diagnostics go to stderr so they don't pollute the captured value.
resolve_time_limit() {
    local env="$1" rule="$2" arch="$3" shards="${4:-1}" key="$1|$2|$3"
    if [[ -n "${TIMES_FILE}" && -n "${TIME_TABLE[$key]:-}" ]]; then
        local est="${TIME_TABLE[$key]}" padded
        padded=$(awk -v e="$est" -v p="$TIME_PAD" -v m="$MIN_TIME_S" \
            'BEGIN{ v=e*p; if(v<m)v=m; printf "%d", (v==int(v)?v:int(v)+1) }')
        if [[ -n "${MAX_TIME}" ]]; then
            local maxs; maxs=$(_slurm_to_secs "${MAX_TIME}")
            if (( padded > maxs )); then
                padded="${maxs}"
                echo "[time] ${key}: padded estimate exceeds MAX_TIME=${MAX_TIME}, capping" >&2
            fi
        fi
        _secs_to_slurm "${padded}"
        return
    fi
    [[ -n "${TIMES_FILE}" ]] && \
        echo "[time] no measurement for ${key}; using flat ${SMALL_TIME}/${BIG_TIME} limit" >&2

    # Flat heuristic (the original proposed limits): small networks -> SMALL_TIME
    # (8 h), every bigger arch -> BIG_TIME (24 h). Independent of env, rule,
    # TIMESTEPS_SCALE and shard count by design — a generous fixed ceiling that
    # won't clip a slice at the wall. For tighter, per-slice sizing, supply
    # measured timings via TIMES_FILE (handled above).
    if [[ "${arch}" == "small" ]]; then
        echo "${SMALL_TIME}"
    else
        echo "${BIG_TIME}"
    fi
}

# ---------------------------------------------------------------------------

SWEEP_OUTDIR="${SWEEP_OUTDIR:-./sweep_results}"
EMAIL="${EMAIL:-}"
DRY_RUN=0

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) echo "unknown flag: $arg"; exit 1 ;;
    esac
done

mkdir -p "${SWEEP_OUTDIR}" logs

# Helper: is this rule in the tau set?
is_tau_rule() {
    local r="$1"
    for t in "${TAU_RULES[@]}"; do
        [[ "${t}" == "${r}" ]] && return 0
    done
    return 1
}

mail_arg=""
if [[ -n "${EMAIL}" ]]; then
    mail_arg="--mail-user=${EMAIL}"
fi

# Poll interval (seconds) when waiting for running tasks to free budget.
POLL_INTERVAL="${POLL_INTERVAL:-90}"

# Count this user's INCOMPLETE array tasks + jobs (pending + running).
# squeue with -r expands each array task into its own line; -t PD,R keeps
# pending and running tasks. Crucially this counts pending tasks too, so a
# task that has been submitted but not yet started still consumes budget —
# that is what stops the sweep from flooding the queue in a burst. Finished
# tasks drop out of squeue and so free their budget.
count_incomplete() {
    squeue -u "$USER" -h -r -t PD,R -o '%i' 2>/dev/null | wc -l | tr -d ' '
}

# Local accounting of queue depth (pending + running tasks) we've committed
# but not yet seen complete. Every task of an --array is registered PENDING by
# SLURM the instant the array is submitted — the `%cap` throttle limits how
# many RUN, not how many sit in the queue — so to bound the queue we must count
# the full number of tasks each submission enqueues, not just its run-cap. We
# therefore submit oversized arrays in index WINDOWS of at most MAX_CONCURRENT
# tasks (see submit_with_gate) and add each window's true task count here. This
# tally is deterministic and does NOT depend on tasks having already appeared
# in squeue, which is what made the old running-only poll race and overshoot.
COMMITTED=0

# Block until COMMITTED + want <= MAX_CONCURRENT, so enqueuing `want` more
# tasks won't push the queue (pending + running) past the cap. Reconciles the
# local COMMITTED tally down against the live incomplete count — finished tasks
# have left squeue, so a lower live count is real freed budget (never grows the
# tally, which would let squeue lag race us). No-op in dry-run (nothing ever
# finishes there). `label` is only for the wait message.
gate_for() {
    local want="$1" label="$2" first=1 incomplete depth
    [[ "${DRY_RUN}" -eq 1 ]] && return
    while :; do
        incomplete=$(count_incomplete)
        # squeue is the authority on your queue depth, which is what makes the
        # cap global: tasks from a second launcher (e.g. launch_supp.sh running
        # at the same time) count too. Exception: on the first check after we
        # submit, our own tasks may not be visible yet, so we keep the local
        # tally rather than let squeue undercount us. After the first sleep
        # squeue has caught up, so we trust it and the tally decays as tasks
        # finish. Reconciling only DOWNWARD (the previous behaviour) let two
        # concurrent launchers each claim the full cap.
        depth="${incomplete}"
        if [[ "${first}" -eq 1 && "${COMMITTED}" -gt "${incomplete}" ]]; then
            depth="${COMMITTED}"
        fi
        COMMITTED="${depth}"
        [[ $((depth + want)) -le "${MAX_CONCURRENT}" ]] && break
        first=0
        echo "[gate] queue depth=${depth}, want=${want}, cap=${MAX_CONCURRENT}; waiting for tasks to finish before ${label}..."
        sleep "${POLL_INTERVAL}"
    done
}

# ---------------------------------------------------------------------------
# Resume support — skip slices that already have a result on disk
# ---------------------------------------------------------------------------
# RESUME=1 makes the launcher submit only the array indices whose per-slice
# result .npy is MISSING. This makes the whole sweep idempotent and restartable:
# re-run the exact same command after failures / admin holds / a partial run and
# it re-does only the slices that never landed, with no duplicates. run_sweep
# writes each .npy atomically on success, so a held/killed/preempted slice
# leaves NO output and is correctly re-submitted; a completed slice is skipped.
# We assume some tasks will fail, so "done" is decided per-slice from disk, not
# from exit codes. Default RESUME=0 = submit the full array exactly as before.
RESUME="${RESUME:-0}"
# A result counts as "done" only if it exists AND is at least this many bytes,
# so a truncated .npy from a job killed mid-write is treated as not-done and
# re-run. A bare empty-array .npy header is ~128 B; real slices are far bigger.
RESUME_MIN_BYTES="${RESUME_MIN_BYTES:-129}"

# Portable file size in bytes (Linux stat -c / BSD stat -f), 0 if absent.
_file_size() { stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || echo 0; }

# Results .npy that array index $idx of this job will write. MIRRORS the STEM
# logic in the sbatch templates (sweep_big.sh / sweep_big_gpu.sh /
# sweep_small_tau.sh / sweep_small_notau.sh) — keep in sync if those change.
# Args: script env rule arch n_taus n_shards idx
slice_out_file() {
    local scr="$1" env="$2" rule="$3" arch="$4" n_taus="$5" n_shards="$6" idx="$7"
    local stem
    case "${scr}" in
        */sweep_small_notau.sh)   # single job, no array index in the name
            stem="${env}_${rule}_${arch}" ;;
        */sweep_small_tau.sh)     # array index IS the tau index
            stem="${env}_${rule}_${arch}_tau${idx}" ;;
        *)  # big / big_gpu: idx = (eta*n_taus + tau)*n_shards + alpha_shard
            local alpha=$(( idx % n_shards ))
            local rest=$(( idx / n_shards ))
            local eta=$(( rest / n_taus ))
            local tau=$(( rest % n_taus ))
            stem="${env}_${rule}_${arch}_eta${eta}"
            [[ "${n_taus}"  -gt 1 ]] && stem="${stem}_tau${tau}"
            [[ "${n_shards}" -gt 1 ]] && stem="${stem}_a${alpha}"
            ;;
    esac
    printf '%s/results/%s.npy' "${SWEEP_OUTDIR}" "${stem}"
}

# True if this slice's result already exists and looks complete.
# Same args as slice_out_file.
slice_present() {
    local f; f="$(slice_out_file "$@")"
    [[ -f "${f}" ]] || return 1
    [[ "$(_file_size "${f}")" -ge "${RESUME_MIN_BYTES}" ]]
}

# Echo the ascending, space-separated list of array indices to submit for one
# job. Normal mode: all of 0..asize-1. RESUME mode: only indices whose result
# is missing/partial. Args: script env rule arch n_taus n_shards asize
indices_to_submit() {
    local scr="$1" env="$2" rule="$3" arch="$4" n_taus="$5" n_shards="$6" asize="$7"
    local i out=()
    for (( i=0; i<asize; i++ )); do
        if [[ "${RESUME}" -eq 1 ]] \
           && slice_present "${scr}" "${env}" "${rule}" "${arch}" "${n_taus}" "${n_shards}" "${i}"; then
            continue
        fi
        out+=("${i}")
    done
    echo "${out[*]}"
}

# Compress an ascending integer list into a SLURM array spec with ranges,
# e.g. "0 1 2 5 7 8 9" -> "0-2,5,7-9". Keeps a full contiguous window as a
# single "lo-hi", so normal (non-resume) submissions are byte-identical.
compress_indices() {
    local -a a=("$@")
    local out="" start="${a[0]}" prev="${a[0]}" x
    for x in "${a[@]:1}"; do
        if [[ "${x}" -eq $((prev + 1)) ]]; then prev="${x}"; continue; fi
        if [[ "${start}" -eq "${prev}" ]]; then out+="${start},"; else out+="${start}-${prev},"; fi
        start="${x}"; prev="${x}"
    done
    if [[ "${start}" -eq "${prev}" ]]; then out+="${start}"; else out+="${start}-${prev}"; fi
    echo "${out}"
}

# Gate + submit one job so the number of concurrent (pending+running) array
# tasks across the whole sweep never exceeds MAX_CONCURRENT.
# Args: job_name is_array idx_list export_vars script time_limit [partition] [gres]
#   - is_array=0: a non-array job. Waits for one free slot, then submits once
#     (idx_list ignored).
#   - is_array=1: idx_list is an ascending, space-separated list of the array
#     indices to submit (all of them normally; only the missing ones under
#     RESUME). They are submitted in windows of at most MAX_CONCURRENT tasks;
#     before each window we gate on COMMITTED+window so the queue never holds
#     more than MAX_CONCURRENT unfinished tasks — even though SLURM would
#     otherwise enqueue every task of an array at once. Each window is emitted
#     as --array=<compressed indices>%cap; the index VALUES are preserved, so
#     the template decodes the identical (eta,tau,alpha) and writes the same
#     per-slice filenames no matter how the set is chunked.
#   - time_limit overrides the #SBATCH --time directive in the script.
#   - partition/gres (optional) add --partition/--gres for GPU archs; empty
#     -> omitted, so CPU jobs are byte-for-byte identical to before.
submit_with_gate() {
    local jn="$1" is_array="$2" idx_list="$3" evars="$4" scr="$5" tlimit="$6"
    local partition="${7:-}" gres="${8:-}"

    local part_arg="" gres_arg=""
    [[ -n "${partition}" ]] && part_arg="--partition=${partition}"
    [[ -n "${gres}" ]] && gres_arg="--gres=${gres}"

    # Non-array job: wants exactly one slot.
    if [[ "${is_array}" -eq 0 ]]; then
        gate_for 1 "${jn}"
        local cmd=(sbatch
                   --job-name="${jn}"
                   --time="${tlimit}"
                   ${mail_arg}
                   ${part_arg}
                   ${gres_arg}
                   --export="${evars}"
                   "${scr}")
        if [[ "${DRY_RUN}" -eq 1 ]]; then
            echo "${cmd[@]}"
        else
            "${cmd[@]}"
        fi
        COMMITTED=$((COMMITTED + 1))
        return
    fi

    # Array job: walk the index list in windows of <= MAX_CONCURRENT tasks.
    # Each window is gated so the whole queue stays <= MAX_CONCURRENT; because
    # a window's size never exceeds the cap and we gate COMMITTED+wsize before
    # it, every task in the window may RUN (cap = wsize) without over-
    # subscribing. A strict queue bound means a window can't enqueue until
    # prior tasks drain, so with mixed task lengths the slowest task in a
    # window gates the next one — bump MAX_CONCURRENT for more throughput.
    local -a idxs
    read -ra idxs <<< "${idx_list}"
    local total="${#idxs[@]}" off=0 wsize cap spec
    while [[ "${off}" -lt "${total}" ]]; do
        wsize=$(( total - off ))
        [[ "${wsize}" -gt "${MAX_CONCURRENT}" ]] && wsize="${MAX_CONCURRENT}"
        spec="$(compress_indices "${idxs[@]:off:wsize}")"

        gate_for "${wsize}" "${jn}[${spec}]"

        cap="${wsize}"
        local cmd=(sbatch
                   --job-name="${jn}"
                   --time="${tlimit}"
                   ${mail_arg}
                   ${part_arg}
                   ${gres_arg}
                   "--array=${spec}%${cap}"
                   --export="${evars}"
                   "${scr}")

        if [[ "${DRY_RUN}" -eq 1 ]]; then
            echo "${cmd[@]}"
        else
            "${cmd[@]}"
        fi

        # Every task in this window is now enqueued (pending) — count them all.
        COMMITTED=$((COMMITTED + wsize))
        off=$(( off + wsize ))
    done
}

echo "=== launch_array ==="
echo "outdir : ${SWEEP_OUTDIR}"
echo "rules  : ${ALL_RULES[*]}"
echo "envs   : ${ALL_ENVS[*]}"
echo "ts scale: ${TIMESTEPS_SCALE}x (converged timesteps = baseline x scale)"
echo "archs  : ${ALL_ARCHS[*]}"
if [[ "${ALPHA_SHARDS}" -gt 1 ]]; then
    echo "alpha  : sharded ${ALPHA_SHARDS}-way (big-style CPU jobs; array size x${ALPHA_SHARDS})"
else
    echo "alpha  : not sharded (ALPHA_SHARDS=1)"
fi
if [[ -n "${GPU_ARCHS}" ]]; then
    echo "gpu    : archs=[${GPU_ARCHS}] script=${GPU_SCRIPT} partition=${PARTITION:-<none>} gres=${GRES}"
else
    echo "gpu    : (none — all CPU)"
fi
echo "max concurrent (global, pending+running) : ${MAX_CONCURRENT}"
if [[ "${RESUME}" -eq 1 ]]; then
    echo "resume : ON — submitting only slices with no result .npy in ${SWEEP_OUTDIR}/results (>= ${RESUME_MIN_BYTES}B)"
else
    echo "resume : off (RESUME=1 to submit only missing slices)"
fi
if [[ -n "${TIMES_FILE}" ]]; then
    echo "times  : ${TIMES_FILE} (${#TIME_TABLE[@]} combos)  pad=${TIME_PAD}  min=${MIN_TIME_S}s  max=${MAX_TIME:-none}"
else
    echo "times  : flat limits — small=${SMALL_TIME}, bigger archs=${BIG_TIME} (set TIMES_FILE for measured per-slice timings)"
fi
echo "dry-run: ${DRY_RUN}"
echo "===================="

n_jobs=0
for env_spec in "${ALL_ENVS[@]}"; do
    env_name="${env_spec%%:*}"
    timesteps="${env_spec##*:}"
    # Apply the converged-timestep scale (integer floor).
    if [[ "${TIMESTEPS_SCALE}" != "1" ]]; then
        timesteps=$(awk -v t="${timesteps}" -v s="${TIMESTEPS_SCALE}" \
            'BEGIN{ printf "%d", int(t*s) }')
    fi
    for rule in "${ALL_RULES[@]}"; do
        for arch in "${ALL_ARCHS[@]}"; do

            tau_rule=0
            is_tau_rule "${rule}" && tau_rule=1

            # Resource class: 'small' is the only literally-small arch.
            # Everything else (big, N*_L*, etc.) goes through the big-resource
            # templates with eta/tau array splitting.
            if [[ "${arch}" == "small" ]]; then
                resource_class="small"
            else
                resource_class="big"
            fi

            # Choose sbatch template + array spec based on (env,resource,tau-rule).
            # For det env we always use big-style eta/tau splits, even for small networks.
            # array_size = number of tasks in this submission (1 = non-array job).
            # job_n_taus is the integer N_TAUS this job's array index is encoded
            # with — needed by the resume check to reproduce each slice's
            # filename (see slice_out_file). It mirrors n_taus_env; the small
            # templates don't use it (small_notau is non-array; small_tau's
            # index IS the tau), so it's set to 1 there.
            if [[ "${env_name}" == "det" ]]; then
                script="slurm/sweep_big.sh"
                if [[ "${tau_rule}" -eq 0 ]]; then
                    array_size=$((N_ETAS))
                    n_taus_env=",N_TAUS=1"
                    job_n_taus=1
                else
                    array_size=$((N_ETAS * N_TAUS))
                    n_taus_env=",N_TAUS=${N_TAUS}"
                    job_n_taus="${N_TAUS}"
                fi
            elif [[ "${resource_class}" == "small" && "${tau_rule}" -eq 0 ]]; then
                script="slurm/sweep_small_notau.sh"
                array_size=1
                n_taus_env=""
                job_n_taus=1
            elif [[ "${resource_class}" == "small" && "${tau_rule}" -eq 1 ]]; then
                script="slurm/sweep_small_tau.sh"
                array_size=$((N_TAUS))
                n_taus_env=""
                job_n_taus=1
            elif [[ "${resource_class}" == "big" && "${tau_rule}" -eq 0 ]]; then
                script="slurm/sweep_big.sh"
                array_size=$((N_ETAS))
                n_taus_env=",N_TAUS=1"
                job_n_taus=1
            else
                # big, tau rule
                script="slurm/sweep_big.sh"
                array_size=$((N_ETAS * N_TAUS))
                n_taus_env=",N_TAUS=${N_TAUS}"
                job_n_taus="${N_TAUS}"
            fi

            job_name="ssb_${env_name}_${rule}_${arch}"
            export_vars="SWEEP_ENV=${env_name},SWEEP_RULE=${rule},SWEEP_ARCH=${arch},SWEEP_TIMESTEPS=${timesteps},SWEEP_OUTDIR=${SWEEP_OUTDIR}${n_taus_env}"

            # GPU routing (opt-in). If this arch is in GPU_ARCHS, swap to the
            # GPU template, attach --partition/--gres, and inject the alpha x
            # beta grid resolution (N200_L2 -> 9x9 by default, else 15x15).
            # CPU archs leave part_arg/gres_arg empty -> unchanged behavior.
            job_partition="" job_gres=""
            if is_gpu_arch "${arch}"; then
                script="${GPU_SCRIPT}"
                job_partition="${PARTITION}"
                job_gres="${GRES}"
                if [[ "${arch}" == "N200_L2" ]]; then
                    na="${N_ALPHA:-9}"; no="${N_BETA:-9}"
                else
                    na="${N_ALPHA:-15}"; no="${N_BETA:-15}"
                fi
                export_vars="${export_vars},N_ALPHA=${na},N_BETA=${no}"
            fi

            # Alpha grid-sharding. Applies ONLY to the CPU big-style template
            # (sweep_big.sh); small templates and the GPU template are left
            # untouched. Grows the array K-fold and passes N_ALPHA_SHARDS so the
            # template decodes an alpha_idx from the flat array index and tags
            # the output filename with _a{A}. With ALPHA_SHARDS=1 this whole
            # block is skipped, so array sizes/filenames/commands are identical
            # to before.
            job_shards=1
            if [[ "${script}" == "slurm/sweep_big.sh" && "${ALPHA_SHARDS}" -gt 1 ]]; then
                array_size=$(( array_size * ALPHA_SHARDS ))
                export_vars="${export_vars},N_ALPHA_SHARDS=${ALPHA_SHARDS}"
                job_shards="${ALPHA_SHARDS}"
            fi

            # Per-job --time: from measured timings (TIMES_FILE) with TIME_PAD
            # headroom, else the buffered model estimate (accounts for arch,
            # env, TIMESTEPS_SCALE and shard count). See resolve_time_limit.
            time_limit="$(resolve_time_limit "${env_name}" "${rule}" "${arch}" "${job_shards}")"

            # Decide array-ness and which indices to submit. Under RESUME the
            # index list drops slices whose result .npy already exists, so this
            # command is idempotent: nothing already done is re-submitted, and
            # a fully-complete job is skipped entirely.
            if [[ "${array_size}" -le 1 ]]; then
                if [[ "${RESUME}" -eq 1 ]] \
                   && slice_present "${script}" "${env_name}" "${rule}" "${arch}" "${job_n_taus}" "${job_shards}" 0; then
                    echo "[resume] skip ${job_name} (result present)"
                    continue
                fi
                submit_with_gate "${job_name}" 0 "" "${export_vars}" "${script}" "${time_limit}" "${job_partition}" "${job_gres}"
            else
                idx_list="$(indices_to_submit "${script}" "${env_name}" "${rule}" "${arch}" "${job_n_taus}" "${job_shards}" "${array_size}")"
                if [[ -z "${idx_list}" ]]; then
                    echo "[resume] skip ${job_name} (all ${array_size} slices present)"
                    continue
                fi
                if [[ "${RESUME}" -eq 1 ]]; then
                    n_missing=$(wc -w <<< "${idx_list}")
                    echo "[resume] ${job_name}: submitting ${n_missing}/${array_size} missing slices"
                fi
                submit_with_gate "${job_name}" 1 "${idx_list}" "${export_vars}" "${script}" "${time_limit}" "${job_partition}" "${job_gres}"
            fi
            n_jobs=$((n_jobs + 1))
        done
    done
done

echo ""
echo "submitted ${n_jobs} sbatch invocations (or printed, if --dry-run)."
echo "individual array tasks: see formula in script header."
echo "monitor   : squeue -u \$USER"
echo "cancel all: squeue -u \$USER -h -o '%i %j' | awk '\$2 ~ /^ssb_/ {print \$1}' | xargs -r scancel"
echo "            (all jobs in this sweep share the 'ssb_' name prefix)"