#!/bin/bash
# Launcher for the supplementary sweeps (reward magnitude, continual learning,
# architecture scan). Companion to launch_array.sh, which drives the master
# (env, rule, arch) parameter sweep; this one drives the cell-sharded sweeps.
#
# It assumes jobs fail. Everything here is built around that: arrays are
# submitted in bounded windows, the queue depth is capped, results are written
# atomically by the drivers, and RESUME=1 decides what is done by looking at
# the files on disk rather than at job exit codes. Re-running the identical
# command after a partial run is always safe and never duplicates work.
#
# Usage
#   bash slurm/launch_supp.sh mag  --dry-run
#   bash slurm/launch_supp.sh cl
#   bash slurm/launch_supp.sh arch --dry-run
#
#   KIND is the first positional argument:
#     mag    reward-magnitude sweep over manifest cells   (small + big)
#     cl     continual-learning sweep over manifest cells (small + big)
#     forget forgetting/interference sweep over manifest cells (small + big):
#            learn the det task, then probe under no-input / constant / reversal
#     arch   reward-magnitude sweep over the FULL grid    (N2_L2, N10_L2, ...)
#
# Common overrides
#   RULES="PC HR"            subset of rules
#   ARCHS="small"            subset of architectures
#   MANIFEST=path.csv        cell manifest (default sweep_results/supp/cells.csv)
#   SUPP_OUTDIR=./out        results root
#   MAX_CONCURRENT=50        cap on queued (pending+running) array tasks
#   CELLS_PER_TASK=25        cells per array task
#   RESUME=1                 submit only shards whose .npz is missing
#   STAGE=s2                 stage tag in filenames (two-stage censoring rerun)
#   TIMESTEPS=20000          per-run budget (stage 1 short, stage 2 long)
#   EMAIL=you@example.org    failure notifications
#
# Two-stage magnitude sweep (see slurm/README_supp.md):
#   bash slurm/launch_supp.sh mag                       # stage 1 @ 20k
#   python -m ssb.analysis.aggregate_supp --stage s1 ... --emit-rerun rerun.csv
#   MANIFEST=rerun.csv STAGE=s2 TIMESTEPS=100000 bash slurm/launch_supp.sh mag

set -euo pipefail

KIND="${1:-}"
shift || true
case "${KIND}" in
    mag|cl|forget|arch) ;;
    *) echo "usage: bash slurm/launch_supp.sh {mag|cl|forget|arch} [--dry-run]" >&2; exit 1 ;;
esac

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) echo "unknown flag: $arg" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ALL_RULES=(PC HR OJA COV BCM LC HZ)
[[ -n "${RULES:-}" ]] && read -ra ALL_RULES <<< "${RULES}"

# Default architectures per kind. mag/cl run the two sizes the manifest covers;
# arch runs the registry scan, skipping N50_L2 (identical to `big`, already
# measured — see slurm/HANDOFF.md).
if [[ "${KIND}" == "arch" ]]; then
    DEFAULT_ARCHS=(N2_L2 N10_L2 N50_L3 N200_L2)
else
    DEFAULT_ARCHS=(small big)
fi
ALL_ARCHS=("${DEFAULT_ARCHS[@]}")
[[ -n "${ARCHS:-}" ]] && read -ra ALL_ARCHS <<< "${ARCHS}"

SUPP_OUTDIR="${SUPP_OUTDIR:-./sweep_results/supp}"
MANIFEST="${MANIFEST:-${SUPP_OUTDIR}/cells.csv}"
STAGE="${STAGE:-s1}"
SEEDS="${SEEDS:-20}"
CELLS_PER_TASK="${CELLS_PER_TASK:-25}"
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"
POLL_INTERVAL="${POLL_INTERVAL:-90}"
RESUME="${RESUME:-0}"
RESUME_MIN_BYTES="${RESUME_MIN_BYTES:-500}"
EMAIL="${EMAIL:-}"
SCRIPT="slurm/sweep_supp.sh"

# Timestep budgets. Stage 1 of the magnitude sweep runs short and re-runs only
# the censored cells long (see README_supp.md); the CL task's budget is per
# reversal block, not total.
if [[ "${KIND}" == "cl" ]]; then
    TIMESTEPS="${TIMESTEPS:-10000}"
    NUM_REVS="${NUM_REVS:-10}"
elif [[ "${KIND}" == "forget" ]]; then
    # per phase (learn, then probe); the driver runs this budget twice
    TIMESTEPS="${TIMESTEPS:-50000}"
    NUM_REVS=""
else
    TIMESTEPS="${TIMESTEPS:-20000}"
    NUM_REVS=""
fi
# Reward levels, COLON-separated. Not commas: sbatch --export takes a
# comma-separated list of assignments, so a value containing commas is split
# by SLURM and the job receives only the first fragment (e.g. "[0.2"), which
# then fails to parse. Colons pass through untouched.
REWARDS="${REWARDS:-0.2:0.4:0.6:0.8:1.0}"
if [[ "${REWARDS}" == *,* ]]; then
    echo "REWARDS must be colon-separated, not comma-separated (sbatch --export" >&2
    echo "would split on the commas). Got: ${REWARDS}" >&2
    echo "Use e.g. REWARDS=0.2:0.6:1.0" >&2
    exit 1
fi

# Per-arch wall clock. Deliberately generous: the drivers write each shard
# atomically, so a job killed at the wall loses that shard entirely — over-
# requesting time is far cheaper than re-running.
SMALL_TIME="${SMALL_TIME:-04:00:00}"
BIG_TIME="${BIG_TIME:-24:00:00}"

# Grid resolution for the architecture sweep. N200_L2 uses the coarser 9x9
# documented in slurm/GPU_CPU_PLAN.md; everything else uses the full 15x15.
N_ALPHA_DEFAULT="${N_ALPHA:-15}"
N_BETA_DEFAULT="${N_BETA:-15}"

mkdir -p "${SUPP_OUTDIR}/results" "${SUPP_OUTDIR}/configs" logs

TAU_RULES=(COV BCM LC HZ)
is_tau_rule() { local r="$1"; for t in "${TAU_RULES[@]}"; do [[ "${t}" == "${r}" ]] && return 0; done; return 1; }

resolve_time() { [[ "$1" == "small" ]] && echo "${SMALL_TIME}" || echo "${BIG_TIME}"; }

# Cells this (rule, arch) will run. For manifest kinds we count the manifest's
# rows; for the arch sweep we compute the grid size from the rule's tau-ness and
# the grid resolution. Counting from the manifest (rather than assuming the
# budget) matters because rules whose seeking region is smaller than the budget
# have fewer cells — OJA/small has 67, not 100.
count_cells() {
    local rule="$1" arch="$2"
    if [[ "${KIND}" == "arch" ]]; then
        local na nb ntau
        na="${N_ALPHA_DEFAULT}"; nb="${N_BETA_DEFAULT}"
        [[ "${arch}" == "N200_L2" && -z "${N_ALPHA:-}" ]] && { na=9; nb=9; }
        ntau=1; is_tau_rule "${rule}" && ntau=4
        echo $(( 6 * ntau * na * nb ))
    else
        [[ -f "${MANIFEST}" ]] || { echo "manifest not found: ${MANIFEST}" >&2; exit 1; }
        awk -F, -v r="${rule}" -v a="${arch}" 'NR>1 && $2==r && $3==a {n++} END{print n+0}' "${MANIFEST}"
    fi
}

# ---------------------------------------------------------------------------
# Queue gating — identical strategy to launch_array.sh
# ---------------------------------------------------------------------------
# SLURM enqueues EVERY task of an --array as PENDING the moment the array is
# submitted (the %cap throttle limits how many RUN, not how many are queued), so
# bounding the queue means submitting oversized arrays in index windows of at
# most MAX_CONCURRENT tasks and gating between windows. COMMITTED is a local
# tally of tasks we have enqueued but not yet seen finish; it is reconciled
# downward against the live squeue count, never upward (squeue lag must not be
# able to inflate our budget).
COMMITTED=0

count_incomplete() {
    squeue -u "$USER" -h -r -t PD,R -o '%i' 2>/dev/null | wc -l | tr -d ' '
}

gate_for() {
    local want="$1" label="$2" first=1 incomplete depth
    [[ "${DRY_RUN}" -eq 1 ]] && return
    while :; do
        incomplete=$(count_incomplete)
        # squeue is the authority on how many of YOUR tasks are queued, which
        # is what makes the cap global: tasks from a second launcher (or any
        # other job of yours) count too. The one exception is the first check
        # after we submit, when our own tasks may not have appeared yet -- there
        # we keep our local tally so squeue cannot undercount us. After the
        # first sleep squeue has caught up, so we trust it and the tally decays
        # correctly as tasks finish.
        depth="${incomplete}"
        if [[ "${first}" -eq 1 && "${COMMITTED}" -gt "${incomplete}" ]]; then
            depth="${COMMITTED}"
        fi
        COMMITTED="${depth}"
        [[ $((depth + want)) -le "${MAX_CONCURRENT}" ]] && break
        first=0
        echo "[gate] queue depth=${depth}, want=${want}, cap=${MAX_CONCURRENT}; waiting before ${label}..."
        sleep "${POLL_INTERVAL}"
    done
}

_file_size() { stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || echo 0; }

# Output file array index $idx will write. MIRRORS the STEM logic in
# sweep_supp.sh — keep the two in sync.
shard_out_file() {
    local rule="$1" arch="$2" idx="$3"
    printf '%s/results/%s_%s_%s_%s_c%d.npz' \
        "${SUPP_OUTDIR}" "${KIND_TAG}" "${rule}" "${arch}" "${STAGE}" \
        $(( idx * CELLS_PER_TASK ))
}

shard_present() {
    local f; f="$(shard_out_file "$@")"
    [[ -f "${f}" ]] || return 1
    [[ "$(_file_size "${f}")" -ge "${RESUME_MIN_BYTES}" ]]
}

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

mail_arg=""
[[ -n "${EMAIL}" ]] && mail_arg="--mail-user=${EMAIL}"

TOTAL_TASKS=0
TOTAL_JOBS=0
SKIPPED=0

submit_windows() {
    local jn="$1" idx_list="$2" evars="$3" tlimit="$4"
    local -a idx=(${idx_list})
    [[ "${#idx[@]}" -eq 0 ]] && return

    local i=0
    while [[ "${i}" -lt "${#idx[@]}" ]]; do
        local -a win=("${idx[@]:i:MAX_CONCURRENT}")
        local n="${#win[@]}"
        gate_for "${n}" "${jn}"
        local spec; spec="$(compress_indices "${win[@]}")"
        local cmd=(sbatch --job-name="${jn}" --array="${spec}%${MAX_CONCURRENT}"
                   --time="${tlimit}" --export="ALL,${evars}")
        [[ -n "${mail_arg}" ]] && cmd+=("${mail_arg}")
        cmd+=("${SCRIPT}")
        if [[ "${DRY_RUN}" -eq 1 ]]; then
            printf '%q ' "${cmd[@]}"; echo
        else
            "${cmd[@]}"
            COMMITTED=$((COMMITTED + n))
        fi
        TOTAL_TASKS=$((TOTAL_TASKS + n))
        TOTAL_JOBS=$((TOTAL_JOBS + 1))
        i=$((i + MAX_CONCURRENT))
    done
}

# Filename tag: the arch sweep is a magnitude sweep, so its driver and its
# output schema are the mag ones; only the cell source differs.
KIND_TAG="mag"; [[ "${KIND}" == "cl" ]] && KIND_TAG="cl"; [[ "${KIND}" == "forget" ]] && KIND_TAG="forget"
DRIVER_KIND="mag"; [[ "${KIND}" == "cl" ]] && DRIVER_KIND="cl"; [[ "${KIND}" == "forget" ]] && DRIVER_KIND="forget"

echo "=== supplementary sweep: ${KIND} (stage ${STAGE}) ==="
echo "rules      : ${ALL_RULES[*]}"
echo "archs      : ${ALL_ARCHS[*]}"
echo "manifest   : $([[ "${KIND}" == "arch" ]] && echo '<full grid>' || echo "${MANIFEST}")"
echo "outdir     : ${SUPP_OUTDIR}"
echo "cells/task : ${CELLS_PER_TASK}"
echo "timesteps  : ${TIMESTEPS}"
echo "max concur : ${MAX_CONCURRENT}"
echo "resume     : ${RESUME}"
echo "dry-run    : ${DRY_RUN}"
echo

for arch in "${ALL_ARCHS[@]}"; do
    for rule in "${ALL_RULES[@]}"; do
        n_cells="$(count_cells "${rule}" "${arch}")"
        if [[ "${n_cells}" -eq 0 ]]; then
            echo "[skip] ${rule}/${arch}: no cells"
            continue
        fi
        n_tasks=$(( (n_cells + CELLS_PER_TASK - 1) / CELLS_PER_TASK ))

        # RESUME: keep only shards whose output is missing or truncated.
        idx=()
        for (( t=0; t<n_tasks; t++ )); do
            if [[ "${RESUME}" -eq 1 ]] && shard_present "${rule}" "${arch}" "${t}"; then
                SKIPPED=$((SKIPPED + 1)); continue
            fi
            idx+=("${t}")
        done
        if [[ "${#idx[@]}" -eq 0 ]]; then
            echo "[done] ${rule}/${arch}: all ${n_tasks} shards present"
            continue
        fi

        evars="SUPP_KIND=${DRIVER_KIND},SUPP_RULE=${rule},SUPP_ARCH=${arch}"
        evars+=",SUPP_OUTDIR=${SUPP_OUTDIR},SUPP_TIMESTEPS=${TIMESTEPS}"
        evars+=",CELLS_PER_TASK=${CELLS_PER_TASK},SUPP_STAGE=${STAGE},SUPP_SEEDS=${SEEDS}"
        [[ "${KIND}" != "arch" ]] && evars+=",SUPP_MANIFEST=${MANIFEST}"
        [[ "${DRIVER_KIND}" == "mag" ]] && evars+=",SUPP_REWARDS=${REWARDS}"
        # The arch sweep enumerates the grid inside the job, so the job MUST be
        # told the same resolution the launcher used to size the array. Without
        # this the launcher sizes N200_L2 for 9x9 while the job enumerates the
        # default 15x15, and the shards silently cover only a prefix of the
        # cell list (eta 0-2) instead of the intended coarse grid.
        if [[ "${KIND}" == "arch" ]]; then
            g_na="${N_ALPHA_DEFAULT}"; g_nb="${N_BETA_DEFAULT}"
            [[ "${arch}" == "N200_L2" && -z "${N_ALPHA:-}" ]] && { g_na=9; g_nb=9; }
            evars+=",SUPP_N_ALPHA=${g_na},SUPP_N_BETA=${g_nb}"
        fi
        [[ -n "${NUM_REVS}" ]] && evars+=",SUPP_NUM_REVS=${NUM_REVS}"

        echo "[submit] ${rule}/${arch}: ${n_cells} cells -> ${#idx[@]}/${n_tasks} shards"
        submit_windows "ssb_${KIND}_${rule}_${arch}" "${idx[*]}" "${evars}" "$(resolve_time "${arch}")"
    done
done

echo
echo "=== summary ==="
echo "sbatch invocations : ${TOTAL_JOBS}"
echo "array tasks        : ${TOTAL_TASKS}"
echo "shards skipped     : ${SKIPPED} (already present)"
[[ "${DRY_RUN}" -eq 1 ]] && echo "(dry run — nothing submitted)"
