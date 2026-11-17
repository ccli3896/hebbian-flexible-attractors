# Steady-state matching-pennies sweep (per-step task)

Replaces the 3-phase `matchingpennies` sweep in `run_sweep.py`, whose 20k-step phases mostly
measured the decay of the *initial* weights (e.g. PC at the old SFig 1d cell: the coin-flip bias is
left over from initialization and never returns on later cycles).

**Protocol** — per cell, network state carried across phases (as in `run_sweep.py`):
computer opponent, then 3 × (coin flip, computer) = 7 phases of 20,000 steps; 20 seeds;
same α × β × η (× τ) grid. Scored per seed over the second half of every phase:
3-bit entropy, side bias |P(1) − 0.5|, P(switch), win rate. Summaries should use phases ≥ 2
(after the first cycle) to measure settled behaviour.

## Files
| File | Purpose |
|---|---|
| `ssb/scripts/run_mp_steady.py` | Sweep driver (one slice per call; skips if output exists; atomic write) |
| `slurm/sweep_mp_steady.sh` | sbatch template (array index → eta, tau, alpha shard) |
| `slurm/launch_mp_steady.sh` | Submits every rule × arch |
| `ssb/analysis/aggregate_mp_steady.py` | Stitches slices into `data/sweeps/matchingpennies_steady_<rule>_<arch>.npz` and prints top cells |

## Run (from the repository root, e.g. ~/local-learning-2025, venv at ~/venvs/ssb as for the main sweep)
```bash
# 1. smoke test on the login node (~1 min): tiny run, then delete it
echo '{"rule":"PC","arch":"small","timesteps":300,"n_cycles":1}' > /tmp/c.json
python -m ssb.scripts.run_mp_steady --config /tmp/c.json --out /tmp/t.npz --seeds 2 --eta-idx 0

# 2. one combination on the cluster first, to check run time
RULES=PC ARCHS=small bash slurm/launch_mp_steady.sh --dry-run
RULES=PC ARCHS=small bash slurm/launch_mp_steady.sh

# 3. everything (small first, then big)
ARCHS=small bash slurm/launch_mp_steady.sh
ARCHS=big   bash slurm/launch_mp_steady.sh

# 4. re-run the same command to fill any failed slices (finished slices exit immediately)

# 5. aggregate
python -m ssb.analysis.aggregate_mp_steady --indir sweep_mp_steady/results --outdir data/sweeps
```
Job counts: small 114 array tasks (6 per PC/HR/OJA, 24 per COV/BCM/LC/HZ); big 1,710 tasks
(15 alpha shards each). Each array runs at most `MAX_RUNNING` (default 50) tasks at once; if the
cluster limits submitted jobs per user, launch one `RULES=...` at a time.
Knobs: `MP_TIMESTEPS` (steps per phase), `MP_CYCLES`, `MP_SEEDS`, `MP_OUTDIR`, `SMALL_TIME`, `BIG_TIME`,
`BIG_ALPHA_SHARDS`, `MAX_RUNNING`.

Check: PC, small, α = 0.0164, β = 1, η = 1e-5 with `--seed 235 --seeds 50` reproduces the
`mp_PC_attractors_sequence_altcell` simulation (median entropy 1.83 bits in the computer phases,
0.89–0.91 in the coin-flip phases; win rate 0.14).
