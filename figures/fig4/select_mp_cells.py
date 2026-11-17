"""
Pick one matching-pennies cell per rule from the settled-behaviour sweeps (small network).

Input : data/sweeps/matchingpennies_steady_<RULE>_<arch>.npz  (from code_release/ssb/analysis/aggregate_mp_steady.py)
        scores shape (eta, tau, alpha, beta, phase, metric, seed); metrics: entropy, bias, switch, win;
        phases: computer, then 3 x [coin flip, computer]  (7 phases, 0-based 0..6).
Rule  : keep cells that settled (repeat phases differ by < 0.1 bits in median entropy: |H3-H5| and |H2-H6|,
        0-based) and are unbiased vs the computer (median side bias over phases 2,4,6 < 0.05); among those,
        take the largest drop in median entropy, competitive (2,4,6) minus coin flip (3,5).
Output: printed table; paste the cells into CELLS in figures/fig4/f4_matchingpennies.ipynb.
Usage : python select_mp_cells.py [small|big]
"""
import sys
from pathlib import Path
import numpy as np
ARCH = sys.argv[1] if len(sys.argv) > 1 else 'small'
SWEEPS = Path(__file__).resolve().parents[2] / 'data' / 'sweeps'   # code_release/data/sweeps
COMP, COIN = [2, 4, 6], [3, 5]
print(f'{"rule":4s} {"eta":>7s} {"tau":>7s} {"alpha":>9s} {"beta":>9s} {"Hc":>5s} {"Hr":>5s} {"drop":>5s} {"Bc":>5s} {"Br":>5s} {"win":>5s} {"Pswitch":>7s}  settled&unbiased')
for rule in ['BCM', 'OJA', 'LC', 'PC', 'COV', 'HZ', 'HR']:
    f = SWEEPS / f'matchingpennies_steady_{rule}_{ARCH}.npz'
    if not f.exists(): print(f'{rule:4s} (missing {f.name})'); continue
    z = np.load(f); S = z['scores']
    med = lambda ps, k: np.nanmedian(S[..., ps, k, :].reshape(S.shape[:4] + (-1,)), -1)
    Hc, Hr, Bc, Br, Wc, SWc = med(COMP, 0), med(COIN, 0), med(COMP, 1), med(COIN, 1), med(COMP, 3), med(COMP, 2)
    drift = np.maximum(abs(med([3], 0) - med([5], 0)), abs(med([2], 0) - med([6], 0)))
    ok = (drift < 0.1) & (Bc < 0.05)
    score = np.where(ok, Hc - Hr, -np.inf); i = np.unravel_index(np.argmax(score), score.shape)
    print(f'{rule:4s} {z["etas"][i[0]]:7.0e} {z["taus"][i[1]]:7.3g} {z["alphas"][i[2]]:9.4g} {z["betas"][i[3]]:9.4g} '
          f'{Hc[i]:5.2f} {Hr[i]:5.2f} {Hc[i]-Hr[i]:5.2f} {Bc[i]:5.2f} {Br[i]:5.2f} {Wc[i]:5.2f} {SWc[i]:7.3f}  {ok.sum()}/{ok.size}')
