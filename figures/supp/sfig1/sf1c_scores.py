"""Supplementary Fig. 1c, scores column: each rule's probability-matching index closest to 0
(small network; top baited, bottom unbaited). Grey band: |S_match| <= 0.35."""
from pathlib import Path
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.ticker as mt
mpl.rcParams['svg.fonttype']='none'; mpl.rcParams['font.family']='DejaVu Sans'
for f in ['Arial','Liberation Sans']:
    try:
        from matplotlib import font_manager as fm; fm.findfont(f, fallback_to_default=False); mpl.rcParams['font.family']=f; break
    except Exception: pass
mpl.rcParams['xtick.direction']='in'; mpl.rcParams['ytick.direction']='in'
ORDER=['HR','PC','BCM','OJA','LC','COV','HZ']; LAB={'HR':'RPE','PC':'PC','BCM':'BCM','OJA':'OJA','LC':'LC','COV':'COV','HZ':'LPL'}
COL={'HR':'#7cba82','PC':'#7189ad','BCM':'#393e96','OJA':'#ea8d50','LC':'#8874b0','COV':'#b8506d','HZ':'#dea43c'}
CUT=0.35   # matching criterion |S_match| <= 0.35 (Methods)
HERE=Path(__file__).resolve().parent
SWEEPS=HERE.parents[2]/'data'/'sweeps'
def best(env,r):
    a=np.load(SWEEPS/f'bandit_{env}_{r}_small.npy')
    if env=='unbait': a=a[...,0]
    f=a.reshape(-1); return float(f[np.nanargmin(np.abs(f))])
fig,axs=plt.subplots(2,1,figsize=(1.55,2.6))
for ax,env,t in zip(axs,['bait','unbait'],['Baited','Unbaited']):
    y=[best(env,r) for r in ORDER]; x=np.arange(7)
    ax.axhspan(-CUT,CUT,color='0.92',lw=0,zorder=0)
    ax.axhline(0,color='0.45',lw=0.4,ls=':',zorder=1)
    ax.scatter(x,y,s=16,c=[COL[r] for r in ORDER],edgecolor='none',zorder=3)
    for sp in ('top','right'): ax.spines[sp].set_visible(False)
    for sp in ('left','bottom'): ax.spines[sp].set_linewidth(0.4)
    ax.set_ylim(-1,0.6); ax.set_yticks([-1,-0.5,0,0.5]); ax.tick_params(labelsize=5.5,width=0.4,length=2)
    ax.set_xlim(-0.6,6.6); ax.set_xticks(x); ax.set_xticklabels([LAB[r] for r in ORDER],fontsize=5,fontweight='bold',rotation=45,ha='right'); ax.tick_params(axis='x',length=0)
    ax.set_ylabel('PM index\n(closest to 0)',fontsize=6)
    print(t,{LAB[r]:round(v,2) for r,v in zip(ORDER,y)})
fig.tight_layout(); fig.savefig(HERE/'sf1c_scores.svg')
