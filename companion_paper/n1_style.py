"""House style shared by the Note 1 figures (matches f6ii_mag.ipynb)."""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings

from matplotlib import font_manager as _fm
_have_arial = any(f.name == 'Arial' for f in _fm.fontManager.ttflist)
mpl.rcParams['font.family'] = 'Arial' if _have_arial else 'Liberation Sans'   # metric-identical to Arial
mpl.rcParams['svg.fonttype'] = 'path'
warnings.filterwarnings('ignore', message='.*not compatible with tight_layout.*')
warnings.filterwarnings('ignore', message='.*findfont.*')

REWARDS = [0.2, 0.3, 0.5, 0.7, 1.0]
REWARD_CMAP = plt.cm.Blues


def reward_color_abs(r, lo=0.2, hi=1.0):
    return REWARD_CMAP(0.35 + 0.55 * (float(r) - lo) / (hi - lo))


def nature_ax(ax, xlabel='', ylabel='', nticks=4, nx=None, ny=None):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_linewidth(0.25)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nx or nticks))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(ny or nticks))
    ax.tick_params(labelsize=7, width=0.25)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)


def side_legend(ax, title=None, **kw):
    ax.legend(fontsize=7, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0),
              title=title, title_fontsize=7, handlelength=1.2, **kw)


def kfmt(ax):
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k' if x else '0'))
