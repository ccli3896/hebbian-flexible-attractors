"""Power-law and heavy-tail analysis for action-length distributions.

Wraps the Clauset-Shalizi-Newman (CSN) protocol via the ``powerlaw``
package. The central use in this project is *distinguishing* heavy-tailed
(open-field foraging) from exponential (lab dynamic foraging) action-length
distributions, and showing the model reproduces that dissociation.

The key function is :func:`fit_and_compare`, which fits both a power law
and an exponential and reports which is favored. A single "is it a power
law" yes/no is not enough — the interesting claim is the *contrast*
between regimes, so we always fit both candidate distributions and compare.

References
----------
Clauset, Shalizi & Newman (2009), "Power-law distributions in empirical
data", SIAM Review 51(4):661-703.
Alstott, Bullmore & Plenz (2014), "powerlaw: a Python package for analysis
of heavy-tailed distributions", PLoS ONE 9(1):e85777.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

try:
    import powerlaw
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The 'powerlaw' package is required. Install with: "
        "pip install powerlaw"
    ) from e


# ---------------------------------------------------------------------------
# Action-length extraction
# ---------------------------------------------------------------------------

def action_lengths(actions: np.ndarray) -> np.ndarray:
    """Run lengths of consecutive identical actions.

    This is the behavioral observable compared across regimes: in lab
    dynamic foraging it is the number of consecutive same-side choices;
    in open-field foraging it is the analogous move-bout length.

    Args:
        actions: 1D array of discrete actions (e.g. 0/1 choices, or
            direction labels). NaNs are treated as breaks.

    Returns:
        1D array of run lengths (each >= 1).

    Example:
        >>> action_lengths(np.array([0, 0, 0, 1, 1, 0]))
        array([3, 2, 1])
    """
    a = np.asarray(actions).ravel()
    if a.size == 0:
        return np.array([], dtype=int)
    # Boundaries where the value changes (NaN-aware: NaN != NaN forces a break)
    change = np.empty(a.size, dtype=bool)
    change[0] = True
    change[1:] = a[1:] != a[:-1]
    # Indices where runs start
    starts = np.flatnonzero(change)
    # Run lengths = diff of start indices (last run goes to the end)
    ends = np.append(starts[1:], a.size)
    return (ends - starts).astype(int)


# ---------------------------------------------------------------------------
# Fitting result container
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Summary of a CSN power-law fit and its comparison to an exponential.

    Attributes:
        n: number of data points fit.
        alpha: power-law exponent (the gamma in p(x) ~ x^-gamma).
        xmin: lower cutoff selected by KS minimization.
        ks_distance: KS distance of the power-law fit above xmin.
        loglik_ratio: log-likelihood ratio, power law vs exponential.
            Positive favors power law, negative favors exponential.
        loglik_ratio_p: significance of the ratio (Vuong test p-value).
            Small p with positive ratio = power law significantly better.
        favored: 'power_law', 'exponential', or 'inconclusive'.
        data_range_decades: log10(max/min) of the data above xmin — how
            many decades the fit actually spans. < 1.5 means the fit is
            weak regardless of statistics.
    """
    n: int
    alpha: float
    xmin: float
    ks_distance: float
    loglik_ratio: float
    loglik_ratio_p: float
    favored: str
    data_range_decades: float

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"n={self.n}  alpha={self.alpha:.3f}  xmin={self.xmin:.3g}  "
            f"KS={self.ks_distance:.3f}  "
            f"LR(PL/exp)={self.loglik_ratio:+.2f} (p={self.loglik_ratio_p:.3f})  "
            f"favored={self.favored}  "
            f"range={self.data_range_decades:.2f} decades"
        )


# ---------------------------------------------------------------------------
# Core fitting
# ---------------------------------------------------------------------------

def fit_and_compare(
    data: np.ndarray,
    discrete: bool = True,
    xmin: Optional[float] = None,
    significance: float = 0.05,
) -> FitResult:
    """Fit a power law and compare it against an exponential.

    Args:
        data: 1D array of positive values (e.g. action lengths). Values
            <= 0 are dropped.
        discrete: True for integer-valued data (action lengths are counts).
        xmin: if given, fix the lower cutoff; otherwise CSN KS-minimization
            selects it.
        significance: threshold for declaring the power-law-vs-exponential
            comparison conclusive.

    Returns:
        FitResult.
    """
    data = np.asarray(data, dtype=float).ravel()
    data = data[data > 0]
    n = data.size
    if n < 2:
        raise ValueError(f"Need at least 2 positive data points, got {n}.")

    # powerlaw prints an xmin-search progress bar to stdout; suppress it.
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        fit = powerlaw.Fit(
            data,
            discrete=discrete,
            xmin=xmin,
            verbose=False,
        )

        # Power law vs exponential comparison.
        # normalized_ratio=True gives the Vuong-style normalized LR + p-value.
        R, p = fit.distribution_compare(
            "power_law", "exponential", normalized_ratio=True
        )

    if p < significance:
        favored = "power_law" if R > 0 else "exponential"
    else:
        favored = "inconclusive"

    above = data[data >= fit.xmin]
    if above.size and above.min() > 0:
        decades = float(np.log10(above.max() / above.min()))
    else:
        decades = 0.0

    # KS distance: ``fit.power_law.D`` is computed during fitting and is
    # available in both powerlaw 1.x and 2.x. The 2.0 backward-compat
    # ``KS()`` shim is broken (it calls an unbound ``compute_distance_metrics``),
    # so read ``.D`` directly instead.
    ks = getattr(fit.power_law, "D", None)
    if ks is None:
        ks = fit.power_law.KS()

    return FitResult(
        n=n,
        alpha=float(fit.power_law.alpha),
        xmin=float(fit.xmin),
        ks_distance=float(ks),
        loglik_ratio=float(R),
        loglik_ratio_p=float(p),
        favored=favored,
        data_range_decades=decades,
    )


def fit_favored_alpha(
    data: np.ndarray,
    discrete: bool = True,
    xmin: Optional[float] = None,
    significance: float = 0.05,
) -> tuple[str, float]:
    """Fast path returning only (favored, alpha).

    Identical fitting to :func:`fit_and_compare` but skips the expensive
    ``power_law.KS()`` computation and the data-range bookkeeping. Use this
    when fitting thousands of cells for an overview (e.g. the det booklet),
    where only the favored verdict and the exponent are plotted. For a
    single headline fit where you want KS, the likelihood-ratio p-value, and
    the decade span, use :func:`fit_and_compare`.

    Returns:
        (favored, alpha) where favored is 'power_law' / 'exponential' /
        'inconclusive'.
    """
    data = np.asarray(data, dtype=float).ravel()
    data = data[data > 0]
    if data.size < 2:
        raise ValueError(f"Need at least 2 positive data points, got {data.size}.")

    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        fit = powerlaw.Fit(data, discrete=discrete, xmin=xmin, verbose=False)
        R, p = fit.distribution_compare(
            "power_law", "exponential", normalized_ratio=True
        )
        alpha = float(fit.power_law.alpha)

    if p < significance:
        favored = "power_law" if R > 0 else "exponential"
    else:
        favored = "inconclusive"
    return favored, alpha


def bootstrap_alpha(
    data: np.ndarray,
    n_boot: int = 500,
    discrete: bool = True,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap a confidence interval for the power-law exponent.

    Resamples the data with replacement, refits, and returns the 2.5 and
    97.5 percentiles of alpha. Used to put error bars on reported exponents
    and to test whether two distributions' exponents overlap.

    Args:
        data: 1D positive values.
        n_boot: number of bootstrap resamples.
        discrete: integer-valued data.
        seed: RNG seed.

    Returns:
        (alpha_lo, alpha_hi) 95% percentile interval.
    """
    data = np.asarray(data, dtype=float).ravel()
    data = data[data > 0]
    rng = np.random.default_rng(seed)
    alphas = np.empty(n_boot)
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(n_boot):
            sample = rng.choice(data, size=data.size, replace=True)
            try:
                fit = powerlaw.Fit(sample, discrete=discrete, verbose=False)
                alphas[i] = fit.power_law.alpha
            except Exception:
                alphas[i] = np.nan
    alphas = alphas[np.isfinite(alphas)]
    return float(np.percentile(alphas, 2.5)), float(np.percentile(alphas, 97.5))