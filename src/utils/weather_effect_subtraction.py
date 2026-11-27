import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.special import ndtr




# ---------------------------------------------------------------------- #
#  Effect removal alarm (PredictiveModel and ProbabilisticModel)          #
# ---------------------------------------------------------------------- #
# Validated two-series palette (reference vs. monitoring); the reserved status color
# for outliers always comes with an 'x' marker and a legend entry, never color alone.
REFERENCE_COLOR = '#2a78d6'
MONITORING_COLOR = '#eb6834'
CRITICAL_COLOR = '#d03b3b'
WARNING_COLOR = '#fab219'
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
GRID = '#e4e3df'


def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def _x_of(index_like, n):
    """ Time axis if the index is a DatetimeIndex, else the measurement number. """
    return index_like if isinstance(index_like, pd.DatetimeIndex) else np.arange(n)


def _format_x(ax, x, max_ticks=6):
    if isinstance(x, pd.DatetimeIndex):
        locator = mdates.AutoDateLocator(maxticks=max_ticks)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set_xlabel('Time')
    else:
        ax.set_xlabel('Measurement')


def plot_effect_alert(d2_ref, d2_mon, marg_ref, marg_mon, limit, alpha, z_single, alert, threshold_ratio,
                      xticks=True, reference_is_check=True, method='', figsize=(12, 7.5)):
    ''' Alarm figure of the effect removal, shared by the predictive and probabilistic models.

        Top: control chart of the squared distance d2 = sum_j w_j^2 of the joint
             scores: reference phase (left) and monitoring phase (right, time axis if
             the index is a DatetimeIndex), chi2(D) limit, monitoring outliers marked.
        Bottom left: box plots of the marginal scores per output, reference vs.
             monitoring (box: 25-75 %, whiskers: 1-99 %), with the single-output limits.
             A shifted box shows the direction of a change (damage lowers frequencies).
        Bottom right: histogram of the marginal percentiles u = Phi(z), pooled over the
             outputs (flat on healthy data).

        Parameters
        ----------
        d2_ref, d2_mon : pd.Series
            Squared distances of the reference and monitoring measurements.
        marg_ref, marg_mon : pd.DataFrame
            Marginal normal scores per output.
        limit : float
            chi2(D) limit of d2 at level alpha.
        alpha : float
            Outlier level of a single measurement.
        z_single : float
            |z_j| limit of a single output.
        alert : bool
            Alert status, shown in the title.
        threshold_ratio : float
            Alert threshold of the outlier ratio.
        xticks : bool, optional
            Show the output names on the per-output panel.
        reference_is_check : bool, optional
            True if the reference scores are an independent calibration check
            (probabilistic models). False if the scores are defined relative to the
            reference sample itself (predictive models): its percentiles are flat by
            construction, so its histogram is not drawn.
        method : str, optional
            Name of the method, shown as figure title.

        Returns
        -------
        matplotlib.figure.Figure '''

    fig = plt.figure(figsize=figsize, facecolor=SURFACE, constrained_layout=True)
    grid = fig.add_gridspec(2, 4, height_ratios=[1.15, 1])
    ax_r = fig.add_subplot(grid[0, 0])
    ax_d = fig.add_subplot(grid[0, 1:], sharey=ax_r)
    ax_box = fig.add_subplot(grid[1, :3])
    ax_u = fig.add_subplot(grid[1, 3])
    for ax in (ax_r, ax_d, ax_box, ax_u):
        _style_axis(ax)
    if method:
        fig.suptitle(method, x=0.01, ha='left', fontsize=11, color=INK_SECONDARY)

    # ---- control chart of d2 -------------------------------------------------
    floor = 1e-3                                           # log axis: keep d2 = 0 visible
    x_ref, x_mon = _x_of(d2_ref.index, len(d2_ref)), _x_of(d2_mon.index, len(d2_mon))
    y_ref, y_mon = np.maximum(d2_ref.to_numpy(), floor), np.maximum(d2_mon.to_numpy(), floor)
    outliers = d2_mon.to_numpy() > limit
    D = marg_mon.shape[1]

    ax_r.scatter(x_ref, y_ref, s=8, color=REFERENCE_COLOR, alpha=0.5, linewidths=0)
    ax_d.scatter(x_mon[~outliers], y_mon[~outliers], s=8, color=MONITORING_COLOR, alpha=0.7, linewidths=0,
                 label='Monitoring')
    ax_d.scatter(x_mon[outliers], y_mon[outliers], s=22, marker='x', color=CRITICAL_COLOR, linewidths=1.2,
                 label=f'Outlier ({outliers.sum()})')
    for ax in (ax_r, ax_d):
        ax.axhline(limit, color=INK_SECONDARY, linestyle='--', linewidth=1.2,
                   label=f'Limit $\\chi^2_{{{D}}}$, $\\alpha$={alpha:g}')
    ax_r.set_yscale('log')
    ax_r.set_ylabel('Distance $d^2 = \\sum_j w_j^2$')
    ax_r.set_title('Reference', loc='left', fontsize=10, color=REFERENCE_COLOR)
    _format_x(ax_r, x_ref, 3)
    _format_x(ax_d, x_mon, 6)
    plt.setp(ax_d.get_yticklabels(), visible=False)
    ax_d.legend(loc='lower right', fontsize=8, frameon=True, facecolor=SURFACE, edgecolor=GRID,
                framealpha=0.95, labelcolor=INK)
    ratio_ref, ratio_mon = float((d2_ref.to_numpy() > limit).mean()), float(outliers.mean())
    ax_d.set_title(f"Monitoring  ·  {'ALERT' if alert else 'No alert'}: {ratio_mon:.1%} outliers "
                   f"(reference {ratio_ref:.1%}, expected {alpha:.1%}, threshold {threshold_ratio:.1%})",
                   loc='left', fontsize=10, color=CRITICAL_COLOR if alert else INK)

    # ---- marginal scores per output ----------------------------------------
    names = list(marg_mon.columns)
    pos = np.arange(len(names))
    width = 0.36

    def box_stats(values):
        values = values[np.isfinite(values)]
        q01, q25, q50, q75, q99 = np.quantile(values, [0.01, 0.25, 0.5, 0.75, 0.99])
        return {'med': q50, 'q1': q25, 'q3': q75, 'whislo': q01, 'whishi': q99, 'fliers': []}

    for data, offset, color, label in ((marg_ref, -width / 2 - 0.02, REFERENCE_COLOR, 'Reference'),
                                       (marg_mon, width / 2 + 0.02, MONITORING_COLOR, 'Monitoring')):
        stats = [box_stats(data[c].to_numpy(dtype=float)) for c in names]
        ax_box.bxp(stats, positions=pos + offset, widths=width, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=color, edgecolor=SURFACE, linewidth=1),
                   medianprops=dict(color=SURFACE, linewidth=1.6),
                   whiskerprops=dict(color=color, linewidth=1.4), capprops=dict(color=color, linewidth=1.4))
    for sign in (-1, 1):
        ax_box.axhline(sign * z_single, color=INK_SECONDARY, linestyle='--', linewidth=1.2)
    ax_box.axhline(0, color=INK_SECONDARY, linewidth=0.6)

    # label the (at most 3) outputs with the largest monitoring outlier ratio above 2 alpha
    exceed = (marg_mon.abs() > z_single).mean().to_numpy()
    for i in [i for i in np.argsort(exceed)[::-1][:3] if exceed[i] > 2 * alpha]:
        top = np.quantile(marg_mon[names[i]], 0.99)
        ax_box.annotate(f'{exceed[i]:.0%}', (pos[i] + width / 2 + 0.02, top), xytext=(0, 3),
                        textcoords='offset points', ha='center', fontsize=8, color=INK)
    many = len(names) > 8
    ax_box.set_xticks(pos)
    ax_box.set_xticklabels(names if xticks else [], rotation=45 if many else 0, ha='right' if many else 'center',
                           fontsize=8)
    ax_box.set_xlim(-0.6, len(names) - 0.4)
    ax_box.set_ylabel('Marginal score $z_j$')
    ax_box.set_title('Per output (labels: monitoring outlier ratio)', loc='left', fontsize=10, color=INK)
    ax_box.legend(handles=[Patch(color=REFERENCE_COLOR, label='Reference'),
                           Patch(color=MONITORING_COLOR, label='Monitoring'),
                           Line2D([], [], color=INK_SECONDARY, linestyle='--', label=f'Limit $\\pm${z_single:.2f}')],
                  fontsize=8, frameon=False, labelcolor=INK, ncol=3, loc='best')

    # ---- percentiles ----------------------------------------------------------
    bins = np.linspace(0, 1, 21)
    if reference_is_check:
        ax_u.hist(ndtr(marg_ref.to_numpy(dtype=float)).ravel(), bins=bins, density=True, histtype='step',
                  linewidth=1.8, color=REFERENCE_COLOR, label='Reference')
    ax_u.hist(ndtr(marg_mon.to_numpy(dtype=float)).ravel(), bins=bins, density=True, histtype='step',
              linewidth=1.8, color=MONITORING_COLOR, label='Monitoring')
    ax_u.axhline(1.0, color=INK_SECONDARY, linestyle='--', linewidth=1.2,
                 label='Uniform' if reference_is_check else 'Uniform (= reference)')
    ax_u.set_xlim(0, 1)
    ax_u.set_xlabel('Percentile $u = \\Phi(z_j)$, all outputs')
    ax_u.set_ylabel('Density')
    ax_u.set_title('Percentiles (flat = healthy)', loc='left', fontsize=10, color=INK)
    ax_u.legend(fontsize=8, frameon=False, labelcolor=INK)

    plt.close(fig)
    return fig


def plot_remained_effects_over_time(reference_effects, monitoring_effects, outputs=None, band=(0.01, 0.99),
                                    max_outputs=6, ratio=0.05):
    ''' Remained (mapped) outputs over time in physical units, one panel per output.

        Three zones per output:
        - Reference region (blue): the quantile range `band` of the reference remained
          effects; the dashed line is their median.
        - Warning regions (yellow): one above and one below the reference region, each
          `ratio` times as wide as the reference region. Samples here are plotted as
          normal monitoring points and counted as warnings.
        - Alert space (no background): everything beyond the warning regions. Samples
          here are marked with a red 'x' and counted as alerts.

        Parameters
        ----------
        reference_effects : pd.DataFrame
            Remained effects of healthy reference data.
        monitoring_effects : pd.DataFrame
            Remained effects of the monitoring data; a DatetimeIndex is used as time axis.
        outputs : list of str, optional
            Outputs to show (default: the first max_outputs columns).
        band : tuple of float, optional
            Lower and upper quantile of the reference band (default 1 % - 99 %).
        max_outputs : int, optional
            Number of outputs shown when outputs is None (default 6).
        ratio : float, optional
            Width of each warning region as a fraction of the width of the reference
            region (default 0.05).

        Returns
        -------
        matplotlib.figure.Figure '''

    if ratio < 0:
        raise ValueError("ratio must be non-negative")
    outputs = list(monitoring_effects.columns[:max_outputs]) if outputs is None else list(outputs)
    missing = [o for o in outputs if o not in monitoring_effects.columns]
    if missing:
        raise ValueError(f"No output(s) called {missing}")
    n = len(outputs)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.7 * n + 0.9), sharex=True, facecolor=SURFACE,
                             constrained_layout=True, squeeze=False)
    axes = axes[:, 0]
    x = _x_of(monitoring_effects.index, len(monitoring_effects))
    for ax, name in zip(axes, outputs):
        _style_axis(ax)
        lo, med, hi = np.quantile(reference_effects[name], [band[0], 0.5, band[1]])
        margin = ratio * (hi - lo)
        warn_lo, warn_hi = lo - margin, hi + margin
        values = monitoring_effects[name].to_numpy(dtype=float)
        alerted = (values < warn_lo) | (values > warn_hi)
        warned = ~alerted & ((values < lo) | (values > hi))
        ax.axhspan(lo, hi, color=REFERENCE_COLOR, alpha=0.15, linewidth=0,
                   label=f'Reference {band[0]:.0%}-{band[1]:.0%}')
        ax.axhspan(hi, warn_hi, color=WARNING_COLOR, alpha=0.35, linewidth=0,
                   label=f'Warning (±{ratio:.0%} of the reference range)')
        ax.axhspan(warn_lo, lo, color=WARNING_COLOR, alpha=0.35, linewidth=0)
        ax.axhline(med, color=REFERENCE_COLOR, linestyle='--', linewidth=1.2, label='Reference median')
        ax.scatter(x[~alerted], values[~alerted], s=6, color=MONITORING_COLOR, alpha=0.7, linewidths=0,
                   label='Monitoring')
        ax.scatter(x[alerted], values[alerted], s=16, marker='x', color=CRITICAL_COLOR, linewidths=1.0,
                   label='Alert (beyond the warning region)')
        ax.set_ylabel(name, rotation=0, ha='right', va='center', fontsize=9, color=INK)
        ax.text(1.0, 1.0, f'{warned.mean():.1%} warning  ·  {alerted.mean():.1%} alert', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8, color=INK_SECONDARY)
    axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=5, fontsize=8, frameon=False, labelcolor=INK)
    _format_x(axes[-1], x)
    fig.suptitle('Remained effects at the reference condition', x=0.01, ha='left', fontsize=11, color=INK_SECONDARY)
    plt.close(fig)
    return fig