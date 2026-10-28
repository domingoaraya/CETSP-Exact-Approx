"""Figures for CETSP experiment results.

Every function takes the tidy frame produced by :mod:`utils.results` plus a
list of formulation labels, and reads the axis extents off the data: the time
axis spans the inferred time limit, the gap axis the observed gaps, and the
colors are assigned per formulation base for any number of series.  No
function assumes a particular formulation set, number of panels, or grid of
instance sizes.

Typical use::

    from utils.results import load_results, select
    from utils.plots import performance_profile

    df = load_results('aws-B-S.xlsx')
    performance_profile(df, select(df, base='PBF'), facet_by='n')

Every figure function draws and returns nothing by default.  Pass
``show=False`` to get the ``Figure`` back for further tweaking instead -- a
shown figure is deliberately not returned as well, because in a notebook the
display hook would then draw a second copy of it.
"""

from __future__ import annotations

import math
import warnings

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from utils.results import (INSTANCE_KEY, common_instances, formulations_in,
                           order_formulations, pairwise_table,
                           parse_formulation, simplify_labels,
                           warn_on_uneven_coverage)

# --------------------------------------------------------------------------
# Colors
# --------------------------------------------------------------------------

#: One hue per formulation base.  ABF and SBF keep the hues used in the
#: original figures so new plots sit beside the old ones.
BASE_PALETTE = {
    'ABF': '#d7263d',
    'SBF': '#1b9aaa',
    'PBF': '#5c3c92',
    'BS':  '#f6ae2d',
}

#: Hues for bases not in BASE_PALETTE.
FALLBACK_PALETTE = ('#003f5c', '#708090', '#2a9d8f', '#e76f51')

#: OKLCH lightness spanned within one base, light (simplest variant) to dark.
#: Inside the 0.43-0.77 band that keeps a mark legible on a white surface;
#: four steps across it sit ~10 apart in OKLab dE, five ~7.7.
LIGHTNESS_BAND = (0.45, 0.76)

#: OKLCH chroma ceiling, so a mid step does not shout over its neighbours.
CHROMA_CAP = 0.16


# -- OKLab, so that shades are perceptually even -------------------------------
#
# Scaling HLS lightness by a multiplier bunches the dark end and washes out the
# light end, which is what makes four shades of one hue hard to tell apart.
# Stepping OKLCH lightness instead spaces them evenly to the eye.

def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


_LMS_FROM_LINEAR = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                             [0.2119034982, 0.6806995451, 0.1073969566],
                             [0.0883024619, 0.2817188376, 0.6299787005]])
_OKLAB_FROM_LMS = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                            [1.9779984951, -2.4285922050, 0.4505937099],
                            [0.0259040371, 0.7827717662, -0.8086757660]])
_LMS_FROM_OKLAB = np.linalg.inv(_OKLAB_FROM_LMS)
_LINEAR_FROM_LMS = np.linalg.inv(_LMS_FROM_LINEAR)


def _to_oklch(color):
    """(lightness, chroma, hue in degrees) for any matplotlib color."""
    linear = _srgb_to_linear(np.array(mcolors.to_rgb(color)))
    lightness, a, b = _OKLAB_FROM_LMS @ np.cbrt(_LMS_FROM_LINEAR @ linear)
    return lightness, float(np.hypot(a, b)), float(np.degrees(np.arctan2(b, a)))


def _from_oklch(lightness, chroma, hue_degrees):
    radians = np.radians(hue_degrees)
    oklab = np.array([lightness, chroma * np.cos(radians),
                      chroma * np.sin(radians)])
    return _LINEAR_FROM_LMS @ (_LMS_FROM_OKLAB @ oklab) ** 3


def _max_chroma(lightness, hue_degrees):
    """Largest chroma at this lightness and hue that stays inside sRGB."""
    low, high = 0.0, 0.4
    for _ in range(30):
        middle = (low + high) / 2
        linear = _from_oklch(lightness, middle, hue_degrees)
        if np.all(linear >= -1e-4) and np.all(linear <= 1 + 1e-4):
            low = middle
        else:
            high = middle
    return low


def shade_ramp(color, count):
    """`count` shades of one hue, light to dark, evenly spaced to the eye."""
    _, _, hue = _to_oklch(color)
    if count == 1:
        levels = [sum(LIGHTNESS_BAND) / 2]
    else:
        levels = np.linspace(LIGHTNESS_BAND[1], LIGHTNESS_BAND[0], count)
    ramp = []
    for lightness in levels:
        chroma = min(CHROMA_CAP, _max_chroma(lightness, hue) * 0.92)
        ramp.append(tuple(_linear_to_srgb(_from_oklch(lightness, chroma, hue))))
    return ramp


def series_colors(formulations, universe=None):
    """Assign a color to each label: hue by base, lightness by variant.

    The shade a formulation gets is its position among *all* the variants of
    its base in ``universe``, not among the ones being drawn, so a color
    follows the formulation rather than its rank in the current plot.  SBF-D
    is the same color whether it is alone or beside every other SBF.
    ``universe`` defaults to the formulations themselves, which is only
    stable when every plot draws the same set -- the plotting functions pass
    the whole results frame instead.

    Returns a list aligned with ``formulations``.
    """
    universe = list(universe) if universe else list(formulations)
    for label in formulations:
        if label not in universe:
            universe.append(label)
    universe = order_formulations(universe)

    def base_of(label):
        try:
            return parse_formulation(label)['base']
        except ValueError:
            return None

    # Variants of each base, in the canonical order: the index into this list
    # is the shade, and it does not move when a plot drops a sibling.
    variants = {}
    for label in universe:
        variants.setdefault(base_of(label), []).append(label)

    hue_of = dict(BASE_PALETTE)
    unknown = [b for b in variants if b not in BASE_PALETTE]
    for index, base in enumerate(unknown):
        hue_of[base] = FALLBACK_PALETTE[index % len(FALLBACK_PALETTE)]

    ramps = {base: shade_ramp(hue_of.get(base, FALLBACK_PALETTE[0]),
                              len(siblings))
             for base, siblings in variants.items()}

    colors = []
    for label in formulations:
        base = base_of(label)
        colors.append(ramps[base][variants[base].index(label)])
    return colors


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def _resolve_colors(df, formulations, colors):
    """Colors for the series, stable across plots by default.

    ``colors`` may be a ``{label: color}`` mapping, a list aligned with
    ``formulations``, or None.  The
    default derives the shade from every formulation present in ``df``, not
    just the ones drawn, so filtering a plot never repaints the survivors.
    """
    if colors is None:
        return series_colors(formulations, universe=formulations_in(df))
    if isinstance(colors, dict):
        missing = [f for f in formulations if f not in colors]
        if missing:
            raise KeyError(f'no color given for {missing}')
        return [colors[f] for f in formulations]
    return list(colors)


def _resolve(df, formulations, align, warn, stacklevel=4):
    """Filter to the chosen formulations, optionally aligning, and warn."""
    formulations = list(formulations or formulations_in(df))
    sub = df[df['Formulation'].isin(formulations)]
    if sub.empty:
        raise ValueError(f'no rows for formulations {formulations}')
    # A crash placeholder carries no result; drawn, it would read as a run
    # with the worst possible gap.  Dropping it here is what lets the
    # coverage warning below report the cell as missing.
    if 'Usable' in sub.columns:
        sub = sub[sub['Usable']]
    if align:
        sub, _ = common_instances(sub, formulations)
    elif warn:
        warn_on_uneven_coverage(sub, formulations, stacklevel=stacklevel)
    return sub, formulations


def _plot_gaps(frame):
    """Gaps for the unsolved rows, with unusable ones marked NaN.

    A run that failed can be recorded with ``Gap = 0`` and an infinite upper
    bound; plotting that as a zero gap would make a failure look like the best
    result on the panel, so it is pushed to the worst end instead.
    """
    gaps = pd.to_numeric(frame['GapPct'], errors='coerce')
    ub = pd.to_numeric(frame['UB'], errors='coerce')
    unusable = ~np.isfinite(ub) & ub.notna().fillna(False)
    gaps = gaps.where(~unusable)
    return gaps.clip(lower=0.0)


def _nice_ceiling(value, fallback=1.0):
    """Round an axis maximum up to a readable number."""
    if not np.isfinite(value) or value <= 0:
        return fallback
    magnitude = 10.0 ** math.floor(math.log10(value))
    for step in (1.0, 1.5, 2.0, 2.5, 5.0, 10.0):
        if value <= step * magnitude:
            return step * magnitude
    return 10.0 * magnitude


def _grid(count, ncols=None, center_last_row=True):
    """Yield axes for `count` panels, optionally centring a short last row."""
    if ncols is None:
        ncols = count if count <= 3 else min(3, math.ceil(math.sqrt(count)))
    ncols = max(1, min(ncols, count))
    nrows = math.ceil(count / ncols)
    return ncols, nrows, center_last_row


def _make_axes(fig, count, ncols, nrows, center_last_row):
    """Create the panel axes, centring the final row when it is short."""
    spec = gridspec.GridSpec(nrows, 2 * ncols, figure=fig)
    axes, index = [], 0
    for row in range(nrows):
        in_row = min(ncols, count - row * ncols)
        offset = (ncols - in_row) if (center_last_row and in_row < ncols) else 0
        for col in range(in_row):
            start = 2 * col + offset
            axes.append(fig.add_subplot(spec[row, start:start + 2]))
            index += 1
    return axes


# --------------------------------------------------------------------------
# Performance profile (instances solved over time, then over gap)
# --------------------------------------------------------------------------

def profile_axes(ax, df, formulations, labels=None, colors=None,
                 time_limit=None, gap_max=None, time_width=300.0,
                 gap_width=300.0, panel_gap=50.0, time_scale='linear',
                 n_reference=None, legend_loc='lower right', legend=True,
                 time_ticks=5, gap_ticks=5):
    """Draw one performance profile on an existing axis.

    The left half counts instances solved to optimality against runtime; the
    right half continues with the unsolved instances ordered by final gap.
    Both halves are mapped onto one axis so a single step curve carries a
    formulation's whole record.

    ``time_limit`` and ``gap_max`` default to the data: the inferred limit for
    the rows being drawn, and a readable ceiling above the largest gap.  Times
    and gaps beyond those are clipped to the panel edge rather than drawn off
    the axis.
    """
    frames = {label: df[df['Formulation'] == label] for label in formulations}
    labels = list(labels or simplify_labels(formulations))
    colors = _resolve_colors(df, formulations, colors)

    if time_limit is None:
        limits = df['TimeLimit'].dropna()
        time_limit = float(limits.max()) if len(limits) else float(df['Time'].max())
    time_limit = max(float(time_limit), 1e-9)

    if gap_max is None:
        observed = pd.concat([_plot_gaps(f[~f['Solved']]) for f in frames.values()
                              if len(f)], ignore_index=True) if frames else pd.Series(dtype=float)
        observed = observed.dropna()
        gap_max = _nice_ceiling(observed.max() if len(observed) else 0.0,
                                fallback=100.0)
    gap_max = max(float(gap_max), 1e-9)

    second_split = time_width + panel_gap

    if time_scale == 'log':
        denominator = math.log10(1.0 + time_limit)

        def time_x(values):
            return time_width * np.log10(1.0 + np.clip(values, 0, time_limit)) / denominator
    elif time_scale == 'linear':
        def time_x(values):
            return time_width * np.clip(values, 0, time_limit) / time_limit
    else:
        raise ValueError("time_scale must be 'linear' or 'log'")

    def gap_x(values):
        return second_split + gap_width * np.clip(values, 0, gap_max) / gap_max

    counts = []
    for label, name, color in zip(formulations, labels, colors):
        frame = frames[label]
        counts.append(len(frame))

        solved = frame[frame['Solved']].sort_values('Time')
        xs, ys = [0.0], [0.0]
        if len(solved):
            xs += list(time_x(solved['Time'].to_numpy()))
            ys += list(range(1, len(solved) + 1))

        unsolved = frame[~frame['Solved']]
        gaps = _plot_gaps(unsolved).dropna().sort_values()
        dropped = len(unsolved) - len(gaps)
        if dropped:
            # Runs with no usable gap sit at the far edge of the gap panel.
            gaps = pd.concat([gaps, pd.Series([gap_max] * dropped)])

        if len(gaps):
            xs += [second_split]
            ys += [ys[-1]]
            xs += list(gap_x(gaps.to_numpy()))
            ys += list(range(len(solved) + 1, len(solved) + len(gaps) + 1))

        ax.step(xs, ys, color=color, label=name, where='post')

    # Axis furniture.
    if time_scale == 'log':
        # Decades crowd together near zero once mapped, so thin them from the
        # right, which keeps the time limit itself labeled.
        decades = [t for t in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0,
                               10000.0) if t < time_limit]
        candidates = np.array(decades + [time_limit])
        positions = time_x(candidates)
        min_sep = 0.12 * time_width
        keep = [len(candidates) - 1]
        for index in range(len(candidates) - 2, -1, -1):
            if positions[keep[-1]] - positions[index] >= min_sep:
                keep.append(index)
        real_times = candidates[sorted(keep)]
    else:
        real_times = np.linspace(0, time_limit, time_ticks)
    real_gaps = np.linspace(0, gap_max, gap_ticks)

    def fmt(value):
        return f'{value:g}'

    ax.set_xticks(list(time_x(real_times)) + list(gap_x(real_gaps)))
    ax.set_xticklabels([fmt(t) for t in real_times] + [fmt(g) for g in real_gaps])

    reference = n_reference if n_reference is not None else (max(counts) if counts else 0)
    ax.axvline(time_width, color='black', lw=0.5, ls='--')
    ax.axvline(second_split, color='black', lw=0.5, ls='--')
    if reference:
        ax.axhline(reference, color='black', lw=0.5, ls='--')

    ax.set_xlim(-5, second_split + gap_width + 5)
    ax.set_ylim(bottom=-1)
    ax.set_xlabel('Time (s)  |  Gap (%)')
    ax.set_ylabel('Instances solved')
    if legend:
        ax.legend(loc=legend_loc, fontsize='small')

    return {'time_limit': time_limit, 'gap_max': gap_max,
            'n_reference': reference}


def performance_profile(df, formulations=None, labels=None, facet_by=None,
                        facets=None, titles=None, title=None, ncols=None,
                        figsize=None, align=False, warn=True, colors=None,
                        share_scale=True, legend='auto', legend_loc='lower right',
                        center_last_row=True, filename=None, dpi=300,
                        show=True, **profile_kwargs):
    """Instances solved over time and gap, for any formulations and facets.

    Parameters
    ----------
    formulations
        Labels to draw; defaults to every formulation in ``df``.  Pair with
        :func:`utils.results.select` to pick a subset by predicate.
    facet_by
        Column to split panels on (``'n'``, ``'r_mean'``, ``'sigma'``,
        ...), or a list of columns for combinations.  Panels are
        laid out on an automatic grid, so any number of values works.
    facets
        Restrict and order the facet values; or, without ``facet_by``, a dict
        mapping a panel title to :func:`utils.results.select` keyword filters.
    share_scale
        Use one time limit and gap ceiling across panels, so they compare.
    align
        Restrict to instance cells common to every formulation first.  The
        default instead warns when coverage is uneven and plots everything.
    """
    sub, formulations = _resolve(df, formulations, align, warn)
    labels = list(labels or simplify_labels(formulations))
    colors = _resolve_colors(df, formulations, colors)

    # Build the panels: (title, frame).
    if isinstance(facets, dict) and facet_by is None:
        from utils.results import select as _select
        panels = [(name, _select(sub, labels_only=False, **filters))
                  for name, filters in facets.items()]
    elif facet_by is not None:
        keys = [facet_by] if isinstance(facet_by, str) else list(facet_by)
        values = sorted(sub.groupby(keys, dropna=False).groups)
        if facets is not None:
            wanted = [v if isinstance(v, tuple) or len(keys) == 1 else tuple(v)
                      for v in facets]
            values = [v for v in wanted if v in values] or values
        panels = []
        for value in values:
            key = value if isinstance(value, tuple) else (value,)
            mask = np.ones(len(sub), dtype=bool)
            for column, item in zip(keys, key):
                mask &= (sub[column] == item).to_numpy()
            name = ', '.join(
                f'{GROUP_LABELS.get(c, c)}='
                + (f'{v:g}' if isinstance(v, (int, float, np.number)) else f'{v}')
                for c, v in zip(keys, key))
            panels.append((name, sub[mask]))
    else:
        panels = [(title or '', sub)]

    if titles is not None:
        panels = [(titles[i] if i < len(titles) else name, frame)
                  for i, (name, frame) in enumerate(panels)]

    count = len(panels)
    ncols, nrows, center = _grid(count, ncols, center_last_row)
    if figsize is None:
        figsize = (5.5 * ncols, 4.5 * nrows)

    fig = plt.figure(figsize=figsize)
    axes = _make_axes(fig, count, ncols, nrows, center)

    shared = {}
    if share_scale and count > 1:
        gaps = pd.concat([_plot_gaps(f[~f['Solved']]) for _, f in panels
                          if len(f)], ignore_index=True).dropna()
        shared['time_limit'] = profile_kwargs.pop(
            'time_limit', float(sub['TimeLimit'].max()))
        shared['gap_max'] = profile_kwargs.pop(
            'gap_max', _nice_ceiling(gaps.max() if len(gaps) else 0.0,
                                     fallback=100.0))
        shared['n_reference'] = profile_kwargs.pop(
            'n_reference',
            max((f['Formulation'].value_counts().max() for _, f in panels
                 if len(f)), default=0))

    show_legend_per_axis = (legend == 'axes') or (legend == 'auto' and count == 1)

    for ax, (name, frame) in zip(axes, panels):
        if frame.empty:
            ax.set_axis_off()
            continue
        profile_axes(ax, frame, formulations, labels=labels, colors=colors,
                     legend=show_legend_per_axis, legend_loc=legend_loc,
                     **shared, **profile_kwargs)
        if name:
            ax.set_title(name)

    if title and count > 1:
        fig.suptitle(title)
    elif title and count == 1 and not panels[0][0]:
        axes[0].set_title(title)

    wants_figure_legend = (legend and not show_legend_per_axis and count > 1)
    if wants_figure_legend:
        # Reserve the strip the legend will occupy before laying out, so it
        # never lands on top of the bottom row's axis labels.
        rows_of_legend = math.ceil(len(formulations) / 5)
        reserved = min(0.04 + 0.03 * rows_of_legend, 0.25)
        fig.tight_layout(rect=(0.0, reserved, 1.0, 1.0))
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc='lower center',
                   ncol=min(len(names), 5), frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
    else:
        fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', transparent=True)
    if show:
        # Returning the figure as well would make Jupyter's display hook draw
        # it a second time, so a shown figure is not handed back.
        plt.show()
        return None
    # Closing is what makes show=False hold: a figure left open is drawn by
    # the notebook's inline backend at the end of the cell regardless.  The
    # returned figure can still be saved, inspected and displayed later.
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------
# Head-to-head scatter
# --------------------------------------------------------------------------

#: Metrics understood by `pairwise_scatter`: column -> (axis label, formatter).
SCATTER_METRICS = {
    'Time': ('Runtime (s)', '{:g}'),
    'GapPct': ('Gap (%)', '{:g}'),
    'Cuts': ('Cuts', '{:g}'),
    'Nodes': ('Nodes', '{:g}'),
    'RootGapPct': ('Root gap (%)', '{:g}'),
}

_MARKERS = ('o', '^', 'D', 's', 'v', 'P', 'X', '*', '<', '>')

#: Short names for the columns a scatter groups by, for the legend.
GROUP_LABELS = {'r_mean': 'r', 'r_min': 'r min', 'r_max': 'r max'}


def pairwise_scatter(df, x=None, y=None, pairs=None,
                     metrics=('Time', 'GapPct'), color_by='r_mean',
                     marker_by='n', labels=None, figsize=None, title=None,
                     log=False, legend=True, share_scale=True, row_pad=2.0,
                     filename=None, dpi=300, show=True, key=INSTANCE_KEY,
                     alpha=0.75):
    """Compare formulations instance by instance, one or several pairs at once.

    Points are paired on :data:`utils.results.INSTANCE_KEY`, so only cells
    both formulations ran are plotted and a point always compares like with
    like.  ``color_by`` and ``marker_by`` may be any columns; their legends
    are built from the values actually present.

    ``pairs`` takes a list of ``(x, y)`` tuples and lays the comparisons out as
    one grid -- a row per pair, a column per metric -- under a single legend,
    which is what a figure in a paper wants.  ``x`` and ``y`` on their own are
    shorthand for one pair.  The color and marker scales are built over every
    pair at once, so that one legend is true of all of them.

    ``log`` is either a flag for every metric or the names of the ones that
    want a log axis: ``log=('Time',)`` spreads out runtimes that span decades
    while keeping the gap axis linear, which matters because a gap of 1e-13 on
    an instance that solved would otherwise set the bottom of the axis.

    ``share_scale`` gives a metric the same axis limits down its column, so
    the rows compare; ``legend=False`` drops the legend, for composing several
    figures by hand; ``row_pad`` is the gap between rows, in font-size units,
    which keeps a row's x label off the panel beneath it.  ``title`` defaults to a description of the comparison
    for a single pair and to nothing for several -- name it explicitly if you
    want one.
    """
    if pairs is None:
        if x is None or y is None:
            raise ValueError('give either x and y, or pairs=[(x, y), ...]')
        pairs = [(x, y)]
    pairs = [tuple(pair) for pair in pairs]

    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    unknown = [m for m in metrics if m not in SCATTER_METRICS]
    if unknown:
        raise KeyError(f'unknown metrics {unknown}; '
                       f'available: {sorted(SCATTER_METRICS)}')

    columns = list(dict.fromkeys(metrics + [c for c in (color_by, marker_by)
                                            if c]))
    tables = []
    for left, right in pairs:
        paired = pairwise_table(df, left, right, metrics=columns, key=key)
        if paired.empty:
            raise ValueError(f'{left!r} and {right!r} share no instance cells')
        tables.append(paired)

    # Grouping columns come through the merge suffixed; they are instance
    # properties, so the two copies agree and either will do.
    def grouping(paired, column):
        if column in paired.columns:
            return paired[column]
        return paired[f'{column}_x']

    # One scale over every pair, so the single legend is true of all panels.
    def values_of(column):
        if not column:
            return [None]
        seen = set()
        for paired in tables:
            seen.update(grouping(paired, column).unique())
        return sorted(seen)

    color_values = values_of(color_by)
    marker_values = values_of(marker_by)

    cmap = plt.get_cmap('viridis' if len(color_values) > 4 else 'tab10')
    color_of = {v: cmap(i / max(1, len(color_values) - 1)
                        if len(color_values) > 4 else i % 10)
                for i, v in enumerate(color_values)}
    marker_of = {v: _MARKERS[i % len(_MARKERS)]
                 for i, v in enumerate(marker_values)}

    # ``log`` may be a flag for every metric, or the metrics that want one:
    # runtime spans decades and wants a log axis, while a gap that is 1e-13 on
    # the instances that solved would drag a log axis down thirteen of them.
    if isinstance(log, bool):
        log_for = {metric: log for metric in metrics}
    else:
        wanted = set(log or ())
        log_for = {metric: metric in wanted for metric in metrics}

    involved = order_formulations([f for pair in pairs for f in pair])
    name_of = dict(zip(involved, simplify_labels(involved)))
    if isinstance(labels, dict):
        name_of.update(labels)
    elif labels is not None and len(pairs) == 1:
        name_of = dict(zip(pairs[0], labels))

    nrows, ncols = len(pairs), len(metrics)
    if figsize is None:
        figsize = (5.0 * ncols, 5.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    # Axis extent per metric, shared down the column so the rows compare.
    extent = {}
    for metric in metrics:
        pool = np.concatenate([
            np.asarray(paired[f'{metric}_{side}'], dtype=float)
            for paired in tables for side in ('x', 'y')])
        pool = pool[np.isfinite(pool)]
        top = _nice_ceiling(pool.max() if len(pool) else 1.0)
        positive = pool[pool > 0]
        bottom = (positive.min() * 0.5 if log_for[metric] and len(positive)
                  else (1e-3 if log_for[metric] else 0.0))
        extent[metric] = (bottom, top)

    for row, (paired, (left, right)) in enumerate(zip(tables, pairs)):
        for column, metric in enumerate(metrics):
            ax = axes[row][column]
            name, _ = SCATTER_METRICS[metric]
            xs, ys = paired[f'{metric}_x'], paired[f'{metric}_y']

            for value in color_values:
                for marker in marker_values:
                    mask = np.ones(len(paired), dtype=bool)
                    if color_by:
                        mask &= (grouping(paired, color_by) == value).to_numpy()
                    if marker_by:
                        mask &= (grouping(paired, marker_by) == marker).to_numpy()
                    if not mask.any():
                        continue
                    ax.scatter(xs[mask], ys[mask], marker=marker_of[marker],
                               color=color_of[value], alpha=alpha,
                               edgecolors='none')

            if share_scale:
                bottom, top = extent[metric]
            else:
                own = np.concatenate([np.asarray(xs, dtype=float),
                                      np.asarray(ys, dtype=float)])
                own = own[np.isfinite(own)]
                top = _nice_ceiling(own.max() if len(own) else 1.0)
                positive = own[own > 0]
                bottom = (positive.min() * 0.5 if log_for[metric] and len(positive)
                          else (1e-3 if log_for[metric] else 0.0))
            if log_for[metric]:
                ax.set_xscale('log')
                ax.set_yscale('log')
            ax.plot([max(bottom, 0), top], [max(bottom, 0), top], 'k--', lw=0.8)
            ax.set_xlim(bottom if log_for[metric] else -0.02 * top, top)
            ax.set_ylim(bottom if log_for[metric] else -0.02 * top, top)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel(name_of.get(left, left))
            ax.set_ylabel(name_of.get(right, right))
            if row == 0:
                ax.set_title(name)
            ax.grid(True, linestyle='--', alpha=0.5)

    def entry(column, value, **style):
        name = GROUP_LABELS.get(column, column)
        shown = f'{value:g}' if isinstance(value, (int, float, np.number))             else f'{value}'
        return plt.Line2D([], [], linestyle='None', label=f'{name}={shown}',
                          **style)

    groups = []
    if legend and marker_by:
        groups.append([entry(marker_by, v, color='black', marker=marker_of[v])
                       for v in marker_values])
    if legend and color_by:
        groups.append([entry(color_by, v, color=color_of[v], marker='s')
                       for v in color_values])

    # A legend fills column-major, so laying the groups out as plain rows
    # would split one across two of them.  Pad each group to the same width
    # and interleave: with one column per entry, each group lands on its own
    # row, markers above colors.
    handles, legend_columns = [], 1
    if groups:
        legend_columns = max(len(group) for group in groups)
        blank = lambda: plt.Line2D([], [], linestyle='None', marker='None',
                                   label=' ')
        padded = [group + [blank()] * (legend_columns - len(group))
                  for group in groups]
        handles = [h for slot in zip(*padded) for h in slot]

    if title is None and len(pairs) == 1:
        left, right = pairs[0]
        title = (f'{name_of.get(left, left)} vs {name_of.get(right, right)}'
                 f'   ({len(tables[0])} paired instances)')
    if title:
        fig.suptitle(title)

    if handles:
        reserved = min((0.05 + 0.035 * len(groups)) / nrows, 0.3)
        fig.tight_layout(rect=(0.0, reserved, 1.0, 1.0), h_pad=row_pad)
        fig.legend(handles=handles, loc='lower center', ncol=legend_columns,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
    else:
        fig.tight_layout(h_pad=row_pad)
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', transparent=True)
    if show:
        # Returning the figure as well would make Jupyter's display hook draw
        # it a second time, so a shown figure is not handed back.
        plt.show()
        return None
    # Closing is what makes show=False hold: a figure left open is drawn by
    # the notebook's inline backend at the end of the cell regardless.  The
    # returned figure can still be saved, inspected and displayed later.
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------
# Scaling and coverage
# --------------------------------------------------------------------------

