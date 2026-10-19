"""Figures for CETSP experiment results.

Every function takes the tidy frame produced by :mod:`utils.results` plus a
list of formulation labels, and reads the axis extents off the data: the time
axis spans the inferred time limit, the gap axis the observed gaps, and the
colours are assigned per formulation base for any number of series.  No
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

import colorsys
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
                           parse_formulation, warn_on_uneven_coverage)

# --------------------------------------------------------------------------
# Colours
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

#: Lightness multipliers spanned within one base, light (simplest variant)
#: to dark.  A single-variant base keeps its hue unchanged.
LIGHTNESS_RANGE = (1.45, 0.70)


def _lightness_variants(hex_color, count):
    """`count` shades of one hue, from light to dark."""
    if count == 1:
        return [mcolors.to_rgb(hex_color)]
    h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(hex_color))
    factors = np.linspace(*LIGHTNESS_RANGE, count)
    return [colorsys.hls_to_rgb(h, min(l * f, 0.95), s) for f in factors]


def series_colors(formulations):
    """Assign a colour to each label: hue by base, lightness by variant.

    Works for any number of series, unlike a fixed ten-colour qualitative
    map, and keeps a base's variants visually grouped.
    """
    bases, unknown = [], []
    for label in formulations:
        try:
            base = parse_formulation(label)['base']
        except ValueError:
            base = None
        bases.append(base)
        if base is not None and base not in BASE_PALETTE and base not in unknown:
            unknown.append(base)

    hue_of = dict(BASE_PALETTE)
    for index, base in enumerate(unknown):
        hue_of[base] = FALLBACK_PALETTE[index % len(FALLBACK_PALETTE)]

    counts = {}
    for base in bases:
        counts[base] = counts.get(base, 0) + 1

    shades = {base: _lightness_variants(hue_of.get(base, FALLBACK_PALETTE[0]),
                                        count)
              for base, count in counts.items()}
    used = {base: 0 for base in counts}

    colors = []
    for base in bases:
        colors.append(shades[base][used[base]])
        used[base] += 1
    return colors


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def _resolve(df, formulations, align, warn, stacklevel=4):
    """Filter to the chosen formulations, optionally aligning, and warn."""
    formulations = list(formulations or formulations_in(df))
    sub = df[df['Formulation'].isin(formulations)]
    if sub.empty:
        raise ValueError(f'no rows for formulations {formulations}')
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
    labels = list(labels or formulations)
    colors = colors or series_colors(formulations)

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
        # right, which keeps the time limit itself labelled.
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
    labels = list(labels or formulations)
    colors = colors or series_colors(formulations)

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
            name = ', '.join(f'{c}={v:g}' if isinstance(v, (int, float, np.number))
                             else f'{c}={v}' for c, v in zip(keys, key))
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


def pairwise_scatter(df, x, y, metrics=('Time', 'GapPct'), color_by='r_mean',
                     marker_by='n', labels=None, figsize=None, title=None,
                     log=False, filename=None, dpi=300, show=True,
                     key=INSTANCE_KEY, alpha=0.75):
    """Compare two formulations instance by instance.

    Points are paired on :data:`utils.results.INSTANCE_KEY`, so only cells
    both formulations ran are plotted and a point always compares like with
    like.  ``color_by`` and ``marker_by`` may be any columns; their legends
    are built from the values actually present.
    """
    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    unknown = [m for m in metrics if m not in SCATTER_METRICS]
    if unknown:
        raise KeyError(f'unknown metrics {unknown}; '
                       f'available: {sorted(SCATTER_METRICS)}')

    columns = list(dict.fromkeys(metrics + [c for c in (color_by, marker_by)
                                            if c]))
    paired = pairwise_table(df, x, y, metrics=columns, key=key)
    if paired.empty:
        raise ValueError(f'{x!r} and {y!r} share no instance cells')

    # Grouping columns come through the merge suffixed; they are instance
    # properties, so the two copies agree and either will do.
    def grouping(column):
        if column in paired.columns:
            return paired[column]
        return paired[f'{column}_x']

    color_values = sorted(grouping(color_by).unique()) if color_by else [None]
    marker_values = sorted(grouping(marker_by).unique()) if marker_by else [None]

    cmap = plt.get_cmap('viridis' if len(color_values) > 4 else 'tab10')
    color_of = {v: cmap(i / max(1, len(color_values) - 1)
                        if len(color_values) > 4 else i % 10)
                for i, v in enumerate(color_values)}
    marker_of = {v: _MARKERS[i % len(_MARKERS)]
                 for i, v in enumerate(marker_values)}

    xlabel, ylabel = labels or (x, y)
    if figsize is None:
        figsize = (5.0 * len(metrics), 5.0)
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, squeeze=False)
    axes = axes[0]

    for ax, metric in zip(axes, metrics):
        name, _ = SCATTER_METRICS[metric]
        xs, ys = paired[f'{metric}_x'], paired[f'{metric}_y']
        for cv in color_values:
            for mv in marker_values:
                mask = np.ones(len(paired), dtype=bool)
                if color_by:
                    mask &= (grouping(color_by) == cv).to_numpy()
                if marker_by:
                    mask &= (grouping(marker_by) == mv).to_numpy()
                if not mask.any():
                    continue
                ax.scatter(xs[mask], ys[mask], marker=marker_of[mv],
                           color=color_of[cv], alpha=alpha, edgecolors='none')

        finite = np.concatenate([xs[np.isfinite(xs)], ys[np.isfinite(ys)]])
        top = _nice_ceiling(finite.max() if len(finite) else 1.0)
        bottom = 0.0
        if log:
            positive = finite[finite > 0]
            bottom = positive.min() * 0.5 if len(positive) else 1e-3
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.plot([max(bottom, 0), top], [max(bottom, 0), top], 'k--', lw=0.8)
        ax.set_xlim(bottom if log else -0.02 * top, top)
        ax.set_ylim(bottom if log else -0.02 * top, top)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(f'{name} -- {xlabel}')
        ax.set_ylabel(f'{name} -- {ylabel}')
        ax.set_title(name)
        ax.grid(True, linestyle='--', alpha=0.5)

    handles = []
    if marker_by:
        handles += [plt.Line2D([], [], linestyle='None', color='black',
                               marker=marker_of[v],
                               label=f'{marker_by}={v:g}'
                               if isinstance(v, (int, float, np.number)) else f'{marker_by}={v}')
                    for v in marker_values]
    if color_by:
        handles += [plt.Line2D([], [], linestyle='None', marker='s',
                               color=color_of[v],
                               label=f'{color_by}={v:g}'
                               if isinstance(v, (int, float, np.number)) else f'{color_by}={v}')
                    for v in color_values]
    fig.suptitle(title or f'{xlabel} vs {ylabel}   ({len(paired)} paired instances)')
    if handles:
        ncol = min(len(handles), 6)
        reserved = min(0.06 + 0.04 * math.ceil(len(handles) / ncol), 0.3)
        fig.tight_layout(rect=(0.0, reserved, 1.0, 1.0))
        fig.legend(handles=handles, loc='lower center', ncol=ncol,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
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
# Scaling and coverage
# --------------------------------------------------------------------------

def scaling_plot(df, formulations=None, metric='Time', over='n', agg='mean',
                 labels=None, colors=None, align=False, warn=True,
                 solved_only=False, min_count=1, size_by_count=True,
                 logy=False, ax=None, figsize=(7, 5), title=None,
                 filename=None, dpi=300, show=True):
    """An aggregate metric against instance size (or any other column).

    ``agg`` may be any pandas aggregation name, or ``'solved'`` for the count
    of instances solved to optimality.

    Every series is drawn over the same x values, with the ones it has no
    data for left as gaps.  A line therefore breaks where a formulation has
    nothing to average -- it is never carried straight across a missing point,
    which would draw a segment through data that does not exist.

    ``solved_only=True`` is the usual source of those gaps: a formulation that
    proved nothing optimal at some size has no runtime to average there, so
    its line stops.  That is reported as a warning, since an absent point and
    an unrun configuration otherwise look identical.

    It also leaves the surviving means resting on very different numbers of
    runs, which flatters whichever formulation solved only the easy subset.
    ``size_by_count`` therefore shrinks a marker in proportion to how thin its
    sample is, leaving a full sample at the normal size -- so the plot is
    unchanged when every point rests on the same number of runs, and thin
    points shrink away when they do not.  ``min_count`` drops them outright.
    For the counts themselves, plot ``agg='solved'``.
    """
    sub, formulations = _resolve(df, formulations, align, warn)
    available = sub[[over, 'Formulation']].copy()
    if solved_only:
        sub = sub[sub['Solved']]
    labels = list(labels or formulations)
    colors = colors or series_colors(formulations)

    own_fig = ax is None
    fig = None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # One x axis for every series, so a gap stays a gap.
    x_values = sorted(available[over].dropna().unique())
    ylabel = 'Instances solved' if agg == 'solved' else f'{agg} {metric}'
    dropped = {}

    # A full sample keeps the normal marker size; thinner ones shrink from
    # it, so nothing grows and no new overplotting is introduced.
    full_marker = plt.rcParams['lines.markersize'] ** 2
    widest = 0

    prepared = []
    for label, name, color in zip(formulations, labels, colors):
        frame = sub[sub['Formulation'] == label]
        if agg == 'solved':
            series = frame.groupby(over)['Solved'].sum()
            counts = frame.groupby(over).size()
        else:
            series = frame.groupby(over)[metric].agg(agg)
            counts = frame.groupby(over)[metric].count()

        series = series.reindex(x_values)
        counts = counts.reindex(x_values).fillna(0).astype(int)
        if min_count > 1:
            series = series.where(counts >= min_count)

        # Distinguish "ran, but nothing to aggregate" from "never run".
        ran = set(available.loc[available['Formulation'] == label, over])
        missing = [x for x in x_values
                   if x in ran and pd.isna(series.get(x))]
        if missing:
            dropped[label] = missing

        widest = max(widest, int(counts.max()) if len(counts) else 0)
        prepared.append((name, color, series, counts))

    handles = []
    for name, color, series, counts in prepared:
        values = series.to_numpy(dtype=float)
        uniform = not size_by_count or widest == 0
        ax.plot(x_values, values, marker='o' if uniform else '',
                color=color, label=name)
        if not uniform:
            # Marker area in proportion to the sample, off a floor so that
            # the thinnest point shrinks noticeably without disappearing.
            share = np.clip(counts.to_numpy(dtype=float) / widest, 0.0, 1.0)
            ax.scatter(x_values, values, s=full_marker * (0.15 + 0.85 * share),
                       color=color, zorder=3)
        handles.append(Line2D([], [], color=color, marker='o', label=name))

    if warn and dropped:
        reason = ('solved nothing there' if solved_only
                  else f'had fewer than {min_count} runs there')
        detail = '; '.join(
            f"{label} at {over}=" + ', '.join(f'{x:g}' for x in missing)
            for label, missing in dropped.items())
        warnings.warn(
            f'line(s) stop short because a formulation {reason}: {detail}.  '
            f'Those points are gaps, not zeros.', stacklevel=3)

    ax.set_xlabel(over)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(handles=handles, fontsize='small')
    if title:
        ax.set_title(title)

    if own_fig:
        fig.tight_layout()
        if filename:
            fig.savefig(filename, dpi=dpi, bbox_inches='tight', transparent=True)
        if show:
            plt.show()
            return None
        plt.close(fig)
    return fig


def coverage_heatmap(df, formulations=None, over='n', figsize=None,
                     title='Runs recorded', filename=None, dpi=300, show=True):
    """Runs recorded per formulation and facet value -- what is still missing.

    Worth a glance before any comparison on a partially complete results file.
    """
    formulations = list(formulations or formulations_in(df))
    sub = df[df['Formulation'].isin(formulations)]
    table = sub.pivot_table(index='Formulation', columns=over, values='Time',
                            aggfunc='size')
    table = table.reindex(order_formulations(table.index))

    if figsize is None:
        figsize = (1.0 * len(table.columns) + 3, 0.30 * len(table) + 1.6)
    fig, ax = plt.subplots(figsize=figsize)

    values = table.to_numpy(dtype=float)
    # Completeness is judged per facet value: a column where every
    # formulation ran 5 instances is complete, even next to one with 45.
    expected = np.nanmax(values, axis=0, keepdims=True)
    expected[~np.isfinite(expected) | (expected == 0)] = 1.0
    ratio = np.where(np.isnan(values), 0.0, values) / expected

    ax.imshow(ratio, aspect='auto', cmap='RdYlGn', vmin=0.0, vmax=1.0)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            text = '-' if np.isnan(value) else f'{int(value)}'
            ax.text(j, i, text, ha='center', va='center', color='black',
                    fontsize=8)

    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([f'{c:g}' if isinstance(c, (int, float, np.number)) else c
                        for c in table.columns])
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index, fontsize=8)
    ax.set_xlabel(over)
    ax.set_title(f'{title} (green = complete for that {over})')

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
