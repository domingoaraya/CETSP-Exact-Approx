"""Loading, selection and tabulation of CETSP experiment results.

Nothing here is specific to a particular set of formulations, instance sizes
or time limits: the formulation label grammar is parsed, the radius regime is
derived from ``r_min``/``r_max``, and the time limit is inferred per source.
The companion module :mod:`utils.plots` turns these frames into figures.

Typical use::

    from utils.results import load_results, select, summary_table

    df = load_results('aws-B-S.xlsx')
    abf = select(df, base='ABF', symmetry=True)
    summary_table(df, formulations=abf)

See ``coverage_report`` before trusting any comparison on partial data.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

#: Columns every results file must provide.
REQUIRED_COLUMNS = ('Formulation', 'n', 'r_min', 'r_max', 'Instance',
                    'Status', 'Gap', 'Time')

#: Columns used when present and filled with NaN otherwise, so that the
#: 9-column files written by older runs and the 15-column files written by
#: current ones can be concatenated into a single frame.
OPTIONAL_COLUMNS = ('UB', 'LB', 'Root Bound', 'Nodes', 'Cuts',
                    'DFJ_Cuts', 'Sub_Failures')

#: Identifies one instance across formulations.  Instances are cached on disk
#: and reused, so a fixed key really is the same instance everywhere.
INSTANCE_KEY = ('n', 'r_min', 'r_max', 'Instance')

#: Status string meaning the run proved optimality of its own model.  For the
#: approximate formulations (and for B&S) this still leaves a discretization
#: gap, which is why gap statistics are reported over all rows as well as over
#: the unsolved ones.
OPTIMAL_STATUS = 'Optimal'


# --------------------------------------------------------------------------
# Formulation label grammar
# --------------------------------------------------------------------------
#
# run_experiment.py builds the label by appending modifiers in a fixed order:
#
#     BASE [-A] [-D] [-enum|-dual|-DE] [-S] [-OC] [-B]
#
# so the label parses unambiguously by walking the slots in that order.  Keep
# this in sync with the label construction in run_experiment.py.

BASES = {'ABF': 'ABF', 'SBF': 'SBF', 'PBF': 'PBF',
         'BS': 'BS', 'B&S': 'BS'}  # 'B&S' is the pre-rename spelling

#: Ordered modifier slots: (field, accepted tokens, value when matched).
MODIFIER_SLOTS = (
    ('extended',      {'A': True}),
    ('decomposition', {'D': True}),
    ('cut',           {'enum': 'enum', 'dual': 'dual', 'DE': 'DE'}),
    ('strengthen',    {'S': True}),
    ('opt_coeff',     {'OC': True}),
    ('symmetry',      {'B': True}),
)

#: Boolean modifier fields, in label order.
FLAG_FIELDS = ('extended', 'decomposition', 'strengthen', 'opt_coeff',
               'symmetry')

#: Every field produced by :func:`parse_formulation`.
FORMULATION_FIELDS = ('base', 'extended', 'decomposition', 'cut',
                      'strengthen', 'opt_coeff', 'symmetry')

_BASE_ORDER = {'ABF': 0, 'SBF': 1, 'PBF': 2, 'BS': 3}
_CUT_ORDER = {None: 0, 'enum': 1, 'dual': 2, 'DE': 3}


def parse_formulation(label):
    """Split a formulation label into its base and modifier flags.

    Returns a dict with keys :data:`FORMULATION_FIELDS` plus ``Formulation``.
    Raises ``ValueError`` on a token that does not fit the grammar, rather
    than silently dropping it -- an unrecognized token means the label scheme
    has changed and this module needs updating.
    """
    tokens = str(label).split('-')
    head = tokens[0]
    if head not in BASES:
        raise ValueError(f"unknown base {head!r} in formulation {label!r}")

    parsed = {'Formulation': str(label), 'base': BASES[head], 'cut': None}
    parsed.update({field: False for field in FLAG_FIELDS})

    slot = 0
    for token in tokens[1:]:
        while slot < len(MODIFIER_SLOTS) and token not in MODIFIER_SLOTS[slot][1]:
            slot += 1
        if slot == len(MODIFIER_SLOTS):
            raise ValueError(
                f"token {token!r} in formulation {label!r} is unknown, or "
                f"appears out of the order {[f for f, _ in MODIFIER_SLOTS]}")
        field, accepted = MODIFIER_SLOTS[slot]
        parsed[field] = accepted[token]
        slot += 1

    return parsed


def parse_formulations(labels):
    """Parse an iterable of labels into a DataFrame indexed by label."""
    unique = pd.unique(pd.Series(list(labels), dtype=object))
    rows = [parse_formulation(label) for label in unique]
    return pd.DataFrame(rows).set_index('Formulation')


def _sort_key(label):
    p = parse_formulation(label)
    return (_BASE_ORDER.get(p['base'], 99), p['extended'], p['decomposition'],
            _CUT_ORDER.get(p['cut'], 9), p['strengthen'], p['opt_coeff'],
            p['symmetry'], label)


def order_formulations(labels):
    """Sort labels by base and then by modifier, for stable colors/legends."""
    return sorted(dict.fromkeys(labels), key=_sort_key)


def describe_formulation(label):
    """A human-readable expansion of a label, for captions."""
    p = parse_formulation(label)
    parts = []
    if p['extended']:
        parts.append('polyhedral')
    if p['decomposition']:
        parts.append('Benders')
    if p['cut']:
        parts.append({'enum': 'enumerative cuts', 'dual': 'dual cuts',
                      'DE': 'dual+enumerative cuts'}[p['cut']])
    if p['strengthen']:
        parts.append('DFJ strengthening')
    if p['opt_coeff']:
        parts.append('optimized coefficients')
    if p['symmetry']:
        parts.append('symmetry breaking')
    return f"{p['base']} ({', '.join(parts)})" if parts else p['base']


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _read_one(source, sheet=None):
    """Read one source into a raw frame.  Returns (frame, description)."""
    if isinstance(source, pd.DataFrame):
        return source.copy(), 'frame'

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in ('.xlsx', '.xlsm', '.xls'):
        tried = []
        with pd.ExcelFile(path) as book:
            candidates = [sheet] if sheet is not None else book.sheet_names
            for name in candidates:
                frame = book.parse(name)
                if set(REQUIRED_COLUMNS).issubset(frame.columns):
                    return frame, f'{path.stem}[{name}]'
                tried.append(name)
        raise ValueError(
            f"no sheet in {path.name} has the columns {REQUIRED_COLUMNS}; "
            f"tried {tried}.  Pivot-table sheets are skipped automatically.")

    return pd.read_csv(path), path.stem


#: Status of a run stopped by the time limit.  Used to infer the limit.
TIME_LIMIT_STATUS = 'Time_Limit'


def _infer_time_limit(frame):
    """Infer the time limit from the runs that hit it, or None.

    Runs stopped by the limit cluster just above it, so the median of their
    times recovers it.  Other unsolved statuses (``Suboptimal``, ``Failed``)
    can stop early and would drag the estimate down, so they are only a
    fallback.  Returns None when the frame carries no evidence at all, which
    lets the caller fall back to a coarser grouping.
    """
    stopped = frame.loc[frame['Status'] == TIME_LIMIT_STATUS, 'Time'].dropna()
    if not len(stopped):
        stopped = frame.loc[~frame['Solved'], 'Time'].dropna()
    if not len(stopped):
        return None
    return float(np.round(np.median(stopped)))


def _time_limit_column(df):
    """Per-n time limit, falling back to the file as a whole.

    A size at which every run was optimal carries no evidence of the limit,
    so it inherits the value inferred from the whole file rather than
    reporting its own longest run as the limit.
    """
    overall = _infer_time_limit(df)
    if overall is None:
        overall = float(np.round(df['Time'].max())) if len(df) else np.nan

    limits = {n: (_infer_time_limit(group) or overall)
              for n, group in df.groupby('n')}
    return df['n'].map(limits)


def load_results(source, sheet=None, time_limit=None,
                 optimal_status=OPTIMAL_STATUS, strict=True):
    """Load a results file into a tidy frame.

    Parameters
    ----------
    source
        Path to a ``.csv`` or ``.xlsx`` file, or a DataFrame.  For a workbook,
        the first sheet carrying :data:`REQUIRED_COLUMNS` is used unless
        ``sheet`` names one, so pivot-table sheets are ignored.
    time_limit
        Override the inferred per-``n`` time limit with a constant.
    strict
        Raise on a formulation label that does not parse.  With ``False`` the
        offending rows are kept with null modifier fields and a warning.

    Files written by older runs carry only nine columns; the rest are filled
    with nulls so that both schemas load the same way.  Where the file came
    from is recorded in ``df.attrs['source']``.

    Added columns
    -------------
    ``r_mean``, ``sigma``, ``Solved``, ``GapPct``, ``TimeLimit``,
    ``RootGapPct`` (when ``UB``/``Root Bound`` are present), and one column per
    entry of :data:`FORMULATION_FIELDS`.
    """
    frame, description = _read_one(source, sheet=sheet)

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f'{description}: missing columns {missing}')
    for column in OPTIONAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan

    df = frame[list(REQUIRED_COLUMNS) + list(OPTIONAL_COLUMNS)].copy()
    df = df.reset_index(drop=True)

    # Radius regime.  run_experiment.py sets r_min = r_mean(1 - sigma) and
    # r_max = r_mean(1 + sigma), so both are recovered exactly.  Rounding
    # absorbs the float error that would otherwise split 0.2 into two groups.
    span = df['r_max'] + df['r_min']
    df['r_mean'] = ((df['r_max'] + df['r_min']) / 2).round(6)
    df['sigma'] = np.where(span > 0, (df['r_max'] - df['r_min']) / span, 0.0)
    df['sigma'] = df['sigma'].round(6)

    df['Solved'] = df['Status'] == optimal_status
    df['GapPct'] = 100.0 * df['Gap']

    if time_limit is not None:
        df['TimeLimit'] = float(time_limit)
    else:
        df['TimeLimit'] = _time_limit_column(df)

    with np.errstate(divide='ignore', invalid='ignore'):
        ub = pd.to_numeric(df['UB'], errors='coerce').replace(
            [np.inf, -np.inf], np.nan)
        df['RootGapPct'] = 100.0 * (ub - df['Root Bound']) / ub

    # Modifier columns.
    try:
        parsed = parse_formulations(df['Formulation'])
    except ValueError:
        if strict:
            raise
        labels, rows = pd.unique(df['Formulation']), []
        for label in labels:
            try:
                rows.append(parse_formulation(label))
            except ValueError as exc:
                warnings.warn(str(exc), stacklevel=2)
                rows.append({'Formulation': label,
                             **{f: None for f in FORMULATION_FIELDS}})
        parsed = pd.DataFrame(rows).set_index('Formulation')

    df = df.join(parsed, on='Formulation')
    df.attrs['source'] = description
    return df


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def _as_set(value):
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series, pd.Index)):
        return set(value)
    return {value}


def select(df, formulations=None, labels_only=True, **predicates):
    """Select formulations by predicate instead of by literal label.

    Any column of ``df`` may be used as a keyword: the modifier fields
    (``base``, ``extended``, ``decomposition``, ``cut``, ``strengthen``,
    ``opt_coeff``, ``symmetry``) as well as ``n``, ``r_mean``, ``sigma`` and
    ``Status``.

    Each keyword narrows the selection, and every keyword is a filter you opt
    into: leaving one out (or passing ``None``) accepts every value of it, so
    *not* mentioning ``decomposition`` keeps both the decomposed and the
    non-decomposed variants.  A list accepts the values it contains, which is
    the explicit way to say the same thing.  For ``cut``, ``False`` means the
    formulation carries no cut-type token, and it may appear inside a list.

    With ``labels_only`` (the default) the ordered list of matching
    formulation labels is returned, ready to hand to a plot or table.  With
    ``labels_only=False`` the filtered frame is returned instead.

    >>> select(df, base='PBF', extended=True)        # -D and non--D alike
    ['PBF-A-S-B', 'PBF-A-D-enum-S-B', 'PBF-A-D-dual-S-OC-B']
    >>> select(df, base='PBF', cut=['dual', False])  # dual cuts, or none
    ['PBF-S-B', 'PBF-D-dual-S-OC-B', 'PBF-A-S-B', 'PBF-A-D-dual-S-OC-B']

    A selection that no single predicate describes is the union of two, which
    :func:`order_formulations` dedupes and re-sorts::

        order_formulations(select(df, base='ABF', decomposition=True)
                           + select(df, base='PBF', decomposition=False))
    """
    mask = pd.Series(True, index=df.index)

    if formulations is not None:
        mask &= df['Formulation'].isin(_as_set(formulations))

    lowered = {str(c).lower(): c for c in df.columns}
    for key, value in predicates.items():
        if value is None:
            continue
        if key not in df.columns:
            if key.lower() not in lowered:
                raise KeyError(
                    f'{key!r} is not a column of the results frame; '
                    f'available: {sorted(df.columns)}')
            key = lowered[key.lower()]
        if key == 'cut':
            # The absence of a cut-type token is a value in its own right,
            # stored as null, so it needs matching by isna rather than isin --
            # including when it appears inside a list alongside real types.
            wanted = _as_set(value)
            no_cut = any(v is False or v is None for v in wanted)
            named = {v for v in wanted if isinstance(v, str)}
            column = (df['cut'].isin(named) if named
                      else pd.Series(False, index=df.index))
            mask &= (column | df['cut'].isna()) if no_cut else column
        else:
            mask &= df[key].isin(_as_set(value))

    filtered = df[mask]
    if not labels_only:
        return filtered
    return order_formulations(filtered['Formulation'].unique())


def formulations_in(df):
    """Every formulation present, in a stable base-then-modifier order."""
    return order_formulations(df['Formulation'].unique())


def frames_for(df, formulations):
    """``{label: sub-frame}`` for the given labels, preserving their order."""
    return {label: df[df['Formulation'] == label] for label in formulations}


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------

def coverage(df, formulations=None, key=INSTANCE_KEY):
    """Run counts per instance cell and formulation (NaN where absent)."""
    formulations = formulations or formulations_in(df)
    sub = df[df['Formulation'].isin(formulations)]
    table = sub.pivot_table(index=list(key), columns='Formulation',
                            values='Time', aggfunc='size')
    return table.reindex(columns=[f for f in formulations
                                  if f in table.columns])


def coverage_report(df, formulations=None, key=INSTANCE_KEY):
    """Per-formulation row counts and the instance cells each one is missing.

    The union of cells over the selected formulations is the reference grid.
    """
    formulations = formulations or formulations_in(df)
    table = coverage(df, formulations, key)
    cells = len(table)
    rows = []
    for label in table.columns:
        present = int(table[label].notna().sum())
        rows.append({'Formulation': label,
                     'Runs': int(df['Formulation'].eq(label).sum()),
                     'Cells': present,
                     'Missing': cells - present,
                     'Coverage (%)': 100.0 * present / cells if cells else np.nan})
    report = pd.DataFrame(rows)
    report.attrs['cells'] = cells
    return report


def missing_cells(df, formulations=None, key=INSTANCE_KEY):
    """Long-form list of (formulation, cell) pairs absent from the data."""
    table = coverage(df, formulations, key)
    absent = table.isna().stack()
    absent = absent[absent]
    if absent.empty:
        return pd.DataFrame(columns=list(key) + ['Formulation'])
    out = absent.index.to_frame(index=False)
    out.columns = list(key) + ['Formulation']
    return out


def common_instances(df, formulations=None, key=INSTANCE_KEY):
    """Restrict to the instance cells covered by every selected formulation.

    Returns ``(aligned_frame, dropped_cells)``.  Use when a comparison must be
    like-for-like; :func:`warn_on_uneven_coverage` is the lighter default.
    """
    formulations = formulations or formulations_in(df)
    sub = df[df['Formulation'].isin(formulations)]
    table = coverage(sub, formulations, key)
    complete = table.dropna(how='any')
    keep = set(map(tuple, complete.index.to_frame(index=False).values))
    cell = list(map(tuple, sub[list(key)].values))
    mask = pd.Series([c in keep for c in cell], index=sub.index)
    dropped = table[table.isna().any(axis=1)].index.to_frame(index=False)
    return sub[mask], dropped


def warn_on_uneven_coverage(df, formulations=None, key=INSTANCE_KEY,
                            stacklevel=3):
    """Warn -- without changing the data -- when coverage is not uniform.

    Comparing profiles over different instance sets is not like-for-like, so
    this makes the discrepancy visible at the point of use.  Returns the
    coverage report.
    """
    report = coverage_report(df, formulations, key)
    short = report[report['Missing'] > 0]
    if len(short):
        detail = ', '.join(f"{r.Formulation} missing {r.Missing}"
                           for r in short.itertuples())
        warnings.warn(
            f"uneven coverage over {report.attrs['cells']} instance cells: "
            f"{detail}.  Curves and means are over different instance sets; "
            f"pass align=True to restrict to the common cells.",
            stacklevel=stacklevel)
    return report


# --------------------------------------------------------------------------
# Summary tables
# --------------------------------------------------------------------------

def _mean(series):
    series = pd.to_numeric(series, errors='coerce')
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    return series.mean() if len(series) else np.nan


#: Metric registry: name -> (callable(sub_frame) -> value, decimals).
#: Add an entry here to make a new column available to ``summary_table``
#: without touching the function.
METRICS = {
    'Runs':        (lambda g: float(len(g)), 0),
    'Opt':         (lambda g: float(g['Solved'].sum()), 0),
    'Opt (%)':     (lambda g: 100.0 * g['Solved'].mean() if len(g) else np.nan, 1),
    'TimeOpt':     (lambda g: _mean(g.loc[g['Solved'], 'Time']), 2),
    'TimeAll':     (lambda g: _mean(g['Time']), 2),
    'Gap (%)':     (lambda g: _mean(g['GapPct']), 2),
    'Gap NoOpt (%)': (lambda g: _mean(g.loc[~g['Solved'], 'GapPct']), 2),
    'RootGap (%)': (lambda g: _mean(g['RootGapPct']), 2),
    'Cuts':        (lambda g: _mean(g['Cuts']), 2),
    'DFJ Cuts':    (lambda g: _mean(g['DFJ_Cuts']), 2),
    'Nodes':       (lambda g: _mean(g['Nodes']), 0),
    'SubFail':     (lambda g: pd.to_numeric(g['Sub_Failures'],
                                            errors='coerce').sum(), 0),
}

#: Columns reported unless ``metrics`` says otherwise.
DEFAULT_METRICS = ('Runs', 'Opt', 'TimeOpt', 'Gap (%)', 'Gap NoOpt (%)', 'Cuts')


def summary_table(df, formulations=None, by=None, metrics=DEFAULT_METRICS,
                  labels=None, align=False, warn=True, numeric=False,
                  dash='-'):
    """Compare formulations on any set of metrics, optionally grouped.

    Parameters
    ----------
    by
        Column or list of columns to break the comparison down by, e.g.
        ``'n'``, ``['n', 'sigma']``.
    metrics
        Names from :data:`METRICS`.
    align
        Restrict to instance cells common to every formulation first.
    numeric
        Return raw numbers instead of formatted strings.
    """
    formulations = formulations or formulations_in(df)
    sub = df[df['Formulation'].isin(formulations)]

    if align:
        sub, _ = common_instances(sub, formulations)
    elif warn:
        warn_on_uneven_coverage(sub, formulations)

    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise KeyError(f'unknown metrics {unknown}; available: {sorted(METRICS)}')

    name_of = dict(zip(formulations, labels)) if labels else {}
    group_cols = [by] if isinstance(by, str) else list(by or [])

    rows = []
    for label in formulations:
        block = sub[sub['Formulation'] == label]
        if group_cols:
            groups = block.groupby(group_cols, dropna=False, sort=True)
            items = [(key if isinstance(key, tuple) else (key,), g)
                     for key, g in groups]
        else:
            items = [((), block)]
        for key, group in items:
            row = {'Formulation': name_of.get(label, label)}
            row.update(dict(zip(group_cols, key)))
            for metric in metrics:
                row[metric] = METRICS[metric][0](group)
            rows.append(row)

    table = pd.DataFrame(rows, columns=['Formulation'] + group_cols +
                         list(metrics))
    if numeric:
        return table

    for metric in metrics:
        decimals = METRICS[metric][1]
        table[metric] = [dash if pd.isna(v) else f'{v:.{decimals}f}'
                         for v in table[metric]]
    return table


def pairwise_table(df, x, y, metrics=('Time', 'GapPct'), key=INSTANCE_KEY):
    """Join two formulations instance by instance.

    Pairing on :data:`INSTANCE_KEY` rather than on row position is what makes
    a head-to-head comparison meaningful when coverage differs.  Returns a
    frame with ``<metric>_x`` / ``<metric>_y`` columns over the shared cells.
    """
    key = list(key)
    # A requested column that is already part of the key stays unsuffixed and
    # must not be selected twice, or the merge sees a duplicate label.
    carried = [c for c in dict.fromkeys(list(metrics) + ['Solved', 'Status'])
               if c not in key]
    left = df[df['Formulation'] == x][key + carried]
    right = df[df['Formulation'] == y][key + carried]
    if left.empty or right.empty:
        missing = x if left.empty else y
        raise ValueError(f'no rows for formulation {missing!r}')
    return left.merge(right, on=key, suffixes=('_x', '_y'))
