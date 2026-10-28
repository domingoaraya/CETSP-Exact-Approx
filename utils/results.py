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
#: approximate formulations (and for B&S) this still leaves a discretisation
#: gap, which is why gap statistics are reported over all rows as well as over
#: the unsolved ones.
OPTIMAL_STATUS = 'Optimal'

#: Statuses meaning the run produced no result at all -- the solver died, or
#: the driver recorded a placeholder in its place.  Such a row occupies an
#: instance cell without filling it, so counting rows would report a grid as
#: complete when part of it never ran.  These rows are excluded from figures
#: and tables, and reported as missing coverage.
FAILED_STATUSES = ('Crashed', 'Failed', 'Error')


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

# Order formulations are presented in: arc, sequence, perspective, then the
# B&S benchmark.  This drives legends and table rows, so it is the one place
# to change the presentation order.  It does not affect which colour a
# formulation gets: the hue comes from its base and the shade from its
# position among that base's own variants, neither of which this reorders.
_BASE_ORDER = {'ABF': 0, 'SBF': 1, 'PBF': 2, 'BS': 3}
_CUT_ORDER = {None: 0, 'enum': 1, 'dual': 2, 'DE': 3}


def parse_formulation(label):
    """Split a formulation label into its base and modifier flags.

    Returns a dict with keys :data:`FORMULATION_FIELDS` plus ``Formulation``.
    Raises ``ValueError`` on a token that does not fit the grammar, rather
    than silently dropping it -- an unrecognised token means the label scheme
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
    """Sort labels by base and then by modifier, for stable colours/legends."""
    return sorted(dict.fromkeys(labels), key=_sort_key)


#: Modifier tokens dropped from display labels by default.  Each is constant
#: across the current experiment, so it separates nothing and only lengthens
#: the legend.
NOISE_TOKENS = ('S', 'OC', 'B')


def simplify_labels(formulations, drop=NOISE_TOKENS, drop_cut='auto'):
    """Shorten labels for a legend, keeping only what separates them.

    ``drop`` removes those modifier tokens outright -- by default ``-S``,
    ``-OC`` and ``-B``, which every formulation in the current runs carries.

    ``drop_cut`` removes the cut-type token.  Under ``'auto'`` it goes only
    where it separates nothing: if no two of the selected formulations share a
    base, extended flag and decomposition flag while differing in cut type,
    then ``PBF-D-dual`` is already unambiguous as ``PBF-D``.  ``True`` always
    drops it, ``False`` never does.  So a plot of PBF alone with only dual
    cuts reads ``PBF-D``, while one that also carries the enumerative variant
    keeps ``PBF-D-dual`` and ``PBF-D-enum``.

    Returns a list aligned with ``formulations``.  Labels that collide after
    shortening keep their distinguishing tokens, so the result never maps two
    formulations onto one name.
    """
    formulations = list(formulations)
    parsed = [parse_formulation(label) for label in formulations]

    def group_of(entry):
        return (entry['base'], entry['extended'], entry['decomposition'])

    # Decided per decomposition, not across the whole selection: PBF-D-dual
    # can shorten to PBF-D while PBF-A-D-dual, which has an enumerative
    # sibling in the same plot, keeps its cut name.
    if drop_cut == 'auto':
        cuts_per_group = {}
        for entry in parsed:
            cuts_per_group.setdefault(group_of(entry), set()).add(entry['cut'])
        ambiguous = {group for group, cuts in cuts_per_group.items()
                     if len(cuts) > 1}
        keeps_cut = [group_of(entry) in ambiguous for entry in parsed]
    else:
        keeps_cut = [not drop_cut] * len(parsed)

    def render(entry, keep_cut, keep_flags):
        parts = [entry['base']]
        if entry['extended']:
            parts.append('A')
        if entry['decomposition']:
            parts.append('D')
        if entry['cut'] and keep_cut:
            parts.append(entry['cut'])
        for field, token in (('strengthen', 'S'), ('opt_coeff', 'OC'),
                             ('symmetry', 'B')):
            if entry[field] and (token not in drop or token in keep_flags):
                parts.append(token)
        return '-'.join(parts)

    labels = [render(entry, keep, ()) for entry, keep in zip(parsed, keeps_cut)]

    # Shortening must never merge two formulations into one legend entry.
    # Restore the least that separates them again: first the cut type, and
    # only for whatever still collides after that, the dropped flags.
    for restore in ((True, ()), (True, drop)):
        collided = {name for name in labels if labels.count(name) > 1}
        if not collided:
            break
        labels = [render(entry, *restore) if name in collided else name
                  for entry, name in zip(parsed, labels)]
    return labels


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
        parts.append('optimised coefficients')
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
    ``r_mean``, ``sigma``, ``Solved``, ``Usable``, ``GapPct``, ``TimeLimit``,
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
    df['Usable'] = ~df['Status'].isin(FAILED_STATUSES)
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


def _predicate_mask(df, predicates):
    """Boolean mask for rows matching every predicate in the dict."""
    mask = pd.Series(True, index=df.index)
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
    return mask


def select(df, formulations=None, labels_only=True, exclude=None,
           **predicates):
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

    ``exclude`` takes a dict of predicates to remove, or a list of such dicts
    as alternatives.  Each dict is a conjunction, so this drops only the PBF
    that is decomposed *and* uses enumerative cuts, and separately every
    monolithic approximation (extended but not decomposed, whose bound is not
    valid for the CETSP and so does not belong on a performance profile)::

        select(df, exclude=[dict(base='PBF', extended=False,
                                 decomposition=True, cut='enum'),
                            dict(extended=True, decomposition=False)])

    A selection that no single predicate describes is the union of two, which
    :func:`order_formulations` dedupes and re-sorts::

        order_formulations(select(df, base='ABF', decomposition=True)
                           + select(df, base='PBF', decomposition=False))
    """
    mask = _predicate_mask(df, predicates)

    if formulations is not None:
        mask &= df['Formulation'].isin(_as_set(formulations))

    if exclude is not None:
        rules = [exclude] if isinstance(exclude, dict) else list(exclude)
        for rule in rules:
            # Each rule is a conjunction, so a row is excluded only when it
            # matches every key of that rule; several rules are alternatives.
            mask &= ~_predicate_mask(df, rule)

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

def failed_runs(df, key=INSTANCE_KEY, include_absent=True):
    """Every run that produced no result, by formulation and instance.

    Covers both ways a cell can end up without one: a row recorded with a
    failure status, where the solver died and the driver wrote a placeholder
    in its place, and a cell carrying no row at all.  The ``Kind`` column
    separates them, since only the first leaves anything behind to inspect.

    Returns an empty frame with the same columns when nothing failed.
    """
    key = list(key)
    columns = ['Kind', 'Formulation'] + key + ['Status', 'Time']

    pieces = []
    recorded = set()
    if 'Usable' in df.columns:
        rows = df[~df['Usable']]
        if len(rows):
            block = rows[['Formulation'] + key + ['Status', 'Time']].copy()
            block.insert(0, 'Kind', 'recorded failure')
            pieces.append(block)
            recorded = set(map(tuple, rows[['Formulation'] + key].values))

    if include_absent:
        # Cells with no row whatsoever.  Counting usable rows here would
        # report the recorded failures a second time, as they hold a cell
        # without filling it.
        absent = missing_cells(df, key=key, usable_only=False)
        if len(absent):
            absent = absent[[tuple(row) not in recorded for row in
                             absent[['Formulation'] + key].values]]
        if len(absent):
            block = absent[['Formulation'] + key].copy()
            block.insert(0, 'Kind', 'no row')
            block['Status'] = 'absent'
            block['Time'] = float('nan')
            pieces.append(block)

    if not pieces:
        return pd.DataFrame(columns=columns)

    out = pd.concat(pieces, ignore_index=True)[columns]
    return out.sort_values(['Kind', 'Formulation'] + key).reset_index(drop=True)


def audit_statuses(df):
    """Check that every status in the data is classified correctly.

    :data:`FAILED_STATUSES` is a list of names, and names drift: the driver
    also emits ``Other_<gurobi code>`` for any termination it does not name,
    and a run recorded by some other script may use a word of its own.
    Matching on the name alone therefore cannot be trusted on a new file.

    What a non-result actually looks like is a *placeholder*: the driver's
    failure branch writes ``UB = inf``, ``LB = 0``, ``Gap = inf``,
    ``Nodes = 0`` and ``Cuts = 0``.  That shape is detectable whatever the
    status is called, so this reports it per status alongside the current
    classification and flags any disagreement.

    Act on the ``Verdict`` column: ``CHECK`` means rows that look like
    placeholders are being counted as results, and the status belongs in
    :data:`FAILED_STATUSES`; ``CHECK (kept?)`` means the reverse.
    """
    numeric = {c: pd.to_numeric(df.get(c), errors='coerce')
               for c in ('UB', 'LB', 'Gap', 'Nodes', 'Cuts')}
    # A file written to the older nine-column schema records no upper bound
    # at all, which would make every row look like it had lost one.  The
    # placeholder test needs the column to mean something.
    records_bounds = bool(numeric['UB'].notna().any())
    no_bound = ~np.isfinite(numeric['UB'])
    placeholder = (no_bound
                   & numeric['Nodes'].fillna(0).eq(0)
                   & numeric['Cuts'].fillna(0).eq(0)
                   & numeric['LB'].fillna(0).eq(0))

    rows = []
    for status, group in df.groupby('Status', dropna=False):
        failed = status in FAILED_STATUSES
        looks = int(placeholder.loc[group.index].sum())
        if not records_bounds:
            verdict = 'n/a (file records no UB)'
        elif failed and looks < len(group):
            verdict = 'CHECK (kept?)' if looks == 0 else 'mixed'
        elif not failed and looks:
            verdict = 'CHECK'
        else:
            verdict = 'ok'
        rows.append({
            'Status': status,
            'Rows': len(group),
            'Treated as': 'failure' if failed else 'result',
            'Placeholder-shaped': looks if records_bounds else '-',
            'No upper bound': (int(no_bound.loc[group.index].sum())
                               if records_bounds else '-'),
            'Verdict': verdict,
        })
    report = pd.DataFrame(rows).sort_values('Rows', ascending=False)
    report.attrs['failed_statuses'] = FAILED_STATUSES
    return report.reset_index(drop=True)


def coverage(df, formulations=None, key=INSTANCE_KEY, usable_only=True):
    """Run counts per instance cell and formulation (NaN where absent).

    Counts only runs that produced a result: a row recording a crash holds a
    cell without filling it, and counting it would call the grid complete.
    Pass ``usable_only=False`` to count every recorded row instead.
    """
    formulations = formulations or formulations_in(df)
    sub = df[df['Formulation'].isin(formulations)]
    if usable_only and 'Usable' in sub.columns:
        sub = sub[sub['Usable']]
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


def missing_cells(df, formulations=None, key=INSTANCE_KEY, usable_only=True):
    """Long-form list of (formulation, cell) pairs with no usable result.

    With ``usable_only=False`` only cells carrying no row at all are listed,
    so a recorded failure does not count as missing.
    """
    table = coverage(df, formulations, key, usable_only=usable_only)
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
    # Gap of the continuous relaxation against the best known bound, read
    # from a separate root-gap file and merged in by the caller.  Absent
    # unless that merge has happened, so it degrades to a dash rather than
    # raising on a frame that does not carry it.
    'Root gap (%)': (lambda g: _mean(g['RootRelaxPct'])
                     if 'RootRelaxPct' in g else np.nan, 2),
    'Cuts':        (lambda g: _mean(g['Cuts']), 2),
    'DFJ Cuts':    (lambda g: _mean(g['DFJ_Cuts']), 2),
    'Nodes':       (lambda g: _mean(g['Nodes']), 0),
    'SubFail':     (lambda g: pd.to_numeric(g['Sub_Failures'],
                                            errors='coerce').sum(), 0),
}

#: Columns reported unless ``metrics`` says otherwise.
DEFAULT_METRICS = ('Opt', 'TimeOpt', 'Gap (%)', 'Gap NoOpt (%)', 'Cuts')


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
    # Rows recording a crash hold no result, so they do not belong in a mean
    # or a count; the coverage warning reports them as missing instead.
    if 'Usable' in sub.columns:
        sub = sub[sub['Usable']]

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
    # A run that produced no result cannot be a point on a comparison: its
    # sentinel time and gap would plot as a real, terrible result.
    if 'Usable' in df.columns:
        df = df[df['Usable']]
    left = df[df['Formulation'] == x][key + carried]
    right = df[df['Formulation'] == y][key + carried]
    if left.empty or right.empty:
        missing = x if left.empty else y
        raise ValueError(f'no rows for formulation {missing!r}')
    return left.merge(right, on=key, suffixes=('_x', '_y'))

def shifted_geometric_mean(values, shift=10.0):
    """Geometric mean of ``values + shift``, less the shift.

    The standard way to average solver runtimes: a plain arithmetic mean is
    dominated by the runs that hit the limit, and a plain geometric mean by
    the ones that finish in milliseconds.  The shift damps both ends.
    """
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not len(values):
        return np.nan
    return float(np.exp(np.mean(np.log(np.maximum(values, 0.0) + shift)))
                 - shift)


#: How ``head_to_head`` reports runtime, and the column prefix each uses.
TIME_SUMMARIES = {
    # Shifted geometric mean over every run, timeouts included at the limit.
    # Both sides are averaged over the same instances, so the columns
    # compare, at the cost of being censored wherever a run hit the limit.
    'sgm': ('SGM', 'all'),
    # Mean over each side's own completed runs.  Uncensored and directly
    # interpretable, but each column is an average over a *different* set of
    # instances -- a formulation that only finishes the easy ones looks fast
    # -- so the two are not comparable with each other.
    'completed': ('Time', 'completed'),
    # Shifted geometric mean over the instances both sides completed.  Same
    # sample and uncensored, but undefined where they share no completions.
    'paired': ('SGM', 'both'),
}


def _time_summary(block, side, how, both, shift):
    """One side's runtime column, under the chosen summary."""
    times = block[f'Time_{side}']
    if how == 'sgm':
        return shifted_geometric_mean(times, shift)
    if how == 'completed':
        done = times[block[f'_s{side}']]
        return done.mean() if len(done) else np.nan
    shared = times[both]
    return shifted_geometric_mean(shared, shift) if len(shared) else np.nan


def head_to_head(df, x, y, by=None, key=INSTANCE_KEY, shift=10.0,
                 solved_gap=None, labels=None, time='sgm'):
    """Compare two formulations instance by instance, as a table.

    The scatter's virtue is that it is paired; this is the same pairing read
    as counts.  Only cells both formulations ran appear, so the comparison is
    like for like, and every column is computed over that shared set.

    ``solved_gap`` decides what counts as solved.  Left as None it is the
    status alone, which is the wrong question for a formulation whose
    ``Optimal`` means it closed its own relaxation rather than the CETSP --
    B&S reports ``Optimal`` on instances with a double-digit gap.  Give it a
    percentage and a run counts as solved only if it is ``Optimal`` *and*
    within that gap.

    ``time`` picks how runtime is summarised, from :data:`TIME_SUMMARIES`:
    ``'sgm'`` (the default) is the shifted geometric mean over every run,
    ``'completed'`` the mean over each side's own completed runs, and
    ``'paired'`` the shifted geometric mean over the instances both
    completed.  Only ``'sgm'`` is defined on every row and only it and
    ``'paired'`` average the two sides over the same instances; see the notes
    in :data:`TIME_SUMMARIES` before putting ``'completed'`` beside another
    formulation's column.

    Columns: how many cells each side solved and how they split, who was
    faster among the cells both solved, who held the smaller gap among the
    cells neither solved, and the runtime summary of each.
    """
    if time not in TIME_SUMMARIES:
        raise KeyError(f'time must be one of {sorted(TIME_SUMMARIES)}')
    time_prefix, time_scope = TIME_SUMMARIES[time]
    metrics = ('Time', 'GapPct', 'Status')
    paired = pairwise_table(df, x, y, metrics=metrics, key=key)
    if paired.empty:
        raise ValueError(f'{x!r} and {y!r} share no instance cells')

    def solved(side):
        done = paired[f'Status_{side}'] == OPTIMAL_STATUS
        if solved_gap is not None:
            done &= paired[f'GapPct_{side}'].abs() <= solved_gap
        return done

    paired = paired.assign(_sx=solved('x'), _sy=solved('y'))

    if by is not None:
        # The grouping column is an instance property, so the merge may have
        # suffixed it; either copy will do.
        if by not in paired.columns:
            paired = paired.assign(**{by: paired[f'{by}_x']})
        groups = list(paired.groupby(by, dropna=False, sort=True))
    else:
        groups = [(None, paired)]

    name_x, name_y = labels or simplify_labels([x, y])
    rows = []
    for value, block in groups:
        both = block['_sx'] & block['_sy']
        neither = ~block['_sx'] & ~block['_sy']
        faster_x = both & (block['Time_x'] < block['Time_y'])
        faster_y = both & (block['Time_y'] < block['Time_x'])
        tighter_x = neither & (block['GapPct_x'] < block['GapPct_y'])
        tighter_y = neither & (block['GapPct_y'] < block['GapPct_x'])

        row = {} if value is None else {by: value}
        row.update({
            'Cells': len(block),
            f'{name_x} solved': int(block['_sx'].sum()),
            f'{name_y} solved': int(block['_sy'].sum()),
            'Both': int(both.sum()),
            f'Only {name_x}': int((block['_sx'] & ~block['_sy']).sum()),
            f'Only {name_y}': int((~block['_sx'] & block['_sy']).sum()),
            'Neither': int(neither.sum()),
            f'{name_x} faster': int(faster_x.sum()),
            f'{name_y} faster': int(faster_y.sum()),
            f'{name_x} tighter': int(tighter_x.sum()),
            f'{name_y} tighter': int(tighter_y.sum()),
            f'{time_prefix} {name_x}': _time_summary(block, 'x', time,
                                                     both, shift),
            f'{time_prefix} {name_y}': _time_summary(block, 'y', time,
                                                     both, shift),
            f'Gap {name_x}': block['GapPct_x'].mean(),
            f'Gap {name_y}': block['GapPct_y'].mean(),
        })
        rows.append(row)
    return pd.DataFrame(rows)
