"""
Bound-consistency check over a results file.

Every run reports a lower and an upper bound on the same optimal tour length,
so the two must bracket it, and the brackets of different formulations on one
instance must overlap. Checking that needs nothing but the reported numbers:

    python utils/check_bounds.py
    python utils/check_bounds.py aws-B-S.xlsx --exclude ABF-S-B

B1  LB <= UB, per run.
B2  max LB <= min UB, over the formulations that ran the instance.

Exits non-zero if either fails.

For ABF, the UB is computed from solving an independent SOCP with the best sequence found fixed. B1 failures on it are explained by numerical issues on the MISOCP leading to a LB higher than the actual SOCP value for a fixed sequence.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.results import INSTANCE_KEY, load_results

DEFAULT_RESULTS = os.path.join('Results', 'experiment_results_all.csv')


def _finite(series):
    """Numeric view of a bound column, with non-numeric and infinite as NaN."""
    values = pd.to_numeric(series, errors='coerce')
    return values.where(np.isfinite(values))


def _relative(lb, ub):
    """How far lb overshoots ub, relative to |ub|. Negative when they bracket.

    The denominator is floored at 1 so that a bound at or near zero is compared
    absolutely instead of dividing by nothing. Tours run 20 to 40 in length, so
    on real data the floor never binds.
    """
    return (lb - ub) / np.maximum(np.abs(ub), 1.0)


def prepare(df, exclude=None):
    """Usable runs with their finite bounds attached as LB_ and UB_.

    Crashed runs are dropped, suboptimal ones are not: whether a bound reported
    after a numerical failure can be trusted is the question being asked.
    """
    runs = df[df['Usable']].copy()
    if exclude:
        runs = runs[~runs['Formulation'].isin(exclude)]
    runs['LB_'] = _finite(runs['LB'])
    runs['UB_'] = _finite(runs['UB'])
    return runs


def check_runs(runs, rtol):
    """B1: a run's own LB must not exceed its own UB."""
    both = runs.dropna(subset=['LB_', 'UB_'])
    excess = _relative(both['LB_'], both['UB_'])
    crossed = excess > rtol
    return (both[crossed].assign(Excess=excess[crossed])
            .sort_values('Excess', ascending=False))


def brackets(runs):
    """B2 inputs: per instance, the highest LB and the lowest UB, with sources."""
    key = list(INSTANCE_KEY)
    columns = key + ['LB_max', 'UB_min', 'LB_from', 'LB_status',
                     'UB_from', 'UB_status']
    rows = []
    for cell, group in runs.groupby(key, sort=True):
        lbs = group.dropna(subset=['LB_'])
        ubs = group.dropna(subset=['UB_'])
        if lbs.empty or ubs.empty:
            continue
        hi = lbs.loc[lbs['LB_'].idxmax()]
        lo = ubs.loc[ubs['UB_'].idxmin()]
        rows.append(dict(zip(key, cell), LB_max=hi['LB_'], UB_min=lo['UB_'],
                         LB_from=hi['Formulation'], LB_status=hi['Status'],
                         UB_from=lo['Formulation'], UB_status=lo['Status']))
    return pd.DataFrame(rows, columns=columns)


def check_cells(cells, rtol):
    """B2: the best LB must not cross the best UB."""
    if cells.empty:
        return cells.assign(Excess=pd.Series(dtype=float))
    excess = _relative(cells['LB_max'], cells['UB_min'])
    crossed = excess > rtol
    return (cells[crossed].assign(Excess=excess[crossed])
            .sort_values('Excess', ascending=False))


def _cell(row):
    return 'n=%-3d r=(%g,%g) i=%-2d' % (row['n'], row['r_min'], row['r_max'],
                                        row['Instance'])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('results', nargs='?', default=DEFAULT_RESULTS,
                    help='results CSV or XLSX (default %s)' % DEFAULT_RESULTS)
    ap.add_argument('--sheet', default=None, help='worksheet, for an XLSX source')
    ap.add_argument('--rtol', type=float, default=1e-3,
                    help='relative tolerance on every comparison (default 1e-3)')
    ap.add_argument('--exclude', nargs='+', default=None, metavar='LABEL',
                    help='formulations to leave out of both checks')
    ap.add_argument('--top', type=int, default=15,
                    help='violations to list per check (default 15, 0 for all)')
    ap.add_argument('--out', default=None,
                    help='write the per-instance brackets to this CSV')
    args = ap.parse_args()

    if not os.path.exists(args.results):
        print('ERROR - no such results file: %s' % args.results)
        return 2

    df = load_results(args.results, sheet=args.sheet)
    runs = prepare(df, args.exclude)
    cells = brackets(runs)
    bad_runs = check_runs(runs, args.rtol)
    bad_cells = check_cells(cells, args.rtol)

    print('results      : %s' % args.results)
    print('runs         : %d usable of %d' % (len(runs), len(df)))
    print('instances    : %d bracketed of %d'
          % (len(cells), df.groupby(list(INSTANCE_KEY)).ngroups))
    print('tolerance    : %g relative' % args.rtol)
    if args.exclude:
        print('excluded     : %s' % ' '.join(args.exclude))
    print()

    # A file without bounds would otherwise pass both checks on an empty set,
    # which is the one outcome a check like this must never report.
    if cells.empty:
        # load_results materialises absent optional columns as NaN, so an
        # empty column reads the same as a missing one.
        empty = [c for c in ('LB', 'UB') if runs['%s_' % c].isna().all()]
        print('ERROR - nothing to check: no instance has both a finite LB and '
              'a finite UB.')
        if empty:
            print('  the file carries no %s.' % ' or '.join(empty))
        return 2

    worst = _relative(cells['LB_max'], cells['UB_min']).max()
    print('worst crossover over all instances: %+.2e relative' % worst)
    print()

    def listing(name, bad, total, describe):
        if bad.empty:
            print('%s PASSED - %d checked.' % (name, total))
            print()
            return
        print('%s FAILED - %d of %d.' % (name, len(bad), total))
        shown = bad if args.top <= 0 else bad.head(args.top)
        for _, row in shown.iterrows():
            print('  ' + describe(row))
        if len(shown) < len(bad):
            print('  ... %d more' % (len(bad) - len(shown)))
        print()

    listing('B1', bad_runs, len(runs.dropna(subset=['LB_', 'UB_'])),
            lambda r: '%s  %-22s %-11s LB %12.6f > UB %12.6f  (%+.2e)'
            % (_cell(r), r['Formulation'], r['Status'], r['LB_'], r['UB_'],
               _relative(r['LB_'], r['UB_'])))
    listing('B2', bad_cells, len(cells),
            lambda r: '%s  LB %12.6f (%s, %s) > UB %12.6f (%s, %s)  (%+.2e)'
            % (_cell(r), r['LB_max'], r['LB_from'], r['LB_status'],
               r['UB_min'], r['UB_from'], r['UB_status'],
               _relative(r['LB_max'], r['UB_min'])))

    if args.out:
        directory = os.path.dirname(os.path.abspath(args.out))
        if directory:
            os.makedirs(directory, exist_ok=True)
        cells.to_csv(args.out, index=False)
        print('brackets     : %s' % args.out)

    failed = len(bad_runs) + len(bad_cells)
    if failed:
        print('FAILED - %d violation(s).' % failed)
        return 1
    print('PASSED - both checks.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
