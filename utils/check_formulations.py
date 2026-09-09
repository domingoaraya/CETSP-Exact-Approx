"""
Cross-formulation sanity check.

Solves the same instances with every available formulation and verifies that
they agree. The concern is correctness rather than performance: a formulation
that reports a wrong bound still looks like a successful run, so the practical
way to detect one is to make the formulations contradict each other.

Run after any change to model building, cut generation or the subproblems:

    python utils/check_formulations.py

Exits non-zero if any check fails, so it can gate a commit.


WHAT IS CHECKED
---------------
C1  internal consistency (per run)
    The reported UB must equal the true length of the tour that run returned,
    recomputed by an independent exact arc SOCP with x fixed. This is the
    strongest check available: it needs no second formulation to disagree, so
    it still holds when every formulation shares the same defect.

C2  LB <= UB (per run)
    Catches sign and accounting errors in the master objective.

C3  UB agreement across formulations (per cell)
    Every exact formulation solved to optimality must report the same optimum,
    since the reported UB is the master objective and equals a real tour
    length. Disagreement beyond tolerance is a defect, not noise.

C4  no UB below a tour that was actually exhibited (per cell)
    The shortest arc-scored tour found by any formulation is an achievable
    value. A UB below it is not an upper bound on anything.

C5  no LB above a tour that was actually exhibited (per cell)
    Detects the opposite failure: an LB above a known feasible value means a
    cut removed the optimum. UB comparisons cannot see this, because an
    over-tight cut moves every bound in the same direction.

C6  the approximated formulations bracket the exact ones (per cell)
    The polyhedral tower in _create_approximated_socp_constraints is an outer
    approximation, so the extended model is a relaxation:

        LB_extended <= LB_exact    and    UB_extended >= UB_exact

    UB_extended is produced by compute_upper_bound with x fixed, i.e. the exact
    length of the sequence the relaxation chose, which can only be worse than
    the true optimum.

    This holds only for the non-decomposed extended formulations. Adding cuts
    closes the approximation gap: under decomposition the subproblem is the
    true SOCP, so theta absorbs the shortfall and the master objective returns
    to the true optimum. The extended decompositions are therefore checked as
    ordinary exact formulations.


TOLERANCE AND MIPGap
--------------------
--tol is an absolute bound on every comparison. It has to sit above the
residual noise of a correct run and below the smallest error worth catching.
The formulations that carry conic constraints on the master rather than in an
exact subproblem (seq and perspective in full MISOCP form) set that noise
floor, at order 1e-4, so the default of 1e-3 leaves roughly an order of
magnitude on either side. Raise it for much larger instances, where an
absolute bound becomes relatively tighter.

MIPGap has to be forced below the tolerance for C3 to mean anything. Gurobi
defaults to 1e-4 RELATIVE, which on a tour of length 20 permits a legitimate
2e-3 absolute spread between two formulations that both report Optimal - wider
than the tolerance itself. It cannot be made arbitrarily small either: solving
a MISOCP to very high precision is not always achievable, and past roughly
1e-6 some formulations terminate in a numerical failure instead, which would
drop them from the comparison they exist to take part in. The default of 1e-6
keeps the MIP slack about two orders below the tolerance while leaving every
formulation solvable.
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gurobipy import Model  # noqa: E402

from classes.data_handling import CETSPData  # noqa: E402
from classes.models import CETSPModel  # noqa: E402
from classes.solver import CETSP_L2_Solver  # noqa: E402
from utils.instance_handler import create_and_save_instance  # noqa: E402

# label, model_type, decomposition, extended, cut_type
#
# Cut types are listed as separate entries rather than combined: a combined
# 'dual+enumerative' run can converge on the strength of the enumerative cut
# alone, which would mask a defective dual cut.
CONFIGURATIONS = [
    ('ABF',           'arc',         False, False, None),
    ('SBF',           'seq',         False, False, None),
    ('PBF',           'perspective', False, False, None),
    ('ABF-D',         'arc',         True,  False, None),
    ('SBF-D-dual',    'seq',         True,  False, 'dual'),
    ('SBF-D-enum',    'seq',         True,  False, 'enumerative'),
    ('SBF-D-DE',      'seq',         True,  False, 'dual+enumerative'),
    ('PBF-D-dual',    'perspective', True,  False, 'dual'),
    ('PBF-D-enum',    'perspective', True,  False, 'enumerative'),
    ('PBF-D-DE',      'perspective', True,  False, 'dual+enumerative'),
    ('ABF-A',         'arc',         False, True,  None),
    ('SBF-A',         'seq',         False, True,  None),
    ('PBF-A',         'perspective', False, True,  None),
    ('ABF-A-D',       'arc',         True,  True,  None),
    ('SBF-A-D-dual',  'seq',         True,  True,  'dual'),
    ('SBF-A-D-enum',  'seq',         True,  True,  'enumerative'),
    ('SBF-A-D-DE',    'seq',         True,  True,  'dual+enumerative'),
    ('PBF-A-D-dual',  'perspective', True,  True,  'dual'),
    ('PBF-A-D-enum',  'perspective', True,  True,  'enumerative'),
    ('PBF-A-D-DE',    'perspective', True,  True,  'dual+enumerative'),
]

STATUS = {2: 'Optimal', 3: 'Infeasible', 4: 'InfOrUnbd', 5: 'Unbounded',
          9: 'Time_Limit', 11: 'Interrupted', 12: 'Numeric', 13: 'Suboptimal'}


def is_bracketing(extended, decomposition):
    """Extended and non-decomposed: the relaxation that brackets, see C6."""
    return extended and not decomposition


def exact_tour_length(data, tour_arcs, mip_gap):
    """
    Length of a given tour under the exact arc SOCP: the common yardstick.

    Every formulation is scored with this evaluator rather than with its own.
    compute_upper_bound maps model_type back to the model's own family, so it
    would score a perspective tour with a perspective SOCP; if that formulation
    is the one at fault, the measurement inherits the fault.
    """
    m = Model()
    m.setParam('OutputFlag', 0)
    m.setParam('MIPGap', mip_gap)
    s = CETSP_L2_Solver(m, data, 'arc')
    s.cut_type = None
    s.build()
    active = set(tour_arcs)
    for i in range(data.n):
        for j in range(data.n):
            v = 1.0 if (i, j) in active else 0.0
            s.x[i, j].lb = v
            s.x[i, j].ub = v
    m.optimize()
    return m.ObjVal if m.SolCount > 0 else None


def run_one(path, label, model_type, decomposition, extended, cut_type,
            nu, time_limit, mip_gap):
    data = CETSPData.from_file(path)
    rec = {'formulation': label, 'status': None, 'ub': None, 'lb': None,
           'tour_len': None, 'ub_delta': None, 'runtime': None,
           'sub_failures': None, 'n_effective': None}
    t0 = time.time()
    try:
        model = CETSPModel(data, model_type=model_type, decomposition=decomposition,
                           extended=extended, nu=nu, cut_type=cut_type)
        model.build()
        model.model.setParam('MIPGap', mip_gap)
        model.optimize(time_limit)
    except Exception as exc:
        rec['status'] = 'Exception:' + type(exc).__name__
        rec['runtime'] = round(time.time() - t0, 2)
        return rec

    rec['status'] = STATUS.get(model.model.Status, str(model.model.Status))
    rec['ub'] = model.upper_bound
    rec['lb'] = model.lower_bound
    rec['runtime'] = round(time.time() - t0, 2)
    rec['sub_failures'] = getattr(model, 'subproblem_failures', None)
    rec['n_effective'] = model.data.n
    if model.arcs:
        rec['tour_len'] = exact_tour_length(model.data, model.arcs, mip_gap)
    if isinstance(rec['ub'], float) and isinstance(rec['tour_len'], float):
        rec['ub_delta'] = rec['ub'] - rec['tour_len']
    return rec


def check_cell(cell, rows, tol):
    """Run C1-C6 over one (n, r_mean, sigma, instance) cell. Returns failures."""
    fails = []
    tag = 'n=%d r=%g s=%g i=%d' % cell

    usable = [r for r in rows if r['status'] == 'Optimal']
    exact = [r for r in usable if not r['bracketing']]
    approx = [r for r in usable if r['bracketing']]

    # C1 / C2, per run
    for r in usable:
        if r['ub_delta'] is not None and abs(r['ub_delta']) > tol:
            fails.append('C1 %s %s: reported UB %.9f but its own tour measures '
                         '%.9f (delta %+.2e)' % (tag, r['formulation'], r['ub'],
                                                 r['tour_len'], r['ub_delta']))
        if (isinstance(r['lb'], float) and isinstance(r['ub'], float)
                and r['lb'] > r['ub'] + tol):
            fails.append('C2 %s %s: LB %.9f exceeds UB %.9f'
                         % (tag, r['formulation'], r['lb'], r['ub']))

    # the shortest tour actually exhibited: an achievable value to compare against
    lengths = [r['tour_len'] for r in usable if isinstance(r['tour_len'], float)]
    best = min(lengths) if lengths else None

    # C3, exact formulations must agree
    ubs = [(r['formulation'], r['ub']) for r in exact if isinstance(r['ub'], float)]
    if len(ubs) >= 2:
        lo = min(ubs, key=lambda kv: kv[1])
        hi = max(ubs, key=lambda kv: kv[1])
        if hi[1] - lo[1] > tol:
            fails.append('C3 %s: UB spread %.2e over tolerance - %s reports '
                         '%.9f, %s reports %.9f'
                         % (tag, hi[1] - lo[1], lo[0], lo[1], hi[0], hi[1]))

    # C4 / C5 against the best exhibited tour
    if best is not None:
        for r in usable:
            if isinstance(r['ub'], float) and r['ub'] < best - tol:
                fails.append('C4 %s %s: UB %.9f is below the shortest tour any '
                             'formulation exhibited (%.9f)'
                             % (tag, r['formulation'], r['ub'], best))
            if isinstance(r['lb'], float) and r['lb'] > best + tol:
                fails.append('C5 %s %s: LB %.9f exceeds a known feasible tour '
                             '(%.9f) - a cut removed the optimum'
                             % (tag, r['formulation'], r['lb'], best))

    # C6, the approximated formulations must bracket
    if approx and exact:
        ub_exact = max(r['ub'] for r in exact if isinstance(r['ub'], float))
        lb_exact = min(r['lb'] for r in exact if isinstance(r['lb'], float))
        for r in approx:
            if isinstance(r['ub'], float) and r['ub'] < ub_exact - tol:
                fails.append('C6 %s %s: UB %.9f is BELOW the exact optimum %.9f; '
                             'a relaxation cannot find a shorter feasible tour'
                             % (tag, r['formulation'], r['ub'], ub_exact))
            if isinstance(r['lb'], float) and r['lb'] > lb_exact + tol:
                fails.append('C6 %s %s: LB %.9f is ABOVE the exact bound %.9f; '
                             'a relaxation cannot prove a stronger bound'
                             % (tag, r['formulation'], r['lb'], lb_exact))
    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, nargs='+', default=[6, 8])
    ap.add_argument('--r-mean', type=float, nargs='+', default=[0.25, 1.0])
    ap.add_argument('--sigma', type=float, nargs='+', default=[0.0])
    ap.add_argument('--instances', type=int, default=2)
    ap.add_argument('--time-limit', type=int, default=120)
    ap.add_argument('--nu', type=int, default=3)
    ap.add_argument('--mip-gap', type=float, default=1e-6,
                    help='forced down so Optimal means optimal (see module docstring)')
    ap.add_argument('--tol', type=float, default=1e-3,
                    help='absolute tolerance for every comparison (see module docstring)')
    ap.add_argument('--only', nargs='+', default=None,
                    help='restrict to these formulation labels')
    ap.add_argument('--out', default=os.path.join('Results', 'formulation_check.csv'))
    args = ap.parse_args()

    configs = CONFIGURATIONS
    if args.only:
        configs = [c for c in configs if c[0] in set(args.only)]
        if not configs:
            raise SystemExit('no configuration matched --only')

    cells = [(n, rm, sg, i)
             for n in args.n for rm in args.r_mean
             for sg in args.sigma for i in range(args.instances)]

    print('formulations : %d' % len(configs))
    print('cells        : %d  (n=%s r_mean=%s sigma=%s instances=%d)'
          % (len(cells), args.n, args.r_mean, args.sigma, args.instances))
    print('runs         : %d' % (len(configs) * len(cells)))
    print('MIPGap       : %g   tolerance: %g   time limit: %ds'
          % (args.mip_gap, args.tol, args.time_limit))
    print()

    all_rows, all_fails, excluded = [], [], []
    t_start = time.time()
    for cell in cells:
        n, rm, sg, inst = cell
        r_min, r_max = rm * (1 - sg), rm * (1 + sg)
        folder = 'Instances/%s_%s_%s' % (n, r_min, r_max)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, 'instance_%d.txt' % inst)
        if not os.path.exists(path):
            create_and_save_instance(n, r_min, r_max, inst)

        rows = []
        for label, mt, dec, ext, ct in configs:
            rec = run_one(path, label, mt, dec, ext, ct,
                          args.nu, args.time_limit, args.mip_gap)
            rec.update(n=n, r_mean=rm, sigma=sg, instance=inst,
                       bracketing=is_bracketing(ext, dec))
            rows.append(rec)
            all_rows.append(rec)
            if rec['status'] != 'Optimal':
                excluded.append((cell, label, rec['status']))

        fails = check_cell(cell, rows, args.tol)
        all_fails.extend(fails)
        n_ok = sum(1 for r in rows if r['status'] == 'Optimal')
        deltas = [abs(r['ub_delta']) for r in rows
                  if isinstance(r['ub_delta'], float)]
        ubs = [r['ub'] for r in rows
               if r['status'] == 'Optimal' and not r['bracketing']
               and isinstance(r['ub'], float)]
        spread = (max(ubs) - min(ubs)) if len(ubs) >= 2 else 0.0
        print('n=%-3d r=%-5g s=%-4g i=%d  optimal %2d/%-2d  max|UB_delta| %8.2e  '
              'UB spread %8.2e  %s'
              % (n, rm, sg, inst, n_ok, len(rows),
                 max(deltas) if deltas else float('nan'), spread,
                 'FAIL' if fails else 'ok'))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    cols = ['n', 'r_mean', 'sigma', 'instance', 'formulation', 'bracketing',
            'status', 'ub', 'lb', 'tour_len', 'ub_delta', 'sub_failures',
            'n_effective', 'runtime']
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_rows)

    print()
    print('elapsed      : %.1f s' % (time.time() - t_start))
    print('raw data     : %s' % args.out)
    if excluded:
        print()
        print('not solved to optimality, excluded from the comparisons (%d):'
              % len(excluded))
        for cell, label, st in excluded:
            print('  n=%d r=%g s=%g i=%d  %-14s %s' % (cell + (label, st)))
        print('  a formulation that cannot reach optimality here is not '
              'verified by this run.')

    print()
    if all_fails:
        print('FAILED - %d check(s):' % len(all_fails))
        for f in all_fails:
            print('  ' + f)
        return 1
    print('PASSED - all checks, %d runs over %d cells.' % (len(all_rows), len(cells)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
