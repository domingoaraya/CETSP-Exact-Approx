"""
Sanity check for the PBF dual cut's coefficient kernel (classes/cut_coefficients.py).

A wrong kernel fails silently: any gamma with ||gamma|| <= 1 still yields a valid
cut, so a defective one produces correct optima with weaker bounds and looks
exactly like a feature that did not help. These checks are the only thing that
distinguishes the two, so they run independently of any solve.

Six checks, over random draws plus forced degenerate cases:

  K1  ||gamma|| <= 1. Cut validity rests on this alone.
  K2  h(gamma) vs h(0): the completion should not come out weaker than the
      trivial one it replaces. The start is not gamma = 0, so monotone descent
      does not make this exact; it is checked as a rate and a magnitude.
  K3  at a high iteration budget, h(gamma) matches an exact SOCP solve. Tests the
      algorithm, independently of the production budget.
  K4  at the production budget, the fraction of available gain retained. Reported
      as a distribution; tests the budget, independently of the algorithm.
  K5  the anchor codes against the closed-form conditions, re-derived here.
  K6  the assembled coefficient equals its definition, and the gamma = 0 case
      reproduces the trivial completion in classes/solver.py exactly.

Exits non-zero if any check fails, so it can gate a commit.
"""
import argparse
import os
import sys

import numpy as np
from gurobipy import GRB, Model

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classes.cut_coefficients import MAX_ITER, TOL_MERGE, _anchor_point, _min_h, fill_coefficients

HIGH_ITER = 20000


def h(g, c, r1, r2, e1, e2):
    return float(c @ g + r1 * np.linalg.norm(g - e1) + r2 * np.linalg.norm(g - e2))


def draw(rng, n_each):
    """Random arcs plus one forced batch per degenerate case."""
    def eta():
        v = rng.normal(size=2)
        return v / np.linalg.norm(v) * rng.uniform(0.0, 1.0) ** 0.5

    def circle():
        v = rng.normal(size=2)
        return v / np.linalg.norm(v)

    def c_vec():
        return rng.normal(size=2) * rng.uniform(0.05, 5.0)

    def r():
        return rng.uniform(0.0, 2.0)

    out = []
    for _ in range(n_each):
        out.append(('generic', c_vec(), r(), r(), eta(), eta()))
        e = eta()
        out.append(('merged', c_vec(), r(), r(), e, e.copy()))
        d = rng.normal(size=2)
        d /= np.linalg.norm(d)
        out.append(('near-merged 1e-9', c_vec(), r(), r(), e, e + 1e-9 * d))
        out.append(('eta on circle', c_vec(), r(), r(), circle(), circle()))
        ec = circle()
        out.append(('merged on circle', c_vec(), r(), r(), ec, ec.copy()))
        out.append(('depot r1=0', c_vec(), 0.0, r(), eta(), eta()))
        out.append(('both radii 0', c_vec(), 0.0, 0.0, eta(), eta()))
        out.append(('c=0 concentric', np.zeros(2), r(), r(), eta(), eta()))
    return out


class Oracle:
    """min <c,g> + r1 t1 + r2 t2  s.t.  ||g|| <= 1, ||rho_k - g|| <= t_k.

    Built once and mutated per arc, so the quadratic constraints stay structural.
    """

    def __init__(self):
        m = Model('completion')
        m.setParam('OutputFlag', 0)
        m.setParam('BarConvTol', 1e-10)
        m.setParam('BarQCPConvTol', 1e-10)
        self.g = [m.addVar(lb=-GRB.INFINITY) for _ in range(2)]
        self.rho1 = [m.addVar(lb=-GRB.INFINITY) for _ in range(2)]
        self.rho2 = [m.addVar(lb=-GRB.INFINITY) for _ in range(2)]
        self.t1 = m.addVar(lb=0.0)
        self.t2 = m.addVar(lb=0.0)
        m.update()
        m.addQConstr(self.g[0] ** 2 + self.g[1] ** 2 <= 1.0)
        m.addQConstr((self.rho1[0] - self.g[0]) ** 2 + (self.rho1[1] - self.g[1]) ** 2 <= self.t1 ** 2)
        m.addQConstr((self.rho2[0] - self.g[0]) ** 2 + (self.rho2[1] - self.g[1]) ** 2 <= self.t2 ** 2)
        self.fix1 = [m.addConstr(self.rho1[k] == 0.0) for k in range(2)]
        self.fix2 = [m.addConstr(self.rho2[k] == 0.0) for k in range(2)]
        m.setObjective(0, GRB.MINIMIZE)
        m.update()
        self.m = m

    def __call__(self, c, r1, r2, e1, e2):
        for k in range(2):
            self.fix1[k].RHS = float(e1[k])
            self.fix2[k].RHS = float(e2[k])
            self.g[k].Obj = float(c[k])
        self.t1.Obj = float(r1)
        self.t2.Obj = float(r2)
        self.m.optimize()
        if self.m.Status != GRB.OPTIMAL:
            return None
        return h(np.array([self.g[0].X, self.g[1].X]), c, r1, r2, e1, e2)


def expected_code(c, r1, r2, e1, e2):
    """The anchor conditions re-derived from the tex, independently of the kernel."""
    def ok(G, r, e):
        ne = float(e @ e)
        nG = float(G @ G)
        if ne < (1.0 - 1e-12) ** 2:
            return nG <= r * r
        ip = float(G @ e)
        if ip >= 0.0:
            return nG <= r * r
        return nG - ip * ip / ne <= r * r

    if float((e1 - e2) @ (e1 - e2)) > TOL_MERGE ** 2:
        u = (e1 - e2) / np.linalg.norm(e1 - e2)
        if ok(c + r2 * u, r1, e1):
            return 1
        if ok(c - r1 * u, r2, e2):
            return 2
        return 0
    if ok(c, r1 + r2, 0.5 * (e1 + e2)):
        return 3
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples', type=int, default=300, help='draws per degenerate case')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min-retained', type=float, default=0.60)
    ap.add_argument('--k2-max-excess', type=float, default=1e-3)
    ap.add_argument('--k2-max-rate', type=float, default=0.01)
    ap.add_argument('--mean-retained', type=float, default=0.99)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cases = draw(rng, args.samples)
    oracle = Oracle()

    kinds = sorted({k for k, *_ in cases})
    stats = {k: {'n': 0, 'k1': 0, 'k2': 0, 'k2_excess': 0.0, 'k3': 0.0, 'k5': 0,
                 'retained': []} for k in kinds}
    failures = []

    for kind, c, r1, r2, e1, e2 in cases:
        s = stats[kind]
        s['n'] += 1

        g = np.array(_min_h(c[0], c[1], r1, r2, e1[0], e1[1], e2[0], e2[1], MAX_ITER))
        g_hi = np.array(_min_h(c[0], c[1], r1, r2, e1[0], e1[1], e2[0], e2[1], HIGH_ITER))

        if not np.isfinite(g).all() or np.linalg.norm(g) > 1.0 + 1e-12:
            s['k1'] += 1
            failures.append(f'K1 {kind}: ||gamma||={np.linalg.norm(g):.17g}')

        hv, h0, h_hi = h(g, c, r1, r2, e1, e2), h(np.zeros(2), c, r1, r2, e1, e2), h(g_hi, c, r1, r2, e1, e2)
        if hv > h0 + 1e-12:
            s['k2'] += 1
            s['k2_excess'] = max(s['k2_excess'], hv - h0)

        ref = oracle(c, r1, r2, e1, e2)
        if ref is not None:
            s['k3'] = max(s['k3'], h_hi - ref)
            avail = h0 - ref
            if avail > 1e-9:
                s['retained'].append((h0 - hv) / avail)

        code, gx, gy = _anchor_point(c[0], c[1], r1, r2, e1[0], e1[1], e2[0], e2[1])
        if code != expected_code(c, r1, r2, e1, e2):
            s['k5'] += 1
            failures.append(f'K5 {kind}: kernel {code} != expected {expected_code(c, r1, r2, e1, e2)}')

    # K2/K3/K4 verdicts
    k2_n = sum(s['k2'] for s in stats.values())
    k2_excess = max(s['k2_excess'] for s in stats.values())
    if k2_excess > args.k2_max_excess:
        failures.append(f'K2: worst excess over h(0) {k2_excess:.3e} > {args.k2_max_excess:.0e}')
    if k2_n / len(cases) > args.k2_max_rate:
        failures.append(f'K2: violation rate {k2_n / len(cases):.4%} > {args.k2_max_rate:.2%}')
    k3_max = max(s['k3'] for s in stats.values())
    all_ret = np.array([x for s in stats.values() for x in s['retained']])
    if k3_max > 1e-6:
        failures.append(f'K3: max gap at {HIGH_ITER} iterations {k3_max:.3e} > 1e-6')
    if all_ret.size and all_ret.mean() < args.mean_retained:
        failures.append(f'K4: mean retained {all_ret.mean():.4f} < {args.mean_retained}')
    if all_ret.size and all_ret.min() < args.min_retained:
        failures.append(f'K4: min retained {all_ret.min():.4f} < {args.min_retained}')

    # K6: the assembled coefficient equals its definition, and gamma = 0 is the
    # trivial completion currently in classes/solver.py.
    n = 7
    centers = np.ascontiguousarray(rng.uniform(0, 10, (n, 2)), dtype=np.float64)
    radii = np.ascontiguousarray(rng.uniform(0.2, 0.9, n), dtype=np.float64)
    radii[0] = 0.0
    etas = np.ascontiguousarray([e / max(1.0, np.linalg.norm(e))
                                 for e in rng.normal(size=(n, 2))], dtype=np.float64)
    perm = np.concatenate(([0], rng.permutation(np.arange(1, n))))
    succ = np.zeros(n, dtype=np.int64)
    for k in range(n):
        succ[perm[k]] = perm[(k + 1) % n]
    C = np.zeros((n, n))
    fill_coefficients(centers, radii, etas, succ, C, MAX_ITER)

    k6a = k6b = 0.0
    for i in range(n):
        for j in range(n):
            if j == i or j == succ[i]:
                continue
            c = centers[j] - centers[i]
            const = float(centers[i] @ etas[i] - centers[j] @ etas[j])
            g = np.array(_min_h(c[0], c[1], radii[i], radii[j],
                                etas[i][0], etas[i][1], etas[j][0], etas[j][1], MAX_ITER))
            k6b = max(k6b, abs(C[i, j] - (const + h(g, c, radii[i], radii[j], etas[i], etas[j]))))
            solver_trivial = (centers[i][0] * etas[i][0] + centers[i][1] * etas[i][1]
                              + np.linalg.norm(etas[i]) * radii[i]
                              - centers[j][0] * etas[j][0] - centers[j][1] * etas[j][1]
                              + np.linalg.norm(etas[j]) * radii[j])
            k6a = max(k6a, abs(solver_trivial - (const + h(np.zeros(2), c, radii[i], radii[j],
                                                           etas[i], etas[j]))))
    if k6a > 1e-12:
        failures.append(f'K6a: trivial completion mismatch {k6a:.3e}')
    if k6b > 1e-12:
        failures.append(f'K6b: assembled coefficient mismatch {k6b:.3e}')

    # Report
    print(f'samples      : {len(cases)}  ({args.samples} per case)')
    print(f'budget       : MAX_ITER={MAX_ITER}   reference budget={HIGH_ITER}\n')
    print(f'{"case":18s} {"n":>5s} {"K1":>4s} {"K2":>4s} {"K2 excess":>11s} {"K5":>4s} '
          f'{"K3 max gap":>12s} {"K4 mean":>9s} {"K4 min":>9s}')
    for k in kinds:
        s = stats[k]
        ret = np.array(s['retained'])
        mean_r = f'{ret.mean() * 100:8.3f}%' if ret.size else f'{"-":>9s}'
        min_r = f'{ret.min() * 100:8.3f}%' if ret.size else f'{"-":>9s}'
        print(f'{k:18s} {s["n"]:5d} {s["k1"]:4d} {s["k2"]:4d} {s["k2_excess"]:11.3e} '
              f'{s["k5"]:4d} {s["k3"]:12.3e} {mean_r} {min_r}')
    print(f'\nK6a trivial-completion identity : {k6a:.3e}')
    print(f'K6b assembled-coefficient identity: {k6b:.3e}')

    if failures:
        print(f'\nFAILED - {len(failures)} problem(s):')
        for f in failures[:20]:
            print(f'  {f}')
        if len(failures) > 20:
            print(f'  ... and {len(failures) - 20} more')
        return 1
    print(f'\nPASSED - all checks, {len(cases)} samples.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
