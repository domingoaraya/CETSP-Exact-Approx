"""
Coefficient completion for the PBF dual cut.

On an arc (i, j) outside the support of the current tour, any gamma in the unit
ball gives a valid cut; the strongest one minimizes

    h(gamma) = <c_j - c_i, gamma> + r_i ||gamma - eta_i|| + r_j ||gamma - eta_j||

over the ball (see the paper). The minimizer is approximated by a closed-form
test of the two kinks of h followed by a projected Weiszfeld iteration. The
returned coefficient includes c_i.eta_i - c_j.eta_j but not m_ij, which the
caller adds.
"""
import numpy as np
from numba import njit
from math import sqrt

MAX_ITER = 20
TOL_MERGE = 1e-8

_TOL_MERGE2 = TOL_MERGE * TOL_MERGE
_INTERIOR = (1.0 - 1e-12) ** 2


def pack_instance(data):
    """Instance geometry as contiguous float64 arrays, row i <-> region i."""
    n = data.n
    centers = np.ascontiguousarray([data.centers[i] for i in range(n)], dtype=np.float64)
    radii = np.ascontiguousarray([data.radii[i] for i in range(n)], dtype=np.float64)
    return centers, radii


@njit
def _h(gx, gy, cx, cy, r1, r2, e1x, e1y, e2x, e2y):
    """h(gamma) for gamma = (gx, gy), c = (cx, cy), anchors e1, e2 and weights r1, r2."""
    ax = gx - e1x
    ay = gy - e1y
    bx = gx - e2x
    by = gy - e2y
    return (cx * gx + cy * gy + r1 * sqrt(ax * ax + ay * ay)
            + r2 * sqrt(bx * bx + by * by))


@njit
def _anchor_ok(Gx, Gy, r, ex, ey):
    """Optimality test of an anchor: 0 in G + r B + N_B(eta), where G is the smooth part of the subdifferential of h there."""
    ne2 = ex * ex + ey * ey
    nG2 = Gx * Gx + Gy * Gy
    if ne2 < _INTERIOR:
        return nG2 <= r * r
    ip = Gx * ex + Gy * ey
    if ip >= 0.0:
        return nG2 <= r * r
    return nG2 - ip * ip / ne2 <= r * r


@njit
def _anchor_point(cx, cy, r1, r2, e1x, e1y, e2x, e2y):
    """
    Closed-form optimality test of the kinks of h, which the iteration cannot
    reach. Returns (code, gx, gy) with code 0 if none is optimal, 1 for eta_1,
    2 for eta_2, and 3 for their common value when they coincide.
    """
    dx = e1x - e2x
    dy = e1y - e2y
    d2 = dx * dx + dy * dy
    if d2 > _TOL_MERGE2:
        d = sqrt(d2)
        ux = dx / d
        uy = dy / d
        if _anchor_ok(cx + r2 * ux, cy + r2 * uy, r1, e1x, e1y):
            return 1, e1x, e1y
        if _anchor_ok(cx - r1 * ux, cy - r1 * uy, r2, e2x, e2y):
            return 2, e2x, e2y
        return 0, 0.0, 0.0
    # Anchors coincide: h collapses to a single Weber term of weight r1 + r2.
    ex = 0.5 * (e1x + e2x)
    ey = 0.5 * (e1y + e2y)
    if _anchor_ok(cx, cy, r1 + r2, ex, ey):
        return 3, ex, ey
    return 0, 0.0, 0.0


@njit
def _min_h(cx, cy, r1, r2, e1x, e1y, e2x, e2y, max_iter):
    """
    Approximate minimizer of h over the unit ball: the kinks in closed form,
    then max_iter projected Weiszfeld steps. The result is never worse than
    gamma = 0 or either anchor, so the cut is at least as strong as the plain
    completion and stays tight on reorderings of straight pass-through
    clusters (see the paper).
    """
    code, ax0, ay0 = _anchor_point(cx, cy, r1, r2, e1x, e1y, e2x, e2y)
    if code != 0:
        return ax0, ay0

    nc = sqrt(cx * cx + cy * cy)
    if r1 == 0.0 and r2 == 0.0:
        # h is linear; the minimizer is on the circle.
        if nc > 0.0:
            return -cx / nc, -cy / nc
        return 0.0, 0.0

    # Starting point: balance of the three pulls (anchors weighted by r_1, r_2,
    # linear term by ||c||), projected onto the ball; gamma = 0 if better.
    den = r1 + r2 + nc
    gx = (r1 * e1x + r2 * e2x - cx) / den
    gy = (r1 * e1y + r2 * e2y - cy) / den
    ng = sqrt(gx * gx + gy * gy)
    if ng > 1.0:
        gx /= ng
        gy /= ng

    if _h(0.0, 0.0, cx, cy, r1, r2, e1x, e1y, e2x, e2y) < _h(gx, gy, cx, cy, r1, r2, e1x, e1y, e2x, e2y):
        gx = 0.0
        gy = 0.0

    for _ in range(max_iter):
        ax = gx - e1x
        ay = gy - e1y
        bx = gx - e2x
        by = gy - e2y
        d1 = sqrt(ax * ax + ay * ay)
        db = sqrt(bx * bx + by * by)
        if d1 == 0.0 or db == 0.0:
            # On an anchor: feasible, stop.
            break
        w1 = r1 / d1
        w2 = r2 / db
        inv = 1.0 / (w1 + w2)
        wx = (w1 * e1x + w2 * e2x - cx) * inv
        wy = (w1 * e1y + w2 * e2y - cy) * inv
        nw = sqrt(wx * wx + wy * wy)
        if nw > 1.0:
            gx = wx / nw
            gy = wy / nw
        else:
            gx = wx
            gy = wy

    # Keep the best of the iterate and the two anchors (projected onto the
    # ball: solver output can exceed norm 1 by rounding).
    hg = _h(gx, gy, cx, cy, r1, r2, e1x, e1y, e2x, e2y)
    s1 = 1.0 / max(1.0, sqrt(e1x * e1x + e1y * e1y))
    h1 = _h(e1x * s1, e1y * s1, cx, cy, r1, r2, e1x, e1y, e2x, e2y)
    if h1 < hg:
        gx, gy, hg = e1x * s1, e1y * s1, h1
    s2 = 1.0 / max(1.0, sqrt(e2x * e2x + e2y * e2y))
    h2 = _h(e2x * s2, e2y * s2, cx, cy, r1, r2, e1x, e1y, e2x, e2y)
    if h2 < hg:
        gx, gy, hg = e2x * s2, e2y * s2, h2
    return gx, gy


@njit
def fill_coefficients(centers, radii, etas, succ, C_out, max_iter):
    """Writes the optimized C_ij into C_out for every arc (i, j) with j != succ[i]; support arcs are left untouched."""
    n = centers.shape[0]
    ce = np.empty(n)
    for i in range(n):
        ce[i] = centers[i, 0] * etas[i, 0] + centers[i, 1] * etas[i, 1]

    for i in range(n):
        cix = centers[i, 0]
        ciy = centers[i, 1]
        r1 = radii[i]
        e1x = etas[i, 0]
        e1y = etas[i, 1]
        si = succ[i]
        for j in range(n):
            if j == i or j == si:
                continue
            cx = centers[j, 0] - cix
            cy = centers[j, 1] - ciy
            r2 = radii[j]
            e2x = etas[j, 0]
            e2y = etas[j, 1]
            gx, gy = _min_h(cx, cy, r1, r2, e1x, e1y, e2x, e2y, max_iter)
            C_out[i, j] = ce[i] - ce[j] + _h(gx, gy, cx, cy, r1, r2, e1x, e1y, e2x, e2y)


def _warmup():
    """Compiles the kernels at import time rather than inside a Gurobi callback."""
    centers = np.ascontiguousarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    radii = np.ascontiguousarray([0.0, 0.5, 0.5], dtype=np.float64)
    etas = np.ascontiguousarray([[0.1, 0.2], [0.3, -0.1], [0.3, -0.1]], dtype=np.float64)
    succ = np.array([1, 2, 0], dtype=np.int64)
    fill_coefficients(centers, radii, etas, succ, np.zeros((3, 3)), MAX_ITER)


_warmup()
