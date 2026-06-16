#!/usr/bin/env python3
"""
roof_composite.py  --  Option B: physically-correct composite roofs
===================================================================

Builds L-shape / T-shape / cross-gable roofs as the UPPER ENVELOPE of several
gable "wings", so that valleys and junction vertices arise correctly.

Core idea
---------
A wing is an axis-aligned rectangular footprint with a symmetric gable, giving
two sloped planes  z = a*x + b*y + c  valid inside the wing's footprint.

For any (x,y) covered by the union of footprints:
    roof_z(x, y) = max over covering planes of plane_z(x, y)      (upper envelope)

* POINT CLOUD: sample (x,y) over the union footprint, evaluate the upper
  envelope. Points therefore lie on the *visible* surface only, and valleys
  (creases where the winning plane switches) appear automatically.

* WIREFRAME (label): structural vertices/edges derived from the SAME planes:
    - eaves      = boundary of the union footprint (outer corners + reflex corners)
    - ridges     = each wing's ridge segment, clipped to where it is on top
    - valleys    = intersection of two wings' planes, where both are the envelope
    - junctions  = ridge/ridge or ridge/valley meeting points
  Connectivity is built from these so the label matches the cloud's creases.

Everything is verified: sampled points are checked to lie on the envelope, and
valley vertices are checked to lie on both contributing planes.
"""

import numpy as np
from itertools import combinations

EPS = 1e-6


# ===========================================================================
# Wing = axis-aligned gable. Ridge runs along 'x' (at y=ymid) or 'y' (at x=xmid).
# ===========================================================================
class Wing:
    def __init__(self, x0, x1, y0, y1, axis, eave_h, ridge_h, roof="gable",
                 hip_frac=0.25):
        self.x0, self.x1 = min(x0, x1), max(x0, x1)
        self.y0, self.y1 = min(y0, y1), max(y0, y1)
        self.axis = axis
        self.eh = eave_h
        self.rh = ridge_h
        self.roof = roof              # 'gable' | 'hip' | 'halfhip'
        self.hip_frac = hip_frac      # how far the hip end slopes inward (frac of length)
        self.xmid = 0.5 * (self.x0 + self.x1)
        self.ymid = 0.5 * (self.y0 + self.y1)
        # Two long planes (the main slopes), as before.
        if axis == "x":
            half = (self.y1 - self.y0) / 2.0
            slope = (self.rh - self.eh) / half
            self.planeA = (0.0, slope, self.eh - slope * self.y0)
            self.planeB = (0.0, -slope, self.eh + slope * self.y1)
            self._long = self.x1 - self.x0
        else:
            half = (self.x1 - self.x0) / 2.0
            slope = (self.rh - self.eh) / half
            self.planeA = (slope, 0.0, self.eh - slope * self.x0)
            self.planeB = (-slope, 0.0, self.eh + slope * self.x1)
            self._long = self.y1 - self.y0
        self._slope = slope

        # End planes for hip / half-hip roofs. A hip end slopes from the eave up
        # to the ridge over a setback `hip_frac*length` at each end. half-hip
        # only partially clips (a small triangular hip above a gable).
        self._end_planes = []
        if roof in ("hip", "halfhip"):
            sb = self.hip_frac * self._long
            if sb > 1e-6:
                eslope = (self.rh - self.eh) / sb
                if axis == "x":
                    # end at x0: z = eh + eslope*(x - x0); end at x1: eh + eslope*(x1 - x)
                    self._end_planes = [(eslope, 0.0, self.eh - eslope * self.x0),
                                        (-eslope, 0.0, self.eh + eslope * self.x1)]
                else:
                    self._end_planes = [(0.0, eslope, self.eh - eslope * self.y0),
                                        (0.0, -eslope, self.eh + eslope * self.y1)]
            # half-hip: the end hip only cuts the TOP portion. We emulate this by
            # raising the end-plane's effective eave so it only clips near the ridge.
            if roof == "halfhip":
                lift = 0.5 * (self.rh - self.eh)
                self._end_planes = [(a, b, c + lift) for (a, b, c) in self._end_planes]

    def contains(self, x, y):
        return (self.x0 - EPS <= x <= self.x1 + EPS) and \
               (self.y0 - EPS <= y <= self.y1 + EPS)

    def contains_vec(self, xy):
        x, y = xy[:, 0], xy[:, 1]
        return (x >= self.x0 - EPS) & (x <= self.x1 + EPS) & \
               (y >= self.y0 - EPS) & (y <= self.y1 + EPS)

    def planes(self):
        return [self.planeA, self.planeB]

    def z_at(self, x, y):
        """
        Roof z at (x,y). Gable = min of the two long slope planes. Hip / half-hip
        additionally clip against the end-slope planes (also via min), so the
        roof comes down at the ends too. The footprint (and thus the 2D outline
        label) is unchanged by roof type.
        """
        za = self.planeA[0] * x + self.planeA[1] * y + self.planeA[2]
        zb = self.planeB[0] * x + self.planeB[1] * y + self.planeB[2]
        z = np.minimum(za, zb)
        for (a, b, c) in self._end_planes:
            z = np.minimum(z, a * x + b * y + c)
        return z

    def ridge_segment(self):
        """Endpoints of this wing's ridge line (3D). For hips the ridge is
        shortened by the hip setback at each end."""
        if self.roof in ("hip",):
            sb = self.hip_frac * self._long
        else:
            sb = 0.0
        if self.axis == "x":
            return (np.array([self.x0 + sb, self.ymid, self.rh]),
                    np.array([self.x1 - sb, self.ymid, self.rh]))
        else:
            return (np.array([self.xmid, self.y0 + sb, self.rh]),
                    np.array([self.xmid, self.y1 - sb, self.rh]))


def plane_z(plane, x, y):
    a, b, c = plane
    return a * x + b * y + c


# ===========================================================================
# Upper envelope
# ===========================================================================
def envelope_z(wings, x, y):
    """Max roof z over all wings covering (x,y). Returns z and winning wing idx."""
    best_z = -np.inf
    best_i = -1
    for i, w in enumerate(wings):
        if w.contains(x, y):
            z = float(w.z_at(np.array([x]), np.array([y]))[0])
            if z > best_z:
                best_z, best_i = z, i
    return best_z, best_i


def envelope_z_vec(wings, xy):
    """Vectorized upper envelope over points xy (M,2). Returns z (M,), inside mask."""
    M = xy.shape[0]
    best = np.full(M, -np.inf)
    inside_any = np.zeros(M, dtype=bool)
    for w in wings:
        m = w.contains_vec(xy)
        if not m.any():
            continue
        z = w.z_at(xy[m, 0], xy[m, 1])
        cur = best[m]
        best[m] = np.maximum(cur, z)
        inside_any |= m
    return best, inside_any


# ===========================================================================
# Point sampling over the union footprint
# ===========================================================================
def sample_cloud(wings, density, rng, jitter=0.0):
    """Grid+jitter sample over union footprint bbox, reject points outside union."""
    x0 = min(w.x0 for w in wings); x1 = max(w.x1 for w in wings)
    y0 = min(w.y0 for w in wings); y1 = max(w.y1 for w in wings)
    area = (x1 - x0) * (y1 - y0)
    n_target = max(800, int(area * density))
    # oversample then reject to union
    xy = np.column_stack([rng.uniform(x0, x1, n_target * 2),
                          rng.uniform(y0, y1, n_target * 2)])
    z, inside = envelope_z_vec(wings, xy)
    xy = xy[inside]; z = z[inside]
    if len(xy) > n_target:
        sel = rng.choice(len(xy), n_target, replace=False)
        xy, z = xy[sel], z[sel]
    pts = np.column_stack([xy, z])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, pts.shape)
    return pts


# ===========================================================================
# Wireframe extraction from the SAME wings (eaves, ridges, valleys, junctions)
# ===========================================================================
def union_footprint_rects(wings):
    """Return wing rectangles as (x0,x1,y0,y1) for boundary computation."""
    return [(w.x0, w.x1, w.y0, w.y1) for w in wings]


def _crease_lines(wings):
    """
    Enumerate candidate structural line SEGMENTS (3D) of the roof, derived
    purely from geometry (no heuristics):

      * RIDGE  : each wing's ridge segment (full).
      * VALLEY/HIP : every pairwise plane-plane intersection line between wings,
        clipped to the xy-region where both wings overlap.
      * GABLE-END RAFTERS : each wing's two gable-end slope lines (eave corner
        up to ridge end) -- these are real creases at the triangular gable end.

    The eave loop is handled separately (footprint boundary). Returns a list of
    (p0, p1) 3D segment endpoints.
    """
    segs = []

    # ridges
    for w in wings:
        r0, r1 = w.ridge_segment()
        segs.append((r0, r1))

    # gable-end rafters: at each gable end, the line from the two eave corners
    # to the ridge end. (axis 'x' -> ends at x0 and x1; axis 'y' -> y0,y1)
    for w in wings:
        if w.axis == "x":
            for xend in (w.x0, w.x1):
                re = np.array([xend, w.ymid, w.rh])
                segs.append((np.array([xend, w.y0, w.eh]), re))
                segs.append((np.array([xend, w.y1, w.eh]), re))
        else:
            for yend in (w.y0, w.y1):
                re = np.array([w.xmid, yend, w.rh])
                segs.append((np.array([w.x0, yend, w.eh]), re))
                segs.append((np.array([w.x1, yend, w.eh]), re))

    # pairwise plane intersections (valleys / interior creases)
    for ia, ib in combinations(range(len(wings)), 2):
        wa, wb = wings[ia], wings[ib]
        # overlap rectangle in xy
        ox0, ox1 = max(wa.x0, wb.x0), min(wa.x1, wb.x1)
        oy0, oy1 = max(wa.y0, wb.y0), min(wa.y1, wb.y1)
        if ox0 >= ox1 - EPS or oy0 >= oy1 - EPS:
            continue  # no overlap
        for pa in wa.planes():
            for pb in wb.planes():
                seg = _plane_intersection_in_rect(pa, pb, ox0, ox1, oy0, oy1)
                if seg is not None:
                    segs.append(seg)
    return segs


def _plane_intersection_in_rect(pa, pb, x0, x1, y0, y1):
    """
    Intersection line of two planes z=a x+b y+c, clipped to rect [x0,x1]x[y0,y1].
    Returns (p0,p1) 3D endpoints or None. Planes here have at most one of a,b
    nonzero each (axis-aligned gables), so the intersection is axis-ish; we solve
    generally via the line a-b.
    """
    a0, b0, c0 = pa
    a1, b1, c1 = pb
    da, db, dc = a0 - a1, b0 - b1, c0 - c1
    # line: da*x + db*y + dc = 0  (equal-height locus of the two planes)
    pts = []
    if abs(db) > 1e-9:
        for x in (x0, x1):
            y = -(da * x + dc) / db
            if y0 - 1e-6 <= y <= y1 + 1e-6:
                pts.append((x, y))
    if abs(da) > 1e-9:
        for y in (y0, y1):
            x = -(db * y + dc) / da
            if x0 - 1e-6 <= x <= x1 + 1e-6:
                pts.append((x, y))
    # dedupe
    uniq = []
    for p in pts:
        if not any(abs(p[0]-q[0])<1e-5 and abs(p[1]-q[1])<1e-5 for q in uniq):
            uniq.append(p)
    if len(uniq) < 2:
        return None
    (xa, ya), (xb, yb) = uniq[0], uniq[1]
    za = a0 * xa + b0 * ya + c0
    zb = a0 * xb + b0 * yb + c0
    return (np.array([xa, ya, za]), np.array([xb, yb, zb]))


def build_outline(wings):
    """
    OUTLINE-ONLY label: the building footprint perimeter (eave loop), at
    eave-height z. No internal ridge / valley / junction structure.

    Returns (verts (V,3), edges list[(i,j)]) where the vertices are the ordered
    corners of the union-footprint boundary and the edges form a single closed
    loop. z is the eave height of the covering wing(s) along each boundary span.

    This is the target for 2D vector outline delineation: the cloud stays 3D
    (real roof points), the label is the planar perimeter graph.
    """
    loop = rectilinear_union_boundary(union_footprint_rects(wings))
    verts = []
    for (x, y) in loop:
        # eave height along the boundary: min eh among covering wings
        ehs = [w.eh for w in wings if w.contains(x, y)]
        z = min(ehs) if ehs else wings[0].eh
        verts.append([x, y, z])
    verts = np.asarray(verts, dtype=float)
    n = len(verts)
    edges = [(i, (i + 1) % n) for i in range(n)]   # closed perimeter loop
    return verts, edges


def build_wireframe(wings, snap=0.30, on_tol=1e-2):
    """
    Deterministic crease-arrangement wireframe.

    1. Eave loop from the union footprint (boundary edges).
    2. Candidate crease segments (ridges, rafters, valleys) from geometry.
    3. Build a planar arrangement: split every segment at all mutual
       intersections (and at eave-boundary crossings).
    4. Keep a sub-segment iff its midpoint lies ON the upper envelope
       (z == envelope_z) -- this discards buried creases (e.g. a ridge passing
       under a taller crossing wing) automatically.
    5. Merge near-coincident vertices; return (verts, edges).

    Every kept edge therefore lies on a real, visible crease of the same surface
    the point cloud was sampled from.

    NOTE: for 2D outline delineation use build_outline() instead -- this full
    wireframe is kept for the 3D-structure use case.
    """
    # ---- eave loop ----
    eave_loop = rectilinear_union_boundary(union_footprint_rects(wings))
    eave_segs = []
    for (x, y), (x2, y2) in zip(eave_loop, eave_loop[1:] + eave_loop[:1]):
        ehs = [w.eh for w in wings if w.contains(0.5*(x+x2), 0.5*(y+y2))]
        z = min(ehs) if ehs else wings[0].eh
        eave_segs.append((np.array([x, y, z]), np.array([x2, y2, z])))

    crease_segs = _crease_lines(wings)
    # tag each segment: 'eave' (boundary, keep always at eave z) or
    # 'crease' (interior ridge/valley/rafter, keep iff on envelope)
    all_segs = eave_segs + crease_segs
    seg_kind = ["eave"] * len(eave_segs) + ["crease"] * len(crease_segs)

    # ---- 2D arrangement: collect split parameters for each segment ----
    seg2d = [(s[0][:2], s[1][:2]) for s in all_segs]
    splits = [set([0.0, 1.0]) for _ in all_segs]
    for i in range(len(seg2d)):
        for j in range(i + 1, len(seg2d)):
            hit = seg_intersection_xy(seg2d[i][0], seg2d[i][1],
                                      seg2d[j][0], seg2d[j][1])
            if hit is None:
                continue
            for k, (p, q) in ((i, seg2d[i]), (j, seg2d[j])):
                d = q - p
                L2 = float(d @ d)
                if L2 < 1e-12:
                    continue
                t = float((hit - p) @ d) / L2
                if 1e-4 < t < 1 - 1e-4:
                    splits[k].add(round(t, 6))

    # ---- build vertices + edges from kept sub-segments ----
    V = []
    def add_vertex(p3):
        p3 = np.asarray(p3, float)
        for k, q in enumerate(V):
            if np.linalg.norm(q - p3) < snap:
                return k
        V.append(p3)
        return len(V) - 1

    edges = set()
    def envelope_at(x, y):
        z, inside = envelope_z_vec(wings, np.array([[x, y]]))
        return z[0], bool(inside[0])

    for s, ts, kind in zip(all_segs, splits, seg_kind):
        p0, p1 = s
        ts = sorted(ts)
        for ta, tb in zip(ts[:-1], ts[1:]):
            pa = p0 + (p1 - p0) * ta
            pb = p0 + (p1 - p0) * tb
            mid = 0.5 * (pa + pb)
            ez, inside = envelope_at(mid[0], mid[1])
            if not inside:
                continue
            if kind == "eave":
                # boundary edge: always kept, keep at eave height (z from segment)
                ia = add_vertex(pa)
                ib = add_vertex(pb)
            else:
                # interior crease: keep iff it lies ON the envelope at the midpoint
                if abs(mid[2] - ez) > on_tol:
                    continue
                za, _ = envelope_at(pa[0], pa[1])
                zb, _ = envelope_at(pb[0], pb[1])
                ia = add_vertex([pa[0], pa[1], za])
                ib = add_vertex([pb[0], pb[1], zb])
            if ia != ib:
                edges.add((min(ia, ib), max(ia, ib)))

    verts = np.array(V, dtype=float)
    edges = sorted(edges)
    # prune orphan vertices (none expected, but keep clean)
    used = sorted(set([i for e in edges for i in e]))
    remap = {old: k for k, old in enumerate(used)}
    verts = verts[used]
    edges = [(remap[i], remap[j]) for i, j in edges]
    return verts, edges


def seg_intersection_xy(p1, p2, p3, p4):
    """Intersection of segments p1p2 and p3p4 in 2D, or None."""
    r = p2 - p1
    s = p4 - p3
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-9:
        return None
    t = ((p3[0] - p1[0]) * s[1] - (p3[1] - p1[1]) * s[0]) / denom
    u = ((p3[0] - p1[0]) * r[1] - (p3[1] - p1[1]) * r[0]) / denom
    if -0.01 <= t <= 1.01 and -0.01 <= u <= 1.01:
        return p1 + t * r
    return None


# ===========================================================================
# Rectilinear union boundary of axis-aligned rectangles (for the eave loop)
# ===========================================================================
def rectilinear_union_boundary(rects, grid_eps=1e-3):
    """
    Compute the outer boundary polygon (CCW) of the union of axis-aligned
    rectangles using a coordinate-grid cell test + marching the boundary.
    Returns a list of (x,y) corner vertices (only true corners, collinear
    points removed).
    """
    xs = sorted(set([r[0] for r in rects] + [r[1] for r in rects]))
    ys = sorted(set([r[2] for r in rects] + [r[3] for r in rects]))

    def covered(cx, cy):
        for (x0, x1, y0, y1) in rects:
            if x0 - grid_eps <= cx <= x1 + grid_eps and \
               y0 - grid_eps <= cy <= y1 + grid_eps:
                return True
        return False

    # build a grid of cell-occupancy between consecutive coords
    nx, ny = len(xs) - 1, len(ys) - 1
    occ = np.zeros((nx, ny), dtype=bool)
    for i in range(nx):
        cx = 0.5 * (xs[i] + xs[i + 1])
        for j in range(ny):
            cy = 0.5 * (ys[j] + ys[j + 1])
            occ[i, j] = covered(cx, cy)

    # collect boundary edges (between occupied and non-occupied / outside)
    edges = []
    for i in range(nx):
        for j in range(ny):
            if not occ[i, j]:
                continue
            x_lo, x_hi = xs[i], xs[i + 1]
            y_lo, y_hi = ys[j], ys[j + 1]
            # left
            if i == 0 or not occ[i - 1, j]:
                edges.append(((x_lo, y_lo), (x_lo, y_hi)))
            # right
            if i == nx - 1 or not occ[i + 1, j]:
                edges.append(((x_hi, y_hi), (x_hi, y_lo)))
            # bottom
            if j == 0 or not occ[i, j - 1]:
                edges.append(((x_hi, y_lo), (x_lo, y_lo)))
            # top
            if j == ny - 1 or not occ[i, j + 1]:
                edges.append(((x_lo, y_hi), (x_hi, y_hi)))

    # stitch edges into a loop
    loop = stitch_edges(edges)
    loop = remove_collinear(loop)
    return loop


def stitch_edges(edges):
    """Order a set of directed boundary edges into a single closed loop."""
    if not edges:
        return []
    from collections import defaultdict
    nxt = defaultdict(list)
    for a, b in edges:
        nxt[_key(a)].append(b)
    start = edges[0][0]
    loop = [start]
    cur = start
    used = set()
    for _ in range(len(edges) + 1):
        cand = nxt.get(_key(cur), [])
        moved = False
        for b in cand:
            ek = (_key(cur), _key(b))
            if ek in used:
                continue
            used.add(ek)
            loop.append(b)
            cur = b
            moved = True
            break
        if not moved:
            break
        if _key(cur) == _key(start):
            loop.pop()  # drop duplicate closing point
            break
    return loop


def _key(p, q=3):
    return (round(p[0], q), round(p[1], q))


def remove_collinear(loop):
    """Drop points lying on a straight run, keeping only true corners."""
    if len(loop) < 3:
        return loop
    out = []
    n = len(loop)
    for i in range(n):
        a = np.array(loop[(i - 1) % n])
        b = np.array(loop[i])
        c = np.array(loop[(i + 1) % n])
        v1 = b - a
        v2 = c - b
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) > 1e-6:        # turn -> corner
            out.append(tuple(b))
    return out


# ===========================================================================
# HIGH-CORNER-COUNT generators (9..80 corners) for diverse building shapes.
# Both produce a single rectilinear outline loop + a matched 3D point cloud,
# reusing the upper-envelope wing engine so cloud and label stay consistent.
# ===========================================================================
def _count_corners(loop):
    return len(remove_collinear(loop))


def _fill_holes(cells):
    """
    Fill any enclosed empty cells so the footprint is simply-connected (a single
    outer boundary, no interior courtyards). Flood-fill 'outside' from a margin;
    any empty cell not reached from outside is interior -> add it to the blob.
    """
    if not cells:
        return cells
    is_, js_ = zip(*cells)
    i0, i1 = min(is_) - 1, max(is_) + 1
    j0, j1 = min(js_) - 1, max(js_) + 1
    from collections import deque
    outside = {(i0, j0)}
    dq = deque([(i0, j0)])
    while dq:
        i, j = dq.popleft()
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (i + di, j + dj)
            if (i0 <= nb[0] <= i1 and j0 <= nb[1] <= j1
                    and nb not in cells and nb not in outside):
                outside.add(nb)
                dq.append(nb)
    filled = set(cells)
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            if (i, j) not in cells and (i, j) not in outside:
                filled.add((i, j))    # enclosed empty cell -> fill it
    return filled


def _cells_to_rects(cells, cell):
    """Convert occupied grid cells {(i,j)} to a list of (x0,x1,y0,y1) rects."""
    return [(i * cell, (i + 1) * cell, j * cell, (j + 1) * cell)
            for (i, j) in cells]


def _grow_polyomino(rng, n_cells, raggedness=0.7):
    """
    Grow a connected set of grid cells by random accretion (a 'blob').
    Higher `raggedness` biases growth toward cells that ADD perimeter (fewer
    existing neighbours), producing more concavities -> more boundary corners.
    """
    cells = {(0, 0)}

    def neighbours(c):
        i, j = c
        return [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]

    while len(cells) < n_cells:
        # candidate frontier = empty cells adjacent to the blob
        frontier = set()
        for c in cells:
            for nb in neighbours(c):
                if nb not in cells:
                    frontier.add(nb)
        if not frontier:
            break
        frontier = list(frontier)
        # weight each candidate: cells with FEWER blob-neighbours add more
        # perimeter (raggedness); cells with more neighbours fill in (compact).
        counts = np.array([sum(1 for nb in neighbours(c) if nb in cells)
                           for c in frontier], dtype=float)
        # prefer count==1 (spiky) when ragged, count>=2 (smooth) when not
        if rng.random() < raggedness:
            w = (counts == 1).astype(float) + 0.05
        else:
            w = (counts >= 2).astype(float) + 0.05
        w = w / w.sum()
        pick = frontier[rng.choice(len(frontier), p=w)]
        cells.add(pick)
    return cells


def _height_field_cloud(rng, rects, density, eh_base):
    """
    Sample a point cloud over a set of rects with a gentle per-region roof.
    Each rect gets a small random ridge (low gable) so the cloud has 3D relief
    without needing a full envelope solve for arbitrary blobs.
    """
    clouds = []
    for (x0, x1, y0, y1) in rects:
        w = min(x1 - x0, y1 - y0)
        area = (x1 - x0) * (y1 - y0)
        n = max(30, int(area * density))
        xs = rng.uniform(x0, x1, n)
        ys = rng.uniform(y0, y1, n)
        # low hip-ish bump: height peaks at cell center, falls to eave at edges
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        rise = rng.uniform(0.8, 2.2)
        dx = 1 - np.abs(xs - cx) / (0.5 * (x1 - x0) + 1e-9)
        dy = 1 - np.abs(ys - cy) / (0.5 * (y1 - y0) + 1e-9)
        zs = eh_base + rise * np.minimum(dx, dy)
        clouds.append(np.column_stack([xs, ys, zs]))
    return np.vstack(clouds)


def gen_grid_polyomino(rng, target_corners):
    """
    Build a rectilinear polyomino footprint targeting ~target_corners corners.
    Returns (points 3D, outline_verts, outline_edges).
    """
    cell = rng.uniform(3.0, 6.0)            # metres per grid cell
    eh = rng.uniform(2.5, 4.0)
    density = rng.uniform(12.0, 24.0)

    # corner count scales roughly with number of cells & raggedness; search.
    best = None
    for _ in range(60):
        # high targets need more cells AND more raggedness
        rag = float(np.clip(0.45 + target_corners / 120.0, 0.45, 0.95))
        n_cells = int(np.clip(target_corners * rng.uniform(0.9, 1.6),
                              3, 400))
        cells = _grow_polyomino(rng, n_cells, raggedness=rag)
        cells = _fill_holes(cells)            # ensure single simple loop, no holes
        rects = _cells_to_rects(cells, cell)
        loop = rectilinear_union_boundary(rects)
        if len(loop) < 3:
            continue
        nc = _count_corners(loop)
        if best is None or abs(nc - target_corners) < abs(best[0] - target_corners):
            best = (nc, cells, rects, loop)
        if 9 <= nc <= 80 and abs(nc - target_corners) <= 4:
            break

    nc, cells, rects, loop = best
    loop = remove_collinear(loop)
    verts = np.array([[x, y, eh] for (x, y) in loop], dtype=float)
    n = len(verts)
    edges = [(i, (i + 1) % n) for i in range(n)]
    pts = _height_field_cloud(rng, rects, density, eh)
    # center both on outline centroid
    c = np.array([verts[:, 0].mean(), verts[:, 1].mean(), 0.0])
    return pts - c, verts - c, edges


def gen_complex_multiwing(rng, target_corners):
    """
    Place several axis-aligned gable wings sharing edges; union footprint gives
    a rectilinear outline. Wing count scales corner count (~4-8 corners/wing).
    Uses the upper-envelope engine -> proper gabled roof cloud.
    Returns (points 3D, outline_verts, outline_edges).
    """
    eh = rng.uniform(2.5, 4.0)
    rr = rng.uniform(1.5, 3.5)
    rh = eh + rr
    density = rng.uniform(14.0, 26.0)

    best = None
    for _ in range(40):
        k = int(np.clip(target_corners / 5.0, 2, 18)) + rng.integers(-1, 2)
        k = max(2, k)
        wings = _random_connected_wings(rng, k, eh, rh)
        rects = union_footprint_rects(wings)
        loop = rectilinear_union_boundary(rects)
        if len(loop) < 3:
            continue
        nc = _count_corners(loop)
        if best is None or abs(nc - target_corners) < abs(best[0] - target_corners):
            best = (nc, wings, loop)
        if 9 <= nc <= 80 and abs(nc - target_corners) <= 5:
            break

    nc, wings, loop = best
    pts = sample_cloud(wings, density, rng)
    verts, edges = build_outline(wings)
    c = np.array([verts[:, 0].mean(), verts[:, 1].mean(), 0.0])
    return pts - c, verts - c, edges


def _random_connected_wings(rng, k, eh, rh):
    """Generate k axis-aligned rectangular wings, each attached to a previous one.
    Wings get random roof types and mild per-wing height variation."""
    def mk(x0, x1, y0, y1, axis):
        roof = ["gable", "hip", "halfhip"][rng.integers(3)]
        e = eh * rng.uniform(0.92, 1.08)
        r = e + (rh - eh) * rng.uniform(0.8, 1.15)
        return Wing(x0, x1, y0, y1, axis, e, r, roof=roof,
                    hip_frac=rng.uniform(0.18, 0.30))
    wings = []
    w0 = rng.uniform(6, 14); h0 = rng.uniform(5, 10)
    axis0 = rng.choice(["x", "y"])
    wings.append(mk(0, w0, 0, h0, axis0))
    boxes = [(0, w0, 0, h0)]
    for _ in range(k - 1):
        bx0, bx1, by0, by1 = boxes[rng.integers(len(boxes))]
        wl = rng.uniform(5, 12); ww = rng.uniform(4, 9)
        side = rng.choice(["N", "S", "E", "W"])
        if side == "N":
            x0 = rng.uniform(bx0, max(bx0, bx1 - ww)); x1 = x0 + ww
            y0 = by1; y1 = by1 + wl; axis = "y"
        elif side == "S":
            x0 = rng.uniform(bx0, max(bx0, bx1 - ww)); x1 = x0 + ww
            y1 = by0; y0 = by0 - wl; axis = "y"
        elif side == "E":
            y0 = rng.uniform(by0, max(by0, by1 - ww)); y1 = y0 + ww
            x0 = bx1; x1 = bx1 + wl; axis = "x"
        else:  # W
            y0 = rng.uniform(by0, max(by0, by1 - ww)); y1 = y0 + ww
            x1 = bx0; x0 = bx0 - wl; axis = "x"
        wings.append(mk(x0, x1, y0, y1, axis))
        boxes.append((min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)))
    return wings


# ===========================================================================
# ARCHITECTURAL massing composer (replaces random polyomino).
# Builds footprints from a small alphabet of real building plans -- combs,
# stepped/terraced setbacks, wings-around-core -- as unions of axis-aligned
# gable wings on a coarse shared grid. Corner count scales through the number
# of teeth/steps. Produces a single rectilinear outline loop + matched cloud.
# ===========================================================================
def _wings_from_boxes(boxes, eh, rh, rng, mixed=True):
    """Turn (x0,x1,y0,y1) boxes into gable Wings with sensible ridge axes.
    When mixed=True, each wing gets a random roof type and small per-wing height
    variation, so composites show diverse roofs rather than uniform gables."""
    wings = []
    for (x0, x1, y0, y1) in boxes:
        axis = "x" if (x1 - x0) >= (y1 - y0) else "y"   # ridge along long side
        if mixed:
            roof = ["gable", "hip", "halfhip"][rng.integers(3)]
            e = eh * rng.uniform(0.92, 1.08)
            r = e + (rh - eh) * rng.uniform(0.8, 1.15)
        else:
            roof, e, r = "gable", eh, rh
        wings.append(Wing(x0, x1, y0, y1, axis, e, r, roof=roof,
                          hip_frac=rng.uniform(0.18, 0.30)))
    return wings


def _plan_bar(rng):
    L = rng.uniform(14, 26); W = rng.uniform(8, 13)
    return [(0, L, 0, W)]


def _plan_L(rng):
    L = rng.uniform(14, 22); W = rng.uniform(7, 11)
    wl = rng.uniform(8, 14); ww = rng.uniform(6, 10)
    return [(0, L, 0, W), (L - ww, L, W, W + wl)]


def _plan_T(rng):
    L = rng.uniform(16, 24); W = rng.uniform(7, 11)
    wl = rng.uniform(8, 14); ww = rng.uniform(6, 9)
    cx = 0.5 * L
    return [(0, L, 0, W), (cx - ww / 2, cx + ww / 2, W, W + wl)]


def _plan_U(rng):
    L = rng.uniform(16, 24); W = rng.uniform(8, 12)
    arm = rng.uniform(8, 13); aw = rng.uniform(5, 8)
    return [(0, L, 0, W),
            (0, aw, W, W + arm),
            (L - aw, L, W, W + arm)]


def _plan_H(rng):
    W = rng.uniform(7, 10); span = rng.uniform(14, 20)
    barw = rng.uniform(6, 9)
    legL = rng.uniform(14, 22)
    return [(0, barw, 0, legL),
            (span, span + barw, 0, legL),
            (barw, span, 0.5 * legL - W / 2, 0.5 * legL + W / 2)]


def _plan_C(rng):
    """Courtyard approximated as a C (notch) so the outline stays a single loop."""
    L = rng.uniform(16, 24); W = rng.uniform(14, 20)
    t = rng.uniform(4, 6)                       # wall thickness
    return [(0, L, 0, t),                        # bottom
            (0, t, 0, W),                        # left
            (0, L, W - t, W),                    # top
            (L - t, L, 0, W)]                    # right  (3 walls + back = C)


def _plan_comb(rng, n_teeth):
    """A spine with n_teeth wings projecting from one side (apartment/comb plan)."""
    spineL = rng.uniform(max(16, 5 * n_teeth), max(20, 7 * n_teeth))
    spineW = rng.uniform(6, 9)
    toothL = rng.uniform(7, 12)
    toothW = rng.uniform(2.5, 4.5)
    boxes = [(0, spineL, 0, spineW)]
    gap = spineL / n_teeth
    for k in range(n_teeth):
        cx = (k + 0.5) * gap
        x0 = cx - toothW / 2; x1 = cx + toothW / 2
        boxes.append((x0, x1, spineW, spineW + toothL))
    return boxes


def _plan_double_comb(rng, n_teeth):
    """Spine with teeth on BOTH sides (sawtooth/industrial)."""
    spineL = rng.uniform(max(16, 5 * n_teeth), max(20, 7 * n_teeth))
    spineW = rng.uniform(6, 9)
    toothL = rng.uniform(6, 10)
    toothW = rng.uniform(2.5, 4.0)
    boxes = [(0, spineL, 0, spineW)]
    gap = spineL / n_teeth
    for k in range(n_teeth):
        cx = (k + 0.5) * gap
        x0 = cx - toothW / 2; x1 = cx + toothW / 2
        boxes.append((x0, x1, spineW, spineW + toothL))       # top teeth
        boxes.append((x0, x1, -toothL, 0))                    # bottom teeth
    return boxes


def _plan_stepped(rng, n_steps):
    """Terraced/ziggurat: a bar that steps inward in n_steps along its length.
    Steps are bounded so the narrowest segment stays a realistic width."""
    n_steps = int(min(n_steps, 8))              # cap: real terraces have few steps
    W0 = rng.uniform(12, 18)
    seg = rng.uniform(5, 8)
    min_w = rng.uniform(5, 7)                    # narrowest segment stays usable
    step = (W0 - min_w) / max(1, n_steps)
    boxes = []
    x = 0.0
    for k in range(n_steps + 1):
        w = W0 - k * step
        boxes.append((x, x + seg, 0, w))
        x += seg
    return boxes


def _plan_wings_around(rng, n_wings):
    """A core block with rectangular projections on its sides (palatial plan)."""
    L = rng.uniform(14, 20); W = rng.uniform(12, 18)
    boxes = [(0, L, 0, W)]
    sides = ["N", "S", "E", "W"]
    for k in range(n_wings):
        s = sides[k % 4]
        pl = rng.uniform(5, 9); pw = rng.uniform(4, 7)
        if s == "N":
            cx = rng.uniform(0, L - pw); boxes.append((cx, cx + pw, W, W + pl))
        elif s == "S":
            cx = rng.uniform(0, L - pw); boxes.append((cx, cx + pw, -pl, 0))
        elif s == "E":
            cy = rng.uniform(0, W - pw); boxes.append((L, L + pl, cy, cy + pw))
        else:
            cy = rng.uniform(0, W - pw); boxes.append((-pl, 0, cy, cy + pw))
    return boxes


def gen_architectural(rng, target_corners):
    """
    Compose an architectural footprint targeting ~target_corners corners.
    Picks a plan whose corner-count range straddles the target, scaling teeth/
    steps/wings to hit it. Returns (points 3D, outline_verts, outline_edges).
    """
    eh = rng.uniform(2.5, 4.0)
    rh = eh + rng.uniform(1.5, 3.5)
    density = rng.uniform(14.0, 24.0)

    def pick(funcs):
        return funcs[rng.integers(len(funcs))](rng)

    def make_boxes(t):
        # low targets: simple named plans
        if t <= 6:
            return pick([_plan_bar, _plan_L])
        if t <= 8:
            return pick([_plan_L, _plan_T, _plan_C])
        if t <= 12:
            return pick([_plan_U, _plan_H, _plan_C])
        # higher targets: parametric repeated-feature plans.
        # stepped caps ~20 corners, so reserve it for the mid range; combs and
        # double-combs (apartment / sawtooth blocks) carry the high end.
        if t <= 22:
            choice = ["comb", "stepped", "wings", "double_comb"][rng.integers(4)]
        else:
            choice = ["comb", "double_comb", "wings"][rng.integers(3)]
        if choice == "comb":
            n = int(np.clip(round((t - 4) / 4), 2, 20))
            return _plan_comb(rng, n)
        if choice == "double_comb":
            n = int(np.clip(round((t - 4) / 8), 2, 12))
            return _plan_double_comb(rng, n)
        if choice == "stepped":
            n = int(np.clip(round((t - 4) / 2), 2, 8))
            return _plan_stepped(rng, n)
        n = int(np.clip(round((t - 4) / 4), 2, 18))
        return _plan_wings_around(rng, n)

    best = None
    for _ in range(50):
        boxes = make_boxes(target_corners)
        loop = rectilinear_union_boundary(boxes)
        if len(loop) < 3:
            continue
        nc = len(remove_collinear(loop))
        if best is None or abs(nc - target_corners) < abs(best[0] - target_corners):
            best = (nc, boxes, loop)
        if 9 <= nc <= 80 and abs(nc - target_corners) <= 4:
            break

    nc, boxes, loop = best
    wings = _wings_from_boxes(boxes, eh, rh, rng)
    pts = sample_cloud(wings, density, rng)
    verts, edges = build_outline(wings)
    c = np.array([verts[:, 0].mean(), verts[:, 1].mean(), 0.0])
    return pts - c, verts - c, edges


# ===========================================================================
# FARMHOUSE: a tall main block + a lower attached wing/shed, each with a
# random roof type (gable / hip / half-hip) and DIFFERENT heights. Matches the
# vernacular reference buildings. Footprint outline = union perimeter.
# ===========================================================================
def _rand_roof(rng):
    return ["gable", "hip", "halfhip"][rng.integers(3)]


def gen_farmhouse(rng):
    """
    Main gabled/hipped block + a lower attached annex wing (shed/barn).
    Returns (points 3D, outline_verts, outline_edges).
    """
    # main block
    L = rng.uniform(12, 20); W = rng.uniform(8, 13)
    eh_main = rng.uniform(3.0, 4.5)
    rh_main = eh_main + rng.uniform(2.5, 4.5)        # tall main roof
    main_axis = "x" if L >= W else "y"
    main = Wing(0, L, 0, W, main_axis, eh_main, rh_main,
                roof=_rand_roof(rng), hip_frac=rng.uniform(0.18, 0.30))

    wings = [main]

    # lower annex wing attached to one side (shed/barn): lower eave AND ridge
    eh_annex = rng.uniform(2.2, eh_main - 0.3)
    rh_annex = eh_annex + rng.uniform(1.2, 2.6)
    rh_annex = min(rh_annex, rh_main - 0.6)          # always lower than main
    al = rng.uniform(6, 12); aw = rng.uniform(5, 9)
    side = ["N", "S", "E", "W"][rng.integers(4)]
    if side == "N":
        x0 = rng.uniform(0, max(0, L - aw)); 
        annex = Wing(x0, x0 + aw, W, W + al, "y", eh_annex, rh_annex,
                     roof=_rand_roof(rng))
    elif side == "S":
        x0 = rng.uniform(0, max(0, L - aw))
        annex = Wing(x0, x0 + aw, -al, 0, "y", eh_annex, rh_annex,
                     roof=_rand_roof(rng))
    elif side == "E":
        y0 = rng.uniform(0, max(0, W - aw))
        annex = Wing(L, L + al, y0, y0 + aw, "x", eh_annex, rh_annex,
                     roof=_rand_roof(rng))
    else:
        y0 = rng.uniform(0, max(0, W - aw))
        annex = Wing(-al, 0, y0, y0 + aw, "x", eh_annex, rh_annex,
                     roof=_rand_roof(rng))
    wings.append(annex)

    # optionally a second small shed (some reference buildings have two annexes)
    if rng.random() < 0.4:
        eh2 = rng.uniform(2.2, eh_main - 0.3)
        rh2 = min(eh2 + rng.uniform(1.0, 2.2), rh_main - 0.6)
        sl = rng.uniform(4, 8); sw = rng.uniform(4, 7)
        y0 = rng.uniform(0, max(0, W - sw))
        wings.append(Wing(-sl, 0, y0, y0 + sw, "x", eh2, rh2, roof=_rand_roof(rng)))

    density = rng.uniform(16.0, 26.0)
    pts = sample_cloud(wings, density, rng)
    verts, edges = build_outline(wings)
    c = np.array([verts[:, 0].mean(), verts[:, 1].mean(), 0.0])
    return pts - c, verts - c, edges


# ===========================================================================
# WALL / FACADE point sampling.
# Given the footprint outline (perimeter at eave height), drop points down each
# wall face from the eave to the ground (z=0), so the synthetic cloud matches
# the real connected-component clouds which include facade returns.
# ===========================================================================
def sample_walls(outline_verts, edges, rng, density, ground_z=0.0, jitter=0.02):
    """
    outline_verts : (V,3) perimeter vertices at eave height (z = local eave).
    edges         : list of (i,j) loop edges.
    density       : approx points per square metre of wall.
    Returns (M,3) wall points spanning ground_z .. eave height along each edge.
    """
    walls = []
    for i, j in edges:
        a = outline_verts[i]
        b = outline_verts[j]
        seg_len = float(np.linalg.norm(b[:2] - a[:2]))
        if seg_len < 1e-6:
            continue
        # eave height varies linearly along the edge between the two corner z's
        wall_h = 0.5 * (a[2] + b[2]) - ground_z
        if wall_h <= 0:
            continue
        n = max(8, int(seg_len * wall_h * density))
        t = rng.random(n)                       # along the edge
        h = rng.random(n)                        # up the wall (0=ground,1=eave)
        x = a[0] + (b[0] - a[0]) * t
        y = a[1] + (b[1] - a[1]) * t
        z_eave = a[2] + (b[2] - a[2]) * t
        z = ground_z + h * (z_eave - ground_z)
        pts = np.column_stack([x, y, z])
        if jitter > 0:
            pts = pts + rng.normal(0, jitter, pts.shape)
        walls.append(pts)
    if not walls:
        return np.empty((0, 3))
    return np.vstack(walls)
