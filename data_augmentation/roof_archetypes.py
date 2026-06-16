#!/usr/bin/env python3
"""
roof_archetypes.py
==================

Procedural generators for parametric roof ARCHETYPES, each producing a matched
pair for Point2Roof:

    - a roof POINT CLOUD  (points sampled on the actual roof planes)        -> (N,3)
    - an outline GRAPH    (structural vertices + model edges)               -> (V,3), [(i,j),...]

Unlike affine augmentation (which only re-poses existing samples), these create
GENUINELY NEW TOPOLOGIES: gable, hip, pyramid (hip with no ridge), L-shape,
T-shape, and cross-gable. This expands the *distribution of roof graphs* the
network sees.

Conventions (matched to the Point2Roof label format)
----------------------------------------------------
* The outline graph is the roof STRUCTURE wireframe: eave corners, ridge ends,
  and hip apexes, connected by the real model edges (eaves, ridges, hips).
  This mirrors the multi-plane sample (14 v / 14 e) rather than a bare footprint
  rectangle.
* Vertices carry true z (eave height for eave corners, ridge height for ridge
  vertices). For a purely-2D-outline target you can flatten z downstream.
* Points are sampled ON the sloped roof planes so the cloud is consistent with
  the wireframe -- a ridge shows up as a height crease exactly where the ridge
  edge is, like in your real samples.

Each generator returns: (points (N,3), verts (V,3), edges list[(i,j)] 0-indexed).
Coordinates are in meters, roughly building-scale; the caller can normalize.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Parameter ranges controlling the random building dimensions (meters).
# Tune to match your real data's scale distribution.
# ---------------------------------------------------------------------------
PARAMS = {
    "len":        (8.0, 22.0),    # primary footprint length
    "wid":        (6.0, 14.0),    # primary footprint width
    "eave_h":     (2.5, 4.0),     # height of the eaves (wall top)
    "ridge_rise": (1.5, 4.0),     # additional height from eave to ridge
    "hip_setback_frac": (0.15, 0.30),  # how far the hip ridge is inset (fraction of length)
    "wing_len":   (6.0, 14.0),    # secondary wing length (L/T/cross)
    "wing_wid":   (5.0, 10.0),    # secondary wing width
    "density":    (12.0, 28.0),   # points per square meter of roof plane
    "min_pts":    (400,),         # floor on total points
}


def _u(rng, key):
    lo, hi = PARAMS[key]
    return rng.uniform(lo, hi)


# ===========================================================================
# Surface sampling helpers
# ===========================================================================
def _sample_triangle(rng, p0, p1, p2, n):
    """Uniformly sample n points on triangle (p0,p1,p2). Returns (n,3)."""
    r1 = np.sqrt(rng.random(n))
    r2 = rng.random(n)
    a = 1 - r1
    b = r1 * (1 - r2)
    c = r1 * r2
    return (a[:, None] * p0 + b[:, None] * p1 + c[:, None] * p2)


def _sample_quad(rng, p0, p1, p2, p3, n):
    """Sample n points on a planar quad (p0->p1->p2->p3) via two triangles."""
    n1 = n // 2
    n2 = n - n1
    s1 = _sample_triangle(rng, p0, p1, p2, n1)
    s2 = _sample_triangle(rng, p0, p2, p3, n2)
    return np.vstack([s1, s2])


def _poly_area_xy(pts):
    """Planar area of a polygon's xy projection (shoelace), pts (k,3)."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _sample_planes(rng, planes, density):
    """
    planes: list of vertex-lists (each a (k,3) array, planar polygon = one roof face).
    Sample each face proportional to its 3D-ish area (use xy area * slope factor ~ ok).
    """
    clouds = []
    for poly in planes:
        poly = np.asarray(poly, dtype=float)
        # approximate true surface area: xy area scaled by slope of the plane
        area_xy = _poly_area_xy(poly)
        # slope factor from plane normal
        v1 = poly[1] - poly[0]
        v2 = poly[2] - poly[0]
        nrm = np.cross(v1, v2)
        nz = abs(nrm[2]) / (np.linalg.norm(nrm) + 1e-9)
        slope_factor = 1.0 / max(nz, 0.25)        # cap steepness (>=~75deg treated same)
        area = area_xy * slope_factor
        n = max(20, min(int(area * density), 8000))   # cap per-face points
        if poly.shape[0] == 3:
            clouds.append(_sample_triangle(rng, poly[0], poly[1], poly[2], n))
        elif poly.shape[0] == 4:
            clouds.append(_sample_quad(rng, poly[0], poly[1], poly[2], poly[3], n))
        else:
            # fan-triangulate arbitrary convex-ish polygon
            for i in range(1, poly.shape[0] - 1):
                clouds.append(_sample_triangle(rng, poly[0], poly[i], poly[i + 1],
                                               max(10, n // (poly.shape[0] - 2))))
    return np.vstack(clouds)


def _finalize(rng, points, verts, edges):
    """Center on footprint centroid, dedupe identical verts, enforce min points."""
    verts = np.asarray(verts, dtype=float)
    points = np.asarray(points, dtype=float)
    # center xy on the footprint centroid (z left as real heights)
    c = np.array([verts[:, 0].mean(), verts[:, 1].mean(), 0.0])
    verts = verts - c
    points = points - c
    return points, verts, edges


# ===========================================================================
# Archetype generators
# Each builds: footprint, roof structural vertices, model edges, roof planes.
# ===========================================================================
def gen_gable(rng):
    """
    Gable roof: rectangular footprint, single ridge along the length.
    6 structural verts: 4 eave corners + 2 ridge ends.  Two sloped rectangles.
    """
    L = _u(rng, "len"); W = _u(rng, "wid")
    eh = _u(rng, "eave_h"); rr = _u(rng, "ridge_rise")
    rh = eh + rr
    x0, x1 = 0.0, L
    y0, y1 = 0.0, W
    ymid = 0.5 * (y0 + y1)

    # eave corners (z = eave height)
    E00 = [x0, y0, eh]; E10 = [x1, y0, eh]
    E11 = [x1, y1, eh]; E01 = [x0, y1, eh]
    # ridge ends (z = ridge height), centered in width, full length
    R0 = [x0, ymid, rh]; R1 = [x1, ymid, rh]

    verts = [E00, E10, E11, E01, R0, R1]
    iE00, iE10, iE11, iE01, iR0, iR1 = range(6)
    edges = [
        (iE00, iE10), (iE10, iE11), (iE11, iE01), (iE01, iE00),  # eaves (footprint loop)
        (iR0, iR1),                                              # ridge
        (iE00, iR0), (iE01, iR0),                                # gable-end rafters (left)
        (iE10, iR1), (iE11, iR1),                                # gable-end rafters (right)
    ]
    planes = [
        np.array([E00, E10, R1, R0]),   # front slope (y0 side)
        np.array([E01, E11, R1, R0]),   # back slope  (y1 side)
    ]
    pts = _sample_planes(rng, planes, _u(rng, "density"))
    return _finalize(rng, pts, verts, edges)


def gen_hip(rng):
    """
    Hip roof: rectangular footprint, ridge shorter than length, 4 sloped faces
    (2 trapezoids + 2 triangular hip ends).
    6 structural verts: 4 eaves + 2 ridge ends (inset along length).
    """
    L = _u(rng, "len"); W = _u(rng, "wid")
    eh = _u(rng, "eave_h"); rr = _u(rng, "ridge_rise")
    rh = eh + rr
    setback = _u(rng, "hip_setback_frac") * L
    x0, x1 = 0.0, L
    y0, y1 = 0.0, W
    ymid = 0.5 * (y0 + y1)

    E00 = [x0, y0, eh]; E10 = [x1, y0, eh]
    E11 = [x1, y1, eh]; E01 = [x0, y1, eh]
    R0 = [x0 + setback, ymid, rh]      # ridge inset from left
    R1 = [x1 - setback, ymid, rh]      # ridge inset from right

    verts = [E00, E10, E11, E01, R0, R1]
    iE00, iE10, iE11, iE01, iR0, iR1 = range(6)
    edges = [
        (iE00, iE10), (iE10, iE11), (iE11, iE01), (iE01, iE00),  # eaves
        (iR0, iR1),                                              # ridge
        (iE00, iR0), (iE01, iR0),                                # left hip edges
        (iE10, iR1), (iE11, iR1),                                # right hip edges
    ]
    planes = [
        np.array([E00, E10, R1, R0]),   # front trapezoid
        np.array([E01, E11, R1, R0]),   # back trapezoid
        np.array([E00, E01, R0]),       # left hip triangle
        np.array([E10, E11, R1]),       # right hip triangle
    ]
    pts = _sample_planes(rng, planes, _u(rng, "density"))
    return _finalize(rng, pts, verts, edges)


def gen_pyramid(rng):
    """
    Pyramid (pyramidal hip): square-ish footprint, single apex (degenerate ridge).
    5 structural verts: 4 eaves + 1 apex.  4 triangular faces.
    """
    L = _u(rng, "len"); W = _u(rng, "wid")
    eh = _u(rng, "eave_h"); rr = _u(rng, "ridge_rise")
    rh = eh + rr
    x0, x1 = 0.0, L
    y0, y1 = 0.0, W
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    E00 = [x0, y0, eh]; E10 = [x1, y0, eh]
    E11 = [x1, y1, eh]; E01 = [x0, y1, eh]
    A = [cx, cy, rh]

    verts = [E00, E10, E11, E01, A]
    iE00, iE10, iE11, iE01, iA = range(5)
    edges = [
        (iE00, iE10), (iE10, iE11), (iE11, iE01), (iE01, iE00),  # eaves
        (iE00, iA), (iE10, iA), (iE11, iA), (iE01, iA),          # 4 hip edges to apex
    ]
    planes = [
        np.array([E00, E10, A]), np.array([E10, E11, A]),
        np.array([E11, E01, A]), np.array([E01, E00, A]),
    ]
    pts = _sample_planes(rng, planes, _u(rng, "density"))
    return _finalize(rng, pts, verts, edges)


# ---- composite footprints: now delegated to the correct upper-envelope
#      builder in roof_composite.py (Option B, with valleys/junctions) --------
import roof_composite as _rc


def _wing_dims(rng):
    eh = _u(rng, "eave_h")
    rh = eh + _u(rng, "ridge_rise")
    return eh, rh


def gen_Lshape(rng):
    """L-shape via two perpendicular gable wings sharing a corner (upper envelope)."""
    eh, rh = _wing_dims(rng)
    L = _u(rng, "len"); W = _u(rng, "wid")
    wl = _u(rng, "wing_len"); ww = min(_u(rng, "wing_wid"), W * 0.95)
    wings = [
        _rc.Wing(0, L, 0, W, "x", eh, rh),
        _rc.Wing(L - ww, L, W, W + wl, "y", eh, rh),
    ]
    pts = _rc.sample_cloud(wings, _u(rng, "density"), rng)
    verts, edges = _rc.build_wireframe(wings)
    return _finalize(rng, pts, verts, edges)


def gen_Tshape(rng):
    """T-shape: main gable + perpendicular wing centered on one long side."""
    eh, rh = _wing_dims(rng)
    L = _u(rng, "len"); W = _u(rng, "wid")
    wl = _u(rng, "wing_len"); ww = min(_u(rng, "wing_wid"), L * 0.5)
    cx = 0.5 * L
    wings = [
        _rc.Wing(0, L, 0, W, "x", eh, rh),
        _rc.Wing(cx - ww / 2, cx + ww / 2, W, W + wl, "y", eh, rh),
    ]
    pts = _rc.sample_cloud(wings, _u(rng, "density"), rng)
    verts, edges = _rc.build_wireframe(wings)
    return _finalize(rng, pts, verts, edges)


def gen_cross_gable(rng):
    """Cross-gable: two comparable gable wings crossing at the center."""
    eh, rh = _wing_dims(rng)
    L = _u(rng, "len"); W = _u(rng, "wid")
    W2 = _u(rng, "wing_wid")
    L2 = _u(rng, "wing_len") + W
    cx, cy = 0.5 * L, 0.5 * W
    wings = [
        _rc.Wing(0, L, cy - W / 2, cy + W / 2, "x", eh, rh),
        _rc.Wing(cx - W2 / 2, cx + W2 / 2, cy - L2 / 2, cy + L2 / 2, "y", eh, rh),
    ]
    pts = _rc.sample_cloud(wings, _u(rng, "density"), rng)
    verts, edges = _rc.build_wireframe(wings)
    return _finalize(rng, pts, verts, edges)


# ===========================================================================
# Outline-only footprint builders (for 2D vector outline delineation).
# Return the ordered perimeter loop (V,3) at eave height + closed-loop edges.
# The point cloud is generated by the full builders above (kept 3D); only the
# LABEL is reduced to the footprint outline.
# ===========================================================================
def _loop_edges(n):
    return [(i, (i + 1) % n) for i in range(n)]


def _rect_outline(L, W, eh):
    verts = np.array([[0, 0, eh], [L, 0, eh], [L, W, eh], [0, W, eh]], float)
    return verts, _loop_edges(4)


def outline_for(archetype, rng):
    """
    Build (points 3D, outline_verts, outline_edges) for an archetype.
    Points come from the full roof (real 3D signal); the label is the footprint
    perimeter at eave height only.

    CRITICAL: the point cloud and the outline are centered on the SAME origin
    (the outline-loop centroid) so they stay spatially locked -- otherwise the
    label would not sit on the cloud's footprint.
    """
    if archetype in ("multiwing", "architectural", "farmhouse"):
        # these generators already return outline-only loops + matched cloud,
        # centered consistently -> use directly.
        return ARCHETYPES[archetype](rng_clone(rng))
    if archetype in ("gable", "hip", "pyramid"):
        pts, vfull, efull = ARCHETYPES[archetype](rng_clone(rng))
        # pts and vfull are already centered (by _finalize) in the SAME frame.
        eh = vfull[:, 2].min()
        xs, ys = vfull[:, 0], vfull[:, 1]
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        verts = np.array([[x0, y0, eh], [x1, y0, eh],
                          [x1, y1, eh], [x0, y1, eh]], float)
        edges = _loop_edges(4)
        # already in the same (centered) frame as pts -> no extra shift
        return pts, verts, edges
    else:
        wings = _composite_wings(archetype, rng)
        pts = _rc.sample_cloud(wings, _u(rng, "density"), rng)
        verts, edges = _rc.build_outline(wings)
        # center BOTH on the outline centroid (single shared origin)
        c = np.array([verts[:, 0].mean(), verts[:, 1].mean(), 0.0])
        return pts - c, verts - c, edges


def rng_clone(rng):
    """A child RNG so the points draw is reproducible but independent."""
    return np.random.default_rng(rng.integers(0, 2**63 - 1))


def _composite_wings(archetype, rng):
    """Wing layout for composites (shared by point sampling + outline)."""
    eh, rh = _wing_dims(rng)
    L = _u(rng, "len"); W = _u(rng, "wid")
    if archetype == "lshape":
        wl = _u(rng, "wing_len"); ww = min(_u(rng, "wing_wid"), W * 0.95)
        return [_rc.Wing(0, L, 0, W, "x", eh, rh),
                _rc.Wing(L - ww, L, W, W + wl, "y", eh, rh)]
    if archetype == "tshape":
        wl = _u(rng, "wing_len"); ww = min(_u(rng, "wing_wid"), L * 0.5)
        cx = 0.5 * L
        return [_rc.Wing(0, L, 0, W, "x", eh, rh),
                _rc.Wing(cx - ww / 2, cx + ww / 2, W, W + wl, "y", eh, rh)]
    if archetype == "cross_gable":
        W2 = _u(rng, "wing_wid"); L2 = _u(rng, "wing_len") + W
        cx, cy = 0.5 * L, 0.5 * W
        return [_rc.Wing(0, L, cy - W / 2, cy + W / 2, "x", eh, rh),
                _rc.Wing(cx - W2 / 2, cx + W2 / 2, cy - L2 / 2, cy + L2 / 2, "y", eh, rh)]
    raise ValueError(archetype)


# ===========================================================================
# Registry
# ===========================================================================
def gen_complex_multiwing(rng):
    """Multi-wing building, ~9-30 corners, gabled roof."""
    tgt = int(rng.integers(9, 31))
    return _rc.gen_complex_multiwing(rng, tgt)


def gen_architectural(rng):
    """Architectural massing (comb/U/H/C/stepped/wings), ~9-80 corners."""
    tgt = int(rng.integers(13, 81))
    return _rc.gen_architectural(rng, tgt)


def gen_farmhouse(rng):
    """Vernacular farmhouse: tall main block + lower annex, mixed roof types."""
    return _rc.gen_farmhouse(rng)


ARCHETYPES = {
    "gable":       gen_gable,
    "hip":         gen_hip,
    "pyramid":     gen_pyramid,
    "lshape":      gen_Lshape,
    "tshape":      gen_Tshape,
    "cross_gable": gen_cross_gable,
    "multiwing":     gen_complex_multiwing,   # 9-30 corners
    "architectural": gen_architectural,       # 13-80 corners, building-like
    "farmhouse":     gen_farmhouse,           # compact, tall main + low annex
}


# Default mode: outline-only labels (footprint perimeter) for 2D delineation.
OUTLINE_ONLY = True

# Match the real samples' coordinate convention: building in the positive
# quadrant with the point cloud's min-xy at the origin and the ground at z=0.
# The real connected-component clouds include WALL/FACADE returns (cloud z runs
# 0 -> roof, with the eave/outline partway up at ~2.5-3 m), so we lift the roof
# to a realistic eave height and add synthetic wall points down to the ground.
MATCH_REAL_FRAME = True
ADD_WALLS = True
EAVE_HEIGHT_RANGE = (2.4, 3.2)   # where the eave/outline sits above ground (m)
WALL_DENSITY = 8.0               # wall points per square metre


def _to_real_frame_with_walls(pts, verts, edges, rng):
    """
    Place the sample in the real coordinate convention and add wall points:
      1. shift so min-xy of the cloud -> origin (positive quadrant);
      2. set the building's lowest roof/eave level so the EAVE sits at a
         realistic height above a ground plane at z=0;
      3. sample wall points from the eave down to z=0 along the outline;
      4. return combined (roof+wall) cloud and the shifted outline.
    Cloud and outline receive the SAME xy/z shift so they stay locked.
    """
    # --- xy: positive quadrant, min corner at origin ---
    sx = pts[:, 0].min()
    sy = pts[:, 1].min()
    pts = pts.copy(); verts = verts.copy()
    pts[:, 0] -= sx; pts[:, 1] -= sy
    verts[:, 0] -= sx; verts[:, 1] -= sy

    # --- z: lift so the eave (outline mean z) sits at a realistic height ---
    eave_now = verts[:, 2].mean()
    eave_target = rng.uniform(*EAVE_HEIGHT_RANGE)
    dz = eave_target - eave_now
    pts[:, 2] += dz
    verts[:, 2] += dz
    # clamp any roof points that dropped below ground
    pts[:, 2] = np.maximum(pts[:, 2], 0.0)

    # --- walls: from eave down to ground z=0 ---
    if ADD_WALLS:
        walls = _rc.sample_walls(verts, edges, rng, WALL_DENSITY, ground_z=0.0)
        if len(walls):
            pts = np.vstack([pts, walls])
    return pts, verts


def generate(archetype, seed):
    """
    Generate one (points, verts, edges) for the named archetype.

    With OUTLINE_ONLY=True (default), the label is the footprint perimeter graph
    at eave height -- no internal ridge/valley/junction structure. The point
    cloud is the full 3D roof (real signal). Set OUTLINE_ONLY=False to get the
    full 3D structural wireframe instead.

    With MATCH_REAL_FRAME=True (default), the sample is placed in the same
    coordinate convention as the real data (positive quadrant, ground at z=0,
    eave at a realistic height) and synthetic wall points are added so the
    cloud's vertical structure matches the real connected-component clouds.
    """
    rng = np.random.default_rng(seed)
    if archetype not in ARCHETYPES:
        raise ValueError(f"unknown archetype '{archetype}'. "
                         f"choices: {list(ARCHETYPES)}")
    if OUTLINE_ONLY:
        pts, verts, edges = outline_for(archetype, rng)
    else:
        pts, verts, edges = ARCHETYPES[archetype](rng)
    # enforce a sensible minimum point count
    min_pts = PARAMS["min_pts"][0]
    if len(pts) < min_pts:
        extra = rng.choice(len(pts), size=min_pts - len(pts), replace=True)
        pts = np.vstack([pts, pts[extra]])
    verts = np.asarray(verts, dtype=float)
    if MATCH_REAL_FRAME:
        pts, verts = _to_real_frame_with_walls(pts, verts, edges, rng)
    return pts, verts, edges
