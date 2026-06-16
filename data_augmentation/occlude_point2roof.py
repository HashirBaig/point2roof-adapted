#!/usr/bin/env python3
"""
occlude_point2roof.py
=====================

Synthesize occluded copies of building point-cloud instances at FOUR fixed
removal levels (25%, 50%, 75%, 90%), to test outline-delineation model
robustness across occlusion severity.

THE GOLDEN RULE
---------------
Occlusion removes points from the cloud. The ground-truth outline
(polygon_n.obj) is the true building boundary -- it does NOT change because
the scan is incomplete. So every occluded copy reuses the SAME, UNCHANGED
label: only points_n.xyz is modified; polygon_n.obj is copied verbatim.

Each input instance produces exactly four occluded copies, written to
separate per-level output directories so the same id (e.g. 000001) refers to
the same source building across all four levels:

    ./<out>/25_perc/000001/{points_n.xyz, polygon_n.obj}
    ./<out>/50_perc/000001/{points_n.xyz, polygon_n.obj}
    ./<out>/75_perc/000001/{points_n.xyz, polygon_n.obj}
    ./<out>/90_perc/000001/{points_n.xyz, polygon_n.obj}

A test.txt file is written inside each level's directory listing every sample
folder path, e.g. ./<out>/25_perc/test.txt contains:

    ../Point2Roof/3d_data_test_occlusion/25_perc/000001
    ../Point2Roof/3d_data_test_occlusion/25_perc/000002
    ...

The path prefix used in test.txt is controlled by --test-prefix; it can differ
from --out (typically does, since the test data is consumed by a separate
inference pipeline with a different relative path).

Usage
-----
    # batch over a directory of <id>/points_n.xyz + polygon_n.obj
    python occlude_point2roof.py --data-dir ./test --out ./occlusion

    # single instance
    python occlude_point2roof.py points_n.xyz polygon_n.obj --out ./occlusion

    # subset of levels
    python occlude_point2roof.py --data-dir ./test --out ./occlusion \\
        --levels 50_perc,75_perc

    # custom test.txt prefix (default: ../Point2Roof/3d_data_test_occlusion/)
    python occlude_point2roof.py --data-dir ./test --out ./occlusion \\
        --test-prefix ../my_project/occluded_data/
"""

import os
import re
import glob
import math
import argparse
import numpy as np

# Fixed filenames inside each input/output sample folder.
POINTS_NAME = "points_n.xyz"
POLYGON_NAME = "polygon_n.obj"

# Name of the index file written inside each level's sub-directory.
TEST_INDEX_NAME = "test.txt"

# Default path prefix written before each sample id in test.txt. Independent
# of --out because the inference pipeline that reads test.txt typically uses
# a different relative path from the directory where this script writes.
DEFAULT_TEST_PREFIX = "../data_augmentation/occlusion_data_90/"

# Never let a cloud drop below this many points (model still needs an input).
MIN_POINTS = 200

# ----------------------------------------------------------------------------
# Three fixed-removal levels. Each carries its OWN output sub-directory and a
# single per-copy removal target. The spatial removal primitives (probabilities
# and severity) escalate with the target so the pattern looks realistic at
# every level rather than just uniformly random.
# ----------------------------------------------------------------------------
LEVELS = {
    "25_perc": {
        "out_subdir": "25_perc",
        "fixed_target": 0.25,
        "random_thinning":     {"p": 1.0, "frac": (0.10, 0.20)},
        "swath_gaps":          {"p": 0.5, "n": (1, 1), "width_frac": (0.04, 0.08)},
        "occlusion_shadow":    {"p": 0.6, "n": (1, 1), "size_frac": (0.06, 0.12)},
        "boundary_erosion":    {"p": 0.5, "frac": (0.10, 0.20), "band_frac": 0.10},
        "directional_dropout": {"p": 0.3, "keep": (0.5, 0.7)},
    },
    "50_perc": {
        "out_subdir": "50_perc",
        "fixed_target": 0.50,
        "random_thinning":     {"p": 1.0, "frac": (0.15, 0.30)},
        "swath_gaps":          {"p": 0.8, "n": (1, 2), "width_frac": (0.06, 0.12)},
        "occlusion_shadow":    {"p": 0.9, "n": (1, 2), "size_frac": (0.10, 0.20)},
        "boundary_erosion":    {"p": 0.8, "frac": (0.25, 0.45), "band_frac": 0.14},
        "directional_dropout": {"p": 0.5, "keep": (0.35, 0.6)},
    },
    "75_perc": {
        "out_subdir": "75_perc",
        "fixed_target": 0.75,
        "random_thinning":     {"p": 1.0, "frac": (0.20, 0.35)},
        "swath_gaps":          {"p": 0.9, "n": (2, 3), "width_frac": (0.08, 0.16)},
        "occlusion_shadow":    {"p": 1.0, "n": (2, 3), "size_frac": (0.15, 0.30)},
        "boundary_erosion":    {"p": 0.9, "frac": (0.35, 0.60), "band_frac": 0.18},
        "directional_dropout": {"p": 0.6, "keep": (0.25, 0.5)},
    },
    "90_perc": {
        "out_subdir": "90_perc",
        "fixed_target": 0.90,
        # Extreme occlusion: roughly 10% of points remain. The final trim step
        # does most of the pinning, but the spatial primitives still establish
        # a realistic, non-uniform pattern (large shadows, multiple swaths,
        # heavy boundary erosion). MIN_POINTS (200) sets the floor for small
        # clouds, so a 500-point input becomes ~200 points rather than 50.
        "random_thinning":     {"p": 1.0, "frac": (0.25, 0.40)},
        "swath_gaps":          {"p": 1.0, "n": (3, 4), "width_frac": (0.10, 0.20)},
        "occlusion_shadow":    {"p": 1.0, "n": (3, 4), "size_frac": (0.20, 0.35)},
        "boundary_erosion":    {"p": 1.0, "frac": (0.50, 0.75), "band_frac": 0.20},
        "directional_dropout": {"p": 0.7, "keep": (0.15, 0.40)},
    },
}


# ============================================================================
# I/O
# ============================================================================
def read_xyz(path):
    pts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            t = line.replace(",", " ").split()
            if len(t) < 3:
                continue
            pts.append([float(t[0]), float(t[1]), float(t[2])])
    if not pts:
        raise ValueError(f"No points parsed from {path}")
    return np.asarray(pts, dtype=np.float64)


def write_xyz(path, pts):
    with open(path, "w") as f:
        for x, y, z in pts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def read_obj_lines(path):
    """Read raw OBJ text so we can copy the label verbatim (label is unchanged)."""
    with open(path) as f:
        return f.read()


def read_obj_verts(path):
    """Parse just the outline vertices (for boundary-aware occlusion)."""
    verts = []
    with open(path) as f:
        for line in f:
            t = line.strip().split()
            if t and t[0] == "v":
                verts.append([float(t[1]), float(t[2]), float(t[3])])
    return np.asarray(verts, dtype=float) if verts else None


# ============================================================================
# Occlusion primitives -- each returns a BOOLEAN KEEP MASK over points.
# (True = keep, False = removed.) Masks are ANDed together.
# ============================================================================
def m_random_thinning(pts, rng, frac):
    return rng.random(len(pts)) >= frac


def m_swath_gaps(pts, rng, n, width_frac):
    """Remove n parallel strips along a random in-plane direction (scan lines)."""
    keep = np.ones(len(pts), dtype=bool)
    ang = rng.uniform(0, math.pi)
    d = np.array([math.cos(ang), math.sin(ang)])
    proj = pts[:, :2] @ d
    lo, hi = proj.min(), proj.max()
    span = hi - lo + 1e-9
    for _ in range(n):
        w = rng.uniform(*width_frac) * span
        c = rng.uniform(lo, hi)
        keep &= ~((proj >= c - w / 2) & (proj <= c + w / 2))
    return keep


def m_occlusion_shadow(pts, rng, n, size_frac):
    """
    Remove n contiguous spatial regions (sensor shadows from adjacent
    structures). Each shadow is a disk centred at a random cloud point,
    optionally clipped to a half-plane to give a wedge/sector shape (more
    shadow-like than a full circle).
    """
    keep = np.ones(len(pts), dtype=bool)
    mn = pts[:, :2].min(0); mx = pts[:, :2].max(0)
    extent = float(np.linalg.norm(mx - mn))
    for _ in range(n):
        r = rng.uniform(*size_frac) * extent
        c = pts[rng.integers(len(pts)), :2]
        dist = np.linalg.norm(pts[:, :2] - c, axis=1)
        bite = dist <= r
        if rng.random() < 0.5:
            ang = rng.uniform(0, 2 * math.pi)
            nrm = np.array([math.cos(ang), math.sin(ang)])
            side = (pts[:, :2] - c) @ nrm
            bite &= side >= 0
        keep &= ~bite
    return keep


def m_boundary_erosion(pts, rng, frac, band_frac, outline_xy):
    """
    Preferentially remove points NEAR the outline boundary. Directly stresses
    edge-localization accuracy. If no outline is available, falls back to
    bounding-box edges.
    """
    keep = np.ones(len(pts), dtype=bool)
    extent = float(np.linalg.norm(pts[:, :2].max(0) - pts[:, :2].min(0)))
    band = band_frac * extent
    dist = _dist_to_polyline(pts[:, :2], outline_xy)
    near = dist <= band
    drop = near & (rng.random(len(pts)) < frac)
    keep &= ~drop
    return keep


def m_directional_dropout(pts, rng, keep_frac):
    """One side of a random line is sparsely sampled (look-angle effect)."""
    keep = np.ones(len(pts), dtype=bool)
    ang = rng.uniform(0, 2 * math.pi)
    nrm = np.array([math.cos(ang), math.sin(ang)])
    c = pts[:, :2].mean(0)
    side = (pts[:, :2] - c) @ nrm
    far = side > 0
    kf = rng.uniform(*keep_frac)
    thin = far & (rng.random(len(pts)) >= kf)
    keep &= ~thin
    return keep


def _dist_to_polyline(xy, poly_xy):
    """Min distance from each xy point to the closed polyline poly_xy."""
    if poly_xy is None or len(poly_xy) < 2:
        mn = xy.min(0); mx = xy.max(0)
        dleft = xy[:, 0] - mn[0]; dright = mx[0] - xy[:, 0]
        dbot = xy[:, 1] - mn[1]; dtop = mx[1] - xy[:, 1]
        return np.minimum.reduce([dleft, dright, dbot, dtop])
    P = poly_xy
    n = len(P)
    best = np.full(len(xy), np.inf)
    for i in range(n):
        a = P[i]; b = P[(i + 1) % n]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            d = np.linalg.norm(xy - a, axis=1)
        else:
            t = np.clip(((xy - a) @ ab) / L2, 0, 1)
            proj = a + t[:, None] * ab
            d = np.linalg.norm(xy - proj, axis=1)
        best = np.minimum(best, d)
    return best


# ============================================================================
# Compose one occluded copy pinned to an exact removal fraction
# ============================================================================
def occlude_to_target(pts, outline_xy, level_cfg, rng):
    """
    Produce one occluded copy whose final removed fraction equals
    level_cfg['fixed_target'] exactly. Spatial primitives run first so the
    removal pattern looks physically plausible; then a trim/restore step pins
    the count.
    """
    n0 = len(pts)
    target = level_cfg["fixed_target"]
    keep = np.ones(n0, dtype=bool)
    applied = []

    def maybe(name, fn):
        cfg = level_cfg.get(name)
        if cfg and rng.random() < cfg["p"]:
            nonlocal keep
            keep &= fn(cfg)
            applied.append(name)

    maybe("random_thinning",
          lambda c: m_random_thinning(pts, rng, rng.uniform(*c["frac"])))
    maybe("swath_gaps",
          lambda c: m_swath_gaps(pts, rng, rng.integers(c["n"][0], c["n"][1] + 1),
                                 c["width_frac"]))
    maybe("occlusion_shadow",
          lambda c: m_occlusion_shadow(pts, rng,
                                       rng.integers(c["n"][0], c["n"][1] + 1),
                                       c["size_frac"]))
    maybe("boundary_erosion",
          lambda c: m_boundary_erosion(pts, rng, rng.uniform(*c["frac"]),
                                       c["band_frac"], outline_xy))
    maybe("directional_dropout",
          lambda c: m_directional_dropout(pts, rng, c["keep"]))

    # pin removed fraction EXACTLY to target
    target_keep = max(MIN_POINTS, int(round((1.0 - target) * n0)))
    target_keep = min(target_keep, n0)
    cur_keep = int(keep.sum())
    if cur_keep > target_keep:
        idx_keep = np.where(keep)[0]
        drop = rng.choice(idx_keep, size=cur_keep - target_keep, replace=False)
        keep[drop] = False
    elif cur_keep < target_keep:
        idx_rem = np.where(~keep)[0]
        restore = rng.choice(idx_rem, size=target_keep - cur_keep, replace=False)
        keep[restore] = True
    applied.append(f"pin{int(target * 100)}")

    out = pts[keep]
    return out, 1.0 - len(out) / n0, applied


# ============================================================================
# Driver
# ============================================================================
class IdMinter:
    def __init__(self, start=1, width=6):
        self.n = start - 1
        self.width = width

    def next(self):
        self.n += 1
        return str(self.n).zfill(self.width)


def find_instances(data_dir):
    out = []
    for xp in glob.glob(os.path.join(data_dir, "**", POINTS_NAME), recursive=True):
        op = os.path.join(os.path.dirname(xp), POLYGON_NAME)
        if os.path.isfile(op):
            rel = os.path.relpath(os.path.dirname(xp), data_dir)
            out.append((rel.replace(os.sep, "_"), xp, op))
        else:
            print(f"  [skip] {os.path.dirname(xp)}: missing {POLYGON_NAME}")
    out.sort()
    return out


def process_instance(xyz_path, obj_path, out_root, levels, minters,
                     ids_per_level, base_seed=0):
    """
    For one input instance, generate one occluded copy per requested level.
    `minters` is a {level_name: IdMinter} dict so each level's output dir gets
    its own sequential numbering -- the SAME id across levels refers to the
    same source building (000001 in 25_perc, 50_perc, 75_perc are aligned).

    `ids_per_level` (mutated): a {level_name: [sid, ...]} dict that collects
    every id written, used by the caller to emit per-level test.txt files.
    """
    pts = read_xyz(xyz_path)
    obj_text = read_obj_lines(obj_path)
    outline_xy = None
    ov = read_obj_verts(obj_path)
    if ov is not None:
        outline_xy = ov[:, :2]

    written = []
    for k, level in enumerate(levels):
        if level not in LEVELS:
            print(f"    [skip] unknown level '{level}'")
            continue
        cfg = LEVELS[level]
        seed = base_seed * 100003 + (k + 1) * 17 + hash(level) % 9973
        rng = np.random.default_rng(seed)
        out, removed, applied = occlude_to_target(pts, outline_xy, cfg, rng)

        sub = os.path.join(out_root, cfg["out_subdir"])
        os.makedirs(sub, exist_ok=True)
        sid = minters[level].next()
        sample_dir = os.path.join(sub, sid)
        os.makedirs(sample_dir, exist_ok=True)
        write_xyz(os.path.join(sample_dir, POINTS_NAME), out)
        with open(os.path.join(sample_dir, POLYGON_NAME), "w") as f:
            f.write(obj_text)                          # label, unchanged
        written.append(sample_dir)
        ids_per_level[level].append(sid)
        print(f"    {cfg['out_subdir']}/{sid}  pts={len(out):5d}  "
              f"removed={removed * 100:4.1f}%  ops=[{','.join(applied)}]")
    return written


def write_test_indexes(out_root, levels, ids_per_level, test_prefix):
    """
    Write one test.txt inside each level's output sub-directory listing every
    sample folder path, using the given path prefix.

    The test_prefix is appended with the level's sub-directory name + sample id,
    one per line. Trailing slash on the prefix is optional.
    """
    prefix = test_prefix.rstrip("/") + "/"
    written_files = []
    for level in levels:
        if level not in LEVELS:
            continue
        sub = LEVELS[level]["out_subdir"]
        ids = ids_per_level.get(level, [])
        if not ids:
            continue
        idx_path = os.path.join(out_root, sub, TEST_INDEX_NAME)
        with open(idx_path, "w") as f:
            for sid in ids:
                f.write(f"{prefix}{sub}/{sid}\n")
        written_files.append((idx_path, len(ids)))
    return written_files


def main():
    ap = argparse.ArgumentParser(
        description="Synthesize occluded copies of building instances at four "
                    "fixed removal levels (25/50/75/90 %), written into separate "
                    "per-level output sub-directories. The ground-truth label "
                    "(polygon_n.obj) is copied UNCHANGED to every output. "
                    "A test.txt is written inside each level's directory.")
    ap.add_argument("xyz", nargs="?", help="input points_n.xyz")
    ap.add_argument("obj", nargs="?", help="input polygon_n.obj")
    ap.add_argument("--data-dir",
                    help="directory of <id>/points_n.xyz + polygon_n.obj")
    ap.add_argument("--out", default="./occlusion",
                    help="output ROOT directory; per-level sub-directories "
                         "(25_perc/, 50_perc/, 75_perc/, 90_perc/) are created "
                         "under it (default: ./occlusion)")
    ap.add_argument("--levels", default="25_perc,50_perc,75_perc,90_perc",
                    help="comma-separated subset of "
                         "25_perc,50_perc,75_perc,90_perc "
                         "(default: all four)")
    ap.add_argument("--test-prefix", default=DEFAULT_TEST_PREFIX,
                    help="path prefix written before each entry in test.txt. "
                         "Each line becomes <prefix>/<level_subdir>/<sample_id>. "
                         f"(default: {DEFAULT_TEST_PREFIX})")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--start-id", type=int, default=1)
    ap.add_argument("--id-width", type=int, default=6)
    args = ap.parse_args()

    levels = [l.strip() for l in args.levels.split(",") if l.strip()]
    unknown = [l for l in levels if l not in LEVELS]
    if unknown:
        ap.error(f"unknown level(s): {unknown}. valid: {list(LEVELS)}")

    os.makedirs(args.out, exist_ok=True)
    # one independent counter per level so same id == same source instance
    minters = {l: IdMinter(start=args.start_id, width=args.id_width)
               for l in levels}
    ids_per_level = {l: [] for l in levels}

    total = 0
    if args.data_dir:
        insts = find_instances(args.data_dir)
        if not insts:
            print(f"No instances found under {args.data_dir}")
            return
        print(f"Found {len(insts)} instance(s); generating {len(levels)} "
              f"occluded copy/copies each -> {len(insts) * len(levels)} total.")
        for i, (sid, xp, op) in enumerate(insts):
            print(f"\n== instance '{sid}'")
            total += len(process_instance(
                xp, op, args.out, levels, minters, ids_per_level,
                base_seed=args.seed + i))
    else:
        if not (args.xyz and args.obj):
            ap.error("provide both xyz and obj, or use --data-dir")
        print(f"== {os.path.basename(args.xyz)} + {os.path.basename(args.obj)}")
        total += len(process_instance(
            args.xyz, args.obj, args.out, levels, minters, ids_per_level,
            base_seed=args.seed))

    # write the per-level test.txt index files
    idx_files = write_test_indexes(args.out, levels, ids_per_level,
                                    args.test_prefix)

    print(f"\nDone. {total} folder(s) written under: {os.path.abspath(args.out)}")
    for l in levels:
        print(f"  {l}: {os.path.join(os.path.abspath(args.out), LEVELS[l]['out_subdir'])}")
    if idx_files:
        print(f"\ntest.txt indexes (prefix: {args.test_prefix}):")
        for path, n in idx_files:
            print(f"  {path}  ({n} entries)")


if __name__ == "__main__":
    main()
