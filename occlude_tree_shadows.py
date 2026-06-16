#!/usr/bin/env python3
"""
occlude_tree_shadows.py
=======================

Add realistic tree-shadow occlusions to Point2Roof samples for RQ 3.1/3.2:
"How does the model perform in less-occluded environments compared to
moderately/heavily occluded urban scenes with trees and narrow streets?"

Occlusion model
---------------
A tree canopy occludes a patch of the building during airborne LiDAR
acquisition. The tree returns are then removed during data cleaning
(vegetation filtering), leaving a SPATIAL GAP in the building's point
cloud where the tree was.

To isolate the "spatial occlusion pattern" variable from "point count",
this script preserves the TOTAL POINT COUNT: points falling under tree
patches are removed, then points are resampled (with sub-cm jitter) from
the remaining visible region to restore the original count. The net effect:
identical n_pts across all severity levels and the source, but with
realistic spatial gaps.

This differs from the existing occlude_point2roof.py (which targets a fixed
removal fraction). Here the variable under study is WHERE the gaps are, not
HOW MANY points remain.

Three severity levels
---------------------
  light    : 1 tree crown over a corner   (open residential)
  moderate : 1-2 trees, corner-biased     (suburban with mature trees)
  heavy    : 2-3 trees, corners + interior (narrow urban streets)

Patch radius scales with the building's bbox diagonal so a 3 m crown on a
3 m shed does not wipe out the entire cloud, while on a 25 m row of houses
it occludes only one corner. Maximum cumulative removal is capped (default
60%) so the cloud never collapses to nothing.

Outputs (same convention as occlude_point2roof.py)
---------------------------------------------------
    <out>/light/<id>/{points_n.xyz, polygon_n.obj}
    <out>/moderate/<id>/...
    <out>/heavy/<id>/...
    <out>/<level>/test.txt          (one per level, with optional path prefix)

The polygon is copied UNCHANGED from the source -- the ground truth
represents the full footprint that the model should predict regardless of
how occluded its input is.

Usage
-----
    python occlude_tree_shadows.py ./test_data --out ./occlusion_trees
    python occlude_tree_shadows.py ./test_data --out ./occlusion_trees \\
        --levels light,moderate
    python occlude_tree_shadows.py ./test_data --out ./occlusion_trees \\
        --preview 5             # save first 5 samples as before/after PNGs
"""

import os
import shutil
import argparse
import numpy as np


POINTS_NAME = "points_n.xyz"
POLYGON_NAME = "polygon_n.obj"
MIN_POINTS = 100   # don't occlude clouds smaller than this; pass through

# --------------------------------------------------------------------------
# Severity levels
# Each level: how many tree patches and how big (as a fraction of the
# building's bbox diagonal). Location is "corner" (corners only) or
# "mixed" (corners + interior).
# --------------------------------------------------------------------------
LEVELS = {
    "light": {
        "out_subdir":     "light",
        "n_patches":      (1, 1),
        "radius_frac":    (0.15, 0.25),
        "location":       "corner",
        "corner_prob":    1.0,
        "max_removal":    0.40,
    },
    "moderate": {
        "out_subdir":     "moderate",
        "n_patches":      (1, 2),
        "radius_frac":    (0.20, 0.35),
        "location":       "corner",
        "corner_prob":    0.85,
        "max_removal":    0.50,
    },
    "heavy": {
        "out_subdir":     "heavy",
        "n_patches":      (2, 3),
        "radius_frac":    (0.25, 0.45),
        "location":       "mixed",
        "corner_prob":    0.6,
        "max_removal":    0.60,
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
            try:
                pts.append([float(t[0]), float(t[1]), float(t[2])])
            except ValueError:
                continue
    return np.asarray(pts, dtype=np.float64) if pts else np.empty((0, 3))


def read_polygon_verts(path):
    """Return polygon vertices (V, 3) -- we only need xy here."""
    verts = []
    with open(path) as f:
        for line in f:
            t = line.split()
            if t and t[0] == "v" and len(t) >= 4:
                verts.append([float(t[1]), float(t[2]), float(t[3])])
    return np.asarray(verts, dtype=np.float64) if verts else np.empty((0, 3))


def write_xyz(path, pts):
    with open(path, "w") as f:
        for x, y, z in pts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


# ============================================================================
# Patch center placement
# ============================================================================
def pick_patch_centers(poly_xy, n_patches, radius_frac_range, bbox_diag,
                       location, corner_prob, rng):
    """
    Return (centers, radii) for `n_patches` tree-crown occlusion patches.

    Corner-biased placement: pick a polygon vertex, then offset by ~half the
    radius in a random direction so the corner sits on the patch edge and is
    about half occluded -- realistic for a tree growing near a building.

    Interior placement (heavy level only): pick a random point uniformly
    within the polygon's axis-aligned bbox. Some such points might fall
    outside the polygon for non-convex shapes; that's fine -- they still
    represent a tree somewhere near the building.
    """
    centers = []
    radii = []
    poly_min = poly_xy.min(axis=0)
    poly_max = poly_xy.max(axis=0)
    for _ in range(n_patches):
        radius = rng.uniform(*radius_frac_range) * bbox_diag

        place_at_corner = (location == "corner") or (rng.random() < corner_prob)
        if place_at_corner and len(poly_xy) > 0:
            v = poly_xy[rng.integers(len(poly_xy))]
            ang = rng.uniform(0, 2 * np.pi)
            dist = radius * rng.uniform(0.3, 0.7)
            cx = v[0] + dist * np.cos(ang)
            cy = v[1] + dist * np.sin(ang)
        else:
            cx = rng.uniform(poly_min[0], poly_max[0])
            cy = rng.uniform(poly_min[1], poly_max[1])
        centers.append((cx, cy))
        radii.append(radius)
    return centers, radii


# ============================================================================
# Occlusion + resampling (preserves total point count)
# ============================================================================
def apply_tree_occlusion(pts, polygon_xy, level_cfg, rng):
    """
    Apply tree-shadow patches and resample to preserve the original point
    count. Returns (new_pts, info_dict).

    Algorithm:
      1. Pick n_patches (centers, radii) using level_cfg.
      2. Mark all points within any patch as 'occluded'.
      3. If occlusion would exceed max_removal of the cloud, randomly UN-mark
         some occluded points so the cap is respected (cloud never collapses).
      4. Remove the marked points.
      5. Resample (with replacement) from the remaining points to refill to
         the original count. Add 2 cm of jitter so resampled points are not
         exact duplicates.
      6. Shuffle so original and resampled rows are interleaved.
    """
    n_target = len(pts)
    info = {"n_patches": 0, "patches": [], "n_removed": 0, "n_resampled": 0}
    if n_target < MIN_POINTS or len(polygon_xy) < 3:
        return pts.copy(), info

    # 1. compute bbox diagonal and pick patches
    poly_min = polygon_xy.min(axis=0)
    poly_max = polygon_xy.max(axis=0)
    bbox_diag = float(np.linalg.norm(poly_max - poly_min))
    if bbox_diag < 1e-6:
        return pts.copy(), info

    n_p = int(rng.integers(level_cfg["n_patches"][0],
                            level_cfg["n_patches"][1] + 1))
    centers, radii = pick_patch_centers(
        polygon_xy, n_p, level_cfg["radius_frac"], bbox_diag,
        level_cfg["location"], level_cfg["corner_prob"], rng)
    info["n_patches"] = n_p
    info["patches"] = list(zip(centers, radii))

    # 2. mark points within any patch
    xy = pts[:, :2]
    mark = np.zeros(len(pts), dtype=bool)
    for (cx, cy), r in zip(centers, radii):
        dx = xy[:, 0] - cx; dy = xy[:, 1] - cy
        mark |= (dx * dx + dy * dy < r * r)

    # 3. enforce max-removal cap by randomly un-marking some flagged points
    max_remove = int(level_cfg["max_removal"] * n_target)
    if mark.sum() > max_remove:
        marked_idx = np.where(mark)[0]
        n_to_unmark = int(mark.sum() - max_remove)
        unmark = rng.choice(marked_idx, size=n_to_unmark, replace=False)
        mark[unmark] = False

    # 4. remove
    kept = pts[~mark]
    n_kept = len(kept)
    info["n_removed"] = int(mark.sum())

    # safety: if for some reason we lost almost everything, pass through
    if n_kept < 16:
        return pts.copy(), info

    # 5. resample with jitter to refill
    n_add = n_target - n_kept
    if n_add > 0:
        idx = rng.choice(n_kept, size=n_add, replace=True)
        resampled = kept[idx].copy()
        resampled += rng.normal(0, 0.02, resampled.shape)  # 2 cm jitter
        out = np.vstack([kept, resampled])
        info["n_resampled"] = int(n_add)
    else:
        out = kept

    # 6. shuffle
    perm = rng.permutation(len(out))
    return out[perm], info


# ============================================================================
# Driver
# ============================================================================
def find_samples(data_dir):
    """List <data_dir>/<id> folders that contain both required files."""
    samples = []
    for name in sorted(os.listdir(data_dir)):
        folder = os.path.join(data_dir, name)
        if not os.path.isdir(folder):
            continue
        xp = os.path.join(folder, POINTS_NAME)
        op = os.path.join(folder, POLYGON_NAME)
        if os.path.isfile(xp) and os.path.isfile(op):
            samples.append((name, xp, op))
    return samples


def find_samples_from_file(samples_file):
    """
    Read sample folder paths (one per line). Lines starting with '#' or blank
    are ignored. Each path is resolved relative to cwd first, with fallback
    to the samples-file's own directory if cwd doesn't have it. Absolute
    paths used as-is.

    Returns the same (sample_id, points_path, polygon_path) tuples as
    find_samples(), so the rest of the script doesn't care which input mode
    was used.
    """
    if not os.path.isfile(samples_file):
        raise FileNotFoundError(f"samples file not found: {samples_file}")
    file_dir = os.path.dirname(os.path.abspath(samples_file))
    cwd = os.getcwd()
    samples = []
    seen = set()
    used_fallback = 0
    missing_lines = []
    with open(samples_file) as f:
        for lineno, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if os.path.isabs(raw):
                candidates = [raw]
            else:
                candidates = [os.path.join(cwd, raw),
                              os.path.join(file_dir, raw)]
            folder = None
            for k, c in enumerate(candidates):
                if os.path.isdir(c):
                    folder = c
                    if k > 0:
                        used_fallback += 1
                    break
            if folder is None:
                missing_lines.append((lineno, raw, "folder not found"))
                continue
            xp = os.path.join(folder, POINTS_NAME)
            op = os.path.join(folder, POLYGON_NAME)
            if not os.path.isfile(xp):
                missing_lines.append((lineno, raw, f"missing {POINTS_NAME}"))
                continue
            if not os.path.isfile(op):
                missing_lines.append((lineno, raw, f"missing {POLYGON_NAME}"))
                continue
            canonical = os.path.realpath(folder)
            if canonical in seen:
                continue
            seen.add(canonical)
            sample_id = os.path.basename(folder.rstrip(os.sep))
            samples.append((sample_id, xp, op))
    if used_fallback:
        print(f"  note: {used_fallback} path(s) resolved against the "
              f"samples-file directory rather than cwd "
              f"('{file_dir}')")
    if missing_lines:
        print(f"  {len(missing_lines)} line(s) skipped:")
        for lineno, raw, why in missing_lines[:5]:
            print(f"    line {lineno}: {raw!r}: {why}")
        if len(missing_lines) > 5:
            print(f"    ... ({len(missing_lines) - 5} more)")
    return samples


def save_preview(src_pts, occ_pts, polygon_xy, info, sample_id, level,
                 out_path):
    """Save a side-by-side before/after PNG to verify occlusions look right."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    closed = np.vstack([polygon_xy, polygon_xy[:1]])
    for ax, (label, pts) in zip(axes, [("source", src_pts), ("occluded", occ_pts)]):
        if len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2],
                       cmap="viridis", s=3, alpha=0.5)
        ax.plot(closed[:, 0], closed[:, 1], "r-", lw=2)
        ax.set_aspect("equal")
        ax.set_title(f"{label}: n={len(pts):,}")
        ax.set_xticks([]); ax.set_yticks([])
    # draw the patch circles on the occluded plot
    for (cx, cy), r in info.get("patches", []):
        circ = plt.Circle((cx, cy), r, fill=False, color="orange",
                          lw=1.5, linestyle="--", alpha=0.8)
        axes[1].add_patch(circ)
    fig.suptitle(f"sample {sample_id}  [level={level}]  "
                 f"n_patches={info['n_patches']}, "
                 f"removed={info['n_removed']:,}, "
                 f"resampled={info['n_resampled']:,}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=92)
    plt.close(fig)


def process_one(level_name, level_cfg, samples, out_root, rng_master,
                test_prefix, preview_first):
    """Process all samples through ONE severity level."""
    level_dir = os.path.join(out_root, level_cfg["out_subdir"])
    os.makedirs(level_dir, exist_ok=True)

    ok_ids = []
    summary = {"applied": 0, "passthrough_small": 0, "errors": 0}

    for i, (sid, xp, op) in enumerate(samples):
        sample_dir = os.path.join(level_dir, sid)
        os.makedirs(sample_dir, exist_ok=True)
        try:
            pts = read_xyz(xp)
            verts = read_polygon_verts(op)
        except Exception as exc:
            print(f"  [error] {sid}: {exc}")
            summary["errors"] += 1
            continue

        # per-sample-per-level seed: deterministic but distinct
        seed = abs(hash((level_name, sid))) % (2 ** 31)
        rng = np.random.default_rng(seed)

        if len(pts) < MIN_POINTS:
            occ_pts = pts.copy()
            info = {"n_patches": 0, "patches": [], "n_removed": 0,
                    "n_resampled": 0}
            summary["passthrough_small"] += 1
        else:
            occ_pts, info = apply_tree_occlusion(
                pts, verts[:, :2], level_cfg, rng)
            summary["applied"] += 1

        write_xyz(os.path.join(sample_dir, POINTS_NAME), occ_pts)
        shutil.copyfile(op, os.path.join(sample_dir, POLYGON_NAME))
        ok_ids.append(sid)

        if i < preview_first:
            preview_path = os.path.join(level_dir, f"preview_{sid}.png")
            save_preview(pts, occ_pts, verts[:, :2], info, sid,
                         level_name, preview_path)

        if (i + 1) % 500 == 0 or (i + 1) == len(samples):
            print(f"  [{level_name}] {i+1:,}/{len(samples):,}  "
                  f"applied={summary['applied']:,}  "
                  f"passthrough={summary['passthrough_small']:,}  "
                  f"errors={summary['errors']:,}")

    # per-level test.txt with optional prefix
    test_path = os.path.join(level_dir, "test.txt")
    prefix = ""
    if test_prefix:
        prefix = test_prefix.replace("\\", "/").rstrip("/") + "/"
    with open(test_path, "w") as f:
        for sid in ok_ids:
            f.write(f"{prefix}{sid}\n")
    print(f"  [{level_name}] wrote {test_path} ({len(ok_ids):,} ids)")

    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Add tree-shadow spatial occlusions to Point2Roof samples "
                    "while preserving total point count. Three severity "
                    "levels (light/moderate/heavy) model the residential -> "
                    "urban-narrow-street progression for RQ 3.1/3.2.")
    ap.add_argument("data_dir", nargs="?",
                    help="root directory of <id>/{points_n.xyz, polygon_n.obj} "
                         "samples. Mutually exclusive with --samples-file.")
    ap.add_argument("--samples-file", default=None, metavar="PATH",
                    help="text file listing sample folder paths (one per line). "
                         "Paths resolved relative to cwd first, with fallback "
                         "to the file's own directory. Mutually exclusive "
                         "with data_dir.")
    ap.add_argument("--out", default="./occlusion_trees",
                    help="output ROOT directory; per-level sub-directories "
                         "(light/, moderate/, heavy/) are created under it.")
    ap.add_argument("--levels", default="light,moderate,heavy",
                    help="comma-separated subset of light,moderate,heavy "
                         "(default: all three).")
    ap.add_argument("--test-prefix",
                    default="../Point2Roof/occlusion_trees/",
                    help="path prepended to each id in the per-level test.txt "
                         "files. Trailing slash optional. Use '' for bare ids.")
    ap.add_argument("--preview", type=int, default=0, metavar="N",
                    help="save before/after PNG previews for the first N "
                         "samples in each level (default 0 = no previews). "
                         "Requires matplotlib.")
    args = ap.parse_args()

    if args.data_dir and args.samples_file:
        ap.error("provide EITHER data_dir OR --samples-file, not both.")
    if not args.data_dir and not args.samples_file:
        ap.error("provide data_dir or --samples-file.")

    if args.samples_file:
        print(f"Reading sample list from {args.samples_file} ...")
        try:
            samples = find_samples_from_file(args.samples_file)
        except FileNotFoundError as exc:
            ap.error(str(exc))
        source_desc = f"samples-file {args.samples_file}"
    else:
        if not os.path.isdir(args.data_dir):
            ap.error(f"data_dir not found: {args.data_dir}")
        samples = find_samples(args.data_dir)
        source_desc = f"directory {args.data_dir}"

    requested = [l.strip() for l in args.levels.split(",") if l.strip()]
    unknown = [l for l in requested if l not in LEVELS]
    if unknown:
        ap.error(f"unknown level(s): {unknown}. valid: {list(LEVELS)}")

    if not samples:
        ap.error(f"no usable samples found in {source_desc}")
    print(f"Found {len(samples):,} sample(s) from {source_desc}")
    print(f"Severity levels: {requested}")
    print(f"Test-list prefix: {args.test_prefix!r}")
    if args.preview > 0:
        print(f"Saving previews for the first {args.preview} sample(s) per level")
    print()

    os.makedirs(args.out, exist_ok=True)
    rng_master = np.random.default_rng(42)

    grand_summary = {}
    for level_name in requested:
        cfg = LEVELS[level_name]
        print(f"=== level: {level_name} ===")
        print(f"  n_patches={cfg['n_patches']}, "
              f"radius_frac={cfg['radius_frac']}, "
              f"location={cfg['location']}, "
              f"corner_prob={cfg['corner_prob']}, "
              f"max_removal={cfg['max_removal']:.0%}")
        s = process_one(level_name, cfg, samples, args.out, rng_master,
                        args.test_prefix, args.preview)
        grand_summary[level_name] = s

    print(f"\n{'='*60}")
    print(f"Done. Outputs at: {os.path.abspath(args.out)}")
    for lvl, s in grand_summary.items():
        print(f"  {lvl}: applied={s['applied']:,}, "
              f"passthrough={s['passthrough_small']:,}, "
              f"errors={s['errors']:,}")


if __name__ == "__main__":
    main()
