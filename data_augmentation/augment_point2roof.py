#!/usr/bin/env python3
"""
augment_point2roof.py
=====================

Synthetic data augmentation for Point2Roof training samples.

A Point2Roof training sample is a pair:
    points_n.xyz   -> roof point cloud, rows of "x y z"
    polygon_n.obj  -> ground-truth outline graph:
                        "v x y z"  vertex lines  (the label vertices)
                        "l i j"    edge lines     (1-indexed, OBJ convention)

The point cloud is the network INPUT; the OBJ is the SUPERVISION
(vertex locations -> detection + offset heads, edges -> PPA edge head).

THE GOLDEN RULE
---------------
Any GEOMETRIC transform must be applied to BOTH the point cloud AND the OBJ
vertices using the SAME matrix, otherwise the point<->vertex<->edge
correspondence (the entire value of the label) is destroyed.
Edge connectivity ("l i j") is topological: it is NEVER changed by a
rigid/affine transform, so edges are simply copied through.

Point-only perturbations (jitter, dropout, duplication, outliers) touch ONLY
the cloud and leave the OBJ untouched -- they model sensor noise / varying
sampling density and teach the network robustness without changing the target.

INPUT SELECTION
---------------
Two mutually-exclusive ways to specify which samples to augment:

  1) Positional `data_dir`: every <data_dir>/<id>/{points_n.xyz, polygon_n.obj}
     under the directory is augmented.

  2) `--samples-file train.txt`: process only the samples listed in the file,
     one folder path per line. Paths in the file are resolved relative to the
     current working directory first; if that fails, against the file's own
     directory (so the script works whether the trainer-style relative paths
     match cwd or not). Absolute paths are honored as-is.

Usage
-----
    # all samples in a directory:
    python augment_point2roof.py data/ --out ./augmented

    # only samples listed in a file:
    python augment_point2roof.py --samples-file train.txt --out ./augmented

    # synthetic only (no real data):
    python augment_point2roof.py --synth 200 --out ./synthetic
"""

import os
import re
import sys
import glob
import math
import argparse
import numpy as np

# ----------------------------------------------------------------------------
# MAIN CONTROL: how many augmented copies to generate per input sample.
# ----------------------------------------------------------------------------
AUG_PER_SAMPLE = 8

# ----------------------------------------------------------------------------
# Augmentation hyper-parameters.  Tune these to your dataset's characteristics.
# Probabilities decide whether each transform is included in a given augmentation.
# ----------------------------------------------------------------------------
CFG = {
    # ---- geometric (applied to BOTH cloud and OBJ vertices) ----
    "rotate_z":        {"p": 1.00},                       # yaw about vertical axis (always on)
    "flip":            {"p": 0.50},                       # mirror across a random vertical plane
    "scale_iso":       {"p": 0.60, "range": (0.85, 1.20)},# uniform xy(z) scale
    "scale_aniso":     {"p": 0.40, "range": (0.85, 1.20)},# independent x/y scale (changes aspect ratio)
    "scale_z":         {"p": 0.30, "range": (0.80, 1.25)},# independent roof-height scale
    "shear_xy":        {"p": 0.25, "amount": 0.12},       # small horizontal shear
    "translate":       {"p": 0.50, "amount": 2.0},        # global xy shift (meters)

    # ---- point-cloud-only (OBJ left untouched) ----
    "jitter":          {"p": 0.80, "sigma": 0.03, "clip": 0.10},  # per-point Gaussian noise (m)
    "dropout":         {"p": 0.60, "max_ratio": 0.30},            # randomly remove up to N% of points
    "resample":        {"p": 0.30, "ratio_range": (0.7, 1.3)},    # change overall point count
    "outliers":        {"p": 0.30, "max_ratio": 0.02, "spread": 1.5},  # add stray noise points
}

RNG = np.random.default_rng()  # reseeded per-augmentation for reproducibility


# ============================================================================
# I/O
# ============================================================================
def read_xyz(path):
    """Read an .xyz point cloud -> (N,3) float array. Ignores extra columns."""
    pts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                continue
            pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not pts:
        raise ValueError(f"No points parsed from {path}")
    return np.asarray(pts, dtype=np.float64)


def read_obj(path):
    """
    Read a wavefront .obj outline graph.
    Returns:
        verts : (V,3) float array
        edges : list of (i, j) 0-indexed integer pairs
    Preserves any 'f' faces as edges too (decomposed into their boundary).
    """
    verts, edges = [], []
    with open(path, "r") as f:
        for line in f:
            tok = line.strip().split()
            if not tok:
                continue
            if tok[0] == "v":
                verts.append([float(tok[1]), float(tok[2]), float(tok[3])])
            elif tok[0] == "l":
                idx = [int(re.split(r"[/]", t)[0]) for t in tok[1:]]
                # a polyline "l a b c" -> edges (a,b),(b,c)
                for a, b in zip(idx[:-1], idx[1:]):
                    edges.append((a - 1, b - 1))   # to 0-indexed
            elif tok[0] == "f":
                idx = [int(re.split(r"[/]", t)[0]) for t in tok[1:]]
                for a, b in zip(idx, idx[1:] + idx[:1]):  # close the face loop
                    edges.append((a - 1, b - 1))
    if not verts:
        raise ValueError(f"No vertices parsed from {path}")
    return np.asarray(verts, dtype=np.float64), edges


def write_xyz(path, pts):
    with open(path, "w") as f:
        for x, y, z in pts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def write_obj(path, verts, edges):
    with open(path, "w") as f:
        for x, y, z in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for i, j in edges:
            f.write(f"l {i + 1} {j + 1}\n")   # back to 1-indexed


# ============================================================================
# Geometric transforms  (return a 3x3 linear matrix M and 3-vector t;
# point' = (point - pivot) @ M.T + pivot + t  applied identically to cloud+verts)
# ============================================================================
def build_affine(rng, pivot):
    """
    Compose a random affine transform from the enabled augmentations.
    Returns (M, t, flags) where the same (M, t, pivot) is applied to both
    the cloud and the OBJ vertices.
    """
    M = np.eye(3)
    t = np.zeros(3)
    flags = []

    # --- yaw rotation about vertical (z) axis ---
    if rng.random() < CFG["rotate_z"]["p"]:
        a = rng.uniform(0, 2 * math.pi)
        c, s = math.cos(a), math.sin(a)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        M = Rz @ M
        flags.append(f"rotz{math.degrees(a):.0f}")

    # --- mirror across a random vertical plane (horizontal flip) ---
    if rng.random() < CFG["flip"]["p"]:
        a = rng.uniform(0, math.pi)            # plane normal direction in xy
        nx, ny = math.cos(a), math.sin(a)
        # Householder reflection in xy, identity in z
        Ref = np.array([
            [1 - 2 * nx * nx, -2 * nx * ny, 0],
            [-2 * nx * ny, 1 - 2 * ny * ny, 0],
            [0, 0, 1],
        ])
        M = Ref @ M
        flags.append("flip")

    # --- isotropic scale (xy + z together) ---
    if rng.random() < CFG["scale_iso"]["p"]:
        s = rng.uniform(*CFG["scale_iso"]["range"])
        M = np.diag([s, s, s]) @ M
        flags.append(f"iso{s:.2f}")

    # --- anisotropic xy scale (changes footprint aspect ratio -> new topology cues) ---
    if rng.random() < CFG["scale_aniso"]["p"]:
        sx = rng.uniform(*CFG["scale_aniso"]["range"])
        sy = rng.uniform(*CFG["scale_aniso"]["range"])
        M = np.diag([sx, sy, 1.0]) @ M
        flags.append(f"aniso{sx:.2f}x{sy:.2f}")

    # --- independent roof-height scale ---
    if rng.random() < CFG["scale_z"]["p"]:
        sz = rng.uniform(*CFG["scale_z"]["range"])
        M = np.diag([1.0, 1.0, sz]) @ M
        flags.append(f"sz{sz:.2f}")

    # --- small horizontal shear ---
    if rng.random() < CFG["shear_xy"]["p"]:
        h = CFG["shear_xy"]["amount"]
        shxy = rng.uniform(-h, h)
        shyx = rng.uniform(-h, h)
        Sh = np.array([[1, shxy, 0], [shyx, 1, 0], [0, 0, 1]])
        M = Sh @ M
        flags.append("shear")

    # --- global xy translation ---
    if rng.random() < CFG["translate"]["p"]:
        amt = CFG["translate"]["amount"]
        t = t + np.array([rng.uniform(-amt, amt), rng.uniform(-amt, amt), 0.0])
        flags.append("trans")

    return M, t, flags


def apply_affine(coords, M, t, pivot):
    """point' = (point - pivot) @ M.T + pivot + t"""
    return (coords - pivot) @ M.T + pivot + t


# ============================================================================
# Point-cloud-only perturbations (OBJ untouched)
# ============================================================================
def perturb_cloud(pts, rng, scale_ref):
    """
    Apply sensor-style perturbations that affect ONLY the input cloud.
    scale_ref is a characteristic size (m) used to scale absolute noise.
    """
    flags = []
    out = pts.copy()

    # --- jitter: per-point Gaussian noise ---
    if rng.random() < CFG["jitter"]["p"]:
        sig = CFG["jitter"]["sigma"]
        clip = CFG["jitter"]["clip"]
        noise = np.clip(rng.normal(0, sig, out.shape), -clip, clip)
        out = out + noise
        flags.append("jit")

    # --- random dropout: remove a fraction of points ---
    if rng.random() < CFG["dropout"]["p"] and len(out) > 64:
        ratio = rng.uniform(0, CFG["dropout"]["max_ratio"])
        keep = rng.random(len(out)) >= ratio
        if keep.sum() >= 64:               # never let it collapse
            out = out[keep]
            flags.append(f"drop{ratio:.2f}")

    # --- resample: change total point count (up- or down-sample) ---
    if rng.random() < CFG["resample"]["p"] and len(out) > 64:
        ratio = rng.uniform(*CFG["resample"]["ratio_range"])
        target = max(64, int(len(out) * ratio))
        idx = rng.choice(len(out), size=target, replace=(target > len(out)))
        out = out[idx]
        # add tiny noise to duplicated points so they are not exact copies
        if target > len(pts):
            out = out + rng.normal(0, CFG["jitter"]["sigma"] * 0.5, out.shape)
        flags.append(f"resamp{ratio:.2f}")

    # --- outliers: inject a few stray points around the cloud ---
    if rng.random() < CFG["outliers"]["p"]:
        ratio = rng.uniform(0, CFG["outliers"]["max_ratio"])
        n_out = int(len(out) * ratio)
        if n_out > 0:
            center = out.mean(axis=0)
            spread = CFG["outliers"]["spread"] * scale_ref * 0.05
            stray = center + rng.normal(0, spread, (n_out, 3))
            out = np.vstack([out, stray])
            flags.append(f"outl{n_out}")

    return out, flags


# ============================================================================
# One augmentation
# ============================================================================
def augment_once(pts, verts, edges, seed):
    rng = np.random.default_rng(seed)

    # pivot = footprint centroid (xy center, mean z) so transforms are local
    pivot = np.array([pts[:, 0].mean(), pts[:, 1].mean(), pts[:, 2].mean()])

    # characteristic size for scaling absolute noise terms
    scale_ref = float(np.linalg.norm(pts.max(0) - pts.min(0)))

    # 1) shared geometric transform on BOTH cloud and OBJ vertices
    M, t, gflags = build_affine(rng, pivot)
    pts_aug = apply_affine(pts, M, t, pivot)
    verts_aug = apply_affine(verts, M, t, pivot)

    # 2) cloud-only perturbations (OBJ stays as verts_aug, edges unchanged)
    pts_aug, pflags = perturb_cloud(pts_aug, rng, scale_ref)

    flags = gflags + pflags
    return pts_aug, verts_aug, edges, flags


# Fixed filenames used inside each sample folder.
POINTS_NAME = "points_n.xyz"
POLYGON_NAME = "polygon_n.obj"


# ============================================================================
# Sequential output-folder ID minter
# ============================================================================
class IdMinter:
    """
    Hands out sequential zero-padded folder IDs (000001, 000002, ...) so every
    written sample -- real augmentations and synthetic -- lands in a numbered
    folder under the output root, matching the input data layout.
    """
    def __init__(self, start=1, width=6):
        self.n = start - 1
        self.width = width

    def next(self):
        self.n += 1
        return str(self.n).zfill(self.width)


def next_available_id(out_root):
    """
    Scan out_root for existing all-digit folder names and return the next index
    after the highest one. If out_root doesn't exist or has no numbered folders,
    returns 1. Non-numeric folder names are ignored. This lets repeated runs
    append to an existing dataset without overwriting.
    """
    if not os.path.isdir(out_root):
        return 1
    highest = 0
    for name in os.listdir(out_root):
        # only treat all-digit names as numbered samples (e.g. '000023')
        if name.isdigit() and os.path.isdir(os.path.join(out_root, name)):
            highest = max(highest, int(name))
    return highest + 1


# ============================================================================
# Driver
# ============================================================================
def process_sample(sample_id, xyz_path, obj_path, out_root, aug_per_sample,
                   minter, base_seed=0, keep_original=True):
    """
    Augment one sample folder. Each generated copy is written to the next
    sequential numbered folder (000001/, 000002/, ...) containing
    points_n.xyz + polygon_n.obj.
    """
    pts = read_xyz(xyz_path)
    verts, edges = read_obj(obj_path)

    def dump(p, v, e):
        d = os.path.join(out_root, minter.next())
        os.makedirs(d, exist_ok=True)
        write_xyz(os.path.join(d, POINTS_NAME), p)
        write_obj(os.path.join(d, POLYGON_NAME), v, e)
        return d

    written = []
    if keep_original:
        written.append(dump(pts, verts, edges))

    for k in range(aug_per_sample):
        seed = base_seed * 100003 + k + 1
        p, v, e, flags = augment_once(pts, verts, edges, seed)
        d = dump(p, v, e)
        written.append(d)
        print(f"    [{k+1}/{aug_per_sample}] -> {os.path.basename(d)}  "
              f"(from '{sample_id}')  pts={len(p):5d}  ops=[{','.join(flags)}]")

    return written


def find_samples(data_dir):
    """
    Find every sample folder under data_dir that contains BOTH the points and
    polygon files.  Layout expected:
        data_dir/<id>/points_n.xyz
        data_dir/<id>/polygon_n.obj
    Returns a sorted list of (sample_id, xyz_path, obj_path).
    Sub-directories are searched recursively, so nested layouts also work.
    """
    samples = []
    for xyz_path in glob.glob(os.path.join(data_dir, "**", POINTS_NAME),
                              recursive=True):
        folder = os.path.dirname(xyz_path)
        obj_path = os.path.join(folder, POLYGON_NAME)
        if not os.path.isfile(obj_path):
            print(f"  [skip] {folder}: missing {POLYGON_NAME}")
            continue
        # sample id = path of the folder relative to data_dir, flattened
        rel = os.path.relpath(folder, data_dir)
        sample_id = rel.replace(os.sep, "_")
        samples.append((sample_id, xyz_path, obj_path))
    samples.sort(key=lambda s: s[0])
    return samples


def find_samples_from_file(samples_file):
    """
    Read a samples-list text file (one folder path per line) and return the
    same (sample_id, xyz_path, obj_path) tuples that find_samples() produces.

    Path resolution: each line is resolved relative to the CURRENT WORKING
    DIRECTORY first. If that folder doesn't exist, we retry relative to the
    samples_file's own directory (so e.g. train.txt generated alongside the
    data still works even if you cd'd somewhere else). Absolute paths are
    used as-is. Lines that are blank or start with '#' are ignored.

    Missing folders / missing required files are skipped with a per-line
    reason; duplicates (same resolved path) are deduplicated with a warning.
    """
    if not os.path.isfile(samples_file):
        raise FileNotFoundError(f"samples file not found: {samples_file}")

    file_dir = os.path.dirname(os.path.abspath(samples_file))
    cwd = os.getcwd()

    samples = []
    seen_paths = set()
    missing = []
    used_fallback = 0

    with open(samples_file) as f:
        for lineno, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue

            # try cwd first, then file_dir; absolute paths used as-is
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
                missing.append((lineno, raw, "folder not found in cwd "
                                "or samples-file dir"))
                continue

            xp = os.path.join(folder, POINTS_NAME)
            op = os.path.join(folder, POLYGON_NAME)
            if not os.path.isfile(xp):
                missing.append((lineno, raw, f"missing {POINTS_NAME}"))
                continue
            if not os.path.isfile(op):
                missing.append((lineno, raw, f"missing {POLYGON_NAME}"))
                continue

            canonical = os.path.realpath(folder)
            if canonical in seen_paths:
                # duplicate -- skip silently; not an error
                continue
            seen_paths.add(canonical)

            # sample id = folder basename (matches the existing project
            # convention; lines in train.txt typically end in the sample id)
            sample_id = os.path.basename(folder.rstrip(os.sep))
            samples.append((sample_id, xp, op))

    # report resolution stats
    if used_fallback:
        print(f"  note: {used_fallback} path(s) resolved against the "
              f"samples-file directory rather than cwd "
              f"(the file's directory is '{file_dir}')")
    if missing:
        print(f"  {len(missing)} line(s) skipped:")
        for lineno, raw, why in missing[:10]:
            print(f"    line {lineno}: {raw!r}: {why}")
        if len(missing) > 10:
            print(f"    ... ({len(missing) - 10} more)")

    # we already deduplicated; preserve list order (input order from the file)
    return samples


# ============================================================================
# Synthetic-archetype generation (topology-varying augmentation)
# ============================================================================
def process_synth(out_root, n_per_arch, archetypes, base_seed, minter,
                  aug_each=0):
    """
    Generate synthetic parametric roof archetypes (gable, hip, pyramid, L, T,
    cross-gable), each as a matched (point cloud, outline graph) pair, written
    to sequential numbered folders (continuing the same counter as the real
    augmentations).

    For each archetype we create `n_per_arch` base roofs with randomized
    dimensions. If aug_each > 0, we additionally apply the affine+noise
    augmentation `aug_each` times to every synthetic roof (re-posing it),
    multiplying variety on top of the new topology.
    """
    import roof_archetypes as ra

    def dump(p, v, e):
        d = os.path.join(out_root, minter.next())
        os.makedirs(d, exist_ok=True)
        write_xyz(os.path.join(d, POINTS_NAME), p)
        write_obj(os.path.join(d, POLYGON_NAME), v, e)
        return d

    written = []
    for arch in archetypes:
        if arch not in ra.ARCHETYPES:
            print(f"  [skip] unknown archetype '{arch}'")
            continue
        print(f"\n== synth archetype '{arch}'  ({n_per_arch} base roof(s))")
        for k in range(n_per_arch):
            seed = base_seed * 100003 + hash(arch) % 9973 * 17 + k + 1
            pts, verts, edges = ra.generate(arch, seed=seed)
            d = dump(pts, verts, edges)
            written.append(d)
            print(f"    base -> {os.path.basename(d)}  ({arch})  "
                  f"pts={len(pts):5d} V={len(verts):2d} E={len(edges):2d}")

            # optional affine+noise re-posing on top of the synthetic roof
            for a in range(aug_each):
                aseed = seed * 7919 + a + 1
                p, v, e, flags = augment_once(pts, verts, edges, aseed)
                d2 = dump(p, v, e)
                written.append(d2)
                print(f"        -> {os.path.basename(d2)}  ({arch})  "
                      f"pts={len(p):5d} ops=[{','.join(flags)}]")
    return written


def main():
    ap = argparse.ArgumentParser(
        description="Augment Point2Roof samples laid out as "
                    "data_dir/<id>/{points_n.xyz, polygon_n.obj}, and/or "
                    "generate synthetic parametric roof archetypes.")
    ap.add_argument("data_dir", nargs="?",
                    help="root data directory containing per-sample folders. "
                         "Mutually exclusive with --samples-file. Optional if "
                         "only using --synth.")
    ap.add_argument("--samples-file", default=None, metavar="PATH",
                    help="text file listing sample folder paths (one per line, "
                         "as produced by inspect_corner_distribution.py "
                         "--write-test). Paths are resolved relative to cwd "
                         "first, with fallback to the file's own directory. "
                         "Mutually exclusive with the positional data_dir.")
    ap.add_argument("--out", default="./augmented",
                    help="output directory (default: ./augmented)")
    ap.add_argument("--aug-per-sample", type=int, default=AUG_PER_SAMPLE,
                    help=f"affine+noise augmentations per real sample "
                         f"(default: {AUG_PER_SAMPLE})")
    ap.add_argument("--no-keep-original", action="store_true",
                    help="do not copy the original sample into the output")
    # --- synthetic archetype options ---
    ap.add_argument("--synth", type=int, default=0, metavar="N",
                    help="generate N synthetic roofs PER archetype "
                         "(topology-varying augmentation). 0 disables it.")
    ap.add_argument("--synth-archetypes", default="all",
                    help="comma-separated subset of "
                         "gable,hip,pyramid,lshape,tshape,cross_gable "
                         "or 'all' (default)")
    ap.add_argument("--synth-aug-each", type=int, default=0, metavar="M",
                    help="additionally apply M affine+noise augmentations to "
                         "each synthetic roof (default: 0)")
    ap.add_argument("--seed", type=int, default=12345,
                    help="global RNG seed for reproducibility")
    ap.add_argument("--start-id", default="auto",
                    help="first folder number to assign. 'auto' (default) scans "
                         "the output directory and starts after the highest "
                         "existing numbered folder, so repeated runs append "
                         "without overwriting. Pass an integer to force a "
                         "specific starting number.")
    ap.add_argument("--id-width", type=int, default=6,
                    help="zero-padding width for folder names (default: 6)")
    args = ap.parse_args()

    global RNG
    RNG = np.random.default_rng(args.seed)

    # ---- validate input selection: data_dir XOR samples-file (or only --synth) ----
    if args.data_dir and args.samples_file:
        ap.error("provide EITHER a positional data_dir OR --samples-file, "
                 "not both.")
    have_input = bool(args.data_dir or args.samples_file)
    if not have_input and args.synth <= 0:
        ap.error("provide data_dir or --samples-file to augment, "
                 "and/or --synth N to generate synthetic archetypes")

    os.makedirs(args.out, exist_ok=True)

    # Resolve start id: 'auto' scans for existing numbered folders so we append
    # rather than overwrite; an explicit integer forces a specific start.
    if str(args.start_id).lower() == "auto":
        start_id = next_available_id(args.out)
        if start_id > 1:
            print(f"Existing numbered folders detected in '{args.out}'; "
                  f"appending from {str(start_id).zfill(args.id_width)}.")
    else:
        try:
            start_id = int(args.start_id)
        except ValueError:
            ap.error(f"--start-id must be 'auto' or an integer, got {args.start_id!r}")

    total_written = 0
    minter = IdMinter(start=start_id, width=args.id_width)

    # ---- 1) augment real samples (from data_dir OR samples-file) ----
    if have_input:
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

        if not samples:
            print(f"No usable sample folders found in {source_desc}")
        else:
            print(f"Found {len(samples)} real sample(s) from {source_desc}")
            for i, (sid, xp, op) in enumerate(samples):
                print(f"\n== [{i+1}/{len(samples)}] real sample '{sid}'")
                try:
                    written = process_sample(
                        sid, xp, op, args.out, args.aug_per_sample, minter,
                        base_seed=args.seed + i,
                        keep_original=not args.no_keep_original)
                    total_written += len(written)
                except Exception as exc:
                    print(f"    [error] failed to process '{sid}': {exc}")

    # ---- 2) synthetic archetypes (if requested) ----
    if args.synth > 0:
        import roof_archetypes as ra
        if args.synth_archetypes.strip().lower() == "all":
            arch_list = list(ra.ARCHETYPES)
        else:
            arch_list = [a.strip() for a in args.synth_archetypes.split(",") if a.strip()]
        written = process_synth(args.out, args.synth, arch_list,
                                base_seed=args.seed, minter=minter,
                                aug_each=args.synth_aug_each)
        total_written += len(written)

    print(f"\nDone. {total_written} folder(s) written to: "
          f"{os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
