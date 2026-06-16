# test_helper.py
# ─────────────────────────────────────────────────────────────────────────────
# PATCHES vs original:
#   [P1] load_data_to_gpu: unified with the patched train_utils.py version.
#        Added handling for:
#          - bool arrays (skipped in original → now converted to bool tensors)
#          - already-on-GPU tensors (ignored in original → now passed through)
#          - already-on-CPU tensors (not moved in original → now moved to CUDA)
#        All numpy float/int handling unchanged.
#   [P2] Removed duplicate assign_targets() function. It was identical to
#        pointnet2.py's version and nothing in the visible codebase imported
#        it from here. Dead code removed to avoid future drift.
#   [P3] denorm_to_points_n_space: added an explicit docstring noting that
#        this returns LOCAL coordinates (minPt-relative), NOT absolute coords.
#        No math changed. This prevents confusion with denorm_to_raw_space.
# Everything else (IO helpers, GeoJSON conversion, geometry helpers) unchanged.
# ─────────────────────────────────────────────────────────────────────────────
import os
import shutil
import json
import itertools
from pathlib import Path

import torch
import numpy as np
import laspy

from model.pointnet_util import *
from model.model_utils import *


DATA_DIR = "easy"



# ── ADD THIS FUNCTION TO utils/test_helper.py ────────────────────────────────
# Raw edge-graph -> GeoJSON MultiLineString (no cycle requirement).
# Mirrors refined_obj_to_geojson's COORDINATE handling exactly (the +minPt
# denorm and the centroid-anchor to src_polygon_geojson) so model_pred.geojson
# overlays pred.geojson in the same CRS — but serializes the edge head's
# thresholded graph AS-IS (MultiLineString), so it is written even when the
# graph is open / branched / disconnected (the very cases worth diagnosing,
# which refined_obj_to_geojson rejects via its strict degree-2 single-cycle
# requirement in _order_cycle_from_edges).

def edge_graph_obj_to_geojson(
    obj_path, out_geojson_path, mm_pt=None, src_polygon_geojson=None, logger=None
):
    """
    Convert a CloudCompare-friendly OBJ wireframe (v + l) into a GeoJSON
    MultiLineString of its edges, WITHOUT requiring the edges to form a closed
    simple polygon. Intended for RAW model graphs (e.g. model_pred.obj, the
    edge head's thresholded pairs) where the topology may be open, branched, or
    disconnected and is itself the thing being inspected.

    Coordinate handling is identical to refined_obj_to_geojson:
      - vertices are denormalized to absolute XY by adding minPt (mm_pt[0]),
      - if src_polygon_geojson exists, the whole graph is centroid-anchored to
        the source polygon's centroid (same shift logic),
      - the source CRS member is copied through when present.
    This guarantees model_pred.geojson and pred.geojson land in the same frame.

    Returns True on success (file written), False otherwise. Unlike the polygon
    converter, an open/branched graph is a SUCCESS here (it is written as lines).
    """
    try:
        if not os.path.exists(obj_path):
            if logger is not None:
                logger.warning(f"[model_pred.geojson] Missing obj: {obj_path}")
            return False

        verts, edges = [], []
        with open(obj_path, "r") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                tag = parts[0].lower()
                if tag == "v":
                    if len(parts) < 4:
                        continue
                    try:
                        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except Exception:
                        continue
                elif tag == "l":
                    if len(parts) < 3:
                        continue
                    try:
                        i = int(parts[1]) - 1
                        j = int(parts[2]) - 1
                    except Exception:
                        continue
                    edges.append([i, j])

        verts = np.asarray(verts, dtype=np.float64)
        edges = np.asarray(edges, dtype=np.int64) if len(edges) > 0 else np.zeros((0, 2), dtype=np.int64)

        if verts.ndim != 2 or verts.shape[0] < 2 or verts.shape[1] < 3:
            if logger is not None:
                logger.warning(f"[model_pred.geojson] Insufficient vertices in {obj_path}")
            return False
        if edges.shape[0] < 1:
            if logger is not None:
                logger.warning(f"[model_pred.geojson] No edges in {obj_path}")
            return False
        # drop out-of-range / self-loop edges defensively
        good = []
        for u, v in edges.tolist():
            if u == v or u < 0 or v < 0 or u >= len(verts) or v >= len(verts):
                continue
            good.append((u, v))
        if len(good) < 1:
            if logger is not None:
                logger.warning(f"[model_pred.geojson] No valid edges in {obj_path}")
            return False
        edges = np.asarray(good, dtype=np.int64)

        xy = verts[:, :2].copy()

        # denorm: add minPt (same as refined_obj_to_geojson with mm_pt)
        if mm_pt is not None:
            mm_pt = np.asarray(mm_pt, dtype=np.float64)
            if mm_pt.shape != (2, 3):
                if logger is not None:
                    logger.warning(f"[model_pred.geojson] Invalid mm_pt shape {mm_pt.shape} for {obj_path}")
                return False
            xy = xy + mm_pt[0][:2]

        # centroid-anchor to source polygon (same as refined_obj_to_geojson)
        src_crs = None
        if src_polygon_geojson is not None and os.path.exists(src_polygon_geojson):
            try:
                with open(src_polygon_geojson, "r", encoding="utf-8") as f:
                    src_gj = json.load(f)
                if isinstance(src_gj, dict) and "crs" in src_gj:
                    src_crs = src_gj["crs"]
                src_feats = src_gj.get("features", [])
                if len(src_feats) > 0:
                    src_geom = src_feats[0].get("geometry", {})
                    if src_geom.get("type", "") == "Polygon":
                        src_ring = np.asarray(src_geom.get("coordinates", [[]])[0], dtype=np.float64)
                        if len(src_ring) >= 2 and np.allclose(src_ring[0], src_ring[-1]):
                            src_ring = src_ring[:-1]
                        if src_ring.ndim == 2 and src_ring.shape[0] >= 3 and src_ring.shape[1] >= 2:
                            # anchor on the graph's VERTEX centroid (matches the
                            # ring-centroid anchor used for the polygon path)
                            pred_centroid = np.mean(xy, axis=0)
                            src_centroid = np.mean(src_ring[:, :2], axis=0)
                            xy = xy + (src_centroid - pred_centroid)
            except Exception as e:
                if logger is not None:
                    logger.warning(f"[model_pred.geojson] Could not anchor to {src_polygon_geojson}: {e}")

        # build MultiLineString: one 2-point line segment per edge
        line_coords = []
        for u, v in edges.tolist():
            line_coords.append([xy[u].tolist(), xy[v].tolist()])

        feature = {
            "type": "Feature",
            "properties": {"source": "model_edge_head_raw", "n_edges": len(line_coords)},
            "geometry": {"type": "MultiLineString", "coordinates": line_coords},
        }
        geojson = {"type": "FeatureCollection", "features": [feature]}
        if src_crs is not None:
            geojson["crs"] = src_crs

        with open(out_geojson_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)
        return True

    except Exception as e:
        if logger is not None:
            logger.warning(f"[model_pred.geojson] Failed converting {obj_path}: {e}")
        return False




# ============================================================
# Dataset path helpers
# ============================================================

def load_train_list(file_path=f"./{DATA_DIR}/test.txt", base_dir=f"./{DATA_DIR}"):
    """
    Accepts either:
      000003
    or:
      ./{DATA_DIR}/000003

    Returns resolved folder Paths.
    """
    file_path = Path(file_path)
    base_dir = Path(base_dir)

    sample_dirs = []
    with open(file_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            p = Path(s)

            if p.parent != Path("."):
                sample_dirs.append(p)
            else:
                sample_dirs.append(base_dir / p)

    return sample_dirs


def build_frame_geojson_lookup(file_path=f"./{DATA_DIR}/test.txt", base_dir=f"./{DATA_DIR}"):
    """
    Build:
      {
        "000003": "/full/path/to/{DATA_DIR}/000003/polygon.geojson",
        ...
      }
    """
    sample_dirs = load_train_list(file_path=file_path, base_dir=base_dir)
    lookup = {}

    for sample_dir in sample_dirs:
        sample_dir = Path(sample_dir).resolve()
        frame_id = sample_dir.name
        geojson_path = sample_dir / "polygon.geojson"

        if geojson_path.exists():
            lookup[str(frame_id)] = str(geojson_path)

    return lookup


# ============================================================
# IO helpers
# ============================================================

def writePoints(points, clsRoad):
    with open(clsRoad, 'w+') as file1:
        for i in range(len(points)):
            point = points[i]
            file1.write(str(point[0]))
            file1.write(' ')
            file1.write(str(point[1]))
            file1.write(' ')
            file1.write(str(point[2]))
            file1.write('\n')


def writeEdges(edges, clsRoad):
    with open(clsRoad, 'w+') as file1:
        for i in range(len(edges)):
            edge = edges[i]
            file1.write(str(edge[0] + 1))
            file1.write(' ')
            file1.write(str(edge[1] + 1))
            file1.write(' ')
            file1.write('\n')


def write_obj_lines(path, verts, edges):
    """
    CloudCompare-friendly wireframe OBJ:
      v x y z
      l i j
    edges are 0-based indices into verts
    """
    verts = np.asarray(verts, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.int64) if edges is not None else np.zeros((0, 2), dtype=np.int64)

    if verts.ndim != 2 or verts.shape[1] < 3:
        raise ValueError(f"verts must be (N,3+) but got {verts.shape}")

    if edges.size == 0:
        edges = edges.reshape(0, 2).astype(np.int64)
    else:
        edges = edges.reshape(-1, 2).astype(np.int64)

    with open(path, "w") as f:
        for x, y, z in verts[:, :3]:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for i, j in edges:
            f.write(f"l {int(i)+1} {int(j)+1}\n")


def write_obj_points(path, verts):
    """
    CloudCompare-friendly OBJ with vertices only:
      v x y z
    """
    verts = np.asarray(verts, dtype=np.float32)
    if verts.size == 0:
        verts = verts.reshape(0, 3)
    if verts.ndim != 2 or verts.shape[1] < 3:
        raise ValueError(f"verts must be (N,3+) but got {verts.shape}")

    with open(path, "w") as f:
        for x, y, z in verts[:, :3]:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")


# ============================================================
# LAS helpers
# ============================================================

def _sanitize_las_extra_name(name):
    name = str(name)
    name = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    if not name:
        name = "extra"
    return name[:32]


def write_las_with_extras(path, xyz, extras=None, scales=(0.001, 0.001, 0.001)):
    """
    Write a LAS file with extra scalar fields.

    Parameters
    ----------
    path : str
        Output LAS path.
    xyz : array-like, shape (N, 3)
        Point coordinates.
    extras : dict[str, np.ndarray]
        Extra scalar fields of length N.
    scales : tuple[float, float, float]
        LAS coordinate scales.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.size == 0:
        xyz = xyz.reshape(0, 3)

    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"xyz must be (N,3+) but got {xyz.shape}")

    extras = extras or {}

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array(scales, dtype=np.float64)

    if xyz.shape[0] > 0:
        header.offsets = np.min(xyz[:, :3], axis=0)
    else:
        header.offsets = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    normalized_extras = {}
    for name, arr in extras.items():
        arr = np.asarray(arr)
        if arr.shape[0] != xyz.shape[0]:
            raise ValueError(
                f"Extra field '{name}' length {arr.shape[0]} != number of points {xyz.shape[0]}"
            )

        safe_name = _sanitize_las_extra_name(name)

        if np.issubdtype(arr.dtype, np.bool_):
            arr = arr.astype(np.uint8)
            extra_type = np.uint8
        elif np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.int32)
            extra_type = np.int32
        else:
            arr = arr.astype(np.float32)
            extra_type = np.float32

        header.add_extra_dim(laspy.ExtraBytesParams(name=safe_name, type=extra_type))
        normalized_extras[safe_name] = arr

    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]

    for name, arr in normalized_extras.items():
        setattr(las, name, arr)

    las.write(path)


def sample_edges_to_points(verts, edges, edge_scores, spacing=0.20, min_points_per_edge=2):
    """
    Convert candidate edges into sampled points so edge confidence can be exported as LAS.
    """
    verts = np.asarray(verts, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.int64)
    edge_scores = np.asarray(edge_scores, dtype=np.float32)

    if verts.size == 0 or edges.size == 0 or edge_scores.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.uint8),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )

    pts_all = []
    conf_all = []
    sel_all = []
    u_all = []
    v_all = []

    for (u, v), s in zip(edges, edge_scores):
        p1 = verts[int(u)]
        p2 = verts[int(v)]
        length = float(np.linalg.norm(p2 - p1))

        if length <= 1e-8:
            n_samples = min_points_per_edge
        else:
            n_samples = max(min_points_per_edge, int(np.ceil(length / spacing)) + 1)

        ts = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
        pts = (1.0 - ts[:, None]) * p1[None, :] + ts[:, None] * p2[None, :]

        pts_all.append(pts)
        conf_all.append(np.full((n_samples,), s, dtype=np.float32))
        sel_all.append(np.full((n_samples,), 1 if s >= 0.5 else 0, dtype=np.uint8))
        u_all.append(np.full((n_samples,), int(u), dtype=np.int32))
        v_all.append(np.full((n_samples,), int(v), dtype=np.int32))

    return (
        np.concatenate(pts_all, axis=0),
        np.concatenate(conf_all, axis=0),
        np.concatenate(sel_all, axis=0),
        np.concatenate(u_all, axis=0),
        np.concatenate(v_all, axis=0),
    )


# ============================================================
# GeoJSON copy helper
# ============================================================

def copy_polygon_geojson(
    sample_dir,
    frame_id=None,
    src_geojson_path=None,
    frame_geojson_lookup=None,
    fallback_data_root="data"
):
    """
    Copy the correct source polygon.geojson into:
      <sample_dir>/polygon.geojson

    Returns:
      (copied_ok: bool, used_src_path: str | None)
    """
    dst_geojson = os.path.join(sample_dir, "polygon.geojson")

    if src_geojson_path is not None:
        if isinstance(src_geojson_path, np.generic):
            src_geojson_path = src_geojson_path.item()
        src_geojson_path = str(src_geojson_path)

        if os.path.exists(src_geojson_path):
            shutil.copy2(src_geojson_path, dst_geojson)
            return True, src_geojson_path

    if frame_id is not None:
        if isinstance(frame_id, np.generic):
            frame_id = frame_id.item()
        frame_id = str(frame_id)

    if frame_id is not None and frame_geojson_lookup is not None:
        src_geojson = frame_geojson_lookup.get(frame_id, None)
        if src_geojson is not None and os.path.exists(src_geojson):
            shutil.copy2(src_geojson, dst_geojson)
            return True, src_geojson

    if frame_id is not None:
        src_geojson = os.path.join(fallback_data_root, frame_id, "polygon.geojson")
        if os.path.exists(src_geojson):
            shutil.copy2(src_geojson, dst_geojson)
            return True, src_geojson

    return False, None


# ============================================================
# refined.obj -> pred.geojson helper
# ============================================================

def _polygon_signed_area_xy(xy):
    x = xy[:, 0]
    y = xy[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _segments_intersect_2d(p1, p2, q1, q2, eps=1e-9):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, c):
        return (
            min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps and
            min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps
        )

    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)

    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and \
       ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True

    if abs(o1) <= eps and on_segment(p1, p2, q1):
        return True
    if abs(o2) <= eps and on_segment(p1, p2, q2):
        return True
    if abs(o3) <= eps and on_segment(q1, q2, p1):
        return True
    if abs(o4) <= eps and on_segment(q1, q2, p2):
        return True

    return False


def _is_simple_polygon_xy(xy, eps=1e-9):
    """
    Check polygon self-intersection in XY.
    xy should be an open ring (no duplicate closing point).
    """
    n = len(xy)
    if n < 3:
        return False

    for i in range(n):
        a1 = xy[i]
        a2 = xy[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i:
                continue
            if (j == i + 1) or ((i == 0) and (j == n - 1)):
                continue
            b1 = xy[j]
            b2 = xy[(j + 1) % n]
            if (i + 1) % n == j:
                continue

            if _segments_intersect_2d(a1, a2, b1, b2, eps=eps):
                return False
    return True


def _order_cycle_from_edges(num_vertices, edges):
    """
    Reconstruct a single simple cycle from undirected edges.
    Returns ordered vertex indices if valid, else None.
    """
    if num_vertices < 3:
        return None

    adj = {i: [] for i in range(num_vertices)}
    for u, v in edges:
        if u == v:
            return None
        adj[u].append(v)
        adj[v].append(u)

    used_vertices = sorted([k for k, nbrs in adj.items() if len(nbrs) > 0])
    if len(used_vertices) < 3:
        return None

    for k in used_vertices:
        if len(adj[k]) != 2:
            return None

    start = used_vertices[0]
    ordered = [start]
    prev = None
    cur = start

    while True:
        nbrs = adj[cur]
        if len(nbrs) != 2:
            return None

        if prev is None:
            nxt = nbrs[0]
        else:
            nxt = nbrs[0] if nbrs[1] == prev else nbrs[1]

        if nxt == start:
            break

        if nxt in ordered:
            return None

        ordered.append(nxt)
        prev, cur = cur, nxt

        if len(ordered) > len(used_vertices):
            return None

    if len(ordered) != len(used_vertices):
        return None

    return ordered


def refined_obj_to_geojson(refined_obj_path, pred_geojson_path, mm_pt=None, src_polygon_geojson=None, logger=None):
    """
    Convert CloudCompare-friendly refined.obj (v + l) into polygon GeoJSON.
    """
    try:
        if not os.path.exists(refined_obj_path):
            if logger is not None:
                logger.warning(f"[pred.geojson] Missing refined.obj: {refined_obj_path}")
            return False

        verts = []
        edges = []

        with open(refined_obj_path, "r") as f:
            for line_no, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                tag = parts[0].lower()

                if tag == "v":
                    if len(parts) < 4:
                        if logger is not None:
                            logger.warning(f"[pred.geojson] Malformed vertex line at {refined_obj_path}:{line_no}")
                        return False
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                    except Exception:
                        if logger is not None:
                            logger.warning(f"[pred.geojson] Non-numeric vertex line at {refined_obj_path}:{line_no}")
                        return False
                    verts.append([x, y, z])

                elif tag == "l":
                    if len(parts) < 3:
                        if logger is not None:
                            logger.warning(f"[pred.geojson] Malformed edge line at {refined_obj_path}:{line_no}")
                        return False
                    try:
                        i = int(parts[1]) - 1
                        j = int(parts[2]) - 1
                    except Exception:
                        if logger is not None:
                            logger.warning(f"[pred.geojson] Non-integer edge indices at {refined_obj_path}:{line_no}")
                        return False
                    edges.append([i, j])

                else:
                    continue

        verts = np.asarray(verts, dtype=np.float64)
        edges = np.asarray(edges, dtype=np.int64) if len(edges) > 0 else np.zeros((0, 2), dtype=np.int64)

        if verts.ndim != 2 or verts.shape[0] < 3 or verts.shape[1] < 3:
            if logger is not None:
                logger.warning(f"[pred.geojson] Invalid or insufficient vertices in {refined_obj_path}")
            return False

        if edges.ndim != 2 or edges.shape[0] < 3 or edges.shape[1] != 2:
            if logger is not None:
                logger.warning(f"[pred.geojson] Invalid or insufficient edges in {refined_obj_path}")
            return False

        if np.any(edges < 0) or np.any(edges >= len(verts)):
            if logger is not None:
                logger.warning(f"[pred.geojson] Edge index out of bounds in {refined_obj_path}")
            return False

        edges_undirected = set()
        for u, v in edges.tolist():
            if u == v:
                if logger is not None:
                    logger.warning(f"[pred.geojson] Self-loop edge in {refined_obj_path}")
                return False
            edges_undirected.add(tuple(sorted((int(u), int(v)))))
        edges = np.asarray(sorted(edges_undirected), dtype=np.int64)

        ordered_idx = _order_cycle_from_edges(len(verts), edges)
        if ordered_idx is None:
            if logger is not None:
                logger.warning(f"[pred.geojson] Could not reconstruct closed polygon cycle from {refined_obj_path}")
            return False

        ring_xyz = verts[ordered_idx]

        if mm_pt is not None:
            mm_pt = np.asarray(mm_pt, dtype=np.float64)
            if mm_pt.shape != (2, 3):
                if logger is not None:
                    logger.warning(f"[pred.geojson] Invalid mm_pt shape {mm_pt.shape} for {refined_obj_path}")
                return False

            minPt = mm_pt[0]
            ring_xyz = ring_xyz + minPt

        ring_xy = ring_xyz[:, :2]

        uniq_xy = np.unique(np.round(ring_xy, decimals=8), axis=0)
        if len(uniq_xy) < 3:
            if logger is not None:
                logger.warning(f"[pred.geojson] Polygon has < 3 unique XY vertices in {refined_obj_path}")
            return False

        area = abs(_polygon_signed_area_xy(ring_xy))
        if area <= 1e-12:
            if logger is not None:
                logger.warning(f"[pred.geojson] Degenerate zero-area polygon in {refined_obj_path}")
            return False

        if not _is_simple_polygon_xy(ring_xy):
            if logger is not None:
                logger.warning(f"[pred.geojson] Self-intersecting polygon in {refined_obj_path}")
            return False

        src_crs = None

        if src_polygon_geojson is not None and os.path.exists(src_polygon_geojson):
            try:
                with open(src_polygon_geojson, "r", encoding="utf-8") as f:
                    src_gj = json.load(f)

                if isinstance(src_gj, dict) and "crs" in src_gj:
                    src_crs = src_gj["crs"]

                src_feats = src_gj.get("features", [])
                if len(src_feats) > 0:
                    src_geom = src_feats[0].get("geometry", {})
                    if src_geom.get("type", "") == "Polygon":
                        src_ring = src_geom.get("coordinates", [[]])[0]
                        src_ring = np.asarray(src_ring, dtype=np.float64)

                        if len(src_ring) >= 2 and np.allclose(src_ring[0], src_ring[-1]):
                            src_ring = src_ring[:-1]

                        if src_ring.ndim == 2 and src_ring.shape[0] >= 3 and src_ring.shape[1] >= 2:
                            src_xy = src_ring[:, :2]
                            pred_centroid = np.mean(ring_xy, axis=0)
                            src_centroid = np.mean(src_xy, axis=0)
                            shift_xy = src_centroid - pred_centroid
                            ring_xy = ring_xy + shift_xy
            except Exception as e:
                if logger is not None:
                    logger.warning(f"[pred.geojson] Could not anchor to {src_polygon_geojson}: {e}")

        uniq_xy = np.unique(np.round(ring_xy, decimals=8), axis=0)
        if len(uniq_xy) < 3:
            if logger is not None:
                logger.warning(f"[pred.geojson] Polygon became invalid after anchoring for {refined_obj_path}")
            return False

        area = abs(_polygon_signed_area_xy(ring_xy))
        if area <= 1e-12:
            if logger is not None:
                logger.warning(f"[pred.geojson] Zero-area polygon after anchoring for {refined_obj_path}")
            return False

        if not _is_simple_polygon_xy(ring_xy):
            if logger is not None:
                logger.warning(f"[pred.geojson] Self-intersection after anchoring for {refined_obj_path}")
            return False

        ring_xy_closed = ring_xy.tolist()
        ring_xy_closed.append(ring_xy_closed[0])

        feature = {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring_xy_closed]
            }
        }

        geojson = {
            "type": "FeatureCollection",
            "features": [feature]
        }

        if src_crs is not None:
            geojson["crs"] = src_crs

        with open(pred_geojson_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)

        return True

    except Exception as e:
        if logger is not None:
            logger.warning(f"[pred.geojson] Failed converting {refined_obj_path}: {e}")
        return False


def raster_polygon_to_geojson(
    poly_norm,
    mm_pt,
    output_path,
    src_polygon_geojson=None,
    epsg=29982,
    logger=None,
):
    """
    Write the raster prior polygon to a GeoJSON file in absolute projected
    coordinates (default EPSG:29982).

    Coordinate chain
    ----------------
    poly_norm  : (K, 2) float32  — raster polygon vertices in model-normalized
                 [0, 1] XY space, as stored in batch_dict['raster_poly'][i].
    mm_pt      : (2, 3) float32  — [[minX,minY,minZ],[maxX,maxY,maxZ]] from
                 batch_dict['minMaxPt'][i].  minPt holds the absolute origin of
                 this tile in the original CRS.
    Conversion : raw_xy = poly_norm * (maxPt - minPt)[:2] + minPt[:2]
                 This is exactly denorm_to_raw_space() restricted to XY and
                 matches the anchor logic used by refined_obj_to_geojson().

    Anchoring  : if src_polygon_geojson exists (the GT polygon.geojson for this
                 tile), the raster polygon centroid is shifted to match the GT
                 centroid, identical to the anchoring in refined_obj_to_geojson().
                 This corrects any residual georeferencing offset between the
                 model's local coordinate frame and the true map position.
                 If src_polygon_geojson is absent, raw_xy is used directly.

    CRS        : always written as EPSG:<epsg> regardless of the src GeoJSON's
                 own CRS declaration, because the raster prior works in the
                 same projected space as the input point cloud.

    Parameters
    ----------
    poly_norm          : (K, 2) numpy array  — raster polygon vertices
    mm_pt              : (2, 3) numpy array  — minPt / maxPt pair
    output_path        : str  — destination file (e.g. .../raster_polygon.geojson)
    src_polygon_geojson: str | None  — path to the GT polygon.geojson for anchoring
    epsg               : int  — EPSG code for the CRS declaration (default 29982)
    logger             : optional logger

    Returns
    -------
    bool — True on success, False on any error.
    """
    try:
        if poly_norm is None or len(poly_norm) < 3:
            if logger is not None:
                logger.warning(f"[raster.geojson] Skipped {output_path}: polygon is None or < 3 vertices")
            return False

        poly_norm = np.asarray(poly_norm, dtype=np.float64)   # (K, 2)
        mm_pt     = np.asarray(mm_pt,    dtype=np.float64)    # (2, 3)

        if mm_pt.shape != (2, 3):
            if logger is not None:
                logger.warning(f"[raster.geojson] Invalid mm_pt shape {mm_pt.shape} for {output_path}")
            return False

        # Convert normalized XY → absolute XY using the same formula as
        # denorm_to_raw_space():  raw = norm * (maxPt - minPt) + minPt
        minPt   = mm_pt[0]                       # (3,) absolute origin
        deltaPt = mm_pt[1] - mm_pt[0]            # (3,) isotropic scale

        # For an isotropic normalization (single scale for all axes) the
        # spatial scale is the same scalar in all three directions.
        # poly_norm is already in the 2D XY plane so we use deltaPt[0]
        # (which equals deltaPt[1] = deltaPt[2] by construction).
        raw_xy = poly_norm * deltaPt[0] + minPt[:2]   # (K, 2)

        # Validate before anchoring
        if len(raw_xy) < 3:
            if logger is not None:
                logger.warning(f"[raster.geojson] < 3 vertices after denorm for {output_path}")
            return False

        # ── Centroid anchor (same logic as refined_obj_to_geojson) ──────────
        # Shift the raster polygon centroid to match the GT polygon centroid so
        # any residual local-to-absolute offset is cancelled.
        if src_polygon_geojson is not None and os.path.exists(src_polygon_geojson):
            try:
                with open(src_polygon_geojson, "r", encoding="utf-8") as f:
                    src_gj = json.load(f)

                src_feats = src_gj.get("features", [])
                if len(src_feats) > 0:
                    src_geom = src_feats[0].get("geometry", {})
                    if src_geom.get("type", "") == "Polygon":
                        src_ring = src_geom.get("coordinates", [[]])[0]
                        src_ring = np.asarray(src_ring, dtype=np.float64)
                        if len(src_ring) >= 2 and np.allclose(src_ring[0], src_ring[-1]):
                            src_ring = src_ring[:-1]
                        if src_ring.ndim == 2 and src_ring.shape[0] >= 3 and src_ring.shape[1] >= 2:
                            raster_centroid = raw_xy.mean(axis=0)
                            src_centroid    = src_ring[:, :2].mean(axis=0)
                            raw_xy = raw_xy + (src_centroid - raster_centroid)
            except Exception as e:
                if logger is not None:
                    logger.warning(f"[raster.geojson] Could not anchor to {src_polygon_geojson}: {e}")

        # Validate after anchoring
        uniq = np.unique(np.round(raw_xy, 8), axis=0)
        if len(uniq) < 3:
            if logger is not None:
                logger.warning(f"[raster.geojson] < 3 unique XY vertices after anchoring for {output_path}")
            return False

        if abs(_polygon_signed_area_xy(raw_xy)) <= 1e-12:
            if logger is not None:
                logger.warning(f"[raster.geojson] Zero-area raster polygon for {output_path}")
            return False

        # Close the ring: GeoJSON spec requires first == last coordinate
        ring_closed = raw_xy.tolist()
        ring_closed.append(ring_closed[0])

        crs_member = {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}
        }

        geojson = {
            "type": "FeatureCollection",
            "crs": crs_member,
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "source": "raster_prior",
                        "n_vertices": len(poly_norm),
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [ring_closed],
                    },
                }
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)

        return True

    except Exception as e:
        if logger is not None:
            logger.warning(f"[raster.geojson] Failed writing {output_path}: {e}")
        return False

def denorm_to_points_n_space(pts_norm, mm_pt):
    """
    Convert model-normalized coordinates back to the local 'points_n' space.

    This returns LOCAL coordinates (relative to minPt), NOT absolute coordinates.
    The result is suitable for writing .obj / .xyz files for CloudCompare
    inspection, where all samples share the same local origin.

    For absolute (georeferenced) coordinates, use denorm_to_raw_space().

    [P3] clarified docstring — no math changed.
    """
    pts_norm = np.asarray(pts_norm, dtype=np.float32)
    mm_pt = np.asarray(mm_pt, dtype=np.float32)

    minPt = mm_pt[0]
    maxPt = mm_pt[1]
    deltaPt = maxPt - minPt

    return pts_norm * deltaPt


def denorm_to_raw_space(pts_norm, mm_pt):
    """
    Convert model-normalized coordinates back to original raw (absolute) coordinates.
    Adds minPt so the result is in the same frame as the original input data.
    """
    pts_norm = np.asarray(pts_norm, dtype=np.float32)
    mm_pt = np.asarray(mm_pt, dtype=np.float32)

    minPt = mm_pt[0]
    maxPt = mm_pt[1]
    deltaPt = maxPt - minPt

    return pts_norm * deltaPt + minPt


def build_fully_connected_edges(num_vertices):
    """
    Build all unordered edges of a complete graph K_n.
    """
    if num_vertices <= 1:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(list(itertools.combinations(range(num_vertices), 2)), dtype=np.int64)


# ============================================================
# Batch move helpers
# ============================================================

def load_data_to_gpu(batch_dict):
    """
    [P1] Unified with patched train_utils.py:
    - float arrays  → float32 CUDA tensor
    - integer arrays → int64 (long) CUDA tensor
    - bool arrays   → bool CUDA tensor  (was silently skipped in original)
    - CPU tensors   → moved to CUDA      (was ignored in original)
    - GPU tensors   → passed through unchanged
    - everything else (str, list, None) → unchanged
    """
    for key, val in batch_dict.items():
        if isinstance(val, torch.Tensor):
            if torch.cuda.is_available() and val.device.type != 'cuda':
                batch_dict[key] = val.cuda()
            continue
        if not isinstance(val, np.ndarray):
            continue
        if np.issubdtype(val.dtype, np.floating):
            batch_dict[key] = torch.from_numpy(val).float().cuda()
        elif np.issubdtype(val.dtype, np.integer):
            batch_dict[key] = torch.from_numpy(val).long().cuda()
        elif np.issubdtype(val.dtype, np.bool_):
            batch_dict[key] = torch.from_numpy(val).bool().cuda()
        # other dtypes (str arrays, object arrays) left as-is


def _to_cpu_np(x):
    """Recursively convert torch tensors inside nested structures to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, list):
        return [_to_cpu_np(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_cpu_np(v) for v in x)
    if isinstance(x, dict):
        return {k: _to_cpu_np(v) for k, v in x.items()}
    return x


def load_data_to_cpu(batch_dict):
    for key in list(batch_dict.keys()):
        batch_dict[key] = _to_cpu_np(batch_dict[key])
