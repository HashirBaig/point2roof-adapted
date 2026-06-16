# ./utils/tile_process_helper.py
import os
import json
import itertools
import numpy as np
import torch
import tqdm

from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from utils.test_helper import (
    build_frame_geojson_lookup,
    copy_polygon_geojson,
    writePoints,
    write_obj_lines,
    write_obj_points,
    denorm_to_points_n_space,
    denorm_to_raw_space,
    load_data_to_gpu,
    load_data_to_cpu,
)


# ============================================================
# Basic geometry validation helpers
# ============================================================

def read_xyz_points(xyz_path):
    """
    Read full points_n.xyz from disk.

    Returns
    -------
    np.ndarray of shape (N, 3), dtype float32
    """
    pts = []
    with open(xyz_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            pts.append([float(parts[0]), float(parts[1]), float(parts[2])])

    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    return np.asarray(pts, dtype=np.float32)

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
    xy is open ring, no repeated closing point.
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
    Reconstruct one simple closed cycle from undirected edges.
    Returns ordered vertex indices or None if malformed.
    """
    if num_vertices < 3:
        return None

    adj = {i: [] for i in range(num_vertices)}

    for u, v in edges:
        u = int(u)
        v = int(v)

        if u == v:
            return None
        if u < 0 or v < 0 or u >= num_vertices or v >= num_vertices:
            return None

        adj[u].append(v)
        adj[v].append(u)

    used_vertices = sorted([k for k, nbrs in adj.items() if len(nbrs) > 0])
    if len(used_vertices) < 3:
        return None

    for k in used_vertices:
        # valid simple cycle requires degree 2
        if len(adj[k]) != 2:
            return None

    start = used_vertices[0]
    ordered = [start]
    prev = None
    cur = start

    while True:
        nbrs = adj[cur]
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


def validate_and_order_graph(vertices_xyz, edges, logger=None, tag="graph"):
    """
    Validate predicted graph and return ordered polygon vertices if valid.

    Returns
    -------
    (is_valid, ordered_ring_xyz, ordered_cycle_edges, reason)
    """
    vertices_xyz = np.asarray(vertices_xyz, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)

    if vertices_xyz.ndim != 2 or vertices_xyz.shape[0] < 3 or vertices_xyz.shape[1] < 3:
        return False, None, None, f"{tag}: invalid vertex array shape {vertices_xyz.shape}"

    if edges.size == 0:
        return False, None, None, f"{tag}: no edges"

    edges = edges.reshape(-1, 2)

    if np.any(edges < 0) or np.any(edges >= len(vertices_xyz)):
        return False, None, None, f"{tag}: edge index out of bounds"

    if np.any(edges[:, 0] == edges[:, 1]):
        return False, None, None, f"{tag}: self-loop edge detected"

    # remove duplicate undirected edges
    edges_set = sorted({tuple(sorted((int(u), int(v)))) for u, v in edges.tolist()})
    edges = np.asarray(edges_set, dtype=np.int64)

    ordered_idx = _order_cycle_from_edges(len(vertices_xyz), edges)
    if ordered_idx is None:
        return False, None, None, f"{tag}: graph is not a single simple cycle"

    ring_xyz = vertices_xyz[ordered_idx]
    ring_xy = ring_xyz[:, :2]

    uniq_xy = np.unique(np.round(ring_xy, decimals=8), axis=0)
    if len(uniq_xy) < 3:
        return False, None, None, f"{tag}: fewer than 3 unique XY vertices"

    area = abs(_polygon_signed_area_xy(ring_xy))
    if area <= 1e-12:
        return False, None, None, f"{tag}: zero-area polygon"

    if not _is_simple_polygon_xy(ring_xy):
        return False, None, None, f"{tag}: self-intersecting polygon"

    n = len(ring_xyz)
    cycle_edges = np.asarray([(i, (i + 1) % n) for i in range(n)], dtype=np.int64)

    return True, ring_xyz.astype(np.float32), cycle_edges, "ok"


# ============================================================
# Point-cloud split helpers
# ============================================================

def split_point_cloud_into_buildings_xy(points_xyz, split_radius=1.0, min_cluster_points=30):
    """
    Split point cloud into separate XY-connected components.

    Parameters
    ----------
    points_xyz : (N, 3)
        Parent point cloud in local metric/sample space.
    split_radius : float
        XY neighbor radius for connectivity.
    min_cluster_points : int
        Minimum number of points required to keep a building instance.

    Returns
    -------
    list[np.ndarray]
        List of index arrays, one per detected building component.
    """
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    n = points_xyz.shape[0]

    if n == 0:
        return []

    if n == 1:
        return [np.array([0], dtype=np.int64)]

    xy = points_xyz[:, :2]
    tree = cKDTree(xy)

    pairs = list(tree.query_pairs(r=float(split_radius)))
    if len(pairs) == 0:
        # If everything is isolated, keep whole cloud as one cluster
        if n >= min_cluster_points:
            return [np.arange(n, dtype=np.int64)]
        return []

    rows = []
    cols = []
    for i, j in pairs:
        rows.extend([i, j])
        cols.extend([j, i])

    data = np.ones(len(rows), dtype=np.uint8)
    graph = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    # include self-connections so isolated vertices are still represented
    graph = graph + coo_matrix(
        (np.ones(n, dtype=np.uint8), (np.arange(n), np.arange(n))),
        shape=(n, n)
    ).tocsr()

    n_comp, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    clusters = []
    for comp_id in range(n_comp):
        idx = np.where(labels == comp_id)[0]
        if idx.shape[0] >= min_cluster_points:
            clusters.append(idx.astype(np.int64))

    # fallback: if splitting produced nothing valid, treat entire parent as one instance
    if len(clusters) == 0 and n >= min_cluster_points:
        clusters = [np.arange(n, dtype=np.int64)]

    return clusters


def normalize_points_like_dataset(points_xyz):
    """
    Match roofn3d_dataset.py normalization logic:
    isotropic min/max cube normalization.
    """
    points_xyz = np.asarray(points_xyz, dtype=np.float32)

    min_pt = np.min(points_xyz, axis=0)
    max_pt = np.max(points_xyz, axis=0)

    maxXYZ = np.max(max_pt)
    minXYZ = np.min(min_pt)

    min_pt[:] = minXYZ
    max_pt[:] = maxXYZ

    delta = max_pt - min_pt
    delta[delta == 0] = 1.0

    points_norm = (points_xyz - min_pt) / delta
    mm_pt = np.stack([min_pt, max_pt], axis=0).astype(np.float32)

    return points_norm.astype(np.float32), mm_pt


def sample_points_to_npoint(points_xyz_norm, npoint):
    """
    Match dataset sampling style.
    """
    points_xyz_norm = np.asarray(points_xyz_norm, dtype=np.float32)
    n = points_xyz_norm.shape[0]

    if n == 0:
        return np.zeros((npoint, 3), dtype=np.float32)

    if n > npoint:
        idx = np.random.randint(0, n, npoint)
    else:
        extra = np.random.randint(0, n, npoint - n)
        idx = np.append(np.arange(0, n), extra)

    np.random.shuffle(idx)
    return points_xyz_norm[idx].astype(np.float32)


def build_single_instance_inference_batch(points_xyz_local, npoint, frame_id):
    """
    Build an inference-only batch dict for one split building instance.
    """
    points_norm, mm_pt = normalize_points_like_dataset(points_xyz_local)
    points_norm = sample_points_to_npoint(points_norm, npoint)

    batch = {
        "points": points_norm[None, ...].astype(np.float32),
        "minMaxPt": mm_pt[None, ...].astype(np.float32),
        "frame_id": [str(frame_id)],
        "batch_size": 1,
    }
    return batch, mm_pt


# ============================================================
# Model output extraction helpers
# ============================================================

def extract_single_instance_graph_from_model_output(batch_out, mm_pt_sub, thr=0.5, refined_z_mode="keep"):
    """
    Extract one predicted graph from a single-instance Point2Roof inference output.

    Returns
    -------
    (vertices_local_xyz, edges, reason)
    """
    # Preferred path: use outline extractor output
    outline_vertices = batch_out.get("outline_vertices", None)
    outline_edges = batch_out.get("outline_edges", None)

    if outline_vertices is not None and len(outline_vertices) > 0:
        V_norm = np.asarray(outline_vertices[0], dtype=np.float32)
        if V_norm.size == 0:
            return None, None, "empty outline_vertices"

        if outline_edges is not None and len(outline_edges) > 0 and outline_edges[0] is not None:
            E = np.asarray(outline_edges[0], dtype=np.int64)
            if E.size == 0:
                return None, None, "empty outline_edges"
            E = E.reshape(-1, 2)
        else:
            return None, None, "outline_edges missing"

        V_local = denorm_to_raw_space(V_norm, mm_pt_sub)

        if refined_z_mode == "mean_pred" and V_local.shape[0] > 0:
            V_local[:, 2] = np.mean(V_local[:, 2])

        return V_local.astype(np.float32), E.astype(np.int64), "ok"

    # Fallback path: use refined keypoints + thresholded pair predictions
    keypoint = np.asarray(batch_out.get("keypoint", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32)
    refined = np.asarray(batch_out.get("refined_keypoint", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    pair_points = np.asarray(batch_out.get("pair_points", np.zeros((0, 2), dtype=np.int64)), dtype=np.int64)
    edge_score = np.asarray(batch_out.get("edge_score", np.zeros((0,), dtype=np.float32)), dtype=np.float32)

    if keypoint.shape[0] == 0 or refined.shape[0] == 0:
        return None, None, "no keypoint/refined_keypoint fallback data"

    mask = keypoint[:, 0].astype(np.int64) == 0
    V_norm = refined[mask]
    if V_norm.shape[0] < 3:
        return None, None, "fallback has < 3 vertices"

    keep = edge_score >= thr
    E = pair_points[keep]
    if E.shape[0] < 3:
        return None, None, "fallback has < 3 edges"

    V_local = denorm_to_raw_space(V_norm, mm_pt_sub)

    if refined_z_mode == "mean_pred" and V_local.shape[0] > 0:
        V_local[:, 2] = np.mean(V_local[:, 2])

    return V_local.astype(np.float32), E.astype(np.int64), "ok"


# ============================================================
# Merge / export helpers
# ============================================================

def merge_ordered_rings_to_graph(polygons_xyz_list):
    """
    Merge multiple polygon rings into one disconnected graph.
    Each polygon ring is expected to be open (no repeated closing vertex).
    """
    merged_vertices = []
    merged_edges = []

    v_offset = 0
    for ring_xyz in polygons_xyz_list:
        ring_xyz = np.asarray(ring_xyz, dtype=np.float32)
        n = ring_xyz.shape[0]
        if n < 3:
            continue

        merged_vertices.append(ring_xyz)
        merged_edges.extend([(v_offset + i, v_offset + ((i + 1) % n)) for i in range(n)])
        v_offset += n

    if len(merged_vertices) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.int64)
        )

    return (
        np.concatenate(merged_vertices, axis=0).astype(np.float32),
        np.asarray(merged_edges, dtype=np.int64)
    )


def write_empty_featurecollection_geojson(path, src_polygon_geojson=None):
    src_crs = None
    if src_polygon_geojson is not None and os.path.exists(src_polygon_geojson):
        try:
            with open(src_polygon_geojson, "r", encoding="utf-8") as f:
                src_gj = json.load(f)
            if isinstance(src_gj, dict) and "crs" in src_gj:
                src_crs = src_gj["crs"]
        except Exception:
            pass

    geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    if src_crs is not None:
        geojson["crs"] = src_crs

    with open(path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)


def write_multipolygon_geojson(polygons_xyz_list, output_path, src_polygon_geojson=None, logger=None):
    """
    Write merged predictions as a single MultiPolygon GeoJSON.
    """
    polygons_xyz_list = [np.asarray(p, dtype=np.float64) for p in polygons_xyz_list if len(p) >= 3]

    if len(polygons_xyz_list) == 0:
        write_empty_featurecollection_geojson(output_path, src_polygon_geojson=src_polygon_geojson)
        return False

    coords = []
    for ring_xyz in polygons_xyz_list:
        ring_xy = ring_xyz[:, :2].tolist()
        ring_xy.append(ring_xy[0])  # close ring
        coords.append([ring_xy])

    src_crs = None
    if src_polygon_geojson is not None and os.path.exists(src_polygon_geojson):
        try:
            with open(src_polygon_geojson, "r", encoding="utf-8") as f:
                src_gj = json.load(f)
            if isinstance(src_gj, dict) and "crs" in src_gj:
                src_crs = src_gj["crs"]
        except Exception as e:
            if logger is not None:
                logger.warning(f"[tile] Could not read CRS from {src_polygon_geojson}: {e}")

    feature = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": coords
        }
    }

    geojson = {
        "type": "FeatureCollection",
        "features": [feature]
    }
    if src_crs is not None:
        geojson["crs"] = src_crs

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    return True


# ============================================================
# Main tile-mode inference per sample
# ============================================================

def run_tile_inference_for_sample(
    model,
    parent_points_local,
    frame_id,
    sample_dir,
    src_polygon_geojson=None,
    thr=0.5,
    refined_z_mode="keep",
    split_radius=1.0,
    min_cluster_points=30,
    logger=None,
):
    """
    Split one parent point cloud into per-building subinstances,
    run Point2Roof on each, validate predicted graphs,
    merge valid predictions, and export MultiPolygon GeoJSON.
    """
    parent_points_local = np.asarray(parent_points_local, dtype=np.float32)

    # export the full original parent point cloud loaded from disk
    writePoints(parent_points_local, os.path.join(sample_dir, "points_n.xyz"))

    # split parent cloud in XY
    cluster_indices = split_point_cloud_into_buildings_xy(
        parent_points_local,
        split_radius=split_radius,
        min_cluster_points=min_cluster_points
    )

    if logger is not None:
        logger.info(f"[tile] frame={frame_id} | detected_subinstances={len(cluster_indices)}")

    # Point2Roof still expects the trained fixed input size
    # Use the model's training-time NPOINT, not the full parent size.
    npoint = 1024

    valid_polygons_xyz = []
    tile_debug_dir = os.path.join(sample_dir, "tiles")
    os.makedirs(tile_debug_dir, exist_ok=True)

    for tile_id, idx in enumerate(cluster_indices):
        sub_points_local = parent_points_local[idx]
        tile_name = f"{tile_id:03d}"
        tile_dir = os.path.join(tile_debug_dir, tile_name)
        os.makedirs(tile_dir, exist_ok=True)

        # --------------------------------------------------------
        # Save the split point-cloud instance first
        # --------------------------------------------------------
        write_obj_points(
            os.path.join(tile_dir, "split_points.obj"),
            sub_points_local
        )

        writePoints(
            sub_points_local,
            os.path.join(tile_dir, "split_points.xyz")
        )

        if sub_points_local.shape[0] < 3:
            if logger is not None:
                logger.warning(f"[tile] frame={frame_id} tile={tile_id}: skipped, < 3 points")
            continue

        # build single-instance inference batch
        sub_batch, mm_pt_sub = build_single_instance_inference_batch(
            sub_points_local,
            npoint=npoint,
            frame_id=f"{frame_id}_tile_{tile_id:03d}"
        )

        # optional: also save normalized sampled points actually fed to the model
        sub_points_norm_for_model = np.asarray(sub_batch["points"][0], dtype=np.float32)
        sub_points_local_for_model = denorm_to_raw_space(sub_points_norm_for_model, mm_pt_sub)

        write_obj_points(
            os.path.join(tile_dir, "model_input_points.obj"),
            sub_points_local_for_model
        )

        writePoints(
            sub_points_local_for_model,
            os.path.join(tile_dir, "model_input_points.xyz")
        )

        # inference
        load_data_to_gpu(sub_batch)
        with torch.no_grad():
            sub_batch = model(sub_batch)
        load_data_to_cpu(sub_batch)

        # --------------------------------------------------------
        # Save intermediate outputs when available
        # --------------------------------------------------------
        keypoint = np.asarray(sub_batch.get("keypoint", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32)
        refined_keypoint = np.asarray(sub_batch.get("refined_keypoint", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)

        if keypoint.shape[0] > 0:
            key_mask = keypoint[:, 0].astype(np.int64) == 0
            key_xyz_norm = keypoint[key_mask][:, 1:4]
            if key_xyz_norm.shape[0] > 0:
                key_xyz_local = denorm_to_raw_space(key_xyz_norm, mm_pt_sub)
                write_obj_points(
                    os.path.join(tile_dir, "centroid_points.obj"),
                    key_xyz_local
                )

        if refined_keypoint.shape[0] > 0:
            ref_xyz_local = denorm_to_raw_space(refined_keypoint, mm_pt_sub)
            write_obj_points(
                os.path.join(tile_dir, "refined_keypoints.obj"),
                ref_xyz_local
            )

        pair_points = np.asarray(sub_batch.get("pair_points", np.zeros((0, 2), dtype=np.int64)), dtype=np.int64)
        edge_score = np.asarray(sub_batch.get("edge_score", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        if refined_keypoint.shape[0] > 0 and pair_points.shape[0] > 0 and edge_score.shape[0] == pair_points.shape[0]:
            keep = edge_score >= thr
            pred_edges = pair_points[keep]
            if pred_edges.shape[0] > 0:
                ref_xyz_local = denorm_to_raw_space(refined_keypoint, mm_pt_sub)
                write_obj_lines(
                    os.path.join(tile_dir, "pred_graph.obj"),
                    ref_xyz_local,
                    pred_edges
                )

        # extract graph
        verts_local, edges, reason = extract_single_instance_graph_from_model_output(
            sub_batch,
            mm_pt_sub=mm_pt_sub,
            thr=thr,
            refined_z_mode=refined_z_mode
        )
        if verts_local is None or edges is None:
            if logger is not None:
                logger.warning(f"[tile] frame={frame_id} tile={tile_id}: invalid graph extract | {reason}")
            continue

        # validate graph before polygon conversion
        ok, ring_xyz, cycle_edges, reason = validate_and_order_graph(
            verts_local,
            edges,
            logger=logger,
            tag=f"{frame_id}_tile_{tile_id:03d}"
        )
        if not ok:
            if logger is not None:
                logger.warning(f"[tile] frame={frame_id} tile={tile_id}: skipped malformed graph | {reason}")
            continue

        valid_polygons_xyz.append(ring_xyz)

        write_obj_lines(
            os.path.join(tile_dir, "refined.obj"),
            ring_xyz,
            cycle_edges
        )

    # merge all valid predictions into one disconnected graph
    merged_vertices, merged_edges = merge_ordered_rings_to_graph(valid_polygons_xyz)

    # export merged graph OBJ
    write_obj_lines(
        os.path.join(sample_dir, "refined.obj"),
        merged_vertices,
        merged_edges
    )

    # export multipolygon geojson
    pred_geojson_path = os.path.join(sample_dir, "pred.geojson")
    write_multipolygon_geojson(
        valid_polygons_xyz,
        pred_geojson_path,
        src_polygon_geojson=src_polygon_geojson,
        logger=logger
    )

    if logger is not None:
        logger.info(
            f"[tile] frame={frame_id} | valid_polygons={len(valid_polygons_xyz)} | "
            f"merged_vertices={len(merged_vertices)} | merged_edges={len(merged_edges)}"
        )

    return {
        "num_detected_subinstances": len(cluster_indices),
        "num_valid_polygons": len(valid_polygons_xyz),
        "merged_vertices": merged_vertices,
        "merged_edges": merged_edges,
    }


# ============================================================
# Tile-mode test loop
# ============================================================

def test_model_tile(
    model,
    data_loader,
    logger,
    test_tag,
    thr=0.5,
    refined_z_mode="keep",
    data_root="data",
    test_list_path="./test_data_easy/test.txt",
    test_base_dir="./test_data_easy",
    split_radius=1.0,
    min_cluster_points=30,
):
    """
    Tile-mode inference:
      parent points_n.xyz
      -> split into per-building subinstances
      -> run Point2Roof on each
      -> validate predicted graphs
      -> merge valid predictions
      -> export one MultiPolygon pred.geojson
    """
    model.use_edge = True
    model.eval()

    frame_geojson_lookup = build_frame_geojson_lookup(
        file_path=test_list_path,
        base_dir=test_base_dir
    )
    if logger is not None:
        logger.info(f"[tile] Loaded {len(frame_geojson_lookup)} GeoJSON paths from {test_list_path}")

    out_dir = os.path.join("output", test_tag, "test", "preds")
    os.makedirs(out_dir, exist_ok=True)

    dataloader_iter = iter(data_loader)

    total_samples = 0
    total_valid_polygons = 0
    total_detected_subinstances = 0

    with tqdm.trange(0, len(data_loader), desc="test_tile", dynamic_ncols=True) as tbar:
        for _ in tbar:
            batch = next(dataloader_iter)

            batch_size = int(batch["batch_size"])

            frame_id = batch.get("frame_id", None)
            if isinstance(frame_id, (list, tuple)):
                frame_ids = list(frame_id)
            elif isinstance(frame_id, np.ndarray) and frame_id.ndim > 0:
                frame_ids = [str(x) for x in frame_id.tolist()]
            else:
                frame_ids = [frame_id] * batch_size

            polygon_geojson_paths = batch.get("polygon_geojson_path", None)
            if isinstance(polygon_geojson_paths, (list, tuple)):
                polygon_geojson_paths = list(polygon_geojson_paths)
            elif isinstance(polygon_geojson_paths, np.ndarray) and polygon_geojson_paths.ndim > 0:
                polygon_geojson_paths = [str(x) for x in polygon_geojson_paths.tolist()]
            else:
                polygon_geojson_paths = [None] * batch_size

            # In tile mode, do NOT split the sampled batch points.
            # Read the full original parent points_n.xyz from disk instead.
            # points_all = np.asarray(batch["points"], dtype=np.float32)
            # mm_all = np.asarray(batch["minMaxPt"], dtype=np.float32)

            for i in range(batch_size):
                fid = str(frame_ids[i])
                sample_dir = os.path.join(out_dir, fid)
                os.makedirs(sample_dir, exist_ok=True)

                src_geojson_i = None
                if polygon_geojson_paths is not None and len(polygon_geojson_paths) > i:
                    src_geojson_i = polygon_geojson_paths[i]

                copied_ok, used_src_geojson = copy_polygon_geojson(
                    sample_dir=sample_dir,
                    frame_id=fid,
                    src_geojson_path=src_geojson_i,
                    frame_geojson_lookup=frame_geojson_lookup,
                    fallback_data_root=data_root
                )

                dst_polygon_geojson = os.path.join(sample_dir, "polygon.geojson")
                if copied_ok:
                    tqdm.tqdm.write(f"[tile][polygon.geojson] Copied {used_src_geojson} -> {dst_polygon_geojson}")
                else:
                    tqdm.tqdm.write(f"[tile][polygon.geojson] Could not find source GeoJSON for sample {fid}")

                src_polygon_geojson = dst_polygon_geojson if os.path.exists(dst_polygon_geojson) else None

                parent_points_path = os.path.join(data_root, fid, "points_n.xyz")
                if not os.path.exists(parent_points_path):
                    raise FileNotFoundError(
                        f"[tile] Missing full parent points_n.xyz for sample {fid}: {parent_points_path}"
                    )

                parent_points_local = read_xyz_points(parent_points_path)

                result = run_tile_inference_for_sample(
                    model=model,
                    parent_points_local=parent_points_local,
                    frame_id=fid,
                    sample_dir=sample_dir,
                    src_polygon_geojson=src_polygon_geojson,
                    thr=thr,
                    refined_z_mode=refined_z_mode,
                    split_radius=split_radius,
                    min_cluster_points=min_cluster_points,
                    logger=logger,
                )

                total_samples += 1
                total_detected_subinstances += result["num_detected_subinstances"]
                total_valid_polygons += result["num_valid_polygons"]

    if logger is not None:
        logger.info("**********************Tile testing done**********************")
        logger.info(f"[tile] total_samples: {total_samples}")
        logger.info(f"[tile] total_detected_subinstances: {total_detected_subinstances}")
        logger.info(f"[tile] total_valid_polygons: {total_valid_polygons}")