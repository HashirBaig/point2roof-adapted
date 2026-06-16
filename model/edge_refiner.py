# edge_refiner.py
import math
import numpy as np
import torch

from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import polygonize, unary_union


# ─── Union-Find for component tracking ────────────────────────────────────────

class _UF:
    """Minimal union-find for premature-closure detection."""
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, x, y):
        self.p[self.find(x)] = self.find(y)
    def same(self, x, y):
        return self.find(x) == self.find(y)


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _angle_deg(a, b, c):
    ba = a - b; bc = c - b
    nba = np.linalg.norm(ba) + 1e-12
    nbc = np.linalg.norm(bc) + 1e-12
    cosang = float(np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0))
    return math.degrees(math.acos(cosang))


def _orthogonalize_polygon_xy(poly_xy, angle_tol_deg=15.0, max_iter=3):
    pts = np.asarray(poly_xy, dtype=np.float64)
    if pts.shape[0] < 4:
        return pts
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    N = pts.shape[0]
    if N < 3:
        return pts

    def unit(v):
        n = np.linalg.norm(v) + 1e-12
        return v / n

    for _ in range(max_iter):
        changed = False
        for i in range(N):
            p_prev = pts[(i - 1) % N]
            p = pts[i]
            p_next = pts[(i + 1) % N]
            ang = _angle_deg(p_prev, p, p_next)
            if min(abs(ang - 90.0), abs(ang - 270.0)) <= angle_tol_deg:
                v_in = unit(p - p_prev)
                perp1 = np.array([-v_in[1], v_in[0]])
                perp2 = -perp1
                v_out = unit(p_next - p)
                v_des = perp1 if np.dot(perp1, v_out) >= np.dot(perp2, v_out) else perp2
                w = p_next - p
                ortho = w - np.dot(w, v_des) * v_des
                new_p = p + ortho
                if not np.allclose(new_p, p, atol=1e-9):
                    pts[i] = new_p
                    changed = True
        if not changed:
            break
    return pts


def _polygon_is_valid_simple(poly_xy):
    try:
        if poly_xy.shape[0] < 3:
            return False
        pg = Polygon(poly_xy)
        return bool(pg.is_valid) and pg.area > 0
    except Exception:
        return False


def _extract_outline_from_edges(points_xy, edges, min_area=1e-6):
    if len(edges) < 3:
        return None, None
    lines = []
    for u, v in edges:
        p1, p2 = points_xy[u], points_xy[v]
        if np.linalg.norm(p1 - p2) < 1e-9:
            continue
        lines.append(LineString([tuple(p1), tuple(p2)]))
    if len(lines) < 3:
        return None, None
    merged = unary_union(MultiLineString(lines))
    polys = [p for p in polygonize(merged) if p.area > min_area]
    if not polys:
        return None, None
    union_geom = unary_union(polys)
    if union_geom.geom_type == 'MultiPolygon':
        union_geom = max(union_geom.geoms, key=lambda p: p.area)
    if union_geom.geom_type != 'Polygon':
        return None, None
    coords = np.asarray(union_geom.exterior.coords, dtype=np.float64)
    boundary = coords[:-1]
    if boundary.shape[0] < 3:
        return None, None
    K = boundary.shape[0]
    ring_edges = [(i, (i + 1) % K) for i in range(K)]
    return boundary, ring_edges


def _convex_hull_polygon(points_xy):
    """
    Fallback: return the convex hull of points_xy as an ordered ring.
    Works for any convex building footprint and gives a reasonable
    approximation for mildly concave ones.
    """
    try:
        from shapely.geometry import MultiPoint
        hull = MultiPoint(points_xy).convex_hull
        if hull.geom_type != 'Polygon':
            return None, None
        coords = np.asarray(hull.exterior.coords, dtype=np.float64)[:-1]
        if coords.shape[0] < 3:
            return None, None
        K = coords.shape[0]
        ring_edges = [(i, (i + 1) % K) for i in range(K)]
        return coords, ring_edges
    except Exception:
        return None, None


# ─── Graph closure ─────────────────────────────────────────────────────────────

def _close_graph(n, edges_set, score_lookup, points_xy):
    """
    Complete a partial edge graph into a single Hamiltonian cycle by greedily
    adding the minimum number of missing edges.
    """
    edges = set(edges_set)
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    uf = _UF(n)
    for u, v in edges:
        uf.union(u, v)

    def _is_premature_close(u, v, current_edges, current_deg):
        if not uf.same(u, v):
            return False
        root = uf.find(u)
        comp_size = sum(1 for i in range(n) if uf.find(i) == root)
        return comp_size < n

    def _candidate_score(u, v):
        key = (min(u, v), max(u, v))
        if key in score_lookup:
            return score_lookup[key]
        d = np.linalg.norm(points_xy[u] - points_xy[v]) + 1e-8
        return 1.0 / d

    max_iters = n * (n - 1) // 2
    for _ in range(max_iters):
        if all(d == 2 for d in deg):
            if all(uf.find(i) == uf.find(0) for i in range(n)):
                return edges, True

        candidates = []
        for u in range(n):
            if deg[u] >= 2:
                continue
            for v in range(u + 1, n):
                if deg[v] >= 2:
                    continue
                if (u, v) in edges:
                    continue
                if _is_premature_close(u, v, edges, deg):
                    continue
                score = _candidate_score(u, v)
                candidates.append((score, u, v))

        if not candidates:
            break

        candidates.sort(reverse=True)
        _, best_u, best_v = candidates[0]

        edges.add((best_u, best_v))
        deg[best_u] += 1
        deg[best_v] += 1
        uf.union(best_u, best_v)

    if all(d == 2 for d in deg) and all(uf.find(i) == uf.find(0) for i in range(n)):
        return edges, True
    return edges, False


# ─── Main extractor ───────────────────────────────────────────────────────────

class NonDiffRoofOutlineExtractor:
    """
    Non-differentiable post-processor that converts edge scores into a closed
    2D polygon outline.

    Produces:
      batch_dict['outline_vertices'] -> list of (K, 3) tensors
      batch_dict['outline_edges']    -> list of (E, 2) tensors
      batch_dict['outline_polygon_valid'] -> list of bool
    """

    def __init__(
        self,
        expected_feature_dim: int,
        edge_score_thresh: float = 0.5,
        angle_tol_deg: float = 15.0,
        orthogonalize_iters: int = 3,
        min_polygon_area: float = 1e-6,
    ):
        self.expected_feature_dim = int(expected_feature_dim)
        self.edge_score_thresh = float(edge_score_thresh)
        self.angle_tol_deg = float(angle_tol_deg)
        self.orthogonalize_iters = int(orthogonalize_iters)
        self.min_polygon_area = float(min_polygon_area)

    @torch.no_grad()
    def __call__(self, batch_dict):
        keypoint = batch_dict.get('keypoint')
        if keypoint is None:
            raise KeyError("batch_dict must contain 'keypoint'")

        device = keypoint.device
        B = int(batch_dict['batch_size'])

        empty_edges = torch.empty((0, 2), dtype=torch.long, device=device)
        empty_verts = torch.empty((0, 3), dtype=torch.float32, device=device)

        if keypoint.numel() == 0:
            batch_dict['outline_edges'] = [empty_edges for _ in range(B)]
            batch_dict['outline_vertices'] = [empty_verts for _ in range(B)]
            batch_dict['outline_polygon_valid'] = [False] * B
            return batch_dict

        batch_idx = keypoint[:, 0].long()

        # Prefer refined_keypoint (N,3) for XY; fall back to keypoint (N,4) with col-0=bidx
        refined = batch_dict.get('refined_keypoint', None)
        if refined is not None and refined.numel() > 0:
            kp_xy_all = refined[:, :2].detach().cpu().numpy()
            kp_z_all  = refined[:, 2].detach().cpu().numpy()
        else:
            kp_xy_all = keypoint[:, 1:3].detach().cpu().numpy()
            kp_z_all  = keypoint[:, 3].detach().cpu().numpy() if keypoint.shape[1] > 3 \
                        else np.zeros(keypoint.shape[0], dtype=np.float32)

        # [RASTER v5] per-sample flag: keypoints already equal the raster ring.
        raster_aligned = batch_dict.get('raster_aligned', None)

        pair_points = batch_dict.get('pair_points')
        edge_score  = batch_dict.get('edge_score')

        if pair_points is None or edge_score is None:
            batch_dict['outline_edges'] = [empty_edges for _ in range(B)]
            batch_dict['outline_vertices'] = [empty_verts for _ in range(B)]
            batch_dict['outline_polygon_valid'] = [False] * B
            return batch_dict

        pair_points_cpu = pair_points.detach().cpu().numpy() if pair_points.numel() > 0 \
            else np.zeros((0, 2), dtype=np.int64)
        edge_score_cpu  = edge_score.detach().cpu().numpy()  if edge_score.numel()  > 0 \
            else np.zeros((0,), dtype=np.float32)

        outline_edges_list   = []
        outline_vertices_list = []
        valid_list            = []

        global_ids_per_roof = []
        for i in range(B):
            mask = batch_idx == i
            gids = torch.nonzero(mask, as_tuple=False).view(-1)
            global_ids_per_roof.append(gids)

        cursor = 0

        for i in range(B):
            gids = global_ids_per_roof[i]
            n    = int(gids.numel())
            m_i  = n * (n - 1) // 2   # always advance cursor by this

            if n < 3:
                outline_edges_list.append(empty_edges)
                outline_vertices_list.append(empty_verts)
                valid_list.append(False)
                cursor += m_i
                continue

            roof_global = gids.detach().cpu().numpy()
            roof_xy     = kp_xy_all[roof_global]                  # (n, 2)
            roof_z      = kp_z_all[roof_global]                   # (n,)

            # ── [RASTER v5 — RING PASSTHROUGH] ─────────────────────────────
            # If this sample's keypoints were aligned to the raster vertex set,
            # they ARE the raster polygon in ring order. Emit the sequential
            # ring directly: skip closure + polygonize + orthogonalization,
            # which would only degrade an already-clean, regularized outline.
            is_aligned = (raster_aligned is not None
                          and i < len(raster_aligned)
                          and bool(raster_aligned[i]))
            if is_aligned:
                K = n
                ring_xyz = np.concatenate(
                    [roof_xy.astype(np.float32), roof_z.astype(np.float32).reshape(-1, 1)],
                    axis=1
                )
                ring_edges = [(j, (j + 1) % K) for j in range(K)]
                out_edges_t = torch.tensor(ring_edges, dtype=torch.long, device=device)
                out_verts_t = torch.tensor(ring_xyz,   dtype=torch.float32, device=device)
                outline_edges_list.append(out_edges_t)
                outline_vertices_list.append(out_verts_t)
                valid_list.append(True)
                cursor += m_i
                continue

            # ── Original path (raster disabled/invalid for this sample) ────
            if cursor + m_i > pair_points_cpu.shape[0]:
                outline_edges_list.append(empty_edges)
                outline_vertices_list.append(empty_verts)
                valid_list.append(False)
                cursor += m_i
                continue

            local_pairs  = pair_points_cpu[cursor:cursor + m_i]  # (m_i, 2) local indices
            local_scores = edge_score_cpu[cursor:cursor + m_i]    # (m_i,)
            cursor += m_i

            # ── Build score lookup for ALL pairs ───────────────────────────
            score_lookup = {}
            for pair_idx_row, score in zip(local_pairs, local_scores):
                u, v = int(pair_idx_row[0]), int(pair_idx_row[1])
                if u >= n or v >= n or u == v:
                    continue
                score_lookup[(min(u, v), max(u, v))] = float(score)

            # ── Initial edge set: pairs above threshold ────────────────────
            edges_above = set()
            for (u, v), sc in score_lookup.items():
                if sc >= self.edge_score_thresh:
                    edges_above.add((u, v))

            # ── Graph closure ──────────────────────────────────────────────
            completed_edges, closed = _close_graph(
                n, edges_above, score_lookup, roof_xy
            )

            # ── Attempt polygonize on completed graph ──────────────────────
            edge_list = list(completed_edges)
            out_poly_xy, ring_edges = _extract_outline_from_edges(
                roof_xy, edge_list, min_area=self.min_polygon_area
            )

            # ── Convex hull fallback ───────────────────────────────────────
            if out_poly_xy is None:
                out_poly_xy, ring_edges = _convex_hull_polygon(roof_xy)

            if out_poly_xy is None or ring_edges is None:
                outline_edges_list.append(empty_edges)
                outline_vertices_list.append(empty_verts)
                valid_list.append(False)
                continue

            # ── Orthogonalize (with validity rollback) ─────────────────────
            ortho_xy = _orthogonalize_polygon_xy(
                out_poly_xy,
                angle_tol_deg=self.angle_tol_deg,
                max_iter=self.orthogonalize_iters,
            )
            if _polygon_is_valid_simple(ortho_xy):
                out_poly_xy = ortho_xy

            K = out_poly_xy.shape[0]
            if K < 3:
                outline_edges_list.append(empty_edges)
                outline_vertices_list.append(empty_verts)
                valid_list.append(False)
                continue

            zeros    = np.zeros((K, 1), dtype=np.float32)
            ring_xyz = np.concatenate([out_poly_xy.astype(np.float32), zeros], axis=1)
            ring_edges = [(j, (j + 1) % K) for j in range(K)]

            out_edges_t = torch.tensor(ring_edges, dtype=torch.long, device=device)
            out_verts_t = torch.tensor(ring_xyz,   dtype=torch.float32, device=device)

            outline_edges_list.append(out_edges_t)
            outline_vertices_list.append(out_verts_t)
            valid_list.append(True)

        batch_dict['outline_edges']           = outline_edges_list
        batch_dict['outline_vertices']        = outline_vertices_list
        batch_dict['outline_polygon_valid']   = valid_list
        return batch_dict
