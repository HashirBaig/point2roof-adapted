import os
import itertools
import numpy as np


# ----------------------------------------------------------------------------- #
# Small writers (CloudCompare-friendly)
# ----------------------------------------------------------------------------- #
def _write_xyz(path, pts, scalars=None):
    """Write an .xyz cloud. Optional per-point scalar appended as extra column(s)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if scalars is not None:
        scalars = np.asarray(scalars, dtype=np.float64).reshape(len(pts), -1)
        data = np.concatenate([pts, scalars], axis=1)
    else:
        data = pts
    with open(path, 'w') as f:
        for row in data:
            f.write(' '.join(repr(float(v)) for v in row) + '\n')


def _write_obj_points(path, pts):
    """Write a vertex-only OBJ. CloudCompare imports this as a point cloud."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    with open(path, 'w') as f:
        for p in pts:
            f.write('v %r %r %r\n' % (float(p[0]), float(p[1]), float(p[2])))


def _write_obj_graph(path, pts, edges):
    """Write an OBJ with vertices and 'l' line elements. edges: (E,2) 0-based."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    with open(path, 'w') as f:
        for p in pts:
            f.write('v %r %r %r\n' % (float(p[0]), float(p[1]), float(p[2])))
        for e in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
            # OBJ is 1-based
            f.write('l %d %d\n' % (int(e[0]) + 1, int(e[1]) + 1))


def _denorm(pts, min_pt, max_pt):
    """Map normalized coords back to world coords. min/max broadcast over xyz."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    delta = (np.asarray(max_pt, dtype=np.float64) - np.asarray(min_pt, dtype=np.float64))
    return pts * delta + np.asarray(min_pt, dtype=np.float64)


# ----------------------------------------------------------------------------- #
# Main per-batch export
# ----------------------------------------------------------------------------- #
def export_batch(batch, out_root, counter, score_thresh, edge_thresh=0.5):
    """
    Export per-sample inference artifacts as CloudCompare-friendly files.

    Must be called AFTER load_data_to_cpu(batch) (all tensors are numpy here).
    `model.use_edge` must be True so the cluster-refine and edge heads ran.

    Parameters
    ----------
    batch        : dict, the model output batch (numpy arrays)
    out_root     : str, directory under which /000001/, /000002/, ... are created
    counter      : int, running sample index BEFORE this batch (e.g. 0 at start)
    score_thresh : float, ClusterRefineNet.ScoreThresh (same value used in forward)
    edge_thresh  : float, threshold on edge_score for the predicted edge set

    Returns
    -------
    counter : int, updated running sample index after this batch
    """
    batch_size = batch['batch_size']

    points        = batch['points']               # (B, N, 3) normalized
    pred_score    = batch['point_pred_score']      # (B, N)
    pred_offset   = batch['point_pred_offset']     # (B, N, 3) already * PosRadius
    keypoint      = batch['keypoint']              # (sum M, 4): [batch_idx, x, y, z]
    refined_kpt   = batch['refined_keypoint']      # (sum M, 3)
    pair_points   = batch['pair_points']           # (sum P, 2) per-sample local idx
    edge_score    = batch['edge_score']            # (sum P,)
    mm_pts        = batch['minMaxPt']              # (B, 2, 3)
    frame_ids     = batch['frame_id']              # list of str

    kpt_batch_idx = keypoint[:, 0].astype(np.int64)

    edge_cursor = 0   # cursor into pair_points / edge_score (concatenated per sample)

    for i in range(batch_size):
        counter += 1
        folder = os.path.join(out_root, '%06d' % counter)
        os.makedirs(folder, exist_ok=True)

        min_pt = mm_pts[i][0]
        max_pt = mm_pts[i][1]

        # --- input cloud ----------------------------------------------------- #
        pts_n = points[i]                          # normalized
        pts_w = _denorm(pts_n, min_pt, max_pt)     # world
        _write_xyz(os.path.join(folder, 'points_n.xyz'), pts_n)
        _write_xyz(os.path.join(folder, 'points_world.xyz'), pts_w)

        # --- ground truth (vectors + edges, rebuilt in world coords) --------- #
        gt_v = batch['vectors'][i]
        gt_v = gt_v[np.sum(gt_v, axis=-1) > -2e1]
        gt_v_w = _denorm(gt_v, min_pt, max_pt)
        gt_e = batch['edges'][i]
        gt_e = gt_e[np.sum(gt_e, axis=-1) > 0]
        _write_obj_graph(os.path.join(folder, 'gt.obj'), gt_v_w, gt_e)

        # --- 1. binary classification head ----------------------------------- #
        score_i = pred_score[i]                    # (N,)
        pos_mask = score_i > score_thresh
        # all points + score as scalar field
        _write_xyz(os.path.join(folder, 'cls_score_all.xyz'), pts_w, score_i)
        # positive set only (what feeds clustering)
        _write_obj_points(os.path.join(folder, 'cls_positive.obj'), pts_w[pos_mask])

        # --- 2. offset regression head --------------------------------------- #
        # shifted seed positions (point + offset) for positive points
        shifted_n = pts_n.copy()
        shifted_n[pos_mask] += pred_offset[i][pos_mask]
        shifted_w = _denorm(shifted_n, min_pt, max_pt)
        _write_obj_points(os.path.join(folder, 'offset_seeds.obj'), shifted_w[pos_mask])
        # raw offset vectors as scalar fields on all points (dx dy dz, normalized)
        _write_xyz(os.path.join(folder, 'offset_vectors.xyz'), pts_w, pred_offset[i])

        # --- 3. DBSCAN centroids --------------------------------------------- #
        kmask = kpt_batch_idx == i
        kpt_n = keypoint[kmask][:, 1:4]            # pre-refine centroids
        ref_n = refined_kpt[kmask]                 # refined centroids
        kpt_w = _denorm(kpt_n, min_pt, max_pt)
        ref_w = _denorm(ref_n, min_pt, max_pt)
        _write_obj_points(os.path.join(folder, 'dbscan_centroids.obj'), kpt_w)
        _write_obj_points(os.path.join(folder, 'refined_keypoints.obj'), ref_w)

        n_kpt = kpt_w.shape[0]

        # --- 4 & 5. fully-connected graph + edge classification -------------- #
        # number of pairs for this sample = C(n_kpt, 2)
        n_pairs = n_kpt * (n_kpt - 1) // 2
        if n_pairs > 0 and n_kpt > 1:
            pairs_i = pair_points[edge_cursor:edge_cursor + n_pairs].astype(np.int64)
            score_e = edge_score[edge_cursor:edge_cursor + n_pairs]
            edge_cursor += n_pairs

            # graph drawn on refined keypoints (local indices already 0-based)
            _write_obj_graph(os.path.join(folder, 'graph_full.obj'), ref_w, pairs_i)
            keep = score_e > edge_thresh
            _write_obj_graph(os.path.join(folder, 'edges_pred.obj'), ref_w, pairs_i[keep])
            # edge scores as a sidecar (v1 v2 score, 1-based to match OBJ)
            with open(os.path.join(folder, 'edge_scores.txt'), 'w') as f:
                for (a, b), s in zip(pairs_i, score_e):
                    f.write('%d %d %r\n' % (int(a) + 1, int(b) + 1, float(s)))
        else:
            # still emit empty graph files for consistency
            _write_obj_graph(os.path.join(folder, 'graph_full.obj'), ref_w, np.zeros((0, 2)))
            _write_obj_graph(os.path.join(folder, 'edges_pred.obj'), ref_w, np.zeros((0, 2)))

    return counter
