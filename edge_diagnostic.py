#!/usr/bin/env python3
"""
edge_diagnostic.py

GT-free per-sample diagnostic for the edge classification head. OBSERVABILITY
ONLY -- it does not gate or change inference output. The actual fallback runs in
postprocess_outlines.py when closure fails; this just lets you see in the log
which samples are likely to need it.

Signals reported per sample:
  n_keypoints           : how many keypoints fed the edge head
  n_pairs               : C(n,2) -- the fully-connected graph size
  n_accepted            : pairs with edge_score > edge_thresh
  degree_ok_frac        : fraction of keypoints with exactly 2 incident accepted edges
                          (== 1.0 means every vertex degree-2, the closed-ring condition)
  components            : number of connected components in the accepted-edge graph
                          (== 1 means topologically one component)
  likely_closes         : bool, predicts closure success: components==1 AND
                          degree_ok_frac >= CLOSURE_DEG_FRAC AND n_accepted >= n_keypoints
"""

import numpy as np


CLOSURE_DEG_FRAC = 0.9   # at least 90% of keypoints must already be degree-2 to predict closure


def _components(n, edges):
    """Count connected components over n nodes given edge list."""
    if n == 0:
        return 0
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for u, v in edges:
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[ru] = rv
    return len({find(i) for i in range(n)})


def diagnose_edge_closure(batch_dict, sample_i, edge_thresh):
    """
    Compute closure-likelihood signals for the edge head on sample `sample_i`.
    Uses only batch_dict keys already populated by the model forward pass.
    """
    keypoint = batch_dict.get('keypoint')
    if keypoint is None or keypoint.size == 0:
        return None

    kpt_batch_idx = keypoint[:, 0].astype(int)
    sample_mask = kpt_batch_idx == sample_i
    n = int(sample_mask.sum())

    out = {
        'n_keypoints': n,
        'n_pairs': 0,
        'n_accepted': 0,
        'degree_ok_frac': 0.0,
        'components': 0,
        'likely_closes': False,
    }
    if n < 3:
        return out

    n_pairs = n * (n - 1) // 2
    out['n_pairs'] = n_pairs

    # Cursor into the concatenated pair_points / edge_score arrays. Sum pairs for
    # samples 0..i-1 to locate this sample's slice.
    cursor = 0
    for j in range(sample_i):
        m = int((kpt_batch_idx == j).sum())
        cursor += m * (m - 1) // 2

    pair_pts = batch_dict.get('pair_points')
    edge_sc  = batch_dict.get('edge_score')
    if pair_pts is None or edge_sc is None:
        return out
    pair_pts = pair_pts[cursor:cursor + n_pairs].astype(int)
    edge_sc  = edge_sc[cursor:cursor + n_pairs]

    accepted = edge_sc > edge_thresh
    n_accepted = int(accepted.sum())
    out['n_accepted'] = n_accepted

    if n_accepted == 0:
        return out

    edges = pair_pts[accepted]
    deg = np.zeros(n, dtype=int)
    for u, v in edges:
        deg[u] += 1; deg[v] += 1
    out['degree_ok_frac'] = float((deg == 2).mean())
    out['components'] = _components(n, edges)
    out['likely_closes'] = bool(
        out['components'] == 1
        and out['degree_ok_frac'] >= CLOSURE_DEG_FRAC
        and n_accepted >= n   # need at least n edges for a closed ring
    )
    return out


def log_edge_closure(batch_dict, edge_thresh, logger):
    """Per-sample diagnostic line. Pure observability -- mutates nothing."""
    if logger is None:
        return
    B = int(batch_dict['batch_size'])
    for i in range(B):
        d = diagnose_edge_closure(batch_dict, i, edge_thresh)
        if d is None:
            continue
        logger.info(
            '[edge diag] sample %d kp=%d accepted=%d/%d deg2=%.0f%% comps=%d closes=%s'
            % (i, d['n_keypoints'], d['n_accepted'], d['n_pairs'],
               100 * d['degree_ok_frac'], d['components'], d['likely_closes'])
        )
