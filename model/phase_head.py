# phase_head.py
"""
PhaseHead -- a drop-in replacement for EdgeAttentionNet that predicts each
refined keypoint's position along the ring as a (cos phi, sin phi) unit vector.
At inference, edges are obtained by sorting keypoints by phi and connecting
consecutive ones, closing the loop -- so no per-pair edge classification.

Interface preserved with EdgeAttentionNet so roofnet.py needs only the toggle:
  forward(batch_dict) reads:    'keypoint' (M_total, 4)  [batch_idx, x, y, z]
                                'keypoint_features' (M_total, C_in)
                                'matches' (training only)
                                'vectors' (training only -- to look up GT order)
                                'batch_size'
  forward(batch_dict) writes:   'pair_points' (P_total, 2) the predicted ring's
                                                 edges as (i,j) local indices
                                'edge_score'  (P_total,) all 1.0 -- existing
                                                 downstream consumers threshold
                                                 at 0.5, so 1.0 keeps every
                                                 predicted ring edge.

Why edge_score = 1.0 for ring edges (and we emit only ring edges, not all pairs):
The original EdgeAttentionNet emitted a score for every C(n,2) pair so eval_process
and the post-processor could threshold. Phase-based prediction is deterministic --
ring edges are sorted by phi, and we emit exactly n of them per sample. By setting
their scores to 1.0 every downstream consumer that thresholds at 0.5 keeps them
all. Non-ring pairs aren't emitted at all (length 0 from their perspective).
This is a clean interface match while changing the prediction semantics.

Supervision:
  Hungarian matches tell us which predicted vertex maps to which GT vertex.
  GT vertices are canonicalized to a fixed traversal (counterclockwise from the
  XY-lex smallest vertex), so each GT vertex has a deterministic ring position
  in [0, n). We convert that to a target angle phi_target = 2*pi*pos/n and
  supervise the predicted (cos, sin) with a cosine-similarity loss.
"""

import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import loss_utils


HIDDEN_DIM = 128


# ---------------------------------------------------------------------------- #
# GT canonicalization: rotate and reflect to a canonical traversal
# ---------------------------------------------------------------------------- #
def _canonicalize_ring(verts_xy):
    """
    verts_xy: (n, 2) numpy or torch tensor of GT vertex positions in ring order.
    Returns the permutation [start, start+1, ..., start-1] (possibly reversed)
    that canonicalises to:
      - start = vertex with the smallest (x, y) lex
      - direction = counterclockwise (signed area positive)
    """
    import numpy as np
    v = verts_xy.detach().cpu().numpy() if isinstance(verts_xy, torch.Tensor) else verts_xy
    n = v.shape[0]
    if n < 3:
        return list(range(n))

    # Direction: signed area; positive = CCW. If negative, reverse.
    s = 0.0
    for i in range(n):
        x1, y1 = v[i]
        x2, y2 = v[(i + 1) % n]
        s += (x2 - x1) * (y2 + y1)
    ccw_order = list(range(n))
    if s > 0:           # shoelace negative for CCW (because y grows down); flip
        ccw_order = list(reversed(ccw_order))
        v = v[ccw_order]

    # Starting vertex: lex-smallest (x, y) in the now-CCW sequence
    lex = sorted(range(n), key=lambda i: (v[i, 0], v[i, 1]))
    start = lex[0]
    perm = ccw_order[start:] + ccw_order[:start]
    return perm


# ---------------------------------------------------------------------------- #
# Phase head
# ---------------------------------------------------------------------------- #
class PhaseHead(nn.Module):
    def __init__(self, model_cfg, input_channel):
        super().__init__()
        self.model_cfg = model_cfg

        self.in_proj = nn.Sequential(
            nn.Linear(input_channel, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
        )
        # Outputs 2D (cos, sin); we L2-normalise at use-time so it stays on the
        # unit circle even before the network has learned to put it there.
        self.phase_fc = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 2),
        )

        self.train_dict = {}
        # loss_weight key reuses 'cls_weight' so existing model_cfg.yaml works
        self.loss_weight = self.model_cfg.LossWeight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------------------------------------------------------------- #
    # forward
    # ---------------------------------------------------------------- #
    def forward(self, batch_dict):
        if self.training:
            self.train_dict = {}
        keypoint = batch_dict['keypoint']
        point_fea = batch_dict['keypoint_features']
        B = int(batch_dict['batch_size'])
        batch_idx = keypoint[:, 0].long()

        # Predict (cos, sin) per refined keypoint
        h = self.in_proj(point_fea)
        cs = self.phase_fc(h)                           # (M_total, 2)
        cs = F.normalize(cs, p=2, dim=-1)               # project to unit circle
        # phi in [-pi, pi], convert to [0, 1) for sort convenience
        phi = (torch.atan2(cs[:, 1], cs[:, 0]) / (2 * math.pi) + 0.5) % 1.0

        pair_points_list = []
        edge_score_list = []
        pred_phi_per_sample = []
        for i in range(B):
            mask = batch_idx == i
            n = int(mask.sum())
            if n < 3:
                pred_phi_per_sample.append((None, None))
                continue
            local_phi = phi[mask]                       # (n,)
            sort_idx = torch.argsort(local_phi)         # (n,)
            # Ring edges: consecutive in sorted order, plus closing edge
            ring_set = set()
            for k in range(n):
                a, b = int(sort_idx[k].item()), int(sort_idx[(k + 1) % n].item())
                ring_set.add((min(a, b), max(a, b)))

            # Emit ALL C(n,2) pairs in itertools.combinations order so
            # eval_process and downstream consumers' cursor arithmetic works.
            # Ring edges get score 1.0; everything else 0.0. The threshold
            # at 0.5 in eval_process will keep exactly the ring edges.
            all_pairs = list(itertools.combinations(range(n), 2))
            pair_tensor = torch.tensor(all_pairs, dtype=torch.float, device=phi.device)
            scores = torch.tensor(
                [1.0 if (a, b) in ring_set else 0.0 for a, b in all_pairs],
                dtype=torch.float, device=phi.device,
            )
            pair_points_list.append(pair_tensor)
            edge_score_list.append(scores)
            pred_phi_per_sample.append((local_phi, cs[mask]))

        if pair_points_list:
            batch_dict['pair_points'] = torch.cat(pair_points_list, dim=0)
            batch_dict['edge_score'] = torch.cat(edge_score_list, dim=0)
        else:
            batch_dict['pair_points'] = point_fea.new_zeros((0, 2))
            batch_dict['edge_score']  = point_fea.new_zeros((0,))

        if self.training:
            self._stash_for_loss(batch_dict, pred_phi_per_sample)
        return batch_dict

    # ---------------------------------------------------------------- #
    # Build per-sample targets from matches and canonicalised GT order
    # ---------------------------------------------------------------- #
    def _stash_for_loss(self, batch_dict, pred_phi_per_sample):
        matches = batch_dict.get('matches', None)
        vectors = batch_dict.get('vectors', None)
        if matches is None or vectors is None:
            return

        keypoint = batch_dict['keypoint']
        batch_idx = keypoint[:, 0].long()
        B = int(batch_dict['batch_size'])

        cs_preds = []
        cs_targets = []
        valid_mask = []
        global_offset = 0
        for i in range(B):
            mask = batch_idx == i
            n = int(mask.sum())
            if n < 3:
                continue
            local_phi, local_cs = pred_phi_per_sample[i]
            if local_cs is None:
                continue

            # GT vertices for this sample
            v_i = vectors[i]
            v_real_mask = v_i.sum(dim=-1) > -2e1
            v_real = v_i[v_real_mask]                            # (n_gt, 3)
            n_gt = int(v_real.shape[0])
            if n_gt < 3:
                global_offset += n
                continue

            # canonicalise GT to CCW starting from XY-lex-smallest
            perm = _canonicalize_ring(v_real[:, :2])
            # pos_in_ring[k] = canonical position (0..n_gt-1) of GT vertex k
            pos_in_ring = [0] * n_gt
            for pos, gt_idx in enumerate(perm):
                pos_in_ring[gt_idx] = pos

            # matches: predicted -> GT vertex index (-1 = unmatched)
            m_i = matches[global_offset:global_offset + n].cpu().numpy()
            for local_k in range(n):
                gt_k = int(m_i[local_k])
                if gt_k < 0 or gt_k >= n_gt:
                    continue
                pos = pos_in_ring[gt_k]
                phi_t = 2 * math.pi * pos / n_gt
                tgt = torch.tensor([math.cos(phi_t), math.sin(phi_t)],
                                   device=local_cs.device, dtype=local_cs.dtype)
                cs_preds.append(local_cs[local_k])
                cs_targets.append(tgt)
                valid_mask.append(True)
            global_offset += n

        if not cs_preds:
            return
        self.train_dict['cs_pred'] = torch.stack(cs_preds, dim=0)       # (M_valid, 2)
        self.train_dict['cs_target'] = torch.stack(cs_targets, dim=0)   # (M_valid, 2)

    # ---------------------------------------------------------------- #
    # loss -- cosine-similarity between predicted and target (cos, sin)
    # ---------------------------------------------------------------- #
    def loss(self, loss_dict, disp_dict):
        # anchor to a real parameter so grad always flows
        zero_with_grad = self.phase_fc[0].weight.sum() * 0.0

        cs_pred = self.train_dict.get('cs_pred', None)
        cs_target = self.train_dict.get('cs_target', None)
        if cs_pred is None or cs_pred.numel() == 0:
            loss_dict.update({'edge_cls_loss': 0.0, 'edge_loss': 0.0})
            disp_dict.update({'edge_acc': 0.0})
            return zero_with_grad, loss_dict, disp_dict

        # cosine similarity: 1 - cos(angle between pred and target)
        sim = (cs_pred * cs_target).sum(dim=-1)            # in [-1, 1]
        loss = (1.0 - sim).mean() * self.loss_weight['cls_weight']

        with torch.no_grad():
            # "edge_acc" repurposed as fraction of vertices within 30 degrees
            # of their target phase -- monitoring signal only
            acc = float((sim >= math.cos(math.radians(30))).float().mean().item())

        loss_dict.update({'edge_cls_loss': loss.item(), 'edge_loss': loss.item()})
        disp_dict.update({'edge_acc': acc})
        return loss, loss_dict, disp_dict
