# edge_gnn_net.py
"""
EdgeGNNNet -- a drop-in replacement for EdgeAttentionNet that uses a GAT
message-passing GNN over a k-NN graph of predicted vertices, followed by a
Sinkhorn assignment layer that enforces degree-2 (closed ring) topology.

Drop-in interface:
  forward(batch_dict) reads:    'keypoint' (M_total, 4)  [batch_idx, x, y, z]
                                'keypoint_features' (M_total, C_in)
                                'edges'  (only in training) (B, max_E, 2)
                                'matches' (only in training, set by ClusterRefineNet)
                                'batch_size'
  forward(batch_dict) writes:   'pair_points' (P_total, 2) local index pairs
                                'edge_score'  (P_total,) post-Sinkhorn scores
                                in the SAME ORDERING as EdgeAttentionNet,
                                i.e. itertools.combinations per sample, concatenated

Why the same ordering: existing exports, post-processor, and eval_process all
index into edge_score using the (n*(n-1)/2) cursor pattern. Keeping the order
lets every downstream consumer work unchanged.

Architecture (per sample, then batched via masking):
  1. project node features C_in -> D
  2. k-NN graph in XY (k=8)
  3. GAT_LAYERS rounds of message passing with masked attention
  4. score every pair as MLP(concat(node_i, node_j))
  5. assemble (n+1) x (n+1) score matrix with dustbins
  6. Sinkhorn iterations to row/col sums of 2 (real rows) and slack (dustbin)
  7. extract upper-triangle real-real entries as edge_score
"""

import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import loss_utils


# --- module constants (tunable in code, not config, since this is research) -- #
K_NN          = 8        # neighbours per node in the k-NN graph
GAT_LAYERS    = 3        # rounds of message passing
HIDDEN_DIM    = 128      # GAT internal width
SINKHORN_ITERS = 20      # iterations of row/col normalization
TARGET_DEGREE = 2.0      # closed ring => every real vertex degree-2
DUSTBIN_INIT  = 1.0      # initial dustbin score (mass that doesn't fit ring)

# Diagnostic prints during the first few training batches. Helpful for
# debugging supervision (matches/edges). Disable in normal runs.
DEBUG_EDGE    = False


# ============================================================================ #
# Sinkhorn with dustbins (log-domain for stability)
# ============================================================================ #
def _log_sinkhorn(log_M, n_iters, target_degree):
    """
    Log-domain Sinkhorn that normalises a square matrix so that:
      - every REAL row (all but last) sums to target_degree
      - every REAL column (all but last) sums to target_degree
      - the dustbin row/col absorbs slack
    log_M : (B, N+1, N+1) log-scores. N is real vertex count.
    Returns log of the normalised matrix, same shape.
    """
    B, M, _ = log_M.shape
    N = M - 1   # number of real vertices
    log_d = math.log(target_degree)

    # marginals in log space: real rows want log(2), dustbin row wants log(N) so
    # that total mass = 2*N (each of N rows contributes 2)
    log_mu = torch.full((B, M), log_d, device=log_M.device, dtype=log_M.dtype)
    log_mu[:, -1] = math.log(N) if N > 0 else 0.0   # dustbin row marginal
    log_nu = log_mu.clone()                          # symmetric

    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(n_iters):
        u = log_mu - torch.logsumexp(log_M + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(log_M + u.unsqueeze(2), dim=1)
    return log_M + u.unsqueeze(2) + v.unsqueeze(1)


# ============================================================================ #
# GAT layer -- hand-rolled, masked attention over a k-NN adjacency
# ============================================================================ #
class GATLayer(nn.Module):
    """Single GAT head over a per-sample mask. We operate on a padded batch of
    nodes (B, N_max, D) plus an adjacency mask (B, N_max, N_max) where 1 means
    'i can attend to j'. Self-loops are included so every node attends to itself.
    """
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W = nn.Linear(d_in, d_out, bias=False)
        # attention coefficients on concat(Wi, Wj)
        self.a = nn.Linear(2 * d_out, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)
        self.d_out = d_out

    def forward(self, x, adj_mask):
        # x : (B, N, d_in)   adj_mask : (B, N, N) {0,1}
        B, N, _ = x.shape
        h = self.W(x)                                          # (B, N, d_out)
        # pairwise concat: (B, N, N, 2*d_out)
        h_i = h.unsqueeze(2).expand(B, N, N, self.d_out)
        h_j = h.unsqueeze(1).expand(B, N, N, self.d_out)
        e = self.leaky(self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1))  # (B,N,N)
        # mask out non-edges with -inf so softmax ignores them
        e = e.masked_fill(adj_mask == 0, float('-inf'))
        # rows that are all -inf (isolated nodes) would NaN; guard by adding self
        # softmax over j (neighbours of i)
        alpha = F.softmax(e, dim=-1)
        # if a row had no neighbours, replace NaN with zeros so output is just zero
        alpha = torch.nan_to_num(alpha, nan=0.0)
        return torch.bmm(alpha, h)                             # (B, N, d_out)


# ============================================================================ #
# Edge GNN with Sinkhorn assignment
# ============================================================================ #
class EdgeGNNNet(nn.Module):
    def __init__(self, model_cfg, input_channel):
        super().__init__()
        self.model_cfg = model_cfg

        # project incoming keypoint features to hidden dim
        self.in_proj = nn.Sequential(
            nn.Linear(input_channel, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
        )
        # GAT stack
        self.gats = nn.ModuleList(
            [GATLayer(HIDDEN_DIM, HIDDEN_DIM) for _ in range(GAT_LAYERS)]
        )
        # pair scorer: scores M[i,j] from concat(node_i, node_j)
        self.pair_scorer = nn.Sequential(
            nn.Linear(2 * HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )
        # learnable dustbin score
        self.dustbin = nn.Parameter(torch.tensor(DUSTBIN_INIT))

        # Always initialise -- self.training is False at __init__ time, so the
        # original 'if self.training' gating would leave these unset and crash
        # the first time loss() is called after model.train().
        self.train_dict = {}
        self.add_module('cls_loss_func', loss_utils.SigmoidBCELoss())
        self.loss_weight = self.model_cfg.LossWeight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -------------------------------------------------------------------- #
    # helper: pad per-sample node lists into (B, N_max, D) plus a valid mask
    # -------------------------------------------------------------------- #
    @staticmethod
    def _pad_batch(features, batch_idx, B, device):
        sizes = [int((batch_idx == i).sum()) for i in range(B)]
        N_max = max(sizes) if sizes else 0
        D = features.shape[1]
        padded = features.new_zeros(B, N_max, D)
        valid = features.new_zeros(B, N_max, dtype=torch.bool)
        for i in range(B):
            n = sizes[i]
            if n == 0:
                continue
            padded[i, :n] = features[batch_idx == i]
            valid[i, :n] = True
        return padded, valid, sizes

    @staticmethod
    def _knn_adj(xy_padded, valid_mask, k):
        """Build (B, N, N) k-NN adjacency. Self-loop included.
        Invalid (padding) rows/columns have all-zero adjacency."""
        B, N, _ = xy_padded.shape
        d2 = torch.cdist(xy_padded, xy_padded, p=2)            # (B,N,N)
        # mask invalid columns with +inf so they're never chosen as neighbours
        col_invalid = ~valid_mask.unsqueeze(1).expand(B, N, N)
        d2 = d2.masked_fill(col_invalid, float('inf'))
        # for each node, take top-k smallest distances (including self at d=0)
        kk = min(k + 1, N)
        _, idx = torch.topk(d2, kk, dim=-1, largest=False)     # (B,N,kk)
        adj = torch.zeros(B, N, N, device=xy_padded.device, dtype=torch.float)
        adj.scatter_(2, idx, 1.0)
        # zero out rows for invalid nodes
        adj = adj * valid_mask.unsqueeze(-1).float()
        # ensure self-loops on valid nodes
        eye = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, N, N)
        adj = torch.maximum(adj, eye * valid_mask.unsqueeze(-1).float())
        return adj

    # -------------------------------------------------------------------- #
    # forward
    # -------------------------------------------------------------------- #
    def forward(self, batch_dict):
        if self.training:
            self.train_dict = {}
        keypoint = batch_dict['keypoint']
        point_fea = batch_dict['keypoint_features']
        B = int(batch_dict['batch_size'])
        batch_idx = keypoint[:, 0].long()

        # Pad to (B, N_max, D); build (B, N_max, 2) XY for k-NN
        feat_pad, valid, sizes = self._pad_batch(point_fea, batch_idx, B, point_fea.device)
        xy_pad, _, _ = self._pad_batch(keypoint[:, 1:3], batch_idx, B, point_fea.device)

        # Build k-NN adjacency in XY (geometric prior on edge candidates)
        adj = self._knn_adj(xy_pad, valid, K_NN)

        # GAT message passing
        h = self.in_proj(feat_pad)
        for gat in self.gats:
            h = h + gat(h, adj)             # residual
            h = F.relu(h)

        # Per-pair scores via concat MLP, producing (B, N_max, N_max)
        N_max = h.shape[1]
        h_i = h.unsqueeze(2).expand(B, N_max, N_max, HIDDEN_DIM)
        h_j = h.unsqueeze(1).expand(B, N_max, N_max, HIDDEN_DIM)
        pair_feat = torch.cat([h_i, h_j], dim=-1)
        scores = self.pair_scorer(pair_feat).squeeze(-1)        # (B, N_max, N_max)

        # Symmetrise (edges are undirected) and mask diagonal
        scores = 0.5 * (scores + scores.transpose(1, 2))
        diag_mask = torch.eye(N_max, device=scores.device, dtype=torch.bool).unsqueeze(0)
        scores = scores.masked_fill(diag_mask, -1e9)
        # mask invalid rows/cols
        valid_pair = valid.unsqueeze(1) & valid.unsqueeze(2)
        scores = scores.masked_fill(~valid_pair, -1e9)

        # Per-sample Sinkhorn (each sample may have different real-N)
        edge_score_list = []
        pair_points_list = []
        train_target_list = []
        train_valid_mask_list = []
        global_offset = 0
        for i in range(B):
            n = sizes[i]
            if n < 2:
                # skip but still advance to keep cursor consistent with downstream
                continue
            S = scores[i, :n, :n]                                # (n,n)
            # add dustbin row/col with learnable score
            db = self.dustbin.to(S.device) * torch.ones(1, n, device=S.device)
            S_aug = torch.cat([S, db.t()], dim=1)                # (n, n+1)
            db_row = self.dustbin.to(S.device) * torch.ones(1, n + 1, device=S.device)
            S_aug = torch.cat([S_aug, db_row], dim=0)            # (n+1, n+1)

            log_M = _log_sinkhorn(S_aug.unsqueeze(0), SINKHORN_ITERS, TARGET_DEGREE)
            P = log_M.exp().squeeze(0)                            # (n+1, n+1) doubly-target
            P_real = P[:n, :n]                                    # (n, n) edge probs

            # emit pair-ordered scores (itertools.combinations order) so downstream
            # code that uses (n*(n-1)/2) cursor still works unchanged
            pair_idx = list(itertools.combinations(range(n), 2))
            pair_idx_t = torch.tensor(pair_idx, dtype=torch.long, device=P.device)
            es = P_real[pair_idx_t[:, 0], pair_idx_t[:, 1]]       # (P_i,)
            edge_score_list.append(es)
            pair_points_list.append(pair_idx_t.float())

            if self.training:
                # Build supervision: for each pair, is it a GT edge?
                t, vmask = self._build_pair_target(batch_dict, i, n, pair_idx_t,
                                                   global_offset)
                train_target_list.append(t)
                train_valid_mask_list.append(vmask)
            global_offset += n

        if not edge_score_list:
            # no usable samples this batch
            batch_dict['pair_points'] = point_fea.new_zeros((0, 2))
            batch_dict['edge_score'] = point_fea.new_zeros((0,))
            return batch_dict

        batch_dict['pair_points'] = torch.cat(pair_points_list, dim=0)
        batch_dict['edge_score'] = torch.cat(edge_score_list, dim=0)

        if self.training:
            self.train_dict['pair_prob'] = torch.cat(edge_score_list, dim=0)
            self.train_dict['pair_target'] = torch.cat(train_target_list, dim=0)
            self.train_dict['pair_valid'] = torch.cat(train_valid_mask_list, dim=0)
            if DEBUG_EDGE:
                v = self.train_dict['pair_valid']
                t = self.train_dict['pair_target']
                p = self.train_dict['pair_prob']
                # only print first batch of training to avoid spam
                if not hasattr(self, '_dbg_count'):
                    self._dbg_count = 0
                if self._dbg_count < 3:
                    print('[edge gnn] pairs=%d  valid=%d  positives=%d  '
                          'prob_range=[%.3f,%.3f]  dustbin=%.3f'
                          % (p.numel(), int(v.sum()), int(t.sum()),
                             float(p.min()), float(p.max()),
                             float(self.dustbin)))
                    self._dbg_count += 1
        return batch_dict

    # -------------------------------------------------------------------- #
    # build per-pair target labels using HungarianMatcher's matches
    # -------------------------------------------------------------------- #
    def _build_pair_target(self, batch_dict, sample_i, n, pair_idx_t, global_offset):
        """
        For each predicted-pair (i,j) of sample sample_i, look at whether the
        Hungarian-matched GT vertex pair is adjacent in the GT polygon.
        Pairs where at least one predicted vertex is unmatched are flagged
        'invalid' so the loss ignores them (Sinkhorn dustbin handles the slack).
        """
        device = pair_idx_t.device
        n_pairs = pair_idx_t.shape[0]
        target = torch.zeros(n_pairs, device=device)
        valid  = torch.zeros(n_pairs, device=device, dtype=torch.bool)

        matches = batch_dict.get('matches', None)
        edges_gt = batch_dict.get('edges', None)
        if matches is None or edges_gt is None:
            if DEBUG_EDGE:
                print('[edge gnn] sample %d: matches=%s edges=%s' %
                      (sample_i, 'None' if matches is None else 'ok',
                       'None' if edges_gt is None else 'ok'))
            return target, valid

        # matches is a 1-D tensor over all batched keypoints; slice this sample's
        m_i = matches[global_offset:global_offset + n].cpu().numpy()

        # GT edges for this sample: (E, 2), padded with rows summing to 0
        e_i = edges_gt[sample_i].cpu().numpy()
        e_set = set()
        for u, v in e_i:
            if u + v <= 0:
                continue
            e_set.add((int(min(u, v)), int(max(u, v))))

        if DEBUG_EDGE and sample_i == 0:
            print('[edge gnn] sample 0: n_kpt=%d  matches=%s  n_matched=%d  '
                  'n_gt_edges=%d  matches_unique=%s'
                  % (n, m_i.tolist(), int((m_i >= 0).sum()),
                     len(e_set), sorted(set(int(x) for x in m_i if x >= 0))[:10]))

        pair_np = pair_idx_t.cpu().numpy()
        for k, (a, b) in enumerate(pair_np):
            ma, mb = int(m_i[a]), int(m_i[b])
            if ma < 0 or mb < 0:
                continue                                    # at least one unmatched
            valid[k] = True
            key = (min(ma, mb), max(ma, mb))
            if key in e_set:
                target[k] = 1.0
        return target, valid

    # -------------------------------------------------------------------- #
    # loss -- BCE on real (matched) pairs only. Dustbin handles the rest.
    # -------------------------------------------------------------------- #
    def loss(self, loss_dict, disp_dict):
        # Even if no valid pairs exist this batch, we must return a tensor with
        # grad_fn connected to a learnable parameter, or backward() crashes with
        # "element 0 ... does not require grad". Anchor to self.dustbin.
        zero_with_grad = self.dustbin * 0.0

        if 'pair_prob' not in self.train_dict or self.train_dict['pair_prob'].numel() == 0:
            loss_dict.update({'edge_cls_loss': 0.0, 'edge_loss': 0.0})
            disp_dict.update({'edge_acc': 0.0})
            return zero_with_grad, loss_dict, disp_dict

        prob   = self.train_dict['pair_prob']
        target = self.train_dict['pair_target']
        valid  = self.train_dict['pair_valid']

        if valid.sum() == 0:
            loss_dict.update({'edge_cls_loss': 0.0, 'edge_loss': 0.0})
            disp_dict.update({'edge_acc': 0.0})
            return zero_with_grad, loss_dict, disp_dict

        prob_v   = prob[valid]
        target_v = target[valid]
        eps = 1e-7
        loss = -(target_v * torch.log(prob_v.clamp(min=eps))
                 + (1 - target_v) * torch.log((1 - prob_v).clamp(min=eps)))
        loss = loss.mean() * self.loss_weight['cls_weight']

        with torch.no_grad():
            pred = (prob_v >= 0.5).float()
            pos = target_v == 1
            edge_acc = float((pred[pos] == 1).float().mean().item()) if pos.any() else 0.0

        loss_dict.update({'edge_cls_loss': loss.item(), 'edge_loss': loss.item()})
        disp_dict.update({'edge_acc': edge_acc})
        return loss, loss_dict, disp_dict
