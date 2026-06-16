#!/usr/bin/env python3
"""
train_stages.py

Helpers for sequential (two-stage) training of RoofNet.

Stage 1 (vertex): train PointNet2 cls/offset heads + ClusterRefineNet refinement.
Stage 2 (edge):   freeze the vertex stack, train only EdgeAttentionNet.

The non-obvious bit is BatchNorm. Freezing parameters via requires_grad=False
does NOT stop BN running statistics from updating. We must also put the frozen
modules in eval() mode -- and re-apply that after every call to net.train(),
because PyTorch's .train() recursively flips every child back to train mode.
"""

import torch


def freeze_vertex(net):
    """Freeze vertex modules for stage-2 (edge) training.

    Effects:
      * Vertex weights -> requires_grad=False (no gradient updates)
      * Vertex BN     -> eval() (running stats don't drift from edge-head batches)
      * Edge head      -> train() (so its BN/dropout work)

    Call this AFTER each `net.train()` in your training loop. The training
    helper handles that automatically.
    """
    for p in net.keypoint_det_net.parameters():
        p.requires_grad = False
    for p in net.cluster_refine_net.parameters():
        p.requires_grad = False
    net.keypoint_det_net.eval()
    net.cluster_refine_net.eval()
    net.edge_att_net.train()


def trainable_params(net):
    """Return the parameters with requires_grad=True. Use this when building
    the optimizer in stage 2 so it only touches edge-head weights."""
    return [p for p in net.parameters() if p.requires_grad]


def set_stage(net, stage):
    """Configure RoofNet for a training stage.

      stage='vertex' : use_edge=False, train_stage='vertex'.
                       Edge head doesn't even run; vertex losses only.
      stage='edge'   : use_edge=True,  train_stage='edge'.
                       Full forward runs; only edge loss contributes.
                       Caller must additionally call freeze_vertex(net).
      stage='all'    : original joint behavior (no change to wiring).
    """
    if stage == 'vertex':
        net.use_edge = False
        net.train_stage = 'vertex'
    elif stage == 'edge':
        net.use_edge = True
        net.train_stage = 'edge'
    elif stage == 'all':
        net.use_edge = False    # original train_model toggles to True after epoch 5
        net.train_stage = 'all'
    else:
        raise ValueError('unknown stage %r' % stage)
