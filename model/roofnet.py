#roofnet.py
from .pointnet2 import PointNet2
from .cluster_refine import ClusterRefineNet
from .edge_pred_net import EdgeAttentionNet
from .edge_gnn_net import EdgeGNNNet
from .phase_head import PhaseHead
import torch.nn as nn


# pick which edge head to use here
# 'attention' = original EdgeAttentionNet (pairwise independent)
# 'gnn'       = EdgeGNNNet (GAT + Sinkhorn closed-ring assignment) [on hold]
# 'phase'     = PhaseHead (predict ring position, edges = sorted-phi consecutive)
EDGE_HEAD = 'phase'


_EDGE_CLASSES = {
    'attention': EdgeAttentionNet,
    'gnn': EdgeGNNNet,
    'phase': PhaseHead,
}


class RoofNet(nn.Module):
    def __init__(self, model_cfg, input_channel=3):
        super().__init__()
        self.use_edge = False
        self.train_stage = 'all'
        self.model_cfg = model_cfg
        self.keypoint_det_net = PointNet2(model_cfg.PointNet2, input_channel)
        self.cluster_refine_net = ClusterRefineNet(
            model_cfg.ClusterRefineNet,
            input_channel=self.keypoint_det_net.num_output_feature,
        )
        edge_cls = _EDGE_CLASSES[EDGE_HEAD]
        self.edge_att_net = edge_cls(
            model_cfg.EdgeAttentionNet,
            input_channel=self.cluster_refine_net.num_output_feature,
        )

    def forward(self, batch_dict):
        batch_dict = self.keypoint_det_net(batch_dict)
        if self.use_edge:
            batch_dict = self.cluster_refine_net(batch_dict)
            batch_dict = self.edge_att_net(batch_dict)
        if self.training:
            loss = 0
            loss_dict = {}
            disp_dict = {}
            if self.train_stage in ('vertex', 'all'):
                tmp_loss, loss_dict, disp_dict = self.keypoint_det_net.loss(loss_dict, disp_dict)
                loss += tmp_loss
                if self.use_edge:
                    tmp_loss, loss_dict, disp_dict = self.cluster_refine_net.loss(loss_dict, disp_dict)
                    loss += tmp_loss
            if self.use_edge and self.train_stage in ('edge', 'all'):
                tmp_loss, loss_dict, disp_dict = self.edge_att_net.loss(loss_dict, disp_dict)
                loss += tmp_loss
            return loss, loss_dict, disp_dict
        else:
            return batch_dict
