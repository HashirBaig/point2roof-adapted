import os
import torch
import torch.nn as nn
import argparse
import datetime
import glob
import torch.distributed as dist
from dataset.data_utils import build_dataloader
from train_utils import train_model
from model.roofnet import RoofNet
from torch import optim
from utils import common_utils
from model import model_utils
from train_stages import set_stage, freeze_vertex, trainable_params


def get_scheduler(optim, last_epoch):
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 20, 0.5, last_epoch=last_epoch)
    return scheduler


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../GithubDeepRoof', help='dataset path')
    parser.add_argument('--cfg_file', type=str, default='./model_cfg.yaml', help='model config for training')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--gpu', type=str, default='1', help='gpu for training')
    parser.add_argument('--extra_tag', type=str, default='pts6', help='extra tag for this experiment')
    parser.add_argument('--epochs', type=int, default=90, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--stage', type=str, default='all', choices=['all', 'vertex', 'edge'],
                        help="'all' = joint (original), 'vertex' = stage 1, 'edge' = stage 2 (loads vertex_*.pth)")
    parser.add_argument('--vertex_ckpt', type=str, default=None,
                        help="(stage=edge only) path to vertex_epoch_*.pth. Default: newest in ckpt_dir.")
    args = parser.parse_args()
    cfg = common_utils.cfg_from_yaml_file(args.cfg_file)
    return args, cfg


def _resolve_vertex_ckpt(args, ckpt_dir):
    if args.vertex_ckpt:
        assert os.path.isfile(args.vertex_ckpt), 'vertex_ckpt not found: %s' % args.vertex_ckpt
        return args.vertex_ckpt
    cands = sorted(glob.glob(str(ckpt_dir / 'vertex_epoch_*.pth')), key=os.path.getmtime)
    assert cands, 'No vertex_epoch_*.pth in %s. Run --stage vertex first or pass --vertex_ckpt.' % str(ckpt_dir)
    return cands[-1]


def main():
    args, cfg = parse_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    extra_tag = args.extra_tag if args.extra_tag is not None \
            else 'model-%s' % datetime.datetime.now().strftime('%Y%m%d')
    output_dir = cfg.ROOT_DIR / 'output' / extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'ckpt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / 'log.txt'
    logger = common_utils.create_logger(log_file)

    logger.info('**********************Start logging**********************')
    logger.info('stage = %s' % args.stage)

    train_loader = build_dataloader(args.data_path, args.batch_size, cfg.DATA, training=True, logger=logger)

    net = RoofNet(cfg.MODEL)
    net.cuda()

    # ---- stage-specific setup ----
    set_stage(net, args.stage)

    start_epoch = it = 0
    last_epoch = -1

    if args.stage == 'edge':
        # Load vertex-only checkpoint, freeze vertex, build optimizer over edge params only.
        vckpt = _resolve_vertex_ckpt(args, ckpt_dir)
        logger.info('loading vertex checkpoint: %s' % vckpt)
        ck = torch.load(vckpt)
        # strict=False: edge_att_net keys aren't in the vertex-only checkpoint
        missing, unexpected = net.load_state_dict(ck['model_state'], strict=False)
        logger.info('missing keys (expected: edge_att_net): %d' % len(missing))
        logger.info('unexpected keys (expected: 0): %d' % len(unexpected))
        freeze_vertex(net)
        params = trainable_params(net)
        n_train = sum(p.numel() for p in params)
        logger.info('stage=edge: trainable params = %d (edge head only)' % n_train)
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=1e-3)
        # do NOT resume start_epoch from vertex ckpt; edge starts fresh
    else:
        # stage='all' or 'vertex': original behavior. Resume from a matching ckpt if present.
        prefix = 'checkpoint' if args.stage == 'all' else 'vertex'
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-3)
        ckpt_list = glob.glob(str(ckpt_dir / ('*%s_epoch_*.pth' % prefix)))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model_utils.load_params_with_optimizer(
                net, ckpt_list[-1], optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1

    scheduler = get_scheduler(optimizer, last_epoch=last_epoch)

    net = net.train()
    # Stage 2: train() above un-froze everything; reapply.
    if args.stage == 'edge':
        freeze_vertex(net)

    logger.info('**********************Start training**********************')

    train_model(net, optimizer, train_loader, scheduler, it, start_epoch, args.epochs, ckpt_dir,
                stage=args.stage)


if __name__ == '__main__':
    main()
