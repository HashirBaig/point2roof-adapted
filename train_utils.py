import glob
import tqdm
import os
import torch
import numpy as np
from test_util import test_model
from train_stages import freeze_vertex
from loss_logger import LossLogger


def train_one_epoch(model, optim, data_loader, accumulated_iter,
                    tbar, leave_pbar=False, stage='all', loss_log=None):
    total_it_each_epoch = len(data_loader)
    dataloader_iter = iter(data_loader)
    pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(data_loader)
            batch = next(dataloader_iter)
            print('new iters')
        try:
            cur_lr = float(optim.lr)
        except:
            cur_lr = optim.param_groups[0]['lr']
        model.train()
        # In stage='edge', net.train() above flipped EVERY submodule back to train
        # mode -- including the frozen vertex stack. Re-apply the freeze so the
        # vertex BN running stats don't drift while only the edge head trains.
        if stage == 'edge':
            freeze_vertex(model)
        optim.zero_grad()
        load_data_to_gpu(batch)
        loss, loss_dict, disp_dict = model(batch)
        loss.backward()
        optim.step()
        accumulated_iter += 1
        disp_dict.update(loss_dict)
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})
        # accumulate per-iter loss values for the epoch CSV
        if loss_log is not None:
            row = dict(loss_dict)
            row['loss'] = loss.item()
            loss_log.add_batch(row)
        # log to console and tensorboard
        pbar.update()
        pbar.set_postfix(dict(total_it=accumulated_iter))
        tbar.set_postfix(disp_dict)
        tbar.refresh()
    pbar.close()
    return accumulated_iter


def train_model(model, optim, data_loader, lr_sch, start_it, start_epoch, total_epochs, ckpt_save_dir, sampler=None,
                max_ckpt_save_num=5, stage='all'):
    """
    stage='all'    -> original behavior: vertex-only for first 5 epochs, then joint.
    stage='vertex' -> vertex losses only for ALL epochs (no edge head).
    stage='edge'   -> assumes vertex is loaded+frozen by caller; edge loss only.
    """
    # per-epoch loss CSV, named by stage so multiple stages in the same dir don't collide
    loss_csv = ckpt_save_dir / ('losses_%s.csv' % stage)
    loss_log = LossLogger(loss_csv)
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True) as tbar:
        accumulated_iter = start_it
        for e in tbar:
            if sampler is not None:
                sampler.set_epoch(e)
            # Original auto-promotion at epoch 5 only applies to joint training.
            if stage == 'all' and e > 5:
                model.use_edge = True
            accumulated_iter = train_one_epoch(model, optim, data_loader, accumulated_iter, tbar,
                                               leave_pbar=(e + 1 == total_epochs),
                                               stage=stage, loss_log=loss_log)
            loss_log.end_epoch(e + 1)
            lr_sch.step()
            lr = max(optim.param_groups[0]['lr'], 1e-6)
            for param_group in optim.param_groups:
                param_group['lr'] = lr
            # checkpoint name prefix per stage so files don't collide
            prefix = 'checkpoint' if stage == 'all' else ('vertex' if stage == 'vertex' else 'edge')
            ckpt_list = glob.glob(str(ckpt_save_dir / ('%s_epoch_*.pth' % prefix)))
            ckpt_list.sort(key=os.path.getmtime)
            if ckpt_list.__len__() >= max_ckpt_save_num:
                for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                    os.remove(ckpt_list[cur_file_idx])
            ckpt_name = ckpt_save_dir / ('%s_epoch_%d' % (prefix, e + 1))
            save_checkpoint(
                checkpoint_state(model, optim, e + 1, accumulated_iter), filename=ckpt_name,
            )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None
    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()
