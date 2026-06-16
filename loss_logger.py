#!/usr/bin/env python3
"""
loss_logger.py

Accumulates per-iteration loss values during training and writes a per-epoch
average to CSV. Designed to be called from train_one_epoch (per iter) and
train_model (per epoch end), with no behavior change otherwise.

The CSV columns are the union of loss keys seen across all epochs, so head-
specific keys appear automatically:
  pts_cls_loss, pts_offset_loss, pts_loss              (PointNet2 head)
  refine_offset_loss, refine_loss                       (ClusterRefineNet)
  edge_cls_loss, edge_loss                              (EdgeAttentionNet)
  loss                                                   (total)
"""

import os
import csv


class LossLogger:
    def __init__(self, csv_path):
        self.csv_path = str(csv_path)
        # accumulators reset per epoch
        self._sums = {}
        self._count = 0
        # all keys ever seen, in stable order
        self._keys = []

    def add_batch(self, loss_dict):
        """Call once per training iteration."""
        for k, v in loss_dict.items():
            if not isinstance(v, (int, float)):
                continue
            if k not in self._sums:
                self._sums[k] = 0.0
                if k not in self._keys:
                    self._keys.append(k)
            self._sums[k] += float(v)
        self._count += 1

    def end_epoch(self, epoch):
        """Call at end of each epoch. Writes the per-epoch average row and resets."""
        if self._count == 0:
            return
        avgs = {k: self._sums[k] / self._count for k in self._sums}
        avgs['epoch'] = epoch

        # rewrite header if columns expanded since last write
        existing = []
        if os.path.isfile(self.csv_path):
            with open(self.csv_path, 'r') as f:
                rdr = csv.reader(f)
                rows = list(rdr)
            if rows:
                existing_cols = rows[0]
                existing = rows[1:]
            else:
                existing_cols = []
        else:
            existing_cols = []

        all_cols = ['epoch'] + [k for k in self._keys if k != 'epoch']
        # extend with any old columns not currently tracked (rare)
        for c in existing_cols:
            if c not in all_cols:
                all_cols.append(c)

        with open(self.csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(all_cols)
            # rewrite previous rows preserving columns
            for row in existing:
                rowmap = dict(zip(existing_cols, row))
                w.writerow([rowmap.get(c, '') for c in all_cols])
            # append this epoch
            w.writerow([avgs.get(c, '') for c in all_cols])

        # reset for next epoch
        self._sums = {}
        self._count = 0
