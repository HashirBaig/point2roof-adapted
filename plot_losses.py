#!/usr/bin/env python3
"""
plot_losses.py

Standalone plotter for the per-epoch loss CSV produced by train_utils.LossLogger.

Usage:
    python plot_losses.py --csv output/run31/ckpt/losses_vertex.csv
    python plot_losses.py --csv output/run31/ckpt/losses_edge.csv \
                          --out output/run31/loss_edge.png

Produces one figure with three subplots so the heads aren't squashed together by
y-scale differences: binary-classification head, offset regression head, and
refinement head. Edge-head columns are plotted too when present.
"""

import os
import csv
import argparse


def read_csv(path):
    with open(path, 'r') as f:
        rdr = csv.reader(f)
        rows = list(rdr)
    if not rows:
        return [], []
    cols = rows[0]
    data = []
    for r in rows[1:]:
        try:
            data.append({c: float(v) if v not in ('', None) else None
                         for c, v in zip(cols, r)})
        except ValueError:
            continue
    return cols, data


def col_series(data, key):
    epochs, vals = [], []
    for row in data:
        v = row.get(key)
        if v is not None and 'epoch' in row and row['epoch'] is not None:
            epochs.append(row['epoch'])
            vals.append(v)
    return epochs, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='per-epoch CSV from training')
    ap.add_argument('--out', default=None, help='output image (default: alongside CSV)')
    ap.add_argument('--logy', action='store_true', help='use log y-axis')
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cols, data = read_csv(args.csv)
    if not data:
        raise SystemExit('empty CSV: %s' % args.csv)

    # Group available columns by head. Edge head only present in joint or stage=edge.
    groups = [
        ('Binary classification head', ['pts_cls_loss']),
        ('Offset regression head',     ['pts_offset_loss']),
        ('Refinement head',            ['refine_offset_loss']),
    ]
    if 'edge_cls_loss' in cols:
        groups.append(('Edge classification head', ['edge_cls_loss']))

    # filter to groups that actually have data
    groups = [(t, ks) for t, ks in groups if any(k in cols for k in ks)]
    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(8.5, 2.6 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (title, keys) in zip(axes, groups):
        for k in keys:
            if k not in cols:
                continue
            ep, v = col_series(data, k)
            if ep:
                ax.plot(ep, v, marker='o', markersize=3, linewidth=1.2, label=k)
        ax.set_title(title)
        ax.set_ylabel('loss')
        ax.grid(True, alpha=0.3)
        if args.logy:
            ax.set_yscale('log')
        ax.legend(loc='upper right', frameon=False, fontsize=9)
    axes[-1].set_xlabel('epoch')
    fig.suptitle(os.path.basename(args.csv), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = args.out or os.path.splitext(args.csv)[0] + '.png'
    fig.savefig(out, dpi=140)
    print('wrote', out)


if __name__ == '__main__':
    main()
