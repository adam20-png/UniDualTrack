"""
Fine-tuning script for SingleVGG_Dual tracker.

Two-phase progressive unfreezing strategy:
  Phase 1 (15 epochs): Frozen backbone (ImageNet weights), train head only
    - ECA attention, RichProjection, fusion conv, SiamFC cross-corr
    - Head LR: 1e-3, backbone LR: 0 (frozen)
  Phase 2 (25 epochs): Unfrozen backbone, differential LR fine-tuning
    - Backbone LR: 1e-4 (10x lower than head)
    - Head LR: 1e-3
    - Total: 40 epochs

Usage (PyCharm): Just click Run — everything is configured below.
You can also use command-line args to override any setting.
"""
from __future__ import absolute_import

import os
import sys
import argparse

# Add project root to path
dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dir_mytest)

from got10k.datasets import GOT10k
from siamfc.siamfc1 import TrackerSiamFC


# ============================================================
# CONFIGURATION — Modify these values, then click Run in PyCharm
# ============================================================

# Checkpoint: full path to .pth file, or "none" for ImageNet only
CHECKPOINT = os.path.join(dir_mytest, 'tools', 'models', 'siamfc_alexnet_e46.pth')

# GOT-10k dataset path
DATA_DIR = os.path.expanduser('~/Desktop/testdata/GOT-10k')

# Where to save new checkpoints
SAVE_DIR = os.path.join(dir_mytest, 'tools', 'models')

# Training hyperparameters
PHASE1_EPOCHS = 15       # Frozen backbone phase
PHASE2_EPOCHS = 25       # Unfrozen backbone phase
BACKBONE_LR = 5e-5       # Backbone learning rate (Phase 2)
HEAD_LR = 1e-2           # Head learning rate (both phases)
BATCH_SIZE = 8
NUM_WORKERS = 8

# ============================================================


def main():
    parser = argparse.ArgumentParser(description='Fine-tune SingleVGG_Dual tracker')
    parser.add_argument('--data', type=str, default=DATA_DIR)
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR)
    parser.add_argument('--p1', type=int, default=PHASE1_EPOCHS)
    parser.add_argument('--p2', type=int, default=PHASE2_EPOCHS)
    parser.add_argument('--blr', type=float, default=BACKBONE_LR)
    parser.add_argument('--hlr', type=float, default=HEAD_LR)
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    # Resolve checkpoint: "none" -> None (skip checkpoint, ImageNet only)
    checkpoint_path = None if args.checkpoint.lower() == 'none' else args.checkpoint
    if checkpoint_path and not os.path.isfile(checkpoint_path):
        print(f'WARNING: Checkpoint not found: {checkpoint_path}')
        print('  -> Falling back to ImageNet initialization only.\n')
        checkpoint_path = None

    print(f'Dataset:      {args.data}')
    print(f'Save dir:     {args.save_dir}')
    print(f'Phase 1:      {args.p1} epochs (frozen backbone, head LR={args.hlr:.1e})')
    print(f'Phase 2:      {args.p2} epochs (unfrozen, backbone LR={args.blr:.1e}, head LR={args.hlr:.1e})')
    print(f'Checkpoint:   {checkpoint_path or "None (ImageNet init only)"}')
    print()

    # Load dataset
    seqs = GOT10k(args.data, subset='train', return_meta=True)
    print(f'Loaded GOT-10k train set: {len(seqs)} sequences')

    # Create tracker (loads ImageNet weights automatically, then checkpoint)
    tracker = TrackerSiamFC(
        net_path=checkpoint_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Run phased fine-tuning
    tracker.train_over_phased(
        seqs=seqs,
        save_dir=args.save_dir,
        phase1_epochs=args.p1,
        phase2_epochs=args.p2,
        backbone_lr=args.blr,
        head_lr=args.hlr,
    )

    print('\nDone! Checkpoints saved to:', args.save_dir)


if __name__ == '__main__':
    main()
