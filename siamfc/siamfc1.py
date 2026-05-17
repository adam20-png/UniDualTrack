from __future__ import absolute_import, division, print_function
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import torchvision.models.vgg

from torch.utils import model_zoo
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker

import sys

sys.path.append(os.path.abspath('.'))

from siamfc import ops
from siamfc.heads import SiamFC
from siamfc.losses import GHMCLoss
from siamfc.losses import FocalLoss
from siamfc.losses import BalancedLoss
from siamfc.datasets import Pair
from siamfc.transforms import SiamFCTransforms
from siamfc import backbones

from siamfc.attention import GlobalAttentionBlock, CBAM
from siamfc.backbones import SELayer1, ECALayer, ECALayer1, _BatchNorm2d
from siamfc.dcn import DeformConv2d
from siamfc.psp import PSA

__all__ = ['TrackerSiamFC']


class RichProjection(nn.Module):
    """Dual-branch bottleneck projection that preserves information better than
    a simple 1×1 convolution when reducing channels before cross-correlation.

    Architecture:
        x (in_channels) ─┬─ DepthWise 3×3 → BN → ReLU → PointWise 1×1 (out_ch) → BN ─┐
                          │                                                           ├→ + → ReLU → out
                          └─ PointWise 1×1 (out_ch) → BN ────────────────────────────┘

    - Depthwise branch: captures spatial context before projection (local patterns)
    - Pointwise branch: direct channel mixing (global relationships)
    - Sum fusion: both streams contribute without information bottleneck

    The output channel count matches the old simple projection, so cross-correlation
    cost is unchanged (128ch → 4× cheaper than 256ch).
    """

    def __init__(self, in_channels=256, out_channels=128):
        super(RichProjection, self).__init__()
        # Depthwise branch: spatial context then project
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1,
            padding=1, groups=in_channels, bias=False)  # depthwise
        self.dw_bn = _BatchNorm2d(in_channels)
        self.dw_relu = nn.ReLU(inplace=True)
        self.dw_pw = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False)  # pointwise proj
        self.dw_pw_bn = _BatchNorm2d(out_channels)

        # Pointwise branch: direct channel projection (no spatial mixing)
        self.pw_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False)
        self.pw_bn = _BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Branch 1: depthwise 3×3 → pointwise 1×1
        dw = self.dw_conv(x)
        dw = self.dw_bn(dw)
        dw = self.dw_relu(dw)
        dw = self.dw_pw(dw)
        dw = self.dw_pw_bn(dw)

        # Branch 2: direct pointwise
        pw = self.pw_conv(x)
        pw = self.pw_bn(pw)

        # Fuse: both branches contribute
        return self.relu(dw + pw)


def load_imagenet_to_vgg166(model):
    """Load official ImageNet VGG16 weights into VGG166 backbone.

    Mapping:
      VGG166 conv1_1..conv4_3 → torchvision VGG16 features[0..21]
    Custom layers (conv5_1, bn5_1, branch1_adapt, branch1_decorr, ECA)
    are NOT covered and keep their random init.

    Args:
        model: TrackerSiamFC instance (model.net contains the VGG166 backbone)

    Returns:
        model with ImageNet weights loaded into backbone conv layers
    """
    # Load official VGG16-ImageNet weights via torchvision
    vgg16_official = torchvision.models.vgg16(weights='IMAGENET1K_V1')
    official_state = vgg16_official.state_dict()

    # Build mapping: VGG166 layer name → torchvision feature index
    # torchvision VGG16 features layout:
    #  [0] Conv2d(3,64,3)   [1] ReLU  [2] Conv2d(64,64,3)   [3] ReLU  [4] MaxPool
    #  [5] Conv2d(64,128,3) [6] ReLU  [7] Conv2d(128,128,3)  [8] ReLU  [9] MaxPool
    #  [10] Conv2d(128,256,3) [11] ReLU [12] Conv2d(256,256,3) [13] ReLU
    #  [14] Conv2d(256,256,3) [15] ReLU [16] MaxPool
    #  [17] Conv2d(256,512,3) [18] ReLU [19] Conv2d(512,512,3) [20] ReLU
    #  [21] Conv2d(512,512,3) [22] ReLU [23] MaxPool
    #  [24] Conv2d(512,512,3) [25] ReLU [26] Conv2d(512,512,3) [27] ReLU
    #  [28] Conv2d(512,512,3) [29] ReLU [30] MaxPool

    conv_map = {
        'conv1_1': 'features.0.weight',  # Conv2d(3, 64)
        'conv1_2': 'features.2.weight',  # Conv2d(64, 64)
        'conv2_1': 'features.5.weight',  # Conv2d(64, 128)
        'conv2_2': 'features.7.weight',  # Conv2d(128, 128)
        'conv3_1': 'features.10.weight', # Conv2d(128, 256)
        'conv3_2': 'features.12.weight', # Conv2d(256, 256)
        'conv3_3': 'features.14.weight', # Conv2d(256, 256)
        'conv4_1': 'features.17.weight', # Conv2d(256, 512)
        'conv4_2': 'features.19.weight', # Conv2d(512, 512)
        'conv4_3': 'features.21.weight', # Conv2d(512, 512)
    }

    backbone = model.net.backbone
    loaded_count = 0

    for vgg166_name, official_key in conv_map.items():
        layer = getattr(backbone, vgg166_name)
        if hasattr(official_state[official_key], 'shape'):
            layer.weight.data.copy_(official_state[official_key])
            loaded_count += 1
            print(f'  ✓ {vgg166_name} ← {official_key} '
                  f'(shape {tuple(official_state[official_key].shape)})')

    print(f'\n  Loaded {loaded_count}/10 conv layers from ImageNet VGG16')
    print(f'  Custom layers (conv5_1, branch1_adapt, branch1_decorr, ECA) '
          f'kept with random init — will be fine-tuned during training.')

    # Free the official model
    del vgg16_official

    return model


# Main network structure
class Net(nn.Module):
    # backbone (VGG166) returns (branch1, branch2) from conv3_3 and conv5_1 respectively.
    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.head = head
        # VGG166 forward() returns two feature branches internally
        self.backbone = backbone
        self.conv = nn.Sequential(nn.Conv2d(2, 1, 1))

        # Attention modules
        self.tematt = ECALayer(256)  # Template attention
        self.detatt = ECALayer(256)
        self.attse = ECALayer1(256)  # Channel attention

        # Rich projection: dual-branch bottleneck
        #   - Depthwise branch: 3×3 spatial context → 128ch (captures local patterns)
        #   - Pointwise branch: 1×1 direct → 128ch (preserves channel relationships)

        self.proj1 = RichProjection(256, 128)  # branch1 rich projection
        self.proj2 = RichProjection(256, 128)  # branch2 rich projection

    def forward(self, z, x, precomputed_kernels=None):
        # === OPTIMISATION: Template caching ===
        # During training (precomputed_kernels=None): compute both branches normally.
        # During inference: skip backbone(z) entirely — kernels are pre-computed
        if precomputed_kernels is None:
            # TRAINING PATH — compute template features from scratch
            z1, z2 = self.backbone(z)

            zf1 = self.tematt(z1)
            zf11 = self.attse(z1)
            z1 = zf1 + zf11 + z1

            zf2 = self.tematt(z2)
            zf22 = self.attse(z2)
            z2 = zf2 + zf22 + z2

            kernel1 = self.proj1(z1)
            kernel2 = self.proj2(z2)
        else:
            # INFERENCE PATH — reuse cached kernels, skip backbone(z) entirely
            kernel1, kernel2 = precomputed_kernels

        # Search branch
        x1, x2 = self.backbone(x)

        xf1 = self.detatt(x1)
        xf11 = self.attse(x1)
        x1 = xf1 + xf11 + x1

        xf2 = self.detatt(x2)
        xf22 = self.attse(x2)
        x2 = xf2 + xf22 + x2

        out1 = self.head(kernel1, self.proj1(x1))
        out2 = self.head(kernel2, self.proj2(x2))

        # Fusion strategy
        out11 = out1 + out2
        out22 = out1 * out2
        out = torch.cat([out11, out22], dim=1)
        out = self.conv(out)
        return out


class TrackerSiamFC(Tracker):
    def __init__(self, net_path=None, **kwargs):
        super(TrackerSiamFC, self).__init__('SiamFC', True)
        self.cfg = self.parse_args(**kwargs)

        # Setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # Setup model
        # VGG166 returns two feature branches.
        self.net = Net(
            backbone=backbones.vgg(),
            head=SiamFC(self.cfg.out_scale),
        )
        ops.init_weights(self.net)

        # Load ImageNet VGG16 weights into backbone BEFORE loading checkpoint.
        # This gives the backbone a strong starting point. The checkpoint
        #
        print('Initializing VGG166 backbone with ImageNet VGG16 weights...')
        load_imagenet_to_vgg166(self)
        print('Done.\n')

        # Load checkpoint if provided (overrides ImageNet weights for matching keys)
        if net_path is not None:
            print(f'Loading checkpoint: {net_path}')
            self.net.load_state_dict(torch.load(
                net_path,
                map_location=lambda storage, loc: storage),
                strict=False)
            print(f'Checkpoint loaded (strict=False — mismatched keys ignored).\n')
        self.net = self.net.to(self.device)

        # Template update variables
        self.frame_count = 0
        self.best_response = 0
        self.best_template = None
        self.current_template_z1 = None
        self.current_template_z2 = None
        self.response_history = []
        self.update_log = []  # Store update history

        # Phase 1: Best template memory for rollback
        self.best_kernel1 = None
        self.best_kernel2 = None
        self.low_response_streak = 0  # Count consecutive low-response frames
        self.rollback_cooldown = 0  # Prevent oscillating rollbacks

        # Setup criterion
        self.criterion = BalancedLoss()

        # Setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)

        # Setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)

    def parse_args(self, **kwargs):
        # Default parameters
        cfg = {
            # Basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,

            # Inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # Adaptive scale: ratio of current response vs history average
            'scale_trigger_threshold': 0.95,

            # Training parameters
            'epoch_num': 36,
            'batch_size': 8,
            'num_workers': 8,
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0,

            # Template update parameters
            'template_update_interval': 25,
            'template_learning_rate': 0.026,
            'quality_threshold': 0.012,
            'response_normalization_factor': 12.0,
            'show_update_conditions': False,

            # Advanced template update parameters
            'adaptive_update': True,
            'max_updates_per_sequence': 20,
            'min_quality_gain': 0.05,
            'emergency_update_threshold': 0.009,

            # Ablation study parameters for template update
            'disable_interval_condition': False,  # Disable interval condition
            'disable_confidence_condition': False,  # Disable confidence condition
            'disable_adaptive_condition': False,  # Disable adaptive condition
            'force_update_every_frame': False,  # Force update every frame
            'handle_occlusions': True, # set to true if you want to enable it
        }

        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)

    @torch.no_grad()
    def _update_template(self, img, max_response, scale_id):
        """Template update based on tracking quality"""
        self.frame_count += 1

        # Store response history
        self.response_history.append(max_response)
        if len(self.response_history) > 20:
            self.response_history.pop(0)

        # Calculate metrics for decision
        normalized_response = max_response / self.cfg.response_normalization_factor
        avg_response = np.mean(self.response_history) if self.response_history else max_response
        adaptive_threshold = avg_response * 0.7

        # Conditions for update
        if self.cfg.force_update_every_frame:
            # Force update every frame (ablation study)
            update_condition = True
            condition_interval = True
            condition_confidence = True
            condition_adaptive = True
        else:
            # Normal conditions with ablation options
            condition_interval = (self.frame_count % self.cfg.template_update_interval == 0)
            condition_confidence = (max_response > self.cfg.quality_threshold)
            condition_adaptive = (max_response > adaptive_threshold)

            # Apply ablation settings
            if self.cfg.disable_interval_condition:
                condition_interval = True  # Always true = condition disabled
            if self.cfg.disable_confidence_condition:
                condition_confidence = True  # Always true = condition disabled
            if self.cfg.disable_adaptive_condition:
                condition_adaptive = True  # Always true = condition disabled

            update_condition = (
                    condition_interval and
                    condition_confidence and
                    condition_adaptive
            )

        # Show conditions if requested
        if self.cfg.show_update_conditions:
            ablation_info = []
            if self.cfg.disable_interval_condition:
                ablation_info.append("interval_disabled")
            if self.cfg.disable_confidence_condition:
                ablation_info.append("confidence_disabled")
            if self.cfg.disable_adaptive_condition:
                ablation_info.append("adaptive_disabled")
            if self.cfg.force_update_every_frame:
                ablation_info.append("force_update")

            print(f"Frame {self.frame_count}: Response={max_response:.3f}, "
                  f"Normalized={normalized_response:.3f}, "
                  f"Avg={avg_response:.3f}, "
                  f"AdaptThresh={adaptive_threshold:.3f}, "
                  f"Interval={condition_interval}, "
                  f"Confidence={condition_confidence}, "
                  f"Adaptive={condition_adaptive}")
            if ablation_info:
                print(f"Ablation settings: {', '.join(ablation_info)}")

        if update_condition:
            try:
                # Extract new template
                z_new = ops.crop_and_resize(
                    img, self.center, self.z_sz,
                    out_size=self.cfg.exemplar_sz,
                    border_value=self.avg_color)

                z_new = torch.from_numpy(z_new).to(
                    self.device).permute(2, 0, 1).unsqueeze(0).float()

                # === OPTIMISATION: single backbone call, no gradient overhead ===
                z1_new, z2_new = self.net.backbone(z_new)

                zf1_new = self.net.tematt(z1_new)
                zf11_new = self.net.attse(z1_new)
                kernel1_new = self.net.proj1(zf1_new + zf11_new + z1_new)

                zf2_new = self.net.tematt(z2_new)
                zf22_new = self.net.attse(z2_new)
                kernel2_new = self.net.proj2(zf2_new + zf22_new + z2_new)
                # === END OPTIMISATION ===

                # === Phase 1 #3a: Adaptive learning rate ===
                # Higher response → more trust in new template → larger LR
                # Lower response → conservative update → smaller LR
                avg_hist = np.mean(self.response_history)
                lr_base = self.cfg.template_learning_rate
                lr = lr_base * np.clip(max_response / (avg_hist + 1e-8), 0.3, 1.5)

                self.kernel1 = (1 - lr) * self.kernel1 + lr * kernel1_new
                self.kernel2 = (1 - lr) * self.kernel2 + lr * kernel2_new

                # === Phase 1 #3b: Track best template for rollback ===
                if max_response > self.best_response:
                    self.best_response = max_response
                    self.best_kernel1 = self.kernel1.clone()
                    self.best_kernel2 = self.kernel2.clone()
                    self.low_response_streak = 0
                else:
                    self.low_response_streak = 0  # Reset on successful update

                # Log update
                update_info = {
                    'frame': self.frame_count,
                    'response': max_response,
                    'normalized_response': normalized_response,
                    'adaptive_threshold': adaptive_threshold,
                    'learning_rate': lr,
                    'lr_base': lr_base,
                    'ablation_type': 'normal'
                }
                if self.cfg.force_update_every_frame:
                    update_info['ablation_type'] = 'force_update'
                elif self.cfg.disable_interval_condition:
                    update_info['ablation_type'] = 'no_interval'
                elif self.cfg.disable_confidence_condition:
                    update_info['ablation_type'] = 'no_confidence'
                elif self.cfg.disable_adaptive_condition:
                    update_info['ablation_type'] = 'no_adaptive'

                self.update_log.append(update_info)

                print(f"✅ Template updated at frame {self.frame_count}, "
                      f"response: {max_response:.3f}, "
                      f"normalized: {normalized_response:.3f}, "
                      f"adaptive_threshold: {adaptive_threshold:.3f}, "
                      f"lr: {lr:.4f} (base: {lr_base:.4f})")

            except Exception as e:
                print(f"❌ Template update failed: {e}")
        else:
            if self.cfg.show_update_conditions and not self.cfg.force_update_every_frame:
                reasons = []
                if not condition_interval:
                    reasons.append(f"interval (need frame % {self.cfg.template_update_interval} == 0)")
                if not condition_confidence:
                    reasons.append(f"confidence (need > {self.cfg.quality_threshold:.3f})")
                if not condition_adaptive:
                    reasons.append(f"adaptive (need > {adaptive_threshold:.3f})")

                if reasons:
                    print(f"⏩ Skip update at frame {self.frame_count}: " + ", ".join(reasons))

    def _handle_occlusions(self, max_response):
        """Handle occlusions to avoid incorrect updates"""
        if len(self.response_history) > 5:
            avg_response = np.mean(self.response_history)
            occlusion_threshold = avg_response * 0.4 # 4.0 default value
        else:
            occlusion_threshold = 5.0 # 5.0 default value

        occlusion_detected = max_response < occlusion_threshold

        if occlusion_detected and self.cfg.show_update_conditions:
            print(f"🚫 Occlusion detected at frame {self.frame_count}, "
                  f"response: {max_response:.3f}, "
                  f"threshold: {occlusion_threshold:.3f}")

        return occlusion_detected

    def _compute_confidence(self, response, max_response):
        """Phase 1 #9: Multi-criteria confidence score.

        Combines Peak-to-Sidelobe Ratio (PSR) with peak sharpness to
        produce a more reliable confidence metric than raw max_response.
        A sharp peak with moderate value is more trustworthy than a broad
        peak with a high value (which often indicates background confusion).
        """
        loc = np.unravel_index(response.argmax(), response.shape)
        h, w = response.shape
        y, x = loc

        # 1. Peak-to-Sidelobe Ratio (PSR)
        #    Mask a 11x11 region around the peak, then compute
        #    (peak - sideload_mean) / sideload_std
        response_masked = response.copy()
        r_mask = 5
        y1, y2 = max(0, y - r_mask), min(h, y + r_mask + 1)
        x1, x2 = max(0, x - r_mask), min(w, x + r_mask + 1)
        response_masked[y1:y2, x1:x2] = 0
        psr = (max_response - response_masked.mean()) / \
              (response_masked.std() + 1e-8)

        # 2. Peak sharpness
        #    Ratio of peak value to mean of 5x5 region around it.
        #    Sharp peaks → high ratio → more reliable localization.
        y1s, y2s = max(0, y - 2), min(h, y + 3)
        x1s, x2s = max(0, x - 2), min(w, x + 3)
        peak_region = response[y1s:y2s, x1s:x2s]
        if peak_region.size > 0:
            sharpness = max_response / (peak_region.mean() + 1e-8)
        else:
            sharpness = 1.0

        return psr * sharpness

    def get_update_statistics(self):
        """Get template update statistics"""
        if not self.update_log:
            return "No template updates performed"

        responses = [log['response'] for log in self.update_log]
        frames = [log['frame'] for log in self.update_log]
        ablation_types = [log.get('ablation_type', 'normal') for log in self.update_log]

        # Count by ablation type
        from collections import Counter
        type_counter = Counter(ablation_types)

        stats = (f"Template Update Statistics:\n"
                 f"  Total updates: {len(self.update_log)}\n"
                 f"  Update frames: {frames}\n"
                 f"  By ablation type: {dict(type_counter)}\n"
                 f"  Average response: {np.mean(responses):.3f}\n"
                 f"  Minimum response: {np.min(responses):.3f}\n"
                 f"  Maximum response: {np.max(responses):.3f}")

        return stats

    @torch.no_grad()
    def init(self, img, box):
        """Initialize tracker with first frame"""
        self.net.eval()

        # Convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # Create Hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # Search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2,
            self.cfg.scale_num)

        # Exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
                    self.cfg.instance_sz / self.cfg.exemplar_sz

        # Exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)

        # Exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()

        # === OPTIMISATION: single backbone call, cache kernels at 128ch ===
        z1, z2 = self.net.backbone(z)

        zf1 = self.net.tematt(z1)
        zf11 = self.net.attse(z1)
        self.kernel1 = self.net.proj1(zf1 + zf11 + z1)

        zf2 = self.net.tematt(z2)
        zf22 = self.net.attse(z2)
        self.kernel2 = self.net.proj2(zf2 + zf22 + z2)
        # === END OPTIMISATION ===

        # Store initial template as best
        self.current_template_z1 = self.kernel1.clone()
        self.current_template_z2 = self.kernel2.clone()
        self.best_response = 1.0
        self.best_kernel1 = self.kernel1.clone()
        self.best_kernel2 = self.kernel2.clone()
        self.low_response_streak = 0
        self.rollback_cooldown = 0
        self.frame_count = 0
        self.response_history = [5.0]  # Initial value
        self.update_log = []

        if self.cfg.show_update_conditions:
            print(f"🎯 Initialization at frame 0 - Template ready for tracking")
            if self.cfg.force_update_every_frame:
                print(f"⚙️  Ablation mode: Force update every frame")
            elif self.cfg.disable_interval_condition:
                print(f"⚙️  Ablation mode: Interval condition disabled")
            elif self.cfg.disable_confidence_condition:
                print(f"⚙️  Ablation mode: Confidence condition disabled")
            elif self.cfg.disable_adaptive_condition:
                print(f"⚙️  Ablation mode: Adaptive condition disabled")

    def _extract_response(self, scale_factors):
        """Run backbone + head for a given list of scale factors.
        Returns (responses_np, scale_id, max_response).
        """
        x = [ops.crop_and_resize(
            self._img_buf, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in scale_factors]
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(self.device).permute(0, 3, 1, 2).float()

        x1, x2 = self.net.backbone(x)

        xf1 = self.net.detatt(x1)
        xf11 = self.net.attse(x1)
        x1 = xf1 + xf11 + x1

        xf2 = self.net.detatt(x2)
        xf22 = self.net.attse(x2)
        x2 = xf2 + xf22 + x2

        r1 = self.net.head(self.kernel1, self.net.proj1(x1))
        r2 = self.net.head(self.kernel2, self.net.proj2(x2))
        responses = self.net.conv(torch.cat([r1 + r2, r1 * r2], dim=1))
        responses = responses.squeeze(1).cpu().numpy()

        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])

        n = len(scale_factors)
        responses[:n // 2] *= self.cfg.scale_penalty
        responses[n // 2 + 1:] *= self.cfg.scale_penalty

        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))
        return responses, scale_id, responses[scale_id].max()

    @torch.no_grad()
    def update(self, img):
        """Update tracker with new frame.

        ADAPTIVE SCALE STRATEGY:
        - Fast path (1 scale)  : used when response is healthy.
          Cost = 1 backbone call. Covers ~95% of normal frames.
        - Full path (3 scales) : triggered when response drops, signaling
          a potential size change or occlusion exit.
          Cost = 3 backbone calls, same as original implementation.

        Gives ~3x speedup on easy frames while preserving scale robustness
        on hard frames where it matters most.
        """
        self.net.eval()
        self._img_buf = img  # shared buffer for _extract_response

        # ── FAST PATH: single scale ───────────────────────────────────────────
        responses_fast, _, max_resp_fast = self._extract_response([1.0])

        # === Phase 1 #9: PSR-based scale trigger ===
        # Also use confidence (PSR*sharpness) to decide if scale search is needed
        avg_hist = np.mean(self.response_history) if self.response_history else max_resp_fast
        use_fast = max_resp_fast >= avg_hist * self.cfg.scale_trigger_threshold
        # Additional check: if confidence history exists, verify PSR health
        if use_fast and hasattr(self, 'response_history_conf') and \
                len(self.response_history_conf) > 3:
            avg_conf = np.mean(self.response_history_conf)
            resp_conf = responses_fast[0].copy()
            resp_conf -= resp_conf.min()
            resp_conf /= (resp_conf.sum() + 1e-16)
            current_conf = self._compute_confidence(resp_conf, max_resp_fast)
            # If confidence dropped sharply, trigger full scale search
            if current_conf < avg_conf * 0.5:
                use_fast = False

        if use_fast:
            responses  = responses_fast
            scale_id   = 0
            scale_used = [1.0]
        else:
            # ── FULL PATH: 3 scales (response too weak) ───────────────────────
            responses, scale_id, _ = self._extract_response(self.scale_factors)
            scale_used = list(self.scale_factors)

        response     = responses[scale_id]
        max_response = response.max()

        # === Phase 1 #9: Compute multi-criteria confidence ===
        # Use PSR-based confidence for more robust decision-making
        response_for_conf = response.copy()
        response_for_conf -= response_for_conf.min()
        response_for_conf /= (response_for_conf.sum() + 1e-16)
        confidence = self._compute_confidence(response_for_conf, max_response)
        self.response_history_conf = getattr(
            self, 'response_history_conf', [])
        self.response_history_conf.append(confidence)
        if len(self.response_history_conf) > 20:
            self.response_history_conf.pop(0)

        # === Phase 1 #3b: Rollback to best template ===
        # If response drops below 40% of best seen, AND we have 3+ consecutive
        # low-response frames, rollback to the best saved template.
        if self.rollback_cooldown > 0:
            self.rollback_cooldown -= 1

        if (self.best_kernel1 is not None
                and self.best_response > 0
                and max_response < 0.4 * self.best_response):
            self.low_response_streak += 1
            if self.low_response_streak >= 3 and self.rollback_cooldown == 0:
                self.kernel1 = self.best_kernel1.clone()
                self.kernel2 = self.best_kernel2.clone()
                self.low_response_streak = 0
                self.rollback_cooldown = 10  # Don't rollback again for 10 frames
                if self.cfg.show_update_conditions:
                    print(f"🔄 ROLLBACK at frame {self.frame_count}: "
                          f"response {max_response:.3f} < "
                          f"0.4 * best {self.best_response:.3f}")
        else:
            self.low_response_streak = 0

        # Handle occlusions and update template
        if self.cfg.handle_occlusions:
            # === Phase 1 #9: Enhanced occlusion detection ===
            # Use both raw response AND confidence (PSR*sharpness)
            occluded_raw = self._handle_occlusions(max_response)
            # Also flag as occluded if PSR-confidence is very low
            avg_conf = np.mean(self.response_history_conf) if \
                self.response_history_conf else confidence
            occluded_conf = confidence < avg_conf * 0.3 and \
                len(self.response_history_conf) > 5
            occluded = occluded_raw or occluded_conf
        else:
            occluded = False

        if not occluded:
            self._update_template(img, max_response, scale_id)

        # Process response
        response -= response.min()
        response /= response.sum() + 1e-16

        # === Phase 1 #5: Dynamic Hann window ===
        # High confidence → reduce spatial penalty (trust the peak location)
        # Low confidence → increase spatial penalty (force re-centering)
        avg_hist = np.mean(self.response_history)
        confidence_ratio = max_response / (avg_hist + 1e-8)
        # Clamp dynamic influence between 0.5× and 1.5× base window_influence
        dynamic_influence = self.cfg.window_influence * np.clip(
            2.0 - confidence_ratio, 0.5, 1.5)
        response = (1 - dynamic_influence) * response + \
                   dynamic_influence * self.hann_window

        loc = np.unravel_index(response.argmax(), response.shape)

        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
                           self.cfg.total_stride / self.cfg.response_up
        disp_in_image    = disp_in_instance * self.x_sz * \
                           scale_used[scale_id] / self.cfg.instance_sz

        self.center += disp_in_image

        scale = (1 - self.cfg.scale_lr) * 1.0 + \
                self.cfg.scale_lr * scale_used[scale_id]
        self.target_sz *= scale
        self.z_sz      *= scale
        self.x_sz      *= scale

        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box

    def track(self, img_files, box, visualize=False):
        """Track object through image sequence"""
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, boxes[f, :])

        # Display final statistics
        if self.cfg.show_update_conditions:
            print("\n" + "=" * 50)
            print(self.get_update_statistics())
            print("=" * 50)

        return boxes, times

    def train_step(self, batch, backward=True):
        """Single training step"""
        self.net.train(backward)

        # Parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # Training path: kernels computed from z inside forward()
            responses = self.net(z, x, precomputed_kernels=None)

            # Calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)

            if backward:
                # Back propagation
                self.optimizer.zero_grad()
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

                loss.backward()
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                self.optimizer.step()

        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None, save_dir='models'):
        """Complete training procedure"""
        self.net.train()

        # Create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)

        dataset = Pair(seqs=seqs, transforms=transforms)

        # Setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)

        # Loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # Loop over dataloader
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()

            # Save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)

            self.lr_scheduler.step(epoch=epoch)

    def _freeze_backbone(self):
        """Freeze all VGG166 backbone parameters (conv1_1..conv5_1, bn, ECA).
        Only ECA attention, RichProjection, fusion head, and SiamFC head remain trainable."""
        for name, param in self.net.backbone.named_parameters():
            param.requires_grad = False
        # Also freeze the backbone-level ECA modules (eca1, eca2, eca3, eca4)
        # They're inside self.net.backbone.eca1/eca2/eca3/eca4
        total_frozen = sum(1 for _, p in self.net.backbone.named_parameters()
                          if not p.requires_grad)
        total_params = sum(1 for _ in self.net.backbone.parameters())
        print(f'  Frozen {total_frozen}/{total_params} backbone parameters')

    def _unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.net.backbone.parameters():
            param.requires_grad = True
        total_trainable = sum(p.numel() for p in self.net.backbone.parameters()
                             if p.requires_grad)
        print(f'  Unfroze backbone — {total_trainable:,} trainable params')

    def _set_backbone_lr(self, lr):
        """Set a specific learning rate for backbone parameters only.
        Returns (param_groups_config) for the optimizer."""
        # Collect param groups: backbone vs non-backbone
        backbone_params = []
        head_params = []
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('backbone.'):
                backbone_params.append(param)
            else:
                head_params.append(param)

        return [
            {'params': backbone_params, 'lr': lr},
            {'params': head_params, 'lr': lr * 10},  # head gets 10× higher LR
        ]

    @torch.enable_grad()
    def train_over_phased(self, seqs, save_dir='models',
                          phase1_epochs=15, phase2_epochs=25,
                          backbone_lr=1e-4, head_lr=1e-3):
        """Two-phase fine-tuning with progressive unfreezing.

        Phase 1 (freeze): Train only head layers (ECA, RichProj, fusion, SiamFC)
          on top of frozen ImageNet backbone weights.
          → Stabilizes the head before touching the backbone.

        Phase 2 (unfreeze): Fine-tune everything with differential LR
          (backbone gets 10× lower LR than head).
          → Adapts backbone features to tracking without destroying ImageNet knowledge.

        Args:
            seqs: GOT-10k training sequences
            save_dir: directory to save checkpoints
            phase1_epochs: epochs for Phase 1 (frozen backbone)
            phase2_epochs: epochs for Phase 2 (unfrozen backbone)
            backbone_lr: learning rate for backbone in Phase 2
            head_lr: learning rate for head in Phase 2
        """
        self.net.train()

        # Create save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Setup dataset & dataloader
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(seqs=seqs, transforms=transforms)
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)

        total_epochs = phase1_epochs + phase2_epochs

        # ============================================================
        # PHASE 1: Frozen backbone — train head only
        # ============================================================
        print('=' * 60)
        print(f'PHASE 1: Frozen backbone ({phase1_epochs} epochs)')
        print(f'  Head LR: {head_lr:.1e}')
        print('=' * 60)

        self._freeze_backbone()

        # Phase 1 optimizer: only trainable (non-backbone) params
        head_params = [p for n, p in self.net.named_parameters() if p.requires_grad]
        optimizer_p1 = optim.SGD(
            head_params,
            lr=head_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)
        scheduler_p1 = ExponentialLR(optimizer_p1, gamma=(
            (self.cfg.ultimate_lr / head_lr) ** (1.0 / phase1_epochs)))

        for epoch in range(phase1_epochs):
            # Loop over dataloader
            for it, batch in enumerate(dataloader):
                loss = self._train_step_with_opt(batch, optimizer_p1)
                print(f'[Phase1] Epoch: {epoch + 1}/{phase1_epochs} '
                      f'[{it + 1}/{len(dataloader)}] Loss: {loss:.5f}')
                sys.stdout.flush()

            # Save checkpoint
            net_path = os.path.join(
                save_dir, f'siamfc_phase1_e{epoch + 1}.pth')
            torch.save(self.net.state_dict(), net_path)
            scheduler_p1.step(epoch=epoch)

        phase1_best_path = os.path.join(save_dir, f'siamfc_phase1_e{phase1_epochs}.pth')
        print(f'\nPhase 1 complete. Best: {phase1_best_path}\n')

        # ============================================================
        # PHASE 2: Unfrozen backbone — fine-tune everything
        # ============================================================
        print('=' * 60)
        print(f'PHASE 2: Unfrozen backbone ({phase2_epochs} epochs)')
        print(f'  Backbone LR: {backbone_lr:.1e}')
        print(f'  Head LR:      {head_lr:.1e}')
        print('=' * 60)

        self._unfreeze_backbone()

        # Phase 2 optimizer: differential LR (backbone 10× lower)
        param_groups = self._set_backbone_lr(backbone_lr)
        optimizer_p2 = optim.SGD(
            param_groups,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay)
        scheduler_p2 = ExponentialLR(optimizer_p2, gamma=(
            (self.cfg.ultimate_lr / backbone_lr) ** (1.0 / phase2_epochs)))

        for epoch in range(phase2_epochs):
            for it, batch in enumerate(dataloader):
                loss = self._train_step_with_opt(batch, optimizer_p2)
                print(f'[Phase2] Epoch: {epoch + 1}/{phase2_epochs} '
                      f'[{it + 1}/{len(dataloader)}] Loss: {loss:.5f}')
                sys.stdout.flush()

            # Save checkpoint (numbered from phase1_epochs+1)
            net_path = os.path.join(
                save_dir, f'siamfc_alexnet_e{phase1_epochs + epoch + 1}.pth')
            torch.save(self.net.state_dict(), net_path)
            scheduler_p2.step(epoch=epoch)

        final_path = os.path.join(
            save_dir, f'siamfc_alexnet_e{total_epochs}.pth')
        print(f'\nTraining complete. Final: {final_path}')

        # Restore the standard optimizer for inference
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)

    def _train_step_with_opt(self, batch, optimizer):
        """Single training step with a given optimizer (for phased training)."""
        self.net.train()

        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        responses = self.net(z, x, precomputed_kernels=None)
        labels = self._create_labels(responses.size())
        loss = self.criterion(responses, labels)

        optimizer.zero_grad()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        loss.backward()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        optimizer.step()

        return loss.item()

    def _create_labels(self, size):
        """Create labels for training"""
        # Skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # Distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # Create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # Repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # Convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()

        return self.labels


# Utility functions for ablation experiments
def create_ablation_configurations():
    """Create different ablation study configurations"""
    configs = {
        # Baseline: all conditions enabled
        'baseline': {
            'disable_interval_condition': False,
            'disable_confidence_condition': False,
            'disable_adaptive_condition': False,
            'force_update_every_frame': False,
            'show_update_conditions': True,
        },

        # Ablation 1: No interval condition
        'no_interval': {
            'disable_interval_condition': True,
            'disable_confidence_condition': False,
            'disable_adaptive_condition': False,
            'force_update_every_frame': False,
            'show_update_conditions': True,
        },

        # Ablation 2: No confidence condition
        'no_confidence': {
            'disable_interval_condition': False,
            'disable_confidence_condition': True,
            'disable_adaptive_condition': False,
            'force_update_every_frame': False,
            'show_update_conditions': True,
        },

        # Ablation 3: No adaptive condition
        'no_adaptive': {
            'disable_interval_condition': False,
            'disable_confidence_condition': False,
            'disable_adaptive_condition': True,
            'force_update_every_frame': False,
            'show_update_conditions': True,
        },

        # Ablation 4: Force update every frame
        'force_update': {
            'disable_interval_condition': False,
            'disable_confidence_condition': False,
            'disable_adaptive_condition': False,
            'force_update_every_frame': True,
            'show_update_conditions': True,
        },

        # Ablation 5: Only interval condition
        'only_interval': {
            'disable_interval_condition': False,
            'disable_confidence_condition': True,
            'disable_adaptive_condition': True,
            'force_update_every_frame': False,
            'show_update_conditions': True,
        },

        # Ablation 6: Only confidence condition
        'only_confidence': {
            'disable_interval_condition': True,
            'disable_confidence_condition': False,
            'disable_adaptive_condition': True,
            'force_update_every_frame': False,
            'show_update_conditions': True,
        },

        # Ablation 7: Only adaptive condition
        'only_adaptive': {
            'disable_interval_condition': True,
            'disable_confidence_condition': True,
            'disable_adaptive_condition': False,
            'force_update_every_frame': False,
            'show_update_conditions': True,
        },
    }

    return configs


def run_ablation_experiment(sequence_path, initial_bbox, model_path, config_name, config):
    """Run a single ablation experiment"""
    print(f"\n{'=' * 60}")
    print(f"Running ablation experiment: {config_name}")
    print(f"Configuration: {config}")
    print(f"{'=' * 60}")

    # Create tracker with specific configuration
    tracker = TrackerSiamFC(
        net_path=model_path,
        **config  # Pass all configuration parameters
    )

    # Run tracking
    boxes, times = tracker.track(sequence_path, initial_bbox)

    # Get statistics
    stats = tracker.get_update_statistics()

    return {
        'config_name': config_name,
        'config': config,
        'boxes': boxes,
        'times': times,
        'statistics': stats,
        'update_log': tracker.update_log
    }


def compare_ablation_results(results):
    """Compare results from different ablation experiments"""
    print(f"\n{'=' * 80}")
    print("ABLATION STUDY RESULTS COMPARISON")
    print(f"{'=' * 80}")

    for result in results:
        print(f"\nExperiment: {result['config_name']}")
        print(f"Configuration: {result['config']}")
        print(f"Total updates: {len(result['update_log'])}")

        if result['update_log']:
            responses = [log['response'] for log in result['update_log']]
            print(f"Average response: {np.mean(responses):.3f}")
            print(f"Update frames: {[log['frame'] for log in result['update_log']]}")

        print("-" * 40)