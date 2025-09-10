# models/octf_backbone.py
import re
import torch
import torch.nn as nn
from typing import Dict
from .core_blocks_V2 import EncoderBlockV1  # your residual downsampling block

__all__ = ["OCTFBackbone", "load_octf_backbone_from_checkpoint"]


class OCTFBackbone(nn.Module):
    """
    OCT-F encoder-only backbone for DeepLabV3/V3+.

    Structure (matches ArchitectureV2Res_new encoder):
      layer1: EncoderBlockV1(3,   32, k=3, stride=2, padding=0)  -> stride 2
      layer2: EncoderBlockV1(32,  64, k=3, stride=2, padding=0)  -> stride 4  (low_level)
      layer3: EncoderBlockV1(64, 128, k=3, stride=2, padding=0)  -> stride 8
      layer4: EncoderBlockV1(128,256, k=3, stride=2, padding=0)  -> stride 16 (out)

    Forward returns:
      {"out": <B,256,H/16,W/16> or stride-8 if output_stride=8,
       "low_level": <B,64,H/4,W/4>}

    Notes
    -----
    * No transformer / decoder / random operations.
    * If input is 1-channel, we lift to 3 via a 1×1 conv initialized to
      approximate "repeat along channels", so your 3-ch weights remain usable.
    """
    def __init__(self, in_channels: int = 1, output_stride: int = 16):
        super().__init__()
        assert output_stride in (8, 16), "output_stride must be 8 or 16"
        self.output_stride = output_stride

        # Adapt grayscale (or N-ch) to the 3-ch encoder expected by your weights
        if in_channels == 3:
            self.input_adapter = nn.Identity()
        else:
            self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
            # initialize like "channel repeat" so pretrained 3ch weights still make sense
            with torch.no_grad():
                self.input_adapter.weight.zero_()
                # if in_channels==1, make all three output channels copy the single input
                for oc in range(min(3, self.input_adapter.out_channels)):
                    ic = 0 if in_channels == 1 else min(oc, in_channels - 1)
                    self.input_adapter.weight[oc, ic] = 1.0

        # ---- Encoder (identical blocks/ordering/channels to OCT-F encoder) ----
        self.layer1 = EncoderBlockV1(3,   32, kernel_size=3, stride=2, padding=1)  # /2
        self.layer2 = EncoderBlockV1(32,  64, kernel_size=3, stride=2, padding=1)  # /4 (low-level)
        self.layer3 = EncoderBlockV1(64, 128, kernel_size=3, stride=2, padding=1)  # /8

        if output_stride == 16:
            # normal downsample to /16
            self.layer4 = EncoderBlockV1(128, 256, kernel_size=3, stride=2, padding=1)  # /16 (out)
        else:
            # keep /8 and add a dilated refinement to reach 256ch without further downsampling
            self.layer4 = _DilatedRefine(in_ch=128, out_ch=256, dilation=2)

        # what decoders typically read
        self.out_channels = 256
        self.low_level_channels = 64

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.input_adapter(x)   # (B,3,H,W)

        x = self.layer1(x)          # (B, 32, H/2,  W/2)
        low = self.layer2(x)        # (B, 64, H/4,  W/4)  -> low-level for V3+
        x = self.layer3(low)        # (B,128, H/8,  W/8)

        out = self.layer4(x)        # (B,256, H/16,W/16) if OS=16, else (B,256, H/8,W/8)

        return {"out": out, "low_level": low}


class _DilatedRefine(nn.Module):
    """
    Simple stride-preserving refinement: 1x1 reduce, 3x3(dilated), 1x1 expand + residual.
    Lets us keep OS=8 while increasing channels to 256.
    """
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 2):
        super().__init__()
        mid = out_ch // 2
        pad = dilation
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn_p = nn.BatchNorm2d(out_ch)

        self.conv1 = nn.Conv2d(in_ch, mid, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, padding=pad, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_ch, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.bn_p(self.proj(x))
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        return self.relu(y + identity)


_PREFIX_RE = re.compile(r'^(?:base_model\.|model\.|module\.)?(?P<rest>.+)$')
_BLOCK_RE  = re.compile(r'^blocks\.(?P<idx>[0-3])\.(?P<rest>.+)$')


@torch.no_grad()
def load_octf_backbone_from_checkpoint(backbone: OCTFBackbone,
                                       checkpoint_path: str,
                                       strict: bool = False,
                                       verbose: bool = True) -> OCTFBackbone:
    """
    Load encoder weights from an OCT-F checkpoint into this backbone.

    Accepts either:
      * a raw state_dict (keys like 'blocks.0.conv1.weight', ...), or
      * a dict with 'model_state_dict'.

    We map:
      blocks.0.* -> layer1.*
      blocks.1.* -> layer2.*
      blocks.2.* -> layer3.*
      blocks.3.* -> layer4.*    (only if OS=16 and layer4 is EncoderBlockV1)
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)

    own = backbone.state_dict()

    # Build a remapped copy
    remapped = {}

    for k, v in state.items():
        m = _PREFIX_RE.match(k)
        if not m:
            continue
        k2 = m.group("rest")

        m2 = _BLOCK_RE.match(k2)
        if m2:
            idx = int(m2.group("idx"))
            rest = m2.group("rest")
            layer_name = f"layer{idx+1}"
            # If OS=8, we only load layer1..3 strictly; layer4 is a different module
            if backbone.output_stride == 8 and layer_name == "layer4":
                # skip: layer4 here is a dilated refine block, not the original EncoderBlockV1
                continue

            new_key = f"{layer_name}.{rest}"
            if new_key in own and own[new_key].shape == v.shape:
                remapped[new_key] = v
            continue

        if k.startswith("input_adapter.") and k in own and own[k].shape == v.shape:
            remapped[k] = v

    # report
    if verbose:
        print(f"[OCTFBackbone] matched {len(remapped)}/{len(own)} keys from {checkpoint_path}")

    # merge + load
    merged = {**own, **remapped}
    backbone.load_state_dict(merged, strict=strict)
    return backbone
