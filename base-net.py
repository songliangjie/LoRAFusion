import importlib.util
import os

import torch
import torch.nn as nn


_MODULE_PATH = os.path.join(os.path.dirname(__file__), "base-module.py")
_SPEC = importlib.util.spec_from_file_location("base_module_dynamic", _MODULE_PATH)
_BASE_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE_MODULE)

ConvBNAct = _BASE_MODULE.ConvBNAct
CrossGatedFusionBlock = _BASE_MODULE.CrossGatedFusionBlock
DownBlock = _BASE_MODULE.DownBlock
UpBlock = _BASE_MODULE.UpBlock


class BaseFusionReconNet(nn.Module):
    """
    Base model for IVIF:
    1) Dual-branch shallow encoder for VI/IR
    2) Multi-scale cross-gated fusion
    3) Lightweight U-Net reconstruction head
    """

    def __init__(self, base_ch=32):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4

        # Modality-specific stems
        self.vi_stem = nn.Sequential(ConvBNAct(3, c1), ConvBNAct(c1, c1))
        self.ir_stem = nn.Sequential(ConvBNAct(1, c1), ConvBNAct(c1, c1))

        # Multi-scale fusion encoder
        self.fuse1 = CrossGatedFusionBlock(c1)
        self.down1 = DownBlock(c1, c2)
        self.fuse2 = CrossGatedFusionBlock(c2)
        self.down2 = DownBlock(c2, c3)
        self.fuse3 = CrossGatedFusionBlock(c3)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBNAct(c3, c3),
            ConvBNAct(c3, c3),
        )

        # U-Net-like reconstruction
        # skip2 comes from down2 output before stride-2 conv -> channels = c3
        self.up2 = UpBlock(c3, c3, c2)
        # skip1 comes from down1 output before stride-2 conv -> channels = c2
        self.up1 = UpBlock(c2, c2, c1)
        self.head = nn.Conv2d(c1, 3, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def _resize_like(x, ref):
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return nn.functional.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, vi, ir, return_features=False):
        # modality stems
        f_vi = self.vi_stem(vi)  # [B, c1, H, W]
        f_ir = self.ir_stem(ir)  # [B, c1, H, W]

        # scale-1 fusion
        f1 = self.fuse1(f_vi, f_ir, vi, ir)

        # downsample + scale-2 fusion
        x2, skip1 = self.down1(f1)
        # Instead of re-injecting resized raw VI/IR into fuse2,
        # use current-scale features for the HF/Sobel path.
        f2 = self.fuse2(x2, x2, x2, x2)

        # downsample + scale-3 fusion
        x3, skip2 = self.down2(f2)
        # Same strategy as fuse2: remove resized raw injection.
        f3 = self.fuse3(x3, x3, x3, x3)

        # reconstruction
        z = self.bottleneck(f3)
        z = self.up2(z, skip2)
        z = self.up1(z, skip1)
        residual = self.head(z)

        # Remove the VI residual skip connection.
        # The network predicts restoration/fusion result directly.
        out = torch.clamp(residual, 0.0, 1.0)
        if return_features:
            return out, z
        return out


if __name__ == "__main__":
    model = BaseFusionReconNet(base_ch=32)
    vi = torch.rand(1, 3, 256, 256)
    ir = torch.rand(1, 1, 256, 256)
    y = model(vi, ir)
    print("Output shape:", y.shape)
