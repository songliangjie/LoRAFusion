import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = nn.InstanceNorm2d(channels, affine=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class SobelGrad(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x):
        # x: [B, C, H, W]
        if x.shape[1] > 1:
            x = torch.mean(x, dim=1, keepdim=True)
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.abs(gx) + torch.abs(gy)


class HaarWaveletHF(nn.Module):
    """
    1-level Haar wavelet high-frequency (LH/HL/HH magnitude) approximation.
    Output is upsampled back to input spatial size so it can be concatenated
    with feature branches inside CrossGatedFusionBlock.
    """

    def __init__(self):
        super().__init__()

        # Haar scaling: s = 1/sqrt(2)
        import math

        s = 1.0 / math.sqrt(2.0)
        lp = torch.tensor([s, s], dtype=torch.float32)   # low-pass
        hp = torch.tensor([-s, s], dtype=torch.float32)  # high-pass

        # 2D separable kernels for LH/HL/HH using outer products
        # LH: low-x, high-y  -> outer(lp, hp)
        # HL: high-x, low-y  -> outer(hp, lp)
        # HH: high-x, high-y -> outer(hp, hp)
        k_LH = torch.ger(lp, hp)
        k_HL = torch.ger(hp, lp)
        k_HH = torch.ger(hp, hp)

        self.register_buffer("k_LH", k_LH.view(1, 1, 2, 2))
        self.register_buffer("k_HL", k_HL.view(1, 1, 2, 2))
        self.register_buffer("k_HH", k_HH.view(1, 1, 2, 2))

    def forward(self, x):
        # x: [B, C, H, W]
        if x.shape[1] > 1:
            x = torch.mean(x, dim=1, keepdim=True)  # -> [B,1,H,W]

        # Make sure kernels match dtype/device under AMP.
        k_LH = self.k_LH.to(device=x.device, dtype=x.dtype)
        k_HL = self.k_HL.to(device=x.device, dtype=x.dtype)
        k_HH = self.k_HH.to(device=x.device, dtype=x.dtype)

        # DWT downsample by factor 2 using stride=2 conv
        LH = F.conv2d(x, k_LH, stride=2, padding=0)
        HL = F.conv2d(x, k_HL, stride=2, padding=0)
        HH = F.conv2d(x, k_HH, stride=2, padding=0)

        hf = torch.abs(LH) + torch.abs(HL) + torch.abs(HH)  # [B,1,H/2,W/2]

        # Upsample to match input resolution for concat downstream
        hf_up = F.interpolate(hf, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return hf_up


class CrossGatedFusionBlock(nn.Module):
    """
    Moderate-complexity fusion block:
    - gated fusion
    - lightweight cross-attention-like modulation
    - high-frequency enhancement
    """

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

        # -------- Path B: DA-BACA (Degradation-Aware Bidirectional Additive Cross-Attention) --------
        self.channels = channels
        # Learnable global weighting vectors w_vi / w_ir in R^C
        self.w_vi = nn.Parameter(torch.randn(channels) * 0.02)
        self.w_ir = nn.Parameter(torch.randn(channels) * 0.02)

        # Query projections
        self.q_proj_vi = nn.Conv2d(channels, channels, kernel_size=1, bias=False)  # Q_vi
        self.q_proj_ir = nn.Conv2d(channels, channels, kernel_size=1, bias=False)  # Q_ir
        # Key projections
        self.k_proj_vi = nn.Conv2d(channels, channels, kernel_size=1, bias=False)  # K_vi
        self.k_proj_ir = nn.Conv2d(channels, channels, kernel_size=1, bias=False)  # K_ir

        # Elementwise interaction then 1x1 projection
        self.interact_conv_ir = nn.Conv2d(channels, channels, kernel_size=1, bias=False)  # for hat{x}_{ir}
        self.interact_conv_vi = nn.Conv2d(channels, channels, kernel_size=1, bias=False)  # for hat{x}_{vi}

        # Environment-Aware Scaling: tau in [0,1] using tiny MLP
        # Input stats: [std, mean] computed from global spatial statistics of features.
        self.tau_mlp = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

        self.hf_refine = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.SiLU(inplace=True),
        )
        # Path C: high-frequency enhancement (Sobel->Haar wavelet HF)
        self.grad = HaarWaveletHF()

        self.merge = nn.Conv2d(channels * 3, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.eca = ECA(channels)
        self.refine = DepthwiseSeparableConv(channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def _environment_aware_tau(self, feat):
        """
        Compute tau in [0,1] from global spatial std (contrast) and mean (brightness).
        feat: [B, C, H, W]
        return: [B, 1, 1, 1]
        """
        # Reduce spatial dims first; then average across channels to get per-sample scalars.
        std_spatial = feat.std(dim=(2, 3), unbiased=False)  # [B, C]
        std = std_spatial.mean(dim=1, keepdim=True)  # [B, 1]
        mean_spatial = feat.mean(dim=(2, 3))  # [B, C]
        mean = mean_spatial.mean(dim=1, keepdim=True)  # [B, 1]
        stats = torch.cat([std, mean], dim=1)  # [B, 2]
        tau = self.tau_mlp(stats)  # [B, 1]
        return tau.view(-1, 1, 1, 1)

    def _additive_global_query(self, Q, w, tau):
        """
        Additive global attention (no O(N^2) matmul):
        - score = sum_c (Q_c * w_c) => [B, 1, H, W]
        - alpha = softmax(score over spatial) => [B, 1, H*W]
        - q = tau * sum_{spatial} (alpha * Q) => [B, C, 1, 1]
        Q: [B, C, H, W], w: [C], tau: [B, 1, 1, 1]
        """
        b, c, h, w_spatial = Q.shape
        # Score map: [B,1,H,W]
        score = (Q * w.view(1, c, 1, 1)).sum(dim=1, keepdim=True)
        n = h * w_spatial
        # Softmax over spatial positions only (alpha sums to 1 per sample).
        # Numerical stability under low precision: scale by sqrt(C) and clamp logits.
        logits = score.view(b, 1, n) / (c ** 0.5)
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        alpha = F.softmax(logits, dim=-1)  # [B,1,N]
        Q_flat = Q.view(b, c, n)  # [B,C,N]
        q = (alpha * Q_flat).sum(dim=-1).view(b, c, 1, 1)  # [B,C,1,1]
        return tau * q

    def forward(self, feat_vi, feat_ir, vi_img, ir_img):
        # Path A: gated fusion
        alpha = self.gate(torch.cat([feat_vi, feat_ir], dim=1))
        f_g = alpha * feat_vi + (1.0 - alpha) * feat_ir

        # Path B: DA-BACA (bidirectional additive cross-attention with degradation-aware tau)
        tau_vi = self._environment_aware_tau(feat_vi)  # [B,1,1,1]
        tau_ir = self._environment_aware_tau(feat_ir)  # [B,1,1,1]

        # Path 1: VI -> IR
        Q_vi = self.q_proj_vi(feat_vi)  # [B,C,H,W]
        q_vi = self._additive_global_query(Q_vi, self.w_vi, tau_vi)  # [B,C,1,1]
        K_ir = self.k_proj_ir(feat_ir)  # [B,C,H,W]
        # hat{x}_{ir} = Conv( K_ir ⊙ q_vi ) + Q_vi
        x_hat_ir = self.interact_conv_ir(K_ir * q_vi) + Q_vi

        # Path 2: IR -> VI
        Q_ir = self.q_proj_ir(feat_ir)  # [B,C,H,W]
        q_ir = self._additive_global_query(Q_ir, self.w_ir, tau_ir)  # [B,C,1,1]
        K_vi = self.k_proj_vi(feat_vi)  # [B,C,H,W]
        # hat{x}_{vi} = Conv( K_vi ⊙ q_ir ) + Q_ir
        x_hat_vi = self.interact_conv_vi(K_vi * q_ir) + Q_ir

        f_attn = x_hat_ir + x_hat_vi

        # Path C: high-frequency enhancement from raw inputs
        hf_map = self.grad(vi_img) + self.grad(ir_img)
        f_hf = self.hf_refine(hf_map)

        out = torch.cat([f_g, f_attn, f_hf], dim=1)
        out = self.merge(out)
        out = self.eca(out)
        out = self.refine(out)
        return f_g + self.gamma * out


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch)
        self.conv2 = ConvBNAct(out_ch, out_ch)
        self.down = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNAct(in_ch, out_ch),
        )
        self.conv1 = ConvBNAct(out_ch + skip_ch, out_ch)
        self.conv2 = ConvBNAct(out_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
