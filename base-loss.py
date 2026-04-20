import torch
import torch.nn as nn
import torch.nn.functional as F

from utils22.loss_ssim import ssim


class SobelGrad(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x):
        if x.shape[1] > 1:
            x = torch.mean(x, dim=1, keepdim=True)
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.abs(gx) + torch.abs(gy)


class BaseFusionLoss(nn.Module):
    """
    Training loss = weighted sum of:
    - intensity loss
    - gradient loss
    - SSIM loss (first branch: teacher luminance if teacher_rgb is passed, else VI)
    """

    def __init__(self, w_intensity=1.0, w_gradient=0.5, w_ssim=0.5, w_color=0.1):
        super().__init__()
        self.w_intensity = w_intensity
        self.w_gradient = w_gradient
        self.w_ssim = w_ssim
        self.w_color = w_color
        self.grad = SobelGrad()

    @staticmethod
    def _rgb_to_ycbcr(x_rgb):
        """
        Convert RGB to YCbCr (Cb/Cr used for color loss).
        x_rgb: [B, 3, H, W] in float range [0,1]
        return: [B, 3, H, W] (Y, Cb, Cr)
        """
        r = x_rgb[:, 0:1, :, :]
        g = x_rgb[:, 1:2, :, :]
        b = x_rgb[:, 2:3, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b
        return torch.cat([y, cb, cr], dim=1)

    def forward(self, out_rgb, vi_rgb, ir_gray, teacher_rgb=None):
        out_y = torch.mean(out_rgb, dim=1, keepdim=True)
        vi_y = torch.mean(vi_rgb, dim=1, keepdim=True)

        # Intensity target: keep strongest response from both modalities
        target_intensity = torch.max(vi_y, ir_gray)
        loss_intensity = F.l1_loss(out_y, target_intensity)

        # Gradient target: preserve strongest edges from both modalities
        grad_out = self.grad(out_y)
        grad_vi = self.grad(vi_y)
        grad_ir = self.grad(ir_gray)
        target_grad = torch.max(grad_vi, grad_ir)
        loss_gradient = F.l1_loss(grad_out, target_grad)

        # Structure similarity: first term uses teacher luminance if given, else VI (same as before).
        ref_y = torch.mean(teacher_rgb, dim=1, keepdim=True) if teacher_rgb is not None else vi_y
        ssim_ref = ssim(out_y, ref_y)
        ssim_ir = ssim(out_y, ir_gray)
        loss_ssim = 1.0 - 0.5 * (ssim_ref + ssim_ir)

        # Color loss: match Cb/Cr chroma channels of fused RGB to visible RGB.
        # (IR doesn't have color, so we only supervise against VI.)
        ycbcr_out = self._rgb_to_ycbcr(out_rgb)
        ycbcr_vi = self._rgb_to_ycbcr(vi_rgb)
        cb_out, cr_out = ycbcr_out[:, 1:2, :, :], ycbcr_out[:, 2:3, :, :]
        cb_vi, cr_vi = ycbcr_vi[:, 1:2, :, :], ycbcr_vi[:, 2:3, :, :]
        loss_color = F.l1_loss(cb_out, cb_vi) + F.l1_loss(cr_out, cr_vi)

        total = (
            self.w_intensity * loss_intensity
            + self.w_gradient * loss_gradient
            + self.w_ssim * loss_ssim
            + self.w_color * loss_color
        )

        logs = {
            "loss_total": total.detach(),
            "loss_intensity": loss_intensity.detach(),
            "loss_gradient": loss_gradient.detach(),
            "loss_ssim": loss_ssim.detach(),
            "loss_color": loss_color.detach(),
        }
        return total, logs
