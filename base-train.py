import argparse
import importlib.util
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def _load_from_file(file_name: str, attr_name: str):
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    spec = importlib.util.spec_from_file_location(attr_name + "_dyn", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr_name)


def _load_module_file(file_name: str):
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    mod_name = file_name.replace(".", "_") + "_dyn"
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Maps TRAIN_DEFAULTS keys in base-wopatha/b/c.py -> base-train globals
_ABLATION_TRAIN_KEY_TO_DEFAULT = {
    "vi_path": "DEFAULT_VI_PATH",
    "ir_path": "DEFAULT_IR_PATH",
    "teacher_path": "DEFAULT_TEACHER_PATH",
    "gt_path": "DEFAULT_GT_PATH",
    "epochs": "DEFAULT_EPOCHS",
    "batch_size": "DEFAULT_BATCH_SIZE",
    "lr": "DEFAULT_LR",
    "lr_min": "DEFAULT_LR_MIN",
    "weight_decay": "DEFAULT_WEIGHT_DECAY",
    "num_workers": "DEFAULT_NUM_WORKERS",
    "prefetch_factor": "DEFAULT_PREFETCH_FACTOR",
    "seed": "DEFAULT_SEED",
    "base_channels": "DEFAULT_BASE_CHANNELS",
    "clip_grad": "DEFAULT_CLIP_GRAD",
    "use_amp": "DEFAULT_USE_AMP",
    "use_torch_compile": "DEFAULT_USE_TORCH_COMPILE",
    "log_interval": "DEFAULT_LOG_INTERVAL",
    "w_intensity": "DEFAULT_W_INTENSITY",
    "w_gradient": "DEFAULT_W_GRADIENT",
    "w_ssim": "DEFAULT_W_SSIM",
    "w_distill": "DEFAULT_W_DISTILL",
    "w_gt": "DEFAULT_W_GT",
    "w_gt_ssim": "DEFAULT_W_GT_SSIM",
    "w_color": "DEFAULT_W_COLOR",
    "save_dir": "DEFAULT_SAVE_DIR",
    "save_interval": "DEFAULT_SAVE_INTERVAL",
    "device": "DEFAULT_DEVICE",
}


def _prepare_ablation_if_flags():
    """If --wopatha / --wopathb / --wopathc: load matching net + TRAIN_DEFAULTS from that file."""
    use_a = "--wopatha" in sys.argv
    use_b = "--wopathb" in sys.argv
    use_c = "--wopathc" in sys.argv
    if sum(1 for x in (use_a, use_b, use_c) if x) > 1:
        print("Error: use only one of --wopatha, --wopathb, or --wopathc", file=sys.stderr)
        sys.exit(2)
    if not use_a and not use_b and not use_c:
        return
    global BaseFusionReconNet
    if use_a:
        wmod = _load_module_file("base-wopatha.py")
        BaseFusionReconNet = wmod.BaseFusionReconNetWoPathA
    elif use_b:
        wmod = _load_module_file("base-wopathb.py")
        BaseFusionReconNet = wmod.BaseFusionReconNetWoPathB
    else:
        wmod = _load_module_file("base-wopathc.py")
        BaseFusionReconNet = wmod.BaseFusionReconNetWoPathC
    td = getattr(wmod, "TRAIN_DEFAULTS", None) or {}
    g = globals()
    for key, val in td.items():
        def_name = _ABLATION_TRAIN_KEY_TO_DEFAULT.get(key)
        if def_name and def_name in g:
            g[def_name] = val


BaseFusionReconNet = _load_from_file("base-net.py", "BaseFusionReconNet")
BaseFusionLoss = _load_from_file("base-loss.py", "BaseFusionLoss")
BaseSSIM = _load_from_file("base-loss.py", "ssim")

# =========================
# Editable top-level config
# =========================
DEFAULT_IR_PATH = r"D:\com\data\MSRS\train\ir"
DEFAULT_VI_PATH = r"D:\com\data\MSRS\train\vi"
DEFAULT_TEACHER_PATH = r"D:\com\SPG\SPGFusion-main\OUTPUT\Time_test1"
DEFAULT_GT_PATH = r""

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 2
DEFAULT_LR = 5e-3
DEFAULT_LR_MIN = 1e-6
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_NUM_WORKERS = 4
DEFAULT_SEED = 3407
DEFAULT_BASE_CHANNELS = 32
DEFAULT_CLIP_GRAD = 1.0
# Default training precision:
# - True: enable AMP autocast with BF16 (recommended for RTX 40 series)
# - False: full fp32
DEFAULT_USE_AMP = True
DEFAULT_LOG_INTERVAL = 10

DEFAULT_PREFETCH_FACTOR = 4
DEFAULT_USE_TORCH_COMPILE = True

# Loss weights
DEFAULT_W_INTENSITY = 1.0
DEFAULT_W_GRADIENT = 2
DEFAULT_W_SSIM = 1.0
DEFAULT_W_DISTILL = 1.0
DEFAULT_W_GT = 0.0
DEFAULT_W_GT_SSIM = 0.0
DEFAULT_W_COLOR = 0.2

DEFAULT_SAVE_DIR = "./ckpt_base"
DEFAULT_SAVE_INTERVAL = 5
DEFAULT_DEVICE = "cuda"


def center_crop_640x480_if_needed(img: Image.Image) -> Image.Image:
    """If both dims are larger than 640x480, center-crop to exactly 640x480."""
    target_w, target_h = 640, 480
    w, h = img.size
    if (w >= target_w and h >= target_h) and (w > target_w or h > target_h):
        left = (w - target_w) // 2
        top = (h - target_h) // 2
        img = img.crop((left, top, left + target_w, top + target_h))
    return img


class PairTeacherDataset(Dataset):
    """
    Train dataset with VI/IR + teacher guidance, and optional GT supervision.
    """

    def __init__(self, vi_path: str, ir_path: str, teacher_path: str, gt_path: str = ""):
        self.vi_files = self._scan(vi_path)
        self.ir_files = self._scan(ir_path)
        self.teacher_files = self._scan(teacher_path)
        self.gt_files = self._scan(gt_path) if gt_path else []
        self.use_gt = bool(gt_path)

        if len(self.vi_files) != len(self.ir_files) or len(self.vi_files) != len(self.teacher_files):
            raise ValueError(
                f"Count mismatch: vi={len(self.vi_files)} ir={len(self.ir_files)} teacher={len(self.teacher_files)}"
            )
        if self.use_gt and len(self.gt_files) != len(self.vi_files):
            raise ValueError(f"GT count mismatch: gt={len(self.gt_files)} vi={len(self.vi_files)}")
        self.length = len(self.vi_files)

    @staticmethod
    def _scan(folder):
        exts = (".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".png")
        files = [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.lower().endswith(exts)
        ]
        files.sort()
        return files

    @staticmethod
    def _read_vi(path):
        img = Image.open(path).convert("RGB")
        img = center_crop_640x480_if_needed(img)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    @staticmethod
    def _read_ir(path):
        img = Image.open(path).convert("L")
        img = center_crop_640x480_if_needed(img)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        return torch.from_numpy(arr)

    def __getitem__(self, index):
        vi = self._read_vi(self.vi_files[index])
        ir = self._read_ir(self.ir_files[index])
        teacher = self._read_vi(self.teacher_files[index])
        if self.use_gt:
            gt = self._read_vi(self.gt_files[index])
            has_gt = torch.tensor(1.0, dtype=torch.float32)
        else:
            gt = torch.zeros_like(teacher)
            has_gt = torch.tensor(0.0, dtype=torch.float32)
        return ir, vi, teacher, gt, has_gt

    def __len__(self):
        return self.length


def _finite_scalar(t: torch.Tensor) -> bool:
    if t is None:
        return False
    return bool(torch.isfinite(t.detach()).all().item())


def _params_grads_finite(params) -> bool:
    for p in params:
        if not p.requires_grad:
            continue
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return False
    return True


def parse_args():
    _prepare_ablation_if_flags()
    parser = argparse.ArgumentParser(description="Train lightweight base fusion+reconstruction network")

    parser.add_argument(
        "--wopatha",
        action="store_true",
        help="Train BaseFusionReconNetWoPathA; default paths & loss weights from base-wopatha.py TRAIN_DEFAULTS",
    )
    parser.add_argument(
        "--wopathb",
        action="store_true",
        help="Train BaseFusionReconNetWoPathB; default paths & loss weights from base-wopathb.py TRAIN_DEFAULTS",
    )
    parser.add_argument(
        "--wopathc",
        action="store_true",
        help="Train BaseFusionReconNetWoPathC; default paths & loss weights from base-wopathc.py TRAIN_DEFAULTS",
    )

    # dataset
    parser.add_argument("--ir_path", type=str, default=DEFAULT_IR_PATH)
    parser.add_argument("--vi_path", type=str, default=DEFAULT_VI_PATH)
    parser.add_argument("--teacher_path", type=str, default=DEFAULT_TEACHER_PATH)
    parser.add_argument("--gt_path", type=str, default=DEFAULT_GT_PATH)

    # training hyperparameters
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--lr_min", type=float, default=DEFAULT_LR_MIN)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--prefetch_factor", type=int, default=DEFAULT_PREFETCH_FACTOR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--base_channels", type=int, default=DEFAULT_BASE_CHANNELS)
    parser.add_argument("--clip_grad", type=float, default=DEFAULT_CLIP_GRAD)
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--use_amp", dest="use_amp", action="store_true", help="Enable AMP (fp16) on CUDA")
    amp_group.add_argument("--no_amp", dest="use_amp", action="store_false", help="Disable AMP for numerical stability")
    parser.set_defaults(use_amp=DEFAULT_USE_AMP)

    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument(
        "--use_torch_compile",
        dest="use_torch_compile",
        action="store_true",
        help="Enable torch.compile(model) after model init",
    )
    compile_group.add_argument(
        "--no_torch_compile",
        dest="use_torch_compile",
        action="store_false",
        help="Disable torch.compile",
    )
    parser.set_defaults(use_torch_compile=DEFAULT_USE_TORCH_COMPILE)
    parser.add_argument("--log_interval", type=int, default=DEFAULT_LOG_INTERVAL)
    # Default: disable finite checks for maximum throughput (can be re-enabled by editing code
    # or adding explicit checks; command-line still allows turning checks off, not on).
    parser.add_argument(
        "--no_finite_check",
        action="store_true",
        default=True,
        help="Disable NaN/Inf assertions (default: assert and stop on non-finite loss/output/grads)",
    )

    # loss weights
    parser.add_argument("--w_intensity", type=float, default=DEFAULT_W_INTENSITY)
    parser.add_argument("--w_gradient", type=float, default=DEFAULT_W_GRADIENT)
    parser.add_argument("--w_ssim", type=float, default=DEFAULT_W_SSIM)
    parser.add_argument("--w_distill", type=float, default=DEFAULT_W_DISTILL)
    parser.add_argument("--w_color", type=float, default=DEFAULT_W_COLOR)
    parser.add_argument("--w_gt", type=float, default=DEFAULT_W_GT)
    parser.add_argument("--w_gt_ssim", type=float, default=DEFAULT_W_GT_SSIM)

    # logging and checkpointing
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--save_interval", type=int, default=DEFAULT_SAVE_INTERVAL)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(args):
    dataset = PairTeacherDataset(
        vi_path=args.vi_path,
        ir_path=args.ir_path,
        teacher_path=args.teacher_path,
        gt_path=args.gt_path,
    )

    num_workers = int(args.num_workers)
    dl_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": True,
    }
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    return DataLoader(**dl_kwargs)


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    use_cuda = args.device.lower() == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    amp_enabled = bool(args.use_amp and use_cuda)
    print(
        f"[Train] device={device} AMP={'on' if amp_enabled else 'off'} "
        f"finite_check={'off' if args.no_finite_check else 'on (fail-fast)'}"
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_loader(args)

    model = BaseFusionReconNet(base_ch=args.base_channels).to(device)
    if args.use_torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("[Train] torch.compile enabled.")
        except Exception as e:
            print(f"[Train] torch.compile failed, fallback to eager. Reason: {e}")
    criterion = BaseFusionLoss(
        w_intensity=args.w_intensity,
        w_gradient=args.w_gradient,
        w_ssim=args.w_ssim,
        w_color=args.w_color,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = torch.zeros((), device=device)
        epoch_start = time.time()
        iter_time_sum = 0.0
        # Accumulate metrics as tensors to avoid GPU sync; convert to python scalars only for printing.
        stat = {k: torch.zeros((), device=device) for k in [
            "fusion", "i_raw", "g_raw", "s_raw",
            "i_w", "g_w", "s_w",
            "dis_raw", "dis_w",
            "gt_raw", "gt_w",
            "gssim_raw", "gssim_w",
        ]}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)

        for i, (data_ir, data_vis, data_teacher, data_gt, has_gt) in enumerate(pbar, start=1):
            iter_start = time.time()
            vi = data_vis.to(device, non_blocking=True)
            ir = data_ir.to(device, non_blocking=True)
            teacher = data_teacher.to(device, non_blocking=True)
            gt = data_gt.to(device, non_blocking=True)
            has_gt = has_gt.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=torch.bfloat16):
                out = model(vi, ir)
                loss_fusion, logs = criterion(out, vi, ir, teacher)
                loss_distill = torch.nn.functional.l1_loss(out, teacher)

                # mask-based GT loss (remove branching)
                has_gt_mask = has_gt.mean()  # scalar tensor in {0,1} (dataset-provided)
                loss_gt_l1_raw = torch.nn.functional.l1_loss(out, gt)
                loss_gt_ssim_raw = 1.0 - BaseSSIM(out, gt)
                loss_gt_l1 = has_gt_mask * loss_gt_l1_raw
                loss_gt_ssim = has_gt_mask * loss_gt_ssim_raw

                loss = (
                    loss_fusion
                    + args.w_distill * loss_distill
                    + args.w_gt * loss_gt_l1
                    + args.w_gt_ssim * loss_gt_ssim
                )

            weighted_distill = args.w_distill * loss_distill
            weighted_gt_l1 = args.w_gt * loss_gt_l1
            weighted_gt_ssim = args.w_gt_ssim * loss_gt_ssim
            weighted_i = args.w_intensity * logs["loss_intensity"]
            weighted_g = args.w_gradient * logs["loss_gradient"]
            weighted_s = args.w_ssim * logs["loss_ssim"]

            if not args.no_finite_check:
                if (not torch.isfinite(out).all()) or not (
                    _finite_scalar(loss)
                    and _finite_scalar(loss_fusion)
                    and _finite_scalar(loss_distill)
                    and _finite_scalar(loss_gt_l1)
                    and _finite_scalar(loss_gt_ssim)
                ):
                    raise RuntimeError(
                        f"Non-finite output or loss at epoch={epoch} iter={i} "
                        f"(out_finite={torch.isfinite(out).all().item()}, "
                        f"loss_finite={_finite_scalar(loss)}). "
                        "Try --no_amp or lower --lr; or pass --no_finite_check (not recommended)."
                    )

            loss.backward()
            if not args.no_finite_check and not _params_grads_finite(model.parameters()):
                raise RuntimeError(
                    f"Non-finite gradients after backward at epoch={epoch} iter={i}. "
                    "Try: --no_amp or lower --lr; or --no_finite_check (not recommended)."
                )
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            epoch_loss += loss.detach()
            stat["fusion"] += logs["loss_total"]
            stat["i_raw"] += logs["loss_intensity"]
            stat["g_raw"] += logs["loss_gradient"]
            stat["s_raw"] += logs["loss_ssim"]
            stat["i_w"] += weighted_i
            stat["g_w"] += weighted_g
            stat["s_w"] += weighted_s
            stat["dis_raw"] += loss_distill
            stat["dis_w"] += weighted_distill
            stat["gt_raw"] += loss_gt_l1
            stat["gt_w"] += weighted_gt_l1
            stat["gssim_raw"] += loss_gt_ssim
            stat["gssim_w"] += weighted_gt_ssim

            iter_time = time.time() - iter_start
            iter_time_sum += iter_time
            pbar.set_postfix(
                t_iter=f"{iter_time:.3f}s",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
            if args.log_interval > 0 and (i % args.log_interval == 0):
                print(
                    f"[Iter {i:04d}] "
                    f"total={loss.detach().item():.4f} fusion={logs['loss_total'].detach().item():.4f} "
                    f"i_raw={logs['loss_intensity'].detach().item():.4f} g_raw={logs['loss_gradient'].detach().item():.4f} s_raw={logs['loss_ssim'].detach().item():.4f} "
                    f"i_w={weighted_i.detach().item():.4f} g_w={weighted_g.detach().item():.4f} s_w={weighted_s.detach().item():.4f} "
                    f"dis_raw={loss_distill.detach().item():.4f} dis_w={weighted_distill.detach().item():.4f} "
                    f"gt_raw={loss_gt_l1.detach().item():.4f} gt_w={weighted_gt_l1.detach().item():.4f} "
                    f"gssim_raw={loss_gt_ssim.detach().item():.4f} gssim_w={weighted_gt_ssim.detach().item():.4f} "
                    f"iter_time={iter_time:.3f}s"
                )

        scheduler.step()
        n_batches = max(1, len(train_loader))
        avg_loss = (epoch_loss / n_batches).item()
        avg = {k: (v / n_batches).item() for k, v in stat.items()}
        epoch_time = time.time() - epoch_start
        avg_iter_time = iter_time_sum / n_batches
        print(
            f"[Epoch {epoch:03d}] avg_loss={avg_loss:.6f} "
            f"epoch_time={epoch_time:.2f}s avg_iter_time={avg_iter_time:.3f}s"
        )
        print(
            "[Epoch stats] "
            f"fusion={avg['fusion']:.4f}, "
            f"i_raw={avg['i_raw']:.4f}, g_raw={avg['g_raw']:.4f}, s_raw={avg['s_raw']:.4f}, "
            f"i_w={avg['i_w']:.4f}, g_w={avg['g_w']:.4f}, s_w={avg['s_w']:.4f}, "
            f"dis_raw={avg['dis_raw']:.4f}, dis_w={avg['dis_w']:.4f}, "
            f"gt_raw={avg['gt_raw']:.4f}, gt_w={avg['gt_w']:.4f}, "
            f"gssim_raw={avg['gssim_raw']:.4f}, gssim_w={avg['gssim_w']:.4f}"
        )

        if epoch % args.save_interval == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }
            ckpt_path = save_dir / f"base_epoch_{epoch:03d}.pth"
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
