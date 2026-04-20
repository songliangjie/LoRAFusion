"""
fusion_test_ckpt_final.py — 使用 ckpt-final 下的 base 与 LoRA 做批量融合推理。

两种模式:
  --mode base          仅加载 base.pth（全量权重）
  --mode base+lora     先加载 base，再加载你指定的 LoRA（与 base 配套）

LoRA 文件格式（自动识别）:
  - lora_state + lora_targets: detect / haze / rain / seg / text 等（与 lora-train.py 一致）
  - trainable_state: light.pth（与 lora-light.py 一致，含 head 等）

用法示例:
  python fusion_test_ckpt_final.py --mode base --vi_path ... --ir_path ... --save_path ...
  python fusion_test_ckpt_final.py --mode base+lora --lora light --vi_path ... --ir_path ... --save_path ...
  python fusion_test_ckpt_final.py --mode base+lora --lora_ckpt ckpt-final/text.pth ...
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm


def _load_from_file(file_name: str, attr_name: str):
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    spec = importlib.util.spec_from_file_location(attr_name + "_dyn", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr_name)


BaseFusionReconNet = _load_from_file("base-net.py", "BaseFusionReconNet")

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_CKPT_DIR = _SCRIPT_DIR / "ckpt-final"
_DEFAULT_BASE = _DEFAULT_CKPT_DIR / "base.pth"

DEFAULT_VI_PATH = ""
DEFAULT_IR_PATH = ""
DEFAULT_SAVE_PATH = str(_SCRIPT_DIR / "output_fusion_test")
DEFAULT_BASE_CHANNELS = 32
DEFAULT_DEVICE = "cuda"
DEFAULT_USE_AMP = True
DEFAULT_USE_TORCH_COMPILE = False

# --- Standard LoRA (lora-train.py) ---


class LoRAConv2dStandard(nn.Module):
    def __init__(self, conv: nn.Conv2d, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.conv = conv
        self.rank = rank
        self.scale = alpha / float(rank)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Conv2d(
            conv.in_channels,
            rank,
            kernel_size=1,
            stride=conv.stride,
            padding=0,
            groups=1,
            bias=False,
        )
        self.lora_b = nn.Conv2d(rank, conv.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for p in self.conv.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.conv(x) + self.scale * self.lora_b(self.lora_a(self.drop(x)))


def replace_module(root: nn.Module, module_name: str, new_module: nn.Module):
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def inject_lora_standard(model: nn.Module, rank: int, alpha: float, dropout: float, target_keywords: list[str]):
    wrapped = []
    use_all = any(k.lower() in {"all", "*"} for k in target_keywords)
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Conv2d):
            continue
        if (not use_all) and (not any(k in name for k in target_keywords)):
            continue
        if isinstance(module, LoRAConv2dStandard):
            continue
        lora_module = LoRAConv2dStandard(module, rank, alpha, dropout)
        lora_module = lora_module.to(device=module.weight.device, dtype=module.weight.dtype)
        replace_module(model, name, lora_module)
        wrapped.append(name)
    return wrapped


# --- Light LoRA (lora-light.py) ---


class LoRALinearLight(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.ref = linear
        for p in self.ref.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scale = alpha / float(rank)
        self.layer_alpha = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.lora_a = nn.Linear(linear.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        out = self.ref(x)
        xa = x.float()
        with torch.cuda.amp.autocast(enabled=False):
            delta = self.lora_b(self.lora_a(xa)) * self.scale * self.layer_alpha
        return out + delta.to(dtype=out.dtype)


class LoRAConv2dLight(nn.Module):
    def __init__(self, conv: nn.Conv2d, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.conv = conv
        self.rank = rank
        self.scale = alpha / float(rank)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.layer_alpha = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.lora_a = nn.Conv2d(
            conv.in_channels, rank, kernel_size=1, stride=conv.stride, padding=0, groups=1, bias=False
        )
        self.lora_b = nn.Conv2d(rank, conv.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for p in self.conv.parameters():
            p.requires_grad = False

    def forward(self, x):
        y = self.conv(x)
        z = self.drop(x).float()
        with torch.cuda.amp.autocast(enabled=False):
            delta = self.lora_b(self.lora_a(z)) * self.scale * self.layer_alpha
        return y + delta.to(dtype=y.dtype)


def inject_lora_vi_stem_light(model: nn.Module, rank: int, alpha: float, dropout: float):
    wrapped = []
    for name, module in list(model.vi_stem.named_modules()):
        if not isinstance(module, nn.Conv2d) or isinstance(module, LoRAConv2dLight):
            continue
        full_name = f"vi_stem.{name}" if name else "vi_stem"
        lora_m = LoRAConv2dLight(module, rank, alpha, dropout)
        lora_m = lora_m.to(device=module.weight.device, dtype=module.weight.dtype)
        replace_module(model, full_name, lora_m)
        wrapped.append(full_name)
    return wrapped


def inject_lora_tau_mlp_light(model: nn.Module, rank: int, alpha: float):
    wrapped = []
    for fuse_name in ("fuse1", "fuse2", "fuse3"):
        block = getattr(model, fuse_name)
        mlp = block.tau_mlp
        new_layers = []
        for i, layer in enumerate(mlp):
            if isinstance(layer, nn.Linear):
                ll = LoRALinearLight(layer, rank, alpha)
                dev = next(layer.parameters()).device
                ll = ll.to(device=dev)
                new_layers.append(ll)
                wrapped.append(f"{fuse_name}.tau_mlp.{i}")
            else:
                new_layers.append(layer)
        block.tau_mlp = nn.Sequential(*new_layers)
    return wrapped


def inject_lora_da_baca_qkv_light(model: nn.Module, rank: int, alpha: float, dropout: float):
    wrapped = []
    for fuse_name in ("fuse1", "fuse2", "fuse3"):
        block = getattr(model, fuse_name)
        for attr in ("q_proj_vi", "q_proj_ir", "k_proj_vi", "k_proj_ir", "interact_conv_ir", "interact_conv_vi"):
            m = getattr(block, attr)
            if isinstance(m, LoRAConv2dLight):
                continue
            lora_m = LoRAConv2dLight(m, rank, alpha, dropout)
            lora_m = lora_m.to(device=m.weight.device, dtype=m.weight.dtype)
            setattr(block, attr, lora_m)
            wrapped.append(f"{fuse_name}.{attr}")
    return wrapped


def _normalize_lora_targets(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [x.strip() for x in raw.split(",") if x.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [str(raw).strip()]


def _resolve_lora_path(ckpt_dir: Path, lora_arg: str | None, lora_ckpt: str | None) -> Path:
    if lora_ckpt:
        p = Path(lora_ckpt)
        if p.is_file():
            return p
        raise FileNotFoundError(f"--lora_ckpt not found: {lora_ckpt}")
    if not lora_arg:
        raise ValueError("base+lora 模式需要 --lora 或 --lora_ckpt")
    stem = lora_arg.strip().lower()
    if stem.endswith(".pth"):
        stem = stem[:-4]
    cand = ckpt_dir / f"{stem}.pth"
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"LoRA 文件不存在: {cand}（可改用 --lora_ckpt 指定完整路径）")


def load_base_state_dict(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt
    raise KeyError(f"{ckpt_path} 中未找到 'model' 键，无法作为 base 权重加载")


def build_model_base_only(base_ckpt: Path, base_channels: int, device: torch.device, use_compile: bool):
    model = BaseFusionReconNet(base_ch=base_channels).to(device)
    state = load_base_state_dict(base_ckpt, device)
    model.load_state_dict(state, strict=True)
    model.eval()
    if use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"[Test] torch.compile skipped: {e}")
    return model


def build_model_base_plus_lora(
    base_ckpt: Path,
    lora_path: Path,
    base_channels: int,
    device: torch.device,
    use_compile: bool,
):
    model = BaseFusionReconNet(base_ch=base_channels).to(device)
    base_state = load_base_state_dict(base_ckpt, device)
    model.load_state_dict(base_state, strict=True)

    lora_obj = torch.load(str(lora_path), map_location=device)
    if not isinstance(lora_obj, dict):
        raise TypeError(f"LoRA 文件顶层应为 dict: {lora_path}")

    # ----- light: trainable_state -----
    if "trainable_state" in lora_obj:
        args_saved = lora_obj.get("args") or {}
        rank = int(args_saved.get("lora_rank", lora_obj.get("lora_rank", 32)))
        alpha = float(args_saved.get("lora_alpha", lora_obj.get("lora_alpha", 24.0)))
        dropout = float(args_saved.get("lora_dropout", lora_obj.get("lora_dropout", 0.0)))
        inject_lora_vi_stem_light(model, rank, alpha, dropout)
        inject_lora_tau_mlp_light(model, rank, alpha)
        inject_lora_da_baca_qkv_light(model, rank, alpha, dropout)
        ts = lora_obj["trainable_state"]
        missing, unexpected = model.load_state_dict(ts, strict=False)
        if unexpected:
            raise RuntimeError(f"trainable_state 含未知键: {unexpected[:24]}")
        print(f"[Test] LoRA(light): {lora_path.name} rank={rank} alpha={alpha} (partial load, missing_base_keys={len(missing)})")
    elif "lora_state" in lora_obj:
        rank = int(lora_obj.get("lora_rank", 32))
        alpha = float(lora_obj.get("lora_alpha", 16.0))
        dropout = float(lora_obj.get("lora_dropout", 0.0))
        targets = _normalize_lora_targets(lora_obj.get("lora_targets"))
        if not targets:
            args_saved = lora_obj.get("args") or {}
            targets = _normalize_lora_targets(args_saved.get("lora_targets"))
        if not targets:
            raise KeyError(f"{lora_path} 缺少 lora_targets（且 args 中也没有）")
        wrapped = inject_lora_standard(model, rank, alpha, dropout, targets)
        if len(wrapped) == 0:
            raise RuntimeError("inject_lora_standard 未匹配到任何 Conv2d，请检查 lora_targets")
        ls = lora_obj["lora_state"]
        missing, unexpected = model.load_state_dict(ls, strict=False)
        if unexpected:
            raise RuntimeError(f"lora_state 含未知键 ({len(unexpected)}): {unexpected[:24]}")
        print(
            f"[Test] LoRA(standard): {lora_path.name} rank={rank} alpha={alpha} targets={targets} "
            f"(lora_tensors={len(ls)}, missing_non_lora_keys={len(missing)} 为正常现象)"
        )
    else:
        raise KeyError(f"{lora_path} 中既没有 trainable_state 也没有 lora_state")

    model.eval()
    if use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"[Test] torch.compile skipped: {e}")
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Test base or base+LoRA (ckpt-final layout)")
    p.add_argument("--mode", type=str, choices=("base", "base+lora"), default="base", help="base: 仅基础模型; base+lora: 基础+指定 LoRA")
    p.add_argument("--ckpt_dir", type=str, default=str(_DEFAULT_CKPT_DIR), help="存放 base.pth 与各 LoRA 的目录")
    p.add_argument("--base_ckpt", type=str, default=str(_DEFAULT_BASE), help="基础模型权重（默认 ckpt-final/base.pth）")
    p.add_argument(
        "--lora",
        type=str,
        default="",
        help="LoRA 简称（不含 .pth），在 --ckpt_dir 下查找，如 light、haze、rain、seg、detect、text",
    )
    p.add_argument("--lora_ckpt", type=str, default="", help="LoRA 权重完整路径（优先于 --lora）")
    p.add_argument("--vi_path", type=str, default=DEFAULT_VI_PATH)
    p.add_argument("--ir_path", type=str, default=DEFAULT_IR_PATH)
    p.add_argument("--save_path", type=str, default=DEFAULT_SAVE_PATH)
    p.add_argument("--base_channels", type=int, default=DEFAULT_BASE_CHANNELS)
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p.add_argument("--crop_640x480", action="store_true")
    amp_g = p.add_mutually_exclusive_group()
    amp_g.add_argument("--use_amp", dest="use_amp", action="store_true")
    amp_g.add_argument("--no_amp", dest="use_amp", action="store_false")
    p.set_defaults(use_amp=DEFAULT_USE_AMP)
    cg = p.add_mutually_exclusive_group()
    cg.add_argument("--use_torch_compile", dest="use_torch_compile", action="store_true")
    cg.add_argument("--no_torch_compile", dest="use_torch_compile", action="store_false")
    p.set_defaults(use_torch_compile=DEFAULT_USE_TORCH_COMPILE)
    return p.parse_args()


def scan_images(folder):
    exts = (".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".png")
    files = [os.path.join(folder, name) for name in os.listdir(folder) if name.lower().endswith(exts)]
    files.sort()
    return files


def center_crop_640x480_if_needed(img: Image.Image) -> Image.Image:
    target_w, target_h = 640, 480
    w, h = img.size
    if (w >= target_w and h >= target_h) and (w > target_w or h > target_h):
        left = (w - target_w) // 2
        top = (h - target_h) // 2
        img = img.crop((left, top, left + target_w, top + target_h))
    return img


def read_vi(path, do_crop: bool):
    img = Image.open(path).convert("RGB")
    if do_crop:
        img = center_crop_640x480_if_needed(img)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0)


def read_ir(path, do_crop: bool):
    img = Image.open(path).convert("L")
    if do_crop:
        img = center_crop_640x480_if_needed(img)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return torch.from_numpy(arr).unsqueeze(0)


def save_rgb_tensor(path, tensor):
    arr = tensor.squeeze(0).detach().to(torch.float32).cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    base_ckpt = Path(args.base_ckpt)

    use_cuda = args.device.lower() == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    amp_enabled = bool(args.use_amp and use_cuda)

    if args.mode == "base+lora":
        lora_path = _resolve_lora_path(ckpt_dir, args.lora or None, args.lora_ckpt or None)
    else:
        lora_path = None

    print(
        f"[fusion_test_ckpt_final] mode={args.mode} device={device} AMP(BF16)={'on' if amp_enabled else 'off'} "
        f"compile={'on' if args.use_torch_compile else 'off'}"
    )
    print(f"  base_ckpt={base_ckpt}")
    if lora_path is not None:
        print(f"  lora_ckpt={lora_path}")

    if not args.vi_path or not os.path.isdir(args.vi_path):
        raise ValueError("请通过 --vi_path 指定可见光图像文件夹（有效目录）")
    if not args.ir_path or not os.path.isdir(args.ir_path):
        raise ValueError("请通过 --ir_path 指定红外图像文件夹（有效目录）")

    vi_files = scan_images(args.vi_path)
    ir_files = scan_images(args.ir_path)
    if len(vi_files) != len(ir_files):
        raise ValueError(f"VI/IR 数量不一致: vi={len(vi_files)} ir={len(ir_files)}")

    save_root = Path(args.save_path)
    save_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "base":
        model = build_model_base_only(base_ckpt, args.base_channels, device, args.use_torch_compile)
    else:
        model = build_model_base_plus_lora(base_ckpt, lora_path, args.base_channels, device, args.use_torch_compile)

    with torch.no_grad():
        for vi_file, ir_file in tqdm(list(zip(vi_files, ir_files)), desc="Infer", ncols=100):
            vi = read_vi(vi_file, args.crop_640x480).to(device, non_blocking=True)
            ir = read_ir(ir_file, args.crop_640x480).to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=torch.bfloat16):
                out = model(vi, ir)
            save_name = os.path.basename(vi_file)
            save_rgb_tensor(str(save_root / save_name), out)

    print(f"Done. Saved to: {save_root}")


if __name__ == "__main__":
    main()
