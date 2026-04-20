import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def _load_from_file(file_name: str, attr_name: str):
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    spec = importlib.util.spec_from_file_location(attr_name + "_dyn", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr_name)


BaseFusionReconNet = _load_from_file("base-net.py", "BaseFusionReconNet")

# =========================
# Editable top-level config
# =========================
DEFAULT_IR_PATH = r"C:\Users\PC\Desktop\com\GRAD\M3FD\IR"
DEFAULT_VI_PATH = r"C:\Users\PC\Desktop\com\GRAD\M3FD\vi"
DEFAULT_CKPT_PATH = r"C:\Users\PC\Desktop\lora\ckpt_base\base_epoch_100.pth"
DEFAULT_SAVE_PATH = r"C:\Users\PC\Desktop\lora\OUTBUT(BASE)\M3FD"
DEFAULT_BASE_CHANNELS = 32
DEFAULT_DEVICE = "cuda"
# Match base-train.py: BF16 inference on CUDA (no GradScaler at test time)
DEFAULT_USE_AMP = True
DEFAULT_USE_TORCH_COMPILE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Test base fusion model")
    parser.add_argument("--ir_path", type=str, default=DEFAULT_IR_PATH)
    parser.add_argument("--vi_path", type=str, default=DEFAULT_VI_PATH)
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--save_path", type=str, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--base_channels", type=int, default=DEFAULT_BASE_CHANNELS)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument(
        "--crop_640x480",
        action="store_true",
        help="If enabled, center-crop inputs to 640x480 when both dims exceed them.",
    )
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--use_amp", dest="use_amp", action="store_true", help="BF16 autocast on CUDA (match base-train)")
    amp_group.add_argument("--no_amp", dest="use_amp", action="store_false", help="Full fp32 inference")
    parser.set_defaults(use_amp=DEFAULT_USE_AMP)
    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument(
        "--use_torch_compile",
        dest="use_torch_compile",
        action="store_true",
        help="torch.compile(model) after load (optional)",
    )
    compile_group.add_argument(
        "--no_torch_compile",
        dest="use_torch_compile",
        action="store_false",
        help="Disable torch.compile",
    )
    parser.set_defaults(use_torch_compile=DEFAULT_USE_TORCH_COMPILE)
    return parser.parse_args()


def scan_images(folder):
    exts = (".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".png")
    files = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(exts)
    ]
    files.sort()
    return files


def center_crop_640x480_if_needed(img: Image.Image) -> Image.Image:
    """If both dims are larger than 640x480, center-crop to exactly 640x480."""
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
    # BF16/fp16 tensors are not supported by numpy directly
    arr = tensor.squeeze(0).detach().to(torch.float32).cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def load_model(ckpt_path, base_channels, device, use_torch_compile: bool):
    model = BaseFusionReconNet(base_ch=base_channels).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    if use_torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("[Test] torch.compile enabled.")
        except Exception as e:
            print(f"[Test] torch.compile skipped: {e}")
    return model


def main():
    args = parse_args()
    use_cuda = args.device.lower() == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    amp_enabled = bool(args.use_amp and use_cuda)
    print(
        f"[Test] device={device} AMP(BF16)={'on' if amp_enabled else 'off'} "
        f"torch_compile={'on' if args.use_torch_compile else 'off'}"
    )

    vi_files = scan_images(args.vi_path)
    ir_files = scan_images(args.ir_path)
    if len(vi_files) != len(ir_files):
        raise ValueError(f"VI/IR count mismatch: vi={len(vi_files)} ir={len(ir_files)}")

    save_root = Path(args.save_path)
    save_root.mkdir(parents=True, exist_ok=True)

    model = load_model(args.ckpt_path, args.base_channels, device, args.use_torch_compile)

    with torch.no_grad():
        for vi_file, ir_file in tqdm(list(zip(vi_files, ir_files)), desc="Testing", ncols=100):
            vi = read_vi(vi_file, args.crop_640x480).to(device, non_blocking=True)
            ir = read_ir(ir_file, args.crop_640x480).to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=torch.bfloat16):
                out = model(vi, ir)

            save_name = os.path.basename(vi_file)
            save_path = save_root / save_name
            save_rgb_tensor(str(save_path), out)

    print(f"Done. Results saved to: {save_root}")


if __name__ == "__main__":
    main()
