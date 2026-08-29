"""
LTX-2 LoRA Training Launcher
使用方式：python train.py configs/ltx2_lora_ryan.yaml

先安裝 repo 內的 ltx-core / ltx-trainer，再用子程序跑 training script，
確保 import 的一定是本 repo 的版本。
"""
import os
import subprocess
import sys
from pathlib import Path

# Ensure triton cache is writable (Slurm jobs may lack $HOME write access)
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/.triton")

ROOT = Path(__file__).resolve().parent
PACKAGES = ROOT / "packages"
TRAIN_SCRIPT = PACKAGES / "ltx-trainer" / "scripts" / "train.py"


def install_packages():
    """從本 repo 的 packages/ 安裝 ltx-core 和 ltx-trainer 到當前 venv"""
    print("Installing ltx-core and ltx-trainer from repo...")
    for pkg in ["ltx-core", "ltx-trainer"]:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--force-reinstall",
             "--no-deps", str(PACKAGES / pkg)],
            stdout=subprocess.DEVNULL,
        )
    # Ensure compatible bitsandbytes version and pin transformers below 5.15
    # (5.15+ has AmbiguousGlobalPerLayerAttributeError breaking Gemma4 init)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade",
         "bitsandbytes>=0.45.0", "transformers>=5.8.0,<5.15"],
        stdout=subprocess.DEVNULL,
    )
    print("Done.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py <config.yaml>")
        sys.exit(1)

    config_path = Path(sys.argv[1]).resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    # 先安裝，再用全新的子程序跑 training（避免舊 import cache）
    install_packages()
    print(f"Config:  {config_path}")
    print(f"Trainer: {TRAIN_SCRIPT}")
    env = os.environ.copy()
    env.setdefault("TRITON_CACHE_DIR", "/tmp/.triton")
    sys.exit(subprocess.call([sys.executable, str(TRAIN_SCRIPT), str(config_path)], env=env))


if __name__ == "__main__":
    main()
