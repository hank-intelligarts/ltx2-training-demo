"""
LTX-2 LoRA Training Launcher
使用方式：python train.py configs/ltx2_lora_ryan.yaml
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGES = ROOT / "packages"
TRAIN_SCRIPT = PACKAGES / "ltx-trainer" / "scripts" / "train.py"


def ensure_installed():
    """確保 ltx-core 和 ltx-trainer 已安裝到當前 venv"""
    try:
        import ltx_trainer  # noqa: F401
        return
    except ImportError:
        pass

    print("Installing ltx-core and ltx-trainer...")
    for pkg in ["ltx-core", "ltx-trainer"]:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-e", str(PACKAGES / pkg)],
            stdout=subprocess.DEVNULL,
        )


def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py <config.yaml>")
        sys.exit(1)

    config_path = Path(sys.argv[1]).resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    ensure_installed()
    print(f"Config:  {config_path}")
    print(f"Trainer: {TRAIN_SCRIPT}")
    sys.exit(subprocess.call([sys.executable, str(TRAIN_SCRIPT), str(config_path)]))


if __name__ == "__main__":
    main()
