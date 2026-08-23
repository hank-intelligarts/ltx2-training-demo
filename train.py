"""
LTX-2 LoRA Training Launcher
使用方式：python train.py configs/ltx2_lora_ryan.yaml
"""
import subprocess
import sys
from pathlib import Path

LTX_REPO_URL = "https://github.com/Lightricks/LTX-Video.git"
LTX_REPO_PATH = Path("/storage/SSD2/repos/LTX-2")
LTX_TRAINER_PATH = LTX_REPO_PATH / "packages" / "ltx-trainer"


def ensure_ltx_trainer():
    """確保 LTX-2 trainer code 存在，不存在就自動 clone 到 local SSD"""
    train_script = LTX_TRAINER_PATH / "scripts" / "train.py"
    if train_script.exists():
        return train_script

    print(f"LTX-2 trainer not found, cloning to {LTX_REPO_PATH}...")
    LTX_REPO_PATH.parent.mkdir(parents=True, exist_ok=True)
    ret = subprocess.call(["git", "clone", "--depth", "1", LTX_REPO_URL, str(LTX_REPO_PATH)])
    if ret != 0:
        print("Failed to clone LTX-2 repo")
        sys.exit(1)

    # Install ltx-trainer + ltx-core into current venv
    print("Installing ltx-trainer and ltx-core...")
    for pkg in ["ltx-core", "ltx-trainer"]:
        pkg_path = LTX_REPO_PATH / "packages" / pkg
        subprocess.call([sys.executable, "-m", "pip", "install", "-e", str(pkg_path)])

    return train_script


def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py <config.yaml>")
        sys.exit(1)

    config_path = Path(sys.argv[1]).resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    train_script = ensure_ltx_trainer()
    print(f"Config:  {config_path}")
    print(f"Trainer: {train_script}")
    sys.exit(subprocess.call([sys.executable, str(train_script), str(config_path)]))


if __name__ == "__main__":
    main()
