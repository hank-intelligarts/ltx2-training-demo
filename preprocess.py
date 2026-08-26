"""
LTX-2 Dataset Preprocessing Launcher
Usage: python preprocess.py <dataset.json> <config.yaml>

Installs repo packages, then runs the preprocessing script to compute
latents and text embeddings from raw video + caption data.
"""
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
PACKAGES = ROOT / "packages"
PREPROCESS_SCRIPT = PACKAGES / "ltx-trainer" / "scripts" / "process_dataset.py"


def install_packages():
    print("Installing ltx-core and ltx-trainer from repo...")
    for pkg in ["ltx-core", "ltx-trainer"]:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--force-reinstall",
             "--no-deps", str(PACKAGES / pkg)],
            stdout=subprocess.DEVNULL,
        )
    # Ensure compatible bitsandbytes version
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", "bitsandbytes>=0.45.0"],
        stdout=subprocess.DEVNULL,
    )
    print("Done.")


def main():
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py <dataset.json> <config.yaml>")
        sys.exit(1)

    dataset_path = Path(sys.argv[1]).resolve()
    config_path = Path(sys.argv[2]).resolve()

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    # Read config to get model paths and data settings
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_path = config["model"]["model_path"]
    text_encoder_path = config["model"]["text_encoder_path"]
    preprocessed_root = config["data"]["preprocessed_data_root"]
    load_8bit = config.get("acceleration", {}).get("load_text_encoder_in_8bit", False)
    with_audio = config.get("training_strategy", {}).get("with_audio", False)

    # Get resolution from validation dims (W x H x F)
    dims = config.get("validation", {}).get("video_dims", [512, 320, 25])
    resolution = f"{dims[0]}x{dims[1]}x{dims[2]}"

    install_packages()

    cmd = [
        sys.executable, str(PREPROCESS_SCRIPT),
        str(dataset_path),
        "--resolution-buckets", resolution,
        "--model-path", model_path,
        "--text-encoder-path", text_encoder_path,
        "--output-dir", preprocessed_root,
    ]
    if load_8bit:
        cmd.append("--load-text-encoder-in-8bit")
    if with_audio:
        cmd.append("--with-audio")

    print(f"Dataset:    {dataset_path}")
    print(f"Config:     {config_path}")
    print(f"Model:      {model_path}")
    print(f"Encoder:    {text_encoder_path}")
    print(f"Output:     {preprocessed_root}")
    print(f"Resolution: {resolution}")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
