"""
LTX-2 v2.5 LoRA Training — One-Command Launcher
=================================================

Usage:
    python train_lora.py /storage/Internal_NAS/<user>/my-dataset

The dataset folder must contain:
    dataset.json   — [{"caption": "...", "media_path": "videos/xxx.mp4"}, ...]
    videos/        — the video files referenced in dataset.json

Everything runs on local SSD for speed. Results stay on local SSD.

Optional flags:
    --steps N          Training steps (default: 1000)
    --rank N           LoRA rank (default: 16)
    --resolution WxHxF Video resolution (default: 512x320x25)
"""
import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/.triton")

ROOT = Path(__file__).resolve().parent
PACKAGES = ROOT / "packages"
PREPROCESS_SCRIPT = PACKAGES / "ltx-trainer" / "scripts" / "process_dataset.py"
TRAIN_SCRIPT = PACKAGES / "ltx-trainer" / "scripts" / "train.py"

# Local SSD scratch space — each node writes here
LOCAL_SCRATCH = Path("/storage/SSD2/training_scratch")

# Checkpoints on NAS (read-only)
CHECKPOINTS = {
    "transformer": "/storage/Internal_NAS/Checkpoints/LTX-2_Release/v2.5/checkpoints/ltx-2.5-22b-dev-transformer-bf16.safetensors",
    "text_encoder": "/storage/Internal_NAS/Checkpoints/LTX-2_Release/v2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video_vae": "/storage/Internal_NAS/Checkpoints/LTX-2_Release/v2.5/vae/ltx-2.5-video-vae-bf16.safetensors",
}


def install_packages():
    print("[1/4] Installing packages...")
    for pkg in ["ltx-core", "ltx-trainer"]:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--force-reinstall",
             "--no-deps", str(PACKAGES / pkg)],
            stdout=subprocess.DEVNULL,
        )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade",
         "bitsandbytes>=0.45.0", "transformers>=5.8.0,<5.15"],
        stdout=subprocess.DEVNULL,
    )
    print("    Done.")


def validate_dataset(dataset_dir: Path) -> list[dict]:
    dataset_json = dataset_dir / "dataset.json"
    if not dataset_json.exists():
        print(f"ERROR: {dataset_json} not found.")
        print()
        print("Your dataset folder should look like:")
        print("  my-dataset/")
        print("    dataset.json")
        print("    videos/")
        print("      clip1.mp4")
        print("      clip2.mp4")
        print()
        print("dataset.json format:")
        print('  [{"caption": "A cat...", "media_path": "videos/clip1.mp4"}, ...]')
        sys.exit(1)

    with open(dataset_json) as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        print("ERROR: dataset.json must be a non-empty JSON array.")
        sys.exit(1)

    entry = data[0]
    if "caption" not in entry or "media_path" not in entry:
        print("ERROR: Each entry needs 'caption' and 'media_path' keys.")
        print(f"  Got: {list(entry.keys())}")
        sys.exit(1)

    missing = [str(dataset_dir / item["media_path"])
               for item in data[:5]
               if not (dataset_dir / item["media_path"]).exists()]
    if missing:
        print("ERROR: Video files not found:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    print(f"    Dataset OK: {len(data)} samples")
    return data


def setup_local_workspace(dataset_dir: Path) -> tuple[Path, Path]:
    """Create local SSD workspaces and return (preprocess_dir, run_dir).

    Preprocess is reusable across runs (keyed by dataset name).
    Each training run gets its own timestamped folder.

    Layout:
        /storage/SSD2/training_scratch/<dataset_name>/
            precomputed/             — shared, reused across runs
            runs/<YYYYMMDD_HHMMSS>/  — per-run output
                config.yaml
                checkpoints/
                samples/
    """
    base = LOCAL_SCRATCH / dataset_dir.name
    preprocess_dir = base / "precomputed"
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    return preprocess_dir, run_dir


def run_preprocess(dataset_dir: Path, preprocess_dir: Path, resolution: str):
    # Check if already preprocessed (latents + conditions dirs with files)
    latents_dir = preprocess_dir / "latents" / "videos"
    conditions_dir = preprocess_dir / "conditions" / "videos"
    if (latents_dir.is_dir() and conditions_dir.is_dir()
            and any(latents_dir.iterdir()) and any(conditions_dir.iterdir())):
        print("[2/3] Preprocessing already done, skipping.")
        return

    print("[2/3] Preprocessing (video latents + text embeddings)...")
    print(f"    Reading from NAS:  {dataset_dir}")
    print(f"    Writing to local:  {preprocess_dir}")

    cmd = [
        sys.executable, str(PREPROCESS_SCRIPT),
        str(dataset_dir / "dataset.json"),
        "--resolution-buckets", resolution,
        "--model-path", CHECKPOINTS["transformer"],
        "--text-encoder-path", CHECKPOINTS["text_encoder"],
        "--video-vae-path", CHECKPOINTS["video_vae"],
        "--output-dir", str(preprocess_dir),
        "--load-text-encoder-in-8bit",
        "--skip-audio",
    ]
    ret = subprocess.call(cmd, env=os.environ.copy())
    if ret != 0:
        print("ERROR: Preprocessing failed.")
        sys.exit(ret)
    print("    Done.")


def generate_config(preprocess_dir: Path, run_dir: Path, args) -> Path:
    w, h, f = [int(x) for x in args.resolution.split("x")]
    precomputed = str(preprocess_dir)
    output_dir = str(run_dir)

    config = f"""# Auto-generated LTX-2 v2.5 LoRA config
model:
  model_path: "{CHECKPOINTS['transformer']}"
  text_encoder_path: "{CHECKPOINTS['text_encoder']}"
  video_vae_path: "{CHECKPOINTS['video_vae']}"
  training_mode: "lora"

lora:
  rank: {args.rank}
  alpha: {args.rank}
  dropout: 0.0
  target_modules: ["to_k", "to_q", "to_v", "to_out.0"]

training_strategy:
  name: "text_to_video"
  first_frame_conditioning_p: 0.5
  with_audio: false

optimization:
  learning_rate: 1e-4
  steps: {args.steps}
  batch_size: 1
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  optimizer_type: "adamw8bit"
  scheduler_type: "cosine"
  scheduler_params: {{}}
  enable_gradient_checkpointing: true

acceleration:
  mixed_precision_mode: "bf16"
  quantization: "int8-quanto"
  load_text_encoder_in_8bit: true

data:
  preprocessed_data_root: "{precomputed}"
  num_dataloader_workers: 2

validation:
  prompts: []
  video_dims: [{w}, {h}, {f}]
  frame_rate: 25.0
  seed: 42
  inference_steps: 30
  interval: {max(args.steps // 5, 100)}
  video_cfg_scale: 4.0
  video_stg_scale: 1.0
  audio_stg_scale: 0.0
  stg_blocks: [29]
  generate_audio: false
  skip_initial_validation: true

checkpoints:
  interval: {max(args.steps // 2, 250)}
  keep_last_n: 3
  precision: "bfloat16"

flow_matching:
  timestep_sampling_mode: "shifted_logit_normal"
  timestep_sampling_params: {{}}

hub:
  push_to_hub: false

wandb:
  enabled: false
  project: "ltx-2-trainer"
  tags: ["ltx2", "lora"]

seed: 42
output_dir: "{output_dir}"
"""
    config_path = run_dir / "config.yaml"
    config_path.write_text(config)
    return config_path


def run_training(config_path: Path):
    print("[3/3] Training...")
    env = os.environ.copy()
    env.setdefault("TRITON_CACHE_DIR", "/tmp/.triton")
    ret = subprocess.call([sys.executable, str(TRAIN_SCRIPT), str(config_path)], env=env)
    if ret != 0:
        print("ERROR: Training failed.")
        sys.exit(ret)
    print("    Done.")


def main():
    parser = argparse.ArgumentParser(
        description="LTX-2 v2.5 LoRA Training — one command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset", help="Dataset folder on NAS (must contain dataset.json + videos/)")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps (default: 1000)")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--resolution", default="512x320x25", help="WxHxF (default: 512x320x25)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).resolve()
    if not dataset_dir.is_dir():
        print(f"ERROR: {dataset_dir} is not a directory.")
        sys.exit(1)

    hostname = socket.gethostname()
    print("=" * 60)
    print("  LTX-2 v2.5 LoRA Training")
    print(f"  Node: {hostname}")
    print("=" * 60)

    # Validate dataset on NAS
    validate_dataset(dataset_dir)

    # Check checkpoints on NAS
    for name, path in CHECKPOINTS.items():
        if not Path(path).exists():
            print(f"ERROR: Checkpoint not found: {path}")
            sys.exit(1)

    # Setup local workspace on SSD
    preprocess_dir, run_dir = setup_local_workspace(dataset_dir)
    print(f"    Preprocess cache: {preprocess_dir}")
    print(f"    Training output:  {run_dir}")

    # Install packages
    install_packages()

    # Preprocess (reads NAS, writes local SSD)
    w, h, f = [int(x) for x in args.resolution.split("x")]
    run_preprocess(dataset_dir, preprocess_dir, f"{w}x{h}x{f}")

    # Generate config (all paths point to local SSD)
    config_path = generate_config(preprocess_dir, run_dir, args)
    print(f"    Config: {config_path}")

    # Train (all on local SSD)
    run_training(config_path)

    print()
    print("=" * 60)
    print("  DONE!")
    print(f"  Node:         {hostname}")
    print(f"  LoRA weights: {run_dir}/checkpoints/")
    print(f"  Validation:   {run_dir}/samples/")
    print("=" * 60)


if __name__ == "__main__":
    main()
