"""
LTX-2 v2.5 LoRA Training — One-Command Launcher
=================================================

Usage:
    python train_lora.py /storage/Internal_NAS/<user>/my-dataset

The dataset folder must contain:
    dataset.json   — list of {"caption": "...", "media_path": "videos/xxx.mp4"}
    videos/        — the video files referenced in dataset.json

Everything else (checkpoints, preprocessing, output) is handled automatically.
The dataset folder MUST be on NAS (/storage/Internal_NAS/) so all nodes can access it.

Optional flags:
    --steps N          Training steps (default: 1000)
    --rank N           LoRA rank (default: 16)
    --resolution WxHxF Video resolution (default: 512x320x25)
    --output DIR       Override output directory
    --skip-preprocess  Skip preprocessing if already done
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/.triton")

ROOT = Path(__file__).resolve().parent
PACKAGES = ROOT / "packages"
PREPROCESS_SCRIPT = PACKAGES / "ltx-trainer" / "scripts" / "process_dataset.py"
TRAIN_SCRIPT = PACKAGES / "ltx-trainer" / "scripts" / "train.py"

# Default checkpoint paths on NAS
DEFAULTS = {
    "transformer": "/storage/Internal_NAS/Checkpoints/LTX-2_Release/v2.5/checkpoints/ltx-2.5-22b-dev-transformer-bf16.safetensors",
    "text_encoder": "/storage/Internal_NAS/Checkpoints/LTX-2_Release/v2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video_vae": "/storage/Internal_NAS/Checkpoints/LTX-2_Release/v2.5/vae/ltx-2.5-video-vae-bf16.safetensors",
}


def install_packages():
    print("[1/3] Installing packages...")
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


def validate_dataset(dataset_dir: Path):
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

    # Check first entry
    entry = data[0]
    if "caption" not in entry or "media_path" not in entry:
        print("ERROR: Each entry in dataset.json must have 'caption' and 'media_path' keys.")
        print(f"  Got: {list(entry.keys())}")
        sys.exit(1)

    # Check a few video files exist
    missing = []
    for item in data[:5]:
        vp = dataset_dir / item["media_path"]
        if not vp.exists():
            missing.append(str(vp))
    if missing:
        print(f"ERROR: Video files not found:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    if not str(dataset_dir).startswith("/storage/Internal_NAS/"):
        print("WARNING: Dataset is not on NAS (/storage/Internal_NAS/).")
        print("         Training may fail if it runs on a different node.")
        print()

    print(f"    Dataset OK: {len(data)} samples in {dataset_dir}")
    return data


def generate_config(dataset_dir: Path, args) -> str:
    """Generate a YAML config string with all defaults filled in."""
    w, h, f = [int(x) for x in args.resolution.split("x")]
    output_dir = args.output or str(dataset_dir.parent / f"{dataset_dir.name}_lora_output")
    precomputed = str(dataset_dir / ".precomputed")

    config = f"""# Auto-generated LTX-2 v2.5 LoRA config
model:
  model_path: "{DEFAULTS['transformer']}"
  text_encoder_path: "{DEFAULTS['text_encoder']}"
  video_vae_path: "{DEFAULTS['video_vae']}"
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
    return config


def run_preprocess(dataset_dir: Path, args):
    print("[2/3] Preprocessing dataset (video latents + text embeddings)...")
    w, h, f = [int(x) for x in args.resolution.split("x")]
    resolution = f"{w}x{h}x{f}"
    precomputed = dataset_dir / ".precomputed"

    cmd = [
        sys.executable, str(PREPROCESS_SCRIPT),
        str(dataset_dir / "dataset.json"),
        "--resolution-buckets", resolution,
        "--model-path", DEFAULTS["transformer"],
        "--text-encoder-path", DEFAULTS["text_encoder"],
        "--video-vae-path", DEFAULTS["video_vae"],
        "--output-dir", str(precomputed),
        "--load-text-encoder-in-8bit",
        "--skip-audio",
    ]
    ret = subprocess.call(cmd, env=os.environ.copy())
    if ret != 0:
        print("ERROR: Preprocessing failed.")
        sys.exit(ret)
    print("    Preprocessing complete.")


def run_training(config_path: str):
    print("[3/3] Starting training...")
    env = os.environ.copy()
    env.setdefault("TRITON_CACHE_DIR", "/tmp/.triton")
    ret = subprocess.call([sys.executable, str(TRAIN_SCRIPT), config_path], env=env)
    if ret != 0:
        print("ERROR: Training failed.")
        sys.exit(ret)
    print("    Training complete!")


def main():
    parser = argparse.ArgumentParser(
        description="LTX-2 v2.5 LoRA Training — one command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dataset", help="Path to dataset folder (must contain dataset.json + videos/)")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps (default: 1000)")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--resolution", default="512x320x25", help="WxHxF (default: 512x320x25)")
    parser.add_argument("--output", default=None, help="Output directory (default: <dataset>_lora_output)")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing if already done")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).resolve()
    if not dataset_dir.is_dir():
        print(f"ERROR: {dataset_dir} is not a directory.")
        sys.exit(1)

    print("=" * 60)
    print("  LTX-2 v2.5 LoRA Training")
    print("=" * 60)

    # Validate
    validate_dataset(dataset_dir)

    # Check checkpoints exist
    for name, path in DEFAULTS.items():
        if not Path(path).exists():
            print(f"ERROR: Checkpoint not found: {path}")
            sys.exit(1)

    # Install packages
    install_packages()

    # Preprocess
    precomputed = dataset_dir / ".precomputed"
    if args.skip_preprocess and precomputed.exists():
        print("[2/3] Skipping preprocessing (--skip-preprocess, .precomputed exists)")
    else:
        run_preprocess(dataset_dir, args)

    # Generate config
    config_yaml = generate_config(dataset_dir, args)
    config_path = str(dataset_dir / "training_config.yaml")
    with open(config_path, "w") as f:
        f.write(config_yaml)
    print(f"    Config saved to {config_path}")

    # Train
    run_training(config_path)

    # Summary
    output_dir = args.output or str(dataset_dir.parent / f"{dataset_dir.name}_lora_output")
    print()
    print("=" * 60)
    print("  DONE!")
    print(f"  LoRA weights: {output_dir}/checkpoints/")
    print(f"  Validation:   {output_dir}/samples/")
    print("=" * 60)


if __name__ == "__main__":
    main()
