"""
Compute 3D depth reference videos for IC-LoRA training using SAM-3D Body.

This script generates depth map reference videos from input videos/images by:
1. Running SAM-3D Body to get 3D human mesh for each frame
2. Rendering the mesh as a depth map
3. Saving as reference video for IC-LoRA training

Basic usage:
    python compute_3d_reference.py videos_dir/ --output videos_dir/dataset.json
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Add SAM-3D Body to path
SAM3D_PATH = Path(__file__).parent.parent.parent.parent / "sam-3d-body"
if SAM3D_PATH.exists():
    sys.path.insert(0, str(SAM3D_PATH))

console = Console()

# Lazy imports for SAM-3D dependencies
_estimator = None
_faces = None


def get_sam3d_estimator(device: str = "cuda"):
    """Lazy load SAM-3D Body estimator."""
    global _estimator, _faces
    
    if _estimator is not None:
        return _estimator, _faces
    
    console.print("[bold blue]Loading SAM-3D Body model...[/]")
    
    try:
        from sam_3d_body import load_sam_3d_body_hf, SAM3DBodyEstimator
        from tools.build_detector import HumanDetector
        from tools.build_fov_estimator import FOVEstimator
        
        # Load model from HuggingFace
        model, model_cfg = load_sam_3d_body_hf(
            "facebook/sam-3d-body-dinov3",
            device=device
        )
        
        # Initialize human detector
        human_detector = HumanDetector(name="vitdet", device=device)
        
        # Initialize FOV estimator
        fov_estimator = FOVEstimator(name="moge2", device=device)
        
        # Create estimator
        _estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=human_detector,
            human_segmentor=None,
            fov_estimator=fov_estimator,
        )
        _faces = _estimator.faces
        
        console.print("[bold green]✓[/] SAM-3D Body model loaded successfully")
        
    except ImportError as e:
        console.print(f"[bold red]Error:[/] Could not import SAM-3D Body: {e}")
        console.print("Please ensure SAM-3D Body is installed and accessible")
        raise
    
    return _estimator, _faces


def render_depth_map(
    vertices: np.ndarray,
    cam_t: np.ndarray,
    focal_length: float,
    image_shape: Tuple[int, int],
    faces: np.ndarray,
) -> np.ndarray:
    """
    Render depth map from 3D mesh vertices.
    
    Args:
        vertices: Mesh vertices (V, 3)
        cam_t: Camera translation (3,)
        focal_length: Camera focal length
        image_shape: Output image shape (H, W)
        faces: Mesh faces (F, 3)
    
    Returns:
        Depth map as uint8 image (H, W, 3) - grayscale, normalized
    """
    import pyrender
    import trimesh
    
    h, w = image_shape
    
    renderer = pyrender.OffscreenRenderer(
        viewport_height=h,
        viewport_width=w,
    )
    
    camera_translation = cam_t.copy()
    camera_translation[0] *= -1.0
    
    # Create mesh
    mesh = trimesh.Trimesh(vertices.copy(), faces.copy())
    
    # Apply rotation (same as SAM-3D Body renderer)
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    
    # Create material (white for depth visibility)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode="OPAQUE",
        baseColorFactor=(1.0, 1.0, 1.0, 1.0),
    )
    
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    
    # Create scene
    scene = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 0.0],
        ambient_light=(0.5, 0.5, 0.5)
    )
    scene.add(mesh, "mesh")
    
    # Add camera
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = camera_translation
    camera_center = [w / 2.0, h / 2.0]
    
    camera = pyrender.IntrinsicsCamera(
        fx=focal_length,
        fy=focal_length,
        cx=camera_center[0],
        cy=camera_center[1],
        zfar=1e12,
    )
    scene.add(camera, pose=camera_pose)
    
    # Add light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=camera_pose)
    
    # Render with depth
    _, depth = renderer.render(scene)
    renderer.delete()
    
    # Normalize depth map
    valid_mask = depth > 0
    if valid_mask.any():
        min_depth = depth[valid_mask].min()
        max_depth = depth[valid_mask].max()
        
        if max_depth > min_depth:
            # Normalize to 0-255, invert so closer = brighter
            depth_normalized = np.zeros_like(depth)
            depth_normalized[valid_mask] = 255 * (1 - (depth[valid_mask] - min_depth) / (max_depth - min_depth))
        else:
            depth_normalized = np.where(valid_mask, 255, 0)
    else:
        depth_normalized = np.zeros_like(depth)
    
    # Convert to 3-channel image
    depth_img = depth_normalized.astype(np.uint8)
    depth_img = np.stack([depth_img] * 3, axis=-1)
    
    return depth_img


def compute_canny_edges(
    frame: np.ndarray,
    threshold1: int = 100,
    threshold2: int = 200,
) -> np.ndarray:
    """
    Compute Canny edge detection on a frame.
    
    Args:
        frame: Input frame (H, W, 3) in BGR format
        threshold1: First threshold for hysteresis
        threshold2: Second threshold for hysteresis
    
    Returns:
        Edge map as uint8 image (H, W, 3) - white edges on black background
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1, threshold2)
    
    # Convert to 3-channel
    edges_3ch = np.stack([edges] * 3, axis=-1)
    
    return edges_3ch


def combine_depth_and_canny(
    depth_map: np.ndarray,
    canny_edges: np.ndarray,
    edge_color: tuple = (255, 255, 255),
) -> np.ndarray:
    """
    Combine depth map and Canny edges into a single image.
    
    Depth map is used as the base, with Canny edges overlaid in the specified color.
    
    Args:
        depth_map: Depth map image (H, W, 3)
        canny_edges: Canny edge image (H, W, 3)
        edge_color: RGB color for edge overlay
    
    Returns:
        Combined image (H, W, 3)
    """
    combined = depth_map.copy()
    
    # Create mask where edges exist
    edge_mask = canny_edges[:, :, 0] > 0
    
    # Overlay edges
    combined[edge_mask] = edge_color
    
    return combined


def process_frame(
    frame: np.ndarray,
    estimator,
    faces: np.ndarray,
    bbox_thresh: float = 0.5,
    include_canny: bool = True,
    canny_threshold1: int = 100,
    canny_threshold2: int = 200,
) -> np.ndarray:
    """
    Process a single frame and generate depth map with optional Canny overlay.
    
    Args:
        frame: Input frame (H, W, 3) in BGR format
        estimator: SAM3DBodyEstimator
        faces: Mesh faces
        bbox_thresh: Detection threshold
        include_canny: Whether to overlay Canny edges
        canny_threshold1: Canny first threshold
        canny_threshold2: Canny second threshold
    
    Returns:
        Depth map image with optional Canny overlay (H, W, 3)
    """
    h, w = frame.shape[:2]
    
    # Convert BGR to RGB for SAM-3D
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run SAM-3D Body inference
    outputs = estimator.process_one_image(
        frame_rgb,
        bbox_thr=bbox_thresh,
        use_mask=False,
    )
    
    # Start with Canny edges as base if no person detected, or black image
    if include_canny:
        canny_edges = compute_canny_edges(frame, canny_threshold1, canny_threshold2)
    else:
        canny_edges = None
    
    if not outputs:
        # No person detected
        if canny_edges is not None:
            # Return just Canny edges (on black background)
            return canny_edges
        else:
            return np.zeros((h, w, 3), dtype=np.uint8)
    
    # Combine depth maps for all detected persons
    combined_depth = np.zeros((h, w, 3), dtype=np.float32)
    
    for person_output in outputs:
        vertices = person_output["pred_vertices"]
        cam_t = person_output["pred_cam_t"]
        focal_length = person_output["focal_length"]
        
        depth_img = render_depth_map(
            vertices=vertices,
            cam_t=cam_t,
            focal_length=focal_length,
            image_shape=(h, w),
            faces=faces,
        )
        
        # Add to combined (take max for overlapping regions)
        combined_depth = np.maximum(combined_depth, depth_img.astype(np.float32))
    
    result = combined_depth.astype(np.uint8)
    
    # Overlay Canny edges if requested
    if canny_edges is not None:
        result = combine_depth_and_canny(result, canny_edges)
    
    return result


def process_video(
    video_path: Path,
    output_path: Path,
    device: str = "cuda",
    bbox_thresh: float = 0.5,
    include_canny: bool = True,
) -> bool:
    """
    Process a video and generate depth reference video.
    
    Args:
        video_path: Path to input video
        output_path: Path to save depth video
        device: CUDA device
        bbox_thresh: Detection threshold
        include_canny: Whether to overlay Canny edges
    
    Returns:
        True if successful
    """
    estimator, faces = get_sam3d_estimator(device)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        console.print(f"[bold red]Error:[/] Could not open video: {video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        depth_frame = process_frame(frame, estimator, faces, bbox_thresh, include_canny)
        out.write(depth_frame)
        
        frame_idx += 1
        if frame_idx % 10 == 0:
            console.print(f"  Processed {frame_idx}/{frame_count} frames")
    
    cap.release()
    out.release()
    
    return True


def process_image(
    image_path: Path,
    output_path: Path,
    device: str = "cuda",
    bbox_thresh: float = 0.5,
    include_canny: bool = True,
) -> bool:
    """
    Process a single image and generate depth map.
    
    Args:
        image_path: Path to input image
        output_path: Path to save depth image
        device: CUDA device
        bbox_thresh: Detection threshold
        include_canny: Whether to overlay Canny edges
    
    Returns:
        True if successful
    """
    estimator, faces = get_sam3d_estimator(device)
    
    # Read image
    frame = cv2.imread(str(image_path))
    if frame is None:
        console.print(f"[bold red]Error:[/] Could not read image: {image_path}")
        return False
    
    # Process frame
    depth_frame = process_frame(frame, estimator, faces, bbox_thresh, include_canny)
    
    # Save
    cv2.imwrite(str(output_path), depth_frame)
    
    return True


def get_meta_data(output_path: Path) -> List[Dict]:
    """Load existing dataset JSON."""
    if not output_path.exists():
        return []
    
    with output_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset_json(
    data: List[Dict],
    output_path: Path,
) -> None:
    """Save dataset JSON with reference paths."""
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    console.print(f"[bold green]✓[/] Dataset saved to [cyan]{output_path}[/]")


app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Compute 3D depth reference videos/images for IC-LoRA training using SAM-3D Body.",
)


@app.command()
def main(
    input_path: Path = typer.Argument(
        ...,
        help="Path to input directory containing videos/images",
        exists=True,
    ),
    output: Path = typer.Option(
        None,
        "--output", "-o",
        help="Path to dataset JSON file (will be updated with reference_path field)",
    ),
    device: str = typer.Option(
        "cuda",
        "--device", "-d",
        help="CUDA device to use",
    ),
    bbox_thresh: float = typer.Option(
        0.5,
        "--bbox-thresh",
        help="Bounding box detection threshold",
    ),
    override: bool = typer.Option(
        False,
        "--override",
        help="Override existing reference files",
    ),
    no_canny: bool = typer.Option(
        False,
        "--no-canny",
        help="Disable Canny edge overlay (output depth only)",
    ),
) -> None:
    """
    Compute 3D depth references for IC-LoRA training.
    
    By default, outputs SAM-3D depth map with Canny edges overlaid.
    Use --no-canny for depth-only output.
    
    Examples:
        # Process all media with depth + canny overlay
        python compute_3d_reference.py videos_dir/ -o videos_dir/dataset.json
        
        # Process with depth only (no canny)
        python compute_3d_reference.py videos_dir/ -o videos_dir/dataset.json --no-canny
    """
    
    include_canny = not no_canny
    
    if output is None:
        output = input_path / "dataset.json"
    
    output = Path(output).resolve()
    mode_str = "Depth + Canny" if include_canny else "Depth only"
    console.print(f"Using dataset file: [bold blue]{output}[/]")
    console.print(f"Output mode: [bold green]{mode_str}[/]")
    
    # Load existing dataset
    if not output.exists():
        console.print(f"[bold yellow]Warning:[/] Dataset file not found, creating new one")
        # Scan for media files
        media_extensions = {".mp4", ".avi", ".mov", ".mkv", ".png", ".jpg", ".jpeg"}
        media_files = []
        for ext in media_extensions:
            media_files.extend(input_path.glob(f"*{ext}"))
            media_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        data = [
            {"media_path": str(f.relative_to(input_path)), "caption": ""}
            for f in sorted(media_files)
        ]
    else:
        data = get_meta_data(output)
    
    if not data:
        console.print("[bold red]Error:[/] No media files found")
        raise typer.Exit(1)
    
    console.print(f"Found [bold]{len(data)}[/] media files to process")
    
    # Process each media file
    base_dir = input_path.resolve()
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    )
    
    # Determine output suffix
    suffix = "_3d_depth_canny" if include_canny else "_3d_depth"
    
    with progress:
        task = progress.add_task("Processing media", total=len(data))
        
        for item in data:
            media_path = base_dir / item["media_path"]
            
            # Determine output path
            reference_path = media_path.parent / (media_path.stem + suffix + media_path.suffix)
            
            progress.update(task, description=f"Processing [bold blue]{media_path.name}[/]")
            
            # Check if already exists
            if reference_path.exists() and not override:
                item["reference_path"] = str(reference_path.relative_to(base_dir))
                progress.advance(task)
                continue
            
            # Process based on media type
            success = False
            if media_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                success = process_video(media_path, reference_path, device, bbox_thresh, include_canny)
            else:
                success = process_image(media_path, reference_path, device, bbox_thresh, include_canny)
            
            if success:
                item["reference_path"] = str(reference_path.relative_to(base_dir))
            
            progress.advance(task)
    
    # Save updated dataset
    save_dataset_json(data, output)
    
    console.print(f"[bold green]✓[/] Processing complete!")
    console.print("Use these files with IC-LoRA training by setting:")
    console.print("  reference_column='[cyan]reference_path[/]'")


if __name__ == "__main__":
    app()
