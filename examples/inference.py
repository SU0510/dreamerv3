"""
Simple inference script for frame prediction.
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import argparse

from frame_prediction_rssm.training import create_train_state, predict


def _load_image(path, size=(64, 64)):
    from PIL import Image
    img = Image.open(path)
    img = img.resize(size)
    frame_array = np.array(img, dtype=np.uint8)
    if len(frame_array.shape) == 2:
        frame_array = np.stack([frame_array] * 3, axis=-1)
    if frame_array.shape[-1] == 4:
        frame_array = frame_array[..., :3]
    return frame_array


def load_frame_sequence_from_directory(frame_dir: str, num_input_frames: int = 4):
    """
    Load the first `num_input_frames` frames and an optional fifth frame.
    Expected: frame_0.png, frame_1.png, ... frame_4.png.
    
    Args:
        frame_dir: Path to directory containing frames
        num_input_frames: Number of conditioning frames to load
        
    Returns:
        frames_1to4: [1, num_input_frames, height, width, 3] frame array
        target_frame_5: [height, width, 3] or None
    """
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        print("PIL not available. Using dummy data instead.")
        frames = np.random.randint(0, 255, (1, num_input_frames, 64, 64, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        return frames, target
    
    frame_dir = Path(frame_dir)
    frame_list = []
    target_frame = None
    
    for i in range(num_input_frames + 1):
        frame_path = frame_dir / f"frame_{i}.png"
        if frame_path.exists():
            frame_array = _load_image(frame_path)
            if i < num_input_frames:
                frame_list.append(frame_array)
            else:
                target_frame = frame_array
        else:
            print(f"Warning: {frame_path} not found")
    
    if len(frame_list) < num_input_frames:
        # Pad with random frames if not enough
        while len(frame_list) < num_input_frames:
            frame_list.append(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    if target_frame is None:
        target_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    frames = np.stack(frame_list[:num_input_frames], axis=0)
    frames = np.expand_dims(frames, axis=0)  # Add batch dimension
    
    return frames, target_frame


def image_to_reference_point_cloud(image: np.ndarray):
    """
    Convert an RGB image into a reference point cloud for visualization.
    This is a visualization proxy, not a real sensor point cloud.
    """
    height, width = image.shape[:2]
    image_f = image.astype(np.float32) / 255.0
    gray = image_f.mean(axis=-1)

    y_coords = np.linspace(-1.0, 1.0, height)
    x_coords = np.linspace(-1.0, 1.0, width)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
    positions = np.stack([xx, yy, gray], axis=-1).reshape(-1, 3)
    features = image_f.reshape(-1, 3)
    confidences = np.ones((height * width,), dtype=np.float32)
    return {
        "positions": positions[None, ...],
        "features": features[None, ...],
        "confidences": confidences[None, ...],
        "depth_map": gray[None, ...],
    }


def visualize_results(frames_1to4, target_frame_5, predicted_frame, point_cloud, output_dir):
    """
    Visualize input frames, target frame, predicted frame, and point clouds.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization.")
        return

    reference_pc = image_to_reference_point_cloud(target_frame_5)
    pred_pc = point_cloud

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 5)

    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(frames_1to4[0, i])
        ax.set_title(f"Input {i + 1}")
        ax.axis("off")

    ax = fig.add_subplot(gs[0, 4])
    ax.imshow(target_frame_5)
    ax.set_title("Ground Truth Frame 5")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(predicted_frame[0])
    ax.set_title("Predicted Frame 5")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(target_frame_5)
    ax.set_title("GT Frame 5")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(reference_pc["depth_map"][0], cmap="viridis")
    ax.set_title("GT Pseudo Point Cloud Depth")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(pred_pc["depth_map"], cmap="viridis")
    ax.set_title("Predicted Point Cloud Depth")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 4], projection="3d")
    ref_positions = reference_pc["positions"][0]
    pred_positions = pred_pc["positions"][0]
    ref_conf = reference_pc["confidences"][0]
    pred_conf = pred_pc["confidences"][0]

    ax.scatter(ref_positions[:, 0], ref_positions[:, 1], ref_positions[:, 2], c=ref_conf, cmap="Blues", s=5, alpha=0.35, label="GT")
    ax.scatter(pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2], c=pred_conf, cmap="Reds", s=5, alpha=0.35, label="Pred")
    ax.set_title("GT vs Pred 3D Points")
    ax.legend(loc="upper right")
    ax.view_init(elev=20, azim=40)

    plt.tight_layout()
    fig_path = output_dir / "comparison_visualization.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Visualization saved to {fig_path}")


def save_point_cloud_ply(point_cloud: dict, output_path: str):
    """
    Save point cloud to PLY format.
    
    Args:
        point_cloud: Point cloud dictionary with positions, features, confidences
        output_path: Output file path
    """
    positions = point_cloud['positions'][0]  # First batch element
    confidences = point_cloud['confidences'][0]
    
    # Filter by confidence
    valid_mask = confidences > 0.3
    valid_positions = positions[valid_mask]
    valid_confidences = confidences[valid_mask]
    
    num_points = len(valid_positions)
    
    # Write PLY header
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write points
        colors = (valid_confidences * 255).astype(np.uint8)
        for i, pos in enumerate(valid_positions):
            x, y, z = pos
            r = colors[i]
            g = colors[i]
            b = colors[i]
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")
    
    print(f"Point cloud saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Frame prediction inference")
    parser.add_argument("--frame_dir", type=str, default="./frames",
                        help="Directory containing input frames")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for predictions")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--model_size", type=str, default="small",
                        choices=["tiny", "small", "medium"],
                        help="Model size")
    parser.add_argument("--visualize", action="store_true",
                        help="Show and save comparison visualization")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Frame Prediction Inference")
    print("=" * 50)
    
    # Set up random number generation
    rng = jax.random.PRNGKey(args.seed)
    
    # Model size configurations
    model_configs = {
        "tiny": {
            "encoder_hidden_dim": 16,
            "encoder_latent_dim": 64,
            "rssm_deter_size": 64,
            "rssm_stoch_size": 8,
            "decoder_hidden_dim": 16,
            "pc_feature_dim": 64,
        },
        "small": {
            "encoder_hidden_dim": 32,
            "encoder_latent_dim": 128,
            "rssm_deter_size": 128,
            "rssm_stoch_size": 16,
            "decoder_hidden_dim": 32,
            "pc_feature_dim": 128,
        },
        "medium": {
            "encoder_hidden_dim": 64,
            "encoder_latent_dim": 256,
            "rssm_deter_size": 256,
            "rssm_stoch_size": 32,
            "decoder_hidden_dim": 64,
            "pc_feature_dim": 256,
        },
    }
    
    config = model_configs[args.model_size]
    
    # Initialize model
    print(f"Initializing {args.model_size} model...")
    train_state_obj, apply_fn = create_train_state(rng, learning_rate=1e-4, **config)
    
    # Load frames
    print(f"Loading frames from {args.frame_dir}...")
    frames_1to4, target_frame_5 = load_frame_sequence_from_directory(args.frame_dir, num_input_frames=4)
    
    # Inference
    print("Running inference...")
    predicted_frame, point_cloud = predict(train_state_obj, frames_1to4, apply_fn)
    
    # Save outputs
    print(f"Saving outputs to {output_dir}...")
    
    # Save predicted frame
    frame_uint8 = (predicted_frame[0] * 255).astype(np.uint8)
    np.save(output_dir / "predicted_frame.npy", frame_uint8)
    
    try:
        from PIL import Image
        Image.fromarray(frame_uint8).save(output_dir / "predicted_frame.png")
    except ImportError:
        pass
    
    # Save point cloud
    np.save(output_dir / "point_cloud_positions.npy", point_cloud['positions'])
    np.save(output_dir / "point_cloud_features.npy", point_cloud['features'])
    np.save(output_dir / "point_cloud_confidences.npy", point_cloud['confidences'])
    
    # Save as PLY
    save_point_cloud_ply(point_cloud, output_dir / "predictions.ply")
    
    # Print statistics
    print("\nInference Results:")
    print(f"  Predicted frame: {predicted_frame.shape}")
    print(f"  Point cloud size: {point_cloud['positions'].shape[1]} points")
    print(f"  Average confidence: {point_cloud['confidences'].mean():.4f}")
    print(f"  Max depth: {point_cloud['depth_map'].max():.4f}")
    print(f"  Min depth: {point_cloud['depth_map'].min():.4f}")

    if args.visualize:
        print("\nCreating visualization...")
        visualize_results(frames_1to4, target_frame_5, predicted_frame, point_cloud, output_dir)
    
    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
