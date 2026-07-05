"""
Example usage of frame prediction pipeline.
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import pickle

from frame_prediction_rssm.pipeline import FramePredictionPipeline
from frame_prediction_rssm.training import create_train_state, train_step, eval_step, predict


def generate_dummy_data(batch_size: int = 4, num_sequences: int = 100):
    """
    Generate dummy frame sequences for testing.
    
    Args:
        batch_size: Batch size
        num_sequences: Number of sequences
        
    Returns:
        frame_sequences: [num_sequences, batch_size, 5, 64, 64, 3] dummy frames
    """
    frame_sequences = []
    for _ in range(num_sequences):
        # Generate 5 consecutive frames (last one is target)
        sequence = np.random.randint(0, 255, (batch_size, 5, 64, 64, 3), dtype=np.uint8)
        frame_sequences.append(sequence)
    
    return np.array(frame_sequences)


def main():
    """
    Simple training loop example.
    """
    print("Frame Prediction with RSSM - Training Example")
    print("=" * 50)
    
    # Hyperparameters
    batch_size = 4
    num_epochs = 1
    learning_rate = 1e-4
    seed = 42
    
    # Set up random number generation
    rng = jax.random.PRNGKey(seed)
    
    # Create dummy data
    print("Generating dummy data...")
    frame_sequences = generate_dummy_data(batch_size, num_sequences=10)
    
    # Initialize model and training state
    print("Initializing model...")
    train_state_obj, apply_fn = create_train_state(
        rng,
        learning_rate=learning_rate,
        encoder_hidden_dim=32,  # Smaller for demo
        encoder_latent_dim=128,
        rssm_deter_size=128,
        rssm_stoch_size=16,
        decoder_hidden_dim=32,
    )
    
    print(f"Model initialized with {sum(p.size for p in jax.tree.leaves(train_state_obj.params))} parameters")
    
    # Training loop
    print("\nTraining...")
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for seq_idx, sequence in enumerate(frame_sequences[:8]):  # Mini training set
            frames_1to4 = sequence[:, :4]  # Taking frames 0-3 instead of 1-4
            target_frame_5 = sequence[:, 4]  # Frame 4 (5th frame)
            
            # Train step
            train_state_obj, metrics = train_step(
                train_state_obj,
                frames_1to4,
                target_frame_5,
                apply_fn
            )
            
            epoch_losses.append(float(metrics['total_loss']))
            
            if (seq_idx + 1) % 2 == 0:
                avg_loss = np.mean(epoch_losses[-2:])
                print(f"  Batch {seq_idx + 1}: Loss = {avg_loss:.6f}")
        
        print(f"Epoch {epoch + 1} - Avg Loss: {np.mean(epoch_losses):.6f}")
    
    # Evaluation
    print("\nEvaluation...")
    eval_state_obj = train_state_obj
    eval_metrics = []
    
    for sequence in frame_sequences[8:]:
        frames_1to4 = sequence[:, :4]
        target_frame_5 = sequence[:, 4]
        
        metrics = eval_step(eval_state_obj, frames_1to4, target_frame_5, apply_fn)
        eval_metrics.append(metrics)
    
    # Print evaluation results
    avg_metrics = {
        k: np.mean([m[k] for m in eval_metrics])
        for k in eval_metrics[0].keys()
    }
    
    print("Evaluation Results:")
    for metric_name, metric_value in avg_metrics.items():
        print(f"  {metric_name}: {metric_value:.6f}")
    
    # Inference example
    print("\nInference Example...")
    test_sequence = frame_sequences[0]
    frames_1to4 = test_sequence[:, :4]
    
    predicted_frame, point_cloud = predict(eval_state_obj, frames_1to4, apply_fn)
    
    print(f"Predicted frame shape: {predicted_frame.shape}")
    print(f"Point cloud positions shape: {point_cloud['positions'].shape}")
    print(f"Point cloud features shape: {point_cloud['features'].shape}")
    print(f"Point cloud confidences shape: {point_cloud['confidences'].shape}")
    
    # Save prediction example
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Save as numpy
    np.save(output_dir / "predicted_frame.npy", predicted_frame)
    np.save(output_dir / "point_cloud_positions.npy", point_cloud['positions'])
    np.save(output_dir / "point_cloud_features.npy", point_cloud['features'])
    
    print(f"\nPredictions saved to {output_dir}/")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
