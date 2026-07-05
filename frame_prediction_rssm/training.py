"""
Training and inference utilities.
"""

import jax
import jax.numpy as jnp
import optax
import flax
from flax.training import train_state
import numpy as np
from typing import Dict, Callable, Tuple
from functools import partial

from .pipeline import FramePredictionPipeline
from .rssm_core import RSSMState


def create_train_state(
    rng: jnp.ndarray,
    learning_rate: float = 1e-4,
    **pipeline_kwargs
) -> Tuple[train_state.TrainState, Callable]:
    """
    Create training state with initialized parameters.
    
    Args:
        rng: Random seed
        learning_rate: Initial learning rate
        **pipeline_kwargs: Arguments for FramePredictionPipeline
        
    Returns:
        train_state: Flax TrainState
        apply_fn: Function to apply model
    """
    # Create dummy input
    dummy_frames = jnp.zeros((1, 4, 64, 64, 3), dtype=jnp.uint8)
    
    # Initialize model
    model = FramePredictionPipeline(**pipeline_kwargs)
    
    # Initialize parameters
    params = model.init(rng, dummy_frames, training=True)
    
    # Create optimizer
    tx = optax.adam(learning_rate)
    
    # Create train state
    train_state_obj = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    
    return train_state_obj, model.apply


def compute_frame_reconstruction_loss(
    predicted_frame: jnp.ndarray,
    target_frame: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute L2 reconstruction loss between predicted and target frames.
    
    Args:
        predicted_frame: [batch, height, width, 3] in [0, 1]
        target_frame: [batch, height, width, 3] in [0, 255] or [0, 1]
        
    Returns:
        loss: scalar, mean squared error
    """
    # Normalize target frame if needed
    if target_frame.dtype == jnp.uint8:
        target_frame = target_frame.astype(jnp.float32) / 255.0
    
    # L2 reconstruction loss
    loss = jnp.mean((predicted_frame - target_frame) ** 2)
    
    return loss


def compute_point_cloud_regularization(
    point_cloud: Dict,
    lambda_confidence: float = 0.1,
    lambda_depth: float = 0.01
) -> jnp.ndarray:
    """
    Compute regularization losses for point cloud generation.
    
    Args:
        point_cloud: Output from PointCloudGenerator
        lambda_confidence: Weight for confidence regularization
        lambda_depth: Weight for depth smoothness regularization
        
    Returns:
        loss: scalar regularization loss
    """
    # Confidence entropy regularization (encourage confident predictions)
    confidences = point_cloud['confidences']
    entropy = -jnp.mean(
        confidences * jnp.log(confidences + 1e-8) + 
        (1 - confidences) * jnp.log(1 - confidences + 1e-8)
    )
    confidence_loss = lambda_confidence * entropy
    
    # Depth smoothness (encourage smooth depth maps)
    depth_map = point_cloud['depth_map']
    dx = jnp.abs(depth_map[:, 1:, :] - depth_map[:, :-1, :])
    dy = jnp.abs(depth_map[:, :, 1:] - depth_map[:, :, :-1])
    depth_smooth_loss = lambda_depth * (jnp.mean(dx) + jnp.mean(dy))
    
    total_loss = confidence_loss + depth_smooth_loss
    
    return total_loss


@partial(jax.jit)
def train_step(
    state: train_state.TrainState,
    frames_1to4: jnp.ndarray,
    target_frame_5: jnp.ndarray,
    apply_fn: Callable
) -> Tuple[train_state.TrainState, Dict]:
    """
    Single training step.
    
    Args:
        state: Current training state
        frames_1to4: [batch, 4, height, width, 3]
        target_frame_5: [batch, height, width, 3]
        apply_fn: Model apply function
        
    Returns:
        new_state: Updated training state
        metrics: Loss metrics
    """
    def loss_fn(params):
        outputs = apply_fn(
            {'params': params},
            frames_1to4,
            training=True
        )
        
        # Frame reconstruction loss
        pred_frame = outputs['predicted_frame']
        frame_loss = compute_frame_reconstruction_loss(pred_frame, target_frame_5)
        
        # Point cloud regularization
        pc_loss = compute_point_cloud_regularization(outputs['point_cloud'])
        
        # Total loss
        total_loss = frame_loss + 0.1 * pc_loss
        
        return total_loss, {
            'frame_loss': frame_loss,
            'pc_loss': pc_loss,
        }
    
    (loss, aux_losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    
    metrics = {
        'total_loss': loss,
        **aux_losses
    }
    
    return new_state, metrics


@partial(jax.jit)
def eval_step(
    state: train_state.TrainState,
    frames_1to4: jnp.ndarray,
    target_frame_5: jnp.ndarray,
    apply_fn: Callable
) -> Dict:
    """
    Evaluation step (no gradient computation).
    
    Args:
        state: Current training state
        frames_1to4: [batch, 4, height, width, 3]
        target_frame_5: [batch, height, width, 3]
        apply_fn: Model apply function
        
    Returns:
        metrics: Evaluation metrics
    """
    outputs = apply_fn(
        {'params': state.params},
        frames_1to4,
        training=False
    )
    
    pred_frame = outputs['predicted_frame']
    frame_loss = compute_frame_reconstruction_loss(pred_frame, target_frame_5)
    pc_loss = compute_point_cloud_regularization(outputs['point_cloud'])
    
    # Additional image quality metrics
    mse = jnp.mean((pred_frame - target_frame_5/255.0) ** 2)
    psnr = 20 * jnp.log10(1.0 / jnp.sqrt(mse + 1e-8))
    
    metrics = {
        'frame_loss': frame_loss,
        'pc_loss': pc_loss,
        'total_loss': frame_loss + 0.1 * pc_loss,
        'mse': mse,
        'psnr': psnr,
    }
    
    return metrics


def predict(
    state: train_state.TrainState,
    frames_1to4: jnp.ndarray,
    apply_fn: Callable
) -> Tuple[jnp.ndarray, Dict]:
    """
    Generate predictions for a batch of frame sequences.
    
    Args:
        state: Training state with model parameters
        frames_1to4: [batch, 4, height, width, 3]
        apply_fn: Model apply function
        
    Returns:
        predicted_frame: [batch, height, width, 3]
        point_cloud: Point cloud data
    """
    outputs = apply_fn(
        {'params': state.params},
        frames_1to4,
        training=False
    )
    
    return outputs['predicted_frame'], outputs['point_cloud']
