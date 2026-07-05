"""
Image encoder for extracting frame features for RSSM input.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Tuple


class FrameEncoder(nn.Module):
    """
    CNN-based encoder to convert frames to latent tokens.
    Processes both spatial and temporal information.
    """
    hidden_dim: int = 64
    latent_dim: int = 256
    num_layers: int = 4
    kernel_size: int = 3
    
    @nn.compact
    def __call__(self, frames: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Encode frames to latent tokens.
        
        Args:
            frames: [batch, time, height, width, channels] or [batch, height, width, channels]
            training: training mode flag
            
        Returns:
            tokens: [batch, time, latent_dim] or [batch, latent_dim] latent tokens
        """
        # Handle both single frame and sequence input
        input_shape = frames.shape
        if len(input_shape) == 5:  # [batch, time, height, width, channels]
            batch, time, height, width, channels = input_shape
            # Reshape to [batch*time, height, width, channels]
            frames = frames.reshape(-1, height, width, channels)
            is_sequence = True
        else:  # [batch, height, width, channels]
            batch = input_shape[0]
            time = 1
            is_sequence = False
        
        # Normalize frames to [-0.5, 0.5]
        x = frames.astype(jnp.float32) / 255.0 - 0.5
        
        # Convolutional layers with progressive downsampling
        for layer_idx in range(self.num_layers):
            out_channels = self.hidden_dim * (2 ** layer_idx)
            
            # Conv layer
            x = nn.Conv(
                features=out_channels,
                kernel_size=(self.kernel_size, self.kernel_size),
                strides=2 if layer_idx > 0 else 1,
                padding='SAME',
                name=f'conv_{layer_idx}'
            )(x)
            
            # Batch norm
            x = nn.BatchNorm(use_running_average=not training, name=f'bn_{layer_idx}')(x)
            
            # Activation
            x = nn.gelu(x)
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))  # [batch*time, hidden_dim * 2^(num_layers-1)]
        
        # Project to latent dimension
        x = nn.Dense(self.latent_dim, name='projection')(x)
        x = nn.gelu(x)
        
        # Reshape back to [batch, time, latent_dim]
        if is_sequence:
            x = x.reshape(batch, time, self.latent_dim)
        
        return x


class TemporalEncoder(nn.Module):
    """
    Encodes temporal relationships in frame sequences.
    Uses Transformer attention for capturing temporal structure.
    """
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 2
    
    @nn.compact
    def __call__(self, tokens: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Encode temporal structure in sequence of frame tokens.
        
        Args:
            tokens: [batch, time, latent_dim] sequence of frame tokens
            training: training mode flag
            
        Returns:
            temporal_features: [batch, time, latent_dim] temporally-aware features
        """
        x = tokens
        
        # Self-attention layers (simplified Transformer)
        for layer_idx in range(self.num_layers):
            # Multi-head attention
            x_attn = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                name=f'attention_{layer_idx}'
            )(x, x)
            
            # Residual connection + layer norm
            x = nn.LayerNorm()(x + x_attn)
            
            # Feed-forward
            ff = nn.Dense(self.hidden_dim, name=f'ff_dense1_{layer_idx}')(x)
            ff = nn.gelu(ff)
            ff = nn.Dense(x.shape[-1], name=f'ff_dense2_{layer_idx}')(ff)
            
            # Residual connection + layer norm
            x = nn.LayerNorm()(x + ff)
        
        return x


class MultiFrameEncoder(nn.Module):
    """
    Combines frame and temporal encoding for multi-frame sequences.
    """
    hidden_dim: int = 64
    latent_dim: int = 256
    num_conv_layers: int = 4
    num_temporal_layers: int = 2
    
    @nn.compact
    def __call__(self, frames: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Encode multi-frame sequence.
        
        Args:
            frames: [batch, time, height, width, channels] frame sequence
            training: training mode flag
            
        Returns:
            encoded_features: [batch, time, latent_dim] temporally-aware frame features
        """
        # Frame-level encoding
        frame_encoder = FrameEncoder(
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=self.num_conv_layers,
            name='frame_encoder'
        )
        frame_tokens = frame_encoder(frames, training)
        
        # Temporal encoding
        temporal_encoder = TemporalEncoder(
            hidden_dim=self.latent_dim,
            num_layers=self.num_temporal_layers,
            name='temporal_encoder'
        )
        encoded = temporal_encoder(frame_tokens, training)
        
        return encoded
