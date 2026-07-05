"""
Frame decoder and point cloud generator from RSSM latent features.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Tuple
import numpy as np


class FeatureDecoder(nn.Module):
    """
    Decodes RSSM latent state back to frame representation.
    Can output image or intermediate feature maps.
    """
    hidden_dim: int = 64
    target_height: int = 64
    target_width: int = 64
    num_layers: int = 4
    kernel_size: int = 3
    
    @nn.compact
    def __call__(self, deter: jnp.ndarray, stoch: jnp.ndarray, 
                 training: bool = True) -> jnp.ndarray:
        """
        Decode latent state to feature map or frame.
        
        Args:
            deter: [batch, deter_size] deterministic state
            stoch: [batch, stoch_size * classes] flattened stochastic state
            training: training mode flag
            
        Returns:
            decoded_frame: [batch, height, width, channels] decoded frame
        """
        batch_size = deter.shape[0]
        
        # Combine deter and stoch features
        combined = jnp.concatenate([deter, stoch], axis=-1)
        
        # Project to spatial dimensions
        spatial_dim = self.target_height // (2 ** (self.num_layers - 1))
        x = nn.Dense(
            spatial_dim * spatial_dim * self.hidden_dim * (2 ** (self.num_layers - 1)),
            name='to_spatial'
        )(combined)
        x = nn.gelu(x)
        
        # Reshape to spatial tensor
        x = x.reshape((batch_size, spatial_dim, spatial_dim, self.hidden_dim * (2 ** (self.num_layers - 1))))
        
        # Progressive upsampling
        for layer_idx in range(self.num_layers - 1, 0, -1):
            out_channels = self.hidden_dim * (2 ** (layer_idx - 1))
            
            # Upsample
            x = jax.image.resize(
                x, 
                shape=(x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]),
                method='nearest'
            )
            
            # Transpose convolution / regular conv
            x = nn.Conv(
                features=out_channels,
                kernel_size=(self.kernel_size, self.kernel_size),
                padding='SAME',
                name=f'deconv_{layer_idx}'
            )(x)
            
            # Batch norm
            x = nn.BatchNorm(use_running_average=not training, name=f'debn_{layer_idx}')(x)
            
            # Activation
            x = nn.gelu(x)
        
        # Final upsample to target resolution
        x = jax.image.resize(
            x,
            shape=(x.shape[0], self.target_height, self.target_width, x.shape[3]),
            method='nearest'
        )
        
        # Output layer: 3 channels for RGB
        x = nn.Conv(
            features=3,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding='SAME',
            name='output'
        )(x)
        
        # Sigmoid to [0, 1]
        x = nn.sigmoid(x)
        
        return x


class DenseFeatureExtractor(nn.Module):
    """
    Extracts dense features from latent representation.
    Each spatial location gets rich features for point cloud generation.
    """
    feature_dim: int = 256
    spatial_height: int = 16
    spatial_width: int = 16
    
    @nn.compact
    def __call__(self, deter: jnp.ndarray, stoch: jnp.ndarray,
                 training: bool = True) -> jnp.ndarray:
        """
        Extract dense features for point cloud.
        
        Args:
            deter: [batch, deter_size] deterministic state
            stoch: [batch, stoch_size, classes] stochastic state
            training: training mode flag
            
        Returns:
            dense_features: [batch, height, width, feature_dim] dense features
        """
        batch_size = deter.shape[0]
        
        # Flatten stochastic state
        stoch_flat = stoch.reshape(batch_size, -1)
        
        # Combine features
        combined = jnp.concatenate([deter, stoch_flat], axis=-1)
        
        # Project to spatial resolution
        spatial_size = self.spatial_height * self.spatial_width * self.feature_dim
        x = nn.Dense(spatial_size, name='to_spatial')(combined)
        x = nn.gelu(x)
        
        # Reshape to spatial tensor
        x = x.reshape((batch_size, self.spatial_height, self.spatial_width, self.feature_dim))
        
        # Refine features with conv layers
        for i in range(2):
            x = nn.Conv(
                features=self.feature_dim,
                kernel_size=(3, 3),
                padding='SAME',
                name=f'refine_{i}'
            )(x)
            x = nn.LayerNorm(name=f'norm_{i}')(x)
            x = nn.gelu(x)
        
        return x


class PointCloudGenerator(nn.Module):
    """
    Generate 4D point cloud (x, y, z, features) from latent representation.
    Extracts dense features and creates 3D coordinates.
    """
    feature_dim: int = 256
    spatial_height: int = 16
    spatial_width: int = 16
    depth_range: float = 1.0  # Max depth value
    
    @nn.compact
    def __call__(self, deter: jnp.ndarray, stoch: jnp.ndarray,
                 training: bool = True) -> Dict[str, jnp.ndarray]:
        """
        Generate 4D point cloud from RSSM latent state.
        
        Args:
            deter: [batch, deter_size] deterministic state
            stoch: [batch, stoch_size, classes] stochastic state
            training: training mode flag
            
        Returns:
            point_cloud: Dict with:
                - 'positions': [batch, height*width, 3] XYZ coordinates
                - 'features': [batch, height*width, feature_dim] point features
                - 'confidences': [batch, height*width] point confidence scores
        """
        batch_size = deter.shape[0]
        
        # Extract dense features
        feature_extractor = DenseFeatureExtractor(
            feature_dim=self.feature_dim,
            spatial_height=self.spatial_height,
            spatial_width=self.spatial_width,
            name='feature_extractor'
        )
        dense_features = feature_extractor(deter, stoch, training)
        
        # Predict depth map for Z coordinates
        depth_predictor = nn.Dense(1, name='depth_predictor')
        depth_map = nn.Conv(
            features=1,
            kernel_size=(3, 3),
            padding='SAME',
            name='depth_conv'
        )(dense_features)
        depth_map = nn.sigmoid(depth_map) * self.depth_range
        depth_map = depth_map[:, :, :, 0]  # [batch, height, width]
        
        # Predict confidence scores
        confidence_map = nn.Conv(
            features=1,
            kernel_size=(3, 3),
            padding='SAME',
            name='confidence_conv'
        )(dense_features)
        confidence_map = nn.sigmoid(confidence_map)[:, :, :, 0]  # [batch, height, width]
        
        # Generate XY grid coordinates
        h, w = self.spatial_height, self.spatial_width
        y_coords = jnp.linspace(-1, 1, h)  # Normalized to [-1, 1]
        x_coords = jnp.linspace(-1, 1, w)
        yy, xx = jnp.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Stack to get 3D positions
        positions_3d = jnp.stack([xx, yy, depth_map], axis=-1)  # [batch, height, width, 3]
        
        # Flatten spatial dimensions
        positions = positions_3d.reshape(batch_size, h * w, 3)  # [batch, h*w, 3]
        features = dense_features.reshape(batch_size, h * w, self.feature_dim)
        confidences = confidence_map.reshape(batch_size, h * w)
        
        point_cloud = {
            'positions': positions,      # [batch, num_points, 3]
            'features': features,         # [batch, num_points, feature_dim]
            'confidences': confidences,   # [batch, num_points]
            'depth_map': depth_map,       # [batch, height, width]
            'confidence_map': confidence_map,  # [batch, height, width]
        }
        
        return point_cloud


class AdaptivePointCloudGenerator(nn.Module):
    """
    Advanced point cloud generator with adaptive point sampling.
    Removes low-confidence points and keeps dense representation where needed.
    """
    feature_dim: int = 256
    spatial_height: int = 16
    spatial_width: int = 16
    depth_range: float = 1.0
    confidence_threshold: float = 0.3
    
    @nn.compact
    def __call__(self, deter: jnp.ndarray, stoch: jnp.ndarray,
                 training: bool = True) -> Dict[str, jnp.ndarray]:
        """
        Generate adaptive 4D point cloud.
        
        Args:
            deter: [batch, deter_size] deterministic state
            stoch: [batch, stoch_size, classes] stochastic state
            training: training mode flag
            
        Returns:
            point_cloud: Dict with variable-length point sets
        """
        # Generate base point cloud
        generator = PointCloudGenerator(
            feature_dim=self.feature_dim,
            spatial_height=self.spatial_height,
            spatial_width=self.spatial_width,
            depth_range=self.depth_range,
            name='base_generator'
        )
        point_cloud = generator(deter, stoch, training)
        
        # Filter by confidence threshold
        batch_size = deter.shape[0]
        h, w = self.spatial_height, self.spatial_width
        
        # Note: In JAX, we can't have ragged arrays easily, so we'll use masking
        confidence_mask = point_cloud['confidences'] > self.confidence_threshold
        
        # Add mask to point cloud dict
        point_cloud['valid_mask'] = confidence_mask
        
        # Apply confidence weighting to features
        weighted_features = point_cloud['features'] * point_cloud['confidences'][..., None]
        point_cloud['weighted_features'] = weighted_features
        
        return point_cloud
