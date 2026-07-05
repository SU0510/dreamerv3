"""
Complete frame prediction pipeline combining encoder, RSSM, decoder, and point cloud generation.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Tuple, Optional
import numpy as np

from .rssm_core import RSSM, RSSMState, RSSMSequence
from .encoder import MultiFrameEncoder
from .decoder import FeatureDecoder, AdaptivePointCloudGenerator


class FramePredictionPipeline(nn.Module):
    """
    Complete pipeline: Frames -> Encoder -> RSSM -> Decoder + PointCloudGenerator
    
    Used for predicting frame 5 from frames 1-4 and generating dense point clouds.
    """
    # Encoder params
    encoder_hidden_dim: int = 64
    encoder_latent_dim: int = 256
    encoder_conv_layers: int = 4
    encoder_temporal_layers: int = 2
    
    # RSSM params
    rssm_deter_size: int = 256
    rssm_stoch_size: int = 32
    rssm_classes: int = 32
    rssm_hidden_size: int = 256
    
    # Decoder params
    decoder_hidden_dim: int = 64
    decoder_target_height: int = 64
    decoder_target_width: int = 64
    
    # Point cloud params
    pc_feature_dim: int = 256
    pc_spatial_height: int = 16
    pc_spatial_width: int = 16
    pc_depth_range: float = 1.0
    
    @nn.compact
    def __call__(self, frames_1to4: jnp.ndarray, 
                 training: bool = True) -> Dict:
        """
        Predict frame 5 and generate point cloud from frames 1-4.
        
        Args:
            frames_1to4: [batch, 4, height, width, 3] - 4 input frames (frames 1-4)
            training: training mode flag
            
        Returns:
            outputs: Dict containing:
                - 'predicted_frame': [batch, height, width, 3] predicted frame 5
                - 'point_cloud': Dict with positions, features, confidences
                - 'rssm_states': List of RSSMState objects
                - 'features': Intermediate features for loss computation
        """
        batch_size = frames_1to4.shape[0]
        
        # ============= ENCODER =============
        # Encode 4 input frames to latent tokens
        encoder = MultiFrameEncoder(
            hidden_dim=self.encoder_hidden_dim,
            latent_dim=self.encoder_latent_dim,
            num_conv_layers=self.encoder_conv_layers,
            num_temporal_layers=self.encoder_temporal_layers,
            name='frame_encoder'
        )
        frame_tokens = encoder(frames_1to4, training)  # [batch, 4, latent_dim]
        
        # ============= RSSM =============
        # Initialize RSSM state
        initial_state = RSSMState(
            deter=jnp.zeros((batch_size, self.rssm_deter_size), dtype=jnp.float32),
            stoch=jnp.zeros((batch_size, self.rssm_stoch_size, self.rssm_classes), dtype=jnp.float32)
        )
        
        # Dummy actions (no action in frame prediction)
        dummy_actions = jnp.zeros((batch_size, 3, 4))  # [batch, 3 steps, 4 action dims]
        
        # Run RSSM on input frames and predict next state
        rssm_sequence = RSSMSequence(
            deter_size=self.rssm_deter_size,
            stoch_size=self.rssm_stoch_size,
            classes=self.rssm_classes,
            name='rssm_sequence'
        )
        
        # Process 4 input frames through RSSM
        states, features_seq = rssm_sequence(
            frame_tokens, dummy_actions, initial_state, training, num_steps=4
        )
        
        # Get the final state after processing 4 frames
        final_state = states[-1]
        
        # Predict frame 5 by one more RSSM step with zero action
        rssm_model = RSSM(
            deter_size=self.rssm_deter_size,
            stoch_size=self.rssm_stoch_size,
            classes=self.rssm_classes,
            hidden_size=self.rssm_hidden_size,
            name='rssm_single'
        )
        
        zero_action = jnp.zeros((batch_size, 4))
        predicted_state, predicted_features = rssm_model(final_state, zero_action, training)
        
        # ============= DECODER =============
        # Flatten stochastic state for decoder
        stoch_flat = predicted_state.stoch.reshape(batch_size, -1)
        
        decoder = FeatureDecoder(
            hidden_dim=self.decoder_hidden_dim,
            target_height=self.decoder_target_height,
            target_width=self.decoder_target_width,
            num_layers=self.encoder_conv_layers,
            name='frame_decoder'
        )
        predicted_frame = decoder(predicted_state.deter, stoch_flat, training)
        
        # ============= POINT CLOUD GENERATION =============
        pc_generator = AdaptivePointCloudGenerator(
            feature_dim=self.pc_feature_dim,
            spatial_height=self.pc_spatial_height,
            spatial_width=self.pc_spatial_width,
            depth_range=self.pc_depth_range,
            confidence_threshold=0.3,
            name='point_cloud_generator'
        )
        point_cloud = pc_generator(predicted_state.deter, predicted_state.stoch, training)
        
        # ============= OUTPUT =============
        outputs = {
            'predicted_frame': predicted_frame,  # Predicted frame 5
            'point_cloud': point_cloud,           # 4D point cloud
            'predicted_state': predicted_state,   # RSSM state for frame 5
            'rssm_states': states,                # States for frames 1-4
            'predicted_features': predicted_features,  # Features for frame 5
            'all_features': features_seq,        # Features for all steps
        }
        
        return outputs


class InferenceModel(nn.Module):
    """
    Inference-only model wrapper with simpler interface.
    """
    pipeline: FramePredictionPipeline
    
    @nn.compact
    def __call__(self, frames_1to4: jnp.ndarray) -> Tuple[jnp.ndarray, Dict]:
        """
        Predict frame 5 and generate point cloud in inference mode.
        
        Args:
            frames_1to4: [batch, 4, height, width, 3] input frames
            
        Returns:
            predicted_frame: [batch, height, width, 3]
            point_cloud: Dict with positions, features, etc.
        """
        outputs = self.pipeline(frames_1to4, training=False)
        return outputs['predicted_frame'], outputs['point_cloud']
