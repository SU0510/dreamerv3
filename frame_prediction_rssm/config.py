"""
Configuration presets for frame prediction.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class PipelineConfig:
    """Configuration for FramePredictionPipeline."""
    
    # Encoder configuration
    encoder_hidden_dim: int = 64
    encoder_latent_dim: int = 256
    encoder_conv_layers: int = 4
    encoder_temporal_layers: int = 2
    
    # RSSM configuration
    rssm_deter_size: int = 256
    rssm_stoch_size: int = 32
    rssm_classes: int = 32
    rssm_hidden_size: int = 256
    
    # Decoder configuration
    decoder_hidden_dim: int = 64
    decoder_target_height: int = 64
    decoder_target_width: int = 64
    
    # Point cloud configuration
    pc_feature_dim: int = 256
    pc_spatial_height: int = 16
    pc_spatial_width: int = 16
    pc_depth_range: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'encoder_hidden_dim': self.encoder_hidden_dim,
            'encoder_latent_dim': self.encoder_latent_dim,
            'encoder_conv_layers': self.encoder_conv_layers,
            'encoder_temporal_layers': self.encoder_temporal_layers,
            'rssm_deter_size': self.rssm_deter_size,
            'rssm_stoch_size': self.rssm_stoch_size,
            'rssm_classes': self.rssm_classes,
            'rssm_hidden_size': self.rssm_hidden_size,
            'decoder_hidden_dim': self.decoder_hidden_dim,
            'decoder_target_height': self.decoder_target_height,
            'decoder_target_width': self.decoder_target_width,
            'pc_feature_dim': self.pc_feature_dim,
            'pc_spatial_height': self.pc_spatial_height,
            'pc_spatial_width': self.pc_spatial_width,
            'pc_depth_range': self.pc_depth_range,
        }


# Predefined configurations
CONFIGS = {
    'tiny': PipelineConfig(
        encoder_hidden_dim=16,
        encoder_latent_dim=64,
        rssm_deter_size=64,
        rssm_stoch_size=8,
        decoder_hidden_dim=16,
        pc_feature_dim=64,
        pc_spatial_height=8,
        pc_spatial_width=8,
    ),
    'small': PipelineConfig(
        encoder_hidden_dim=32,
        encoder_latent_dim=128,
        rssm_deter_size=128,
        rssm_stoch_size=16,
        decoder_hidden_dim=32,
        pc_feature_dim=128,
        pc_spatial_height=12,
        pc_spatial_width=12,
    ),
    'medium': PipelineConfig(
        encoder_hidden_dim=64,
        encoder_latent_dim=256,
        rssm_deter_size=256,
        rssm_stoch_size=32,
        decoder_hidden_dim=64,
        pc_feature_dim=256,
        pc_spatial_height=16,
        pc_spatial_width=16,
    ),
    'large': PipelineConfig(
        encoder_hidden_dim=128,
        encoder_latent_dim=512,
        rssm_deter_size=512,
        rssm_stoch_size=64,
        rssm_hidden_size=512,
        decoder_hidden_dim=128,
        decoder_target_height=128,
        decoder_target_width=128,
        pc_feature_dim=512,
        pc_spatial_height=24,
        pc_spatial_width=24,
    ),
}


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    
    learning_rate: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 100
    
    # Loss weights
    frame_loss_weight: float = 1.0
    pc_loss_weight: float = 0.1
    
    # Regularization
    lambda_confidence: float = 0.1
    lambda_depth: float = 0.01
    
    # Optimization
    gradient_clip: float = 1.0
    warmup_steps: int = 1000
    
    # Logging
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 1000


def get_config(config_name: str = 'small') -> PipelineConfig:
    \"\"\"Get predefined configuration.\"\"\"
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[config_name]
