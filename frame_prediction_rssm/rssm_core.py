"""
Simplified RSSM (Recurrent State Space Model) for frame prediction.
Based on DreamerV3's RSSM architecture.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Tuple, NamedTuple
import math


class RSSMState(NamedTuple):
    """RSSM state representation"""
    deter: jnp.ndarray  # Deterministic state [batch, deter_size]
    stoch: jnp.ndarray  # Stochastic state [batch, stoch_size, classes]


class RSSMCore(nn.Module):
    """
    Core RSSM dynamics model.
    Predicts next deterministic state from current state and action.
    """
    deter_size: int = 256  # Deterministic state dimension
    stoch_size: int = 32   # Number of stochastic latent variables
    classes: int = 32      # Number of classes per stochastic variable
    hidden_size: int = 256
    num_layers: int = 2
    
    @nn.compact
    def __call__(self, deter: jnp.ndarray, stoch: jnp.ndarray, 
                 action: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Update deterministic state using current state and action.
        
        Args:
            deter: [batch, deter_size] deterministic state
            stoch: [batch, stoch_size, classes] stochastic state
            action: [batch, action_size] action vector
            training: training mode flag
            
        Returns:
            new_deter: [batch, deter_size] updated deterministic state
        """
        # Flatten stochastic state
        batch_size = deter.shape[0]
        stoch_flat = stoch.reshape(batch_size, -1)  # [batch, stoch_size * classes]
        
        # Process each input stream
        deter_embed = nn.Dense(self.hidden_size, name='deter_embed')(deter)
        deter_embed = nn.gelu(deter_embed)
        
        stoch_embed = nn.Dense(self.hidden_size, name='stoch_embed')(stoch_flat)
        stoch_embed = nn.gelu(stoch_embed)
        
        action_embed = nn.Dense(self.hidden_size, name='action_embed')(action)
        action_embed = nn.gelu(action_embed)
        
        # Concatenate embeddings
        x = jnp.concatenate([deter_embed, stoch_embed, action_embed], axis=-1)
        
        # Process through MLP layers
        for i in range(self.num_layers):
            x = nn.Dense(self.hidden_size, name=f'layer_{i}')(x)
            x = nn.LayerNorm()(x)
            x = nn.gelu(x)
        
        # Final layer: output new deterministic state
        new_deter = nn.Dense(self.deter_size, name='deter_out')(x)
        
        # Mix with previous deterministic state (RNN-like gating)
        gate = nn.Dense(self.deter_size, name='gate')(x)
        gate = nn.sigmoid(gate)
        new_deter = gate * new_deter + (1 - gate) * deter
        
        return new_deter


class StochasticEncoder(nn.Module):
    """
    Encodes deterministic state to stochastic latent space.
    """
    stoch_size: int = 32
    classes: int = 32
    hidden_size: int = 256
    
    @nn.compact
    def __call__(self, deter: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Encode deterministic state to stochastic logits.
        
        Args:
            deter: [batch, deter_size] deterministic state
            
        Returns:
            logits: [batch, stoch_size, classes] logits for each stochastic variable
        """
        x = deter
        x = nn.Dense(self.hidden_size, name='enc_layer_0')(x)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)
        
        x = nn.Dense(self.hidden_size, name='enc_layer_1')(x)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)
        
        # Output logits for each stochastic dimension
        logits = nn.Dense(self.stoch_size * self.classes, name='logits')(x)
        logits = logits.reshape((-1, self.stoch_size, self.classes))
        
        return logits


class RSSM(nn.Module):
    """
    Complete RSSM model combining dynamics and stochastic encoding.
    """
    deter_size: int = 256
    stoch_size: int = 32
    classes: int = 32
    hidden_size: int = 256
    
    @nn.compact
    def __call__(self, state: RSSMState, action: jnp.ndarray, 
                 training: bool = True) -> Tuple[RSSMState, Dict]:
        """
        Single RSSM step: predict next state given current state and action.
        
        Args:
            state: Current RSSMState
            action: [batch, action_size] action vector
            training: training mode flag
            
        Returns:
            next_state: Predicted RSSMState
            features: Dictionary with intermediate features for loss computation
        """
        # Update deterministic state
        dynamics = RSSMCore(
            deter_size=self.deter_size,
            stoch_size=self.stoch_size,
            classes=self.classes,
            hidden_size=self.hidden_size,
            name='dynamics'
        )
        new_deter = dynamics(state.deter, state.stoch, action, training)
        
        # Generate stochastic state from deterministic
        stoch_encoder = StochasticEncoder(
            stoch_size=self.stoch_size,
            classes=self.classes,
            hidden_size=self.hidden_size,
            name='stoch_encoder'
        )
        logits = stoch_encoder(new_deter, training)
        
        # Sample stochastic state from logits
        # In training: use gumbel-softmax or relaxed sampling
        # In inference: use argmax or sampling
        if training:
            # Temperature-based sampling for differentiability
            new_stoch = jax.nn.softmax(logits / 0.5, axis=-1)
        else:
            # Deterministic: one-hot encode argmax
            indices = jnp.argmax(logits, axis=-1)
            new_stoch = jax.nn.one_hot(indices, self.classes, dtype=jnp.float32)
        
        next_state = RSSMState(deter=new_deter, stoch=new_stoch)
        
        features = {
            'deter': new_deter,
            'stoch': new_stoch,
            'logits': logits,
        }
        
        return next_state, features


class RSSMSequence(nn.Module):
    """
    Process a sequence of frames through RSSM.
    Maintains internal state across time steps.
    """
    deter_size: int = 256
    stoch_size: int = 32
    classes: int = 32
    
    @nn.compact
    def __call__(self, frame_tokens: jnp.ndarray, actions: jnp.ndarray,
                 initial_state: RSSMState, training: bool = True,
                 num_steps: int = None):
        """
        Process frame sequence through RSSM.
        
        Args:
            frame_tokens: [batch, time, feature_size] encoded frame features
            actions: [batch, time-1, action_size] or zeros if no actions
            initial_state: Initial RSSMState
            training: training mode flag
            num_steps: number of steps to unroll (default: time dimension)
            
        Returns:
            states_sequence: List of RSSMState for each step
            features_sequence: List of feature dicts
        """
        if num_steps is None:
            num_steps = frame_tokens.shape[1]
        
        rssm = RSSM(
            deter_size=self.deter_size,
            stoch_size=self.stoch_size,
            classes=self.classes,
            name='rssm'
        )
        
        states = []
        features = []
        state = initial_state
        
        for t in range(num_steps):
            # Get action (or zero if at first step)
            if t < actions.shape[1]:
                action = actions[:, t]
            else:
                action = jnp.zeros((frame_tokens.shape[0], actions.shape[-1]))
            
            # RSSM step
            next_state, feat = rssm(state, action, training)
            
            states.append(next_state)
            features.append(feat)
            
            state = next_state
        
        return states, features
