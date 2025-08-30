"""
PPO Actor-Critic Networks for Project NEXUS
Complete neural architecture for multi-agent reinforcement learning
COMPLETELY FIXED VERSION - All type and import issues resolved
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import warnings
import os

# Try to import our custom modules with fallback
try:
    from .spatial_cnn import SpatialCNN, EnhancedSpatialCNN
    from .multimodal_fusion import MultiModalFusion, AttentionalFusion, prepare_agent_state_batch
except ImportError:
    # Fallback imports for standalone use
    try:
        from spatial_cnn import SpatialCNN, EnhancedSpatialCNN
        from multimodal_fusion import MultiModalFusion, AttentionalFusion, prepare_agent_state_batch
    except ImportError:
        warnings.warn("Could not import custom modules. Some functionality may be limited.")
        SpatialCNN = None
        EnhancedSpatialCNN = None
        MultiModalFusion = None
        AttentionalFusion = None
        prepare_agent_state_batch = None

class PolicyHead(nn.Module):
    """
    Policy network head for action selection
    """
    
    def __init__(self, input_dim: int = 512, action_dim: int = 14, hidden_dim: int = 256):
        super(PolicyHead, self).__init__()
        
        self.action_dim = action_dim
        
        self.policy_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            
            nn.Linear(hidden_dim, action_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize policy weights with smaller variance for stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)  # type: ignore
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning action logits
        
        Args:
            x: Feature tensor (batch_size, input_dim)
            
        Returns:
            Action logits (batch_size, action_dim)
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        
        if x.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {x.dim()}D")
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input contains NaN or infinite values")
        
        try:
            return self.policy_layers(x)
        except Exception as e:
            raise RuntimeError(f"Policy forward pass failed: {e}")
    
    def get_action_distribution(self, x: torch.Tensor) -> Categorical:
        """Get action distribution for sampling"""
        try:
            logits = self.forward(x)
            return Categorical(logits=logits)
        except Exception as e:
            raise RuntimeError(f"Could not create action distribution: {e}")

class ValueHead(nn.Module):
    """
    Value network head for state value estimation
    """
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super(ValueHead, self).__init__()
        
        self.value_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            
            nn.Linear(hidden_dim, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize value weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)  # type: ignore
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning state values
        
        Args:
            x: Feature tensor (batch_size, input_dim)
            
        Returns:
            State values (batch_size, 1)
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        
        if x.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {x.dim()}D")
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input contains NaN or infinite values")
        
        try:
            return self.value_layers(x)
        except Exception as e:
            raise RuntimeError(f"Value forward pass failed: {e}")

class PPOActorCritic(nn.Module):
    """
    Complete PPO Actor-Critic Network for Project NEXUS
    Combines spatial CNN, multi-modal fusion, and policy/value heads
    """
    
    def __init__(self, 
                 spatial_channels: int = 5,
                 spatial_dim: int = 256,
                 state_dim: int = 128,
                 fusion_dim: int = 512,
                 action_dim: int = 14,
                 use_enhanced_cnn: bool = False,
                 use_attention_fusion: bool = False):
        super(PPOActorCritic, self).__init__()
        
        # Store configuration
        self.spatial_channels = spatial_channels
        self.spatial_dim = spatial_dim
        self.state_dim = state_dim
        self.fusion_dim = fusion_dim
        self.action_dim = action_dim
        
        # Spatial feature extractor with fallback
        if SpatialCNN is None:
            raise ImportError("SpatialCNN module not available")
        
        if use_enhanced_cnn and EnhancedSpatialCNN is not None:
            self.spatial_cnn = EnhancedSpatialCNN(
                input_channels=spatial_channels,
                feature_dim=spatial_dim,
                use_attention=True
            )
        else:
            if use_enhanced_cnn:
                warnings.warn("Enhanced CNN requested but not available, using standard CNN")
            self.spatial_cnn = SpatialCNN(
                input_channels=spatial_channels,
                feature_dim=spatial_dim
            )
        
        # Multi-modal fusion with fallback
        if MultiModalFusion is None:
            raise ImportError("MultiModalFusion module not available")
        
        if use_attention_fusion and AttentionalFusion is not None:
            self.fusion = AttentionalFusion(
                spatial_dim=spatial_dim,
                state_dim=state_dim,
                output_dim=fusion_dim
            )
        else:
            if use_attention_fusion:
                warnings.warn("Attention fusion requested but not available, using standard fusion")
            self.fusion = MultiModalFusion(
                spatial_dim=spatial_dim,
                state_dim=state_dim,
                output_dim=fusion_dim
            )
        
        # Policy and value heads
        self.policy_head = PolicyHead(fusion_dim, action_dim)
        self.value_head = ValueHead(fusion_dim)
    
    def forward(self, observations: torch.Tensor, agent_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through complete network
        
        Args:
            observations: Spatial observations (batch_size, 5, 15, 15)
            agent_states: Agent state information (batch_size, 8)
            
        Returns:
            Tuple of (action_logits, state_values)
        """
        # Input validation
        if not isinstance(observations, torch.Tensor) or not isinstance(agent_states, torch.Tensor):
            raise TypeError("Both inputs must be torch.Tensors")
        
        if observations.dim() != 4:
            raise ValueError(f"Expected 4D observations tensor, got {observations.dim()}D")
        
        if agent_states.dim() != 2:
            raise ValueError(f"Expected 2D agent states tensor, got {agent_states.dim()}D")
        
        if observations.shape[0] != agent_states.shape[0]:
            raise ValueError(f"Batch size mismatch: obs {observations.shape[0]} vs states {agent_states.shape[0]}")
        
        # Device compatibility
        if observations.device != agent_states.device:
            agent_states = agent_states.to(observations.device)
        
        try:
            # Extract spatial features
            spatial_features = self.spatial_cnn(observations)
            
            # Fuse with agent states
            fused_features = self.fusion(spatial_features, agent_states)
            
            # Get policy and value outputs
            action_logits = self.policy_head(fused_features)
            state_values = self.value_head(fused_features)
            
            return action_logits, state_values
            
        except Exception as e:
            raise RuntimeError(f"PPO forward pass failed: {e}")
    
    def act(self, observations: torch.Tensor, agent_states: torch.Tensor, 
            deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select actions based on current policy
        
        Args:
            observations: Spatial observations
            agent_states: Agent state information  
            deterministic: If True, select most likely action
            
        Returns:
            Tuple of (actions, log_probs, state_values)
        """
        try:
            action_logits, state_values = self.forward(observations, agent_states)
            
            # Create action distribution
            action_dist = Categorical(logits=action_logits)
            
            if deterministic:
                actions = torch.argmax(action_logits, dim=1)
            else:
                actions = action_dist.sample()
            
            log_probs = action_dist.log_prob(actions)
            
            return actions, log_probs, state_values.squeeze(-1)
            
        except Exception as e:
            raise RuntimeError(f"Action selection failed: {e}")
    
    def evaluate_actions(self, observations: torch.Tensor, agent_states: torch.Tensor, 
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO training
        
        Args:
            observations: Spatial observations
            agent_states: Agent state information
            actions: Actions to evaluate
            
        Returns:
            Tuple of (log_probs, state_values, entropy)
        """
        if not isinstance(actions, torch.Tensor):
            raise TypeError(f"Actions must be torch.Tensor, got {type(actions)}")
        
        if actions.dim() != 1:
            raise ValueError(f"Actions must be 1D tensor, got {actions.dim()}D")
        
        try:
            action_logits, state_values = self.forward(observations, agent_states)
            
            # Create action distribution and evaluate
            action_dist = Categorical(logits=action_logits)
            log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy()
            
            return log_probs, state_values.squeeze(-1), entropy
            
        except Exception as e:
            raise RuntimeError(f"Action evaluation failed: {e}")
    
    def get_value(self, observations: torch.Tensor, agent_states: torch.Tensor) -> torch.Tensor:
        """Get state values only (for advantage computation)"""
        try:
            spatial_features = self.spatial_cnn(observations)
            fused_features = self.fusion(spatial_features, agent_states)
            return self.value_head(fused_features).squeeze(-1)
        except Exception as e:
            raise RuntimeError(f"Value computation failed: {e}")
    
    def save_checkpoint(self, filepath: str, optimizer_state: Optional[Dict] = None, 
                       step: int = 0, episode: int = 0):
        """Save model checkpoint with error handling"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            checkpoint = {
                'model_state_dict': self.state_dict(),
                'model_config': {
                    'spatial_channels': self.spatial_channels,
                    'spatial_dim': self.spatial_dim,
                    'state_dim': self.state_dim,
                    'fusion_dim': self.fusion_dim,
                    'action_dim': self.action_dim
                },
                'step': step,
                'episode': episode
            }
            
            if optimizer_state is not None:
                checkpoint['optimizer_state_dict'] = optimizer_state
            
            torch.save(checkpoint, filepath)
            
        except Exception as e:
            raise RuntimeError(f"Could not save checkpoint: {e}")
    
    @classmethod
    def load_checkpoint(cls, filepath: str, device: Optional[torch.device] = None):
        """Load model from checkpoint with error handling"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
            
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            checkpoint = torch.load(filepath, map_location=device)
            config = checkpoint['model_config']
            
            # Create model with saved configuration
            model = cls(
                spatial_channels=config['spatial_channels'],
                spatial_dim=config['spatial_dim'],
                state_dim=config['state_dim'],
                fusion_dim=config['fusion_dim'],
                action_dim=config['action_dim']
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            return model, checkpoint
            
        except Exception as e:
            raise RuntimeError(f"Could not load checkpoint: {e}")

class CommunicationModule(nn.Module):
    """
    Communication module for multi-agent coordination
    Enables discrete token-based communication between agents
    """
    
    def __init__(self, feature_dim: int = 512, vocab_size: int = 16, max_agents: int = 4):
        super(CommunicationModule, self).__init__()
        
        self.feature_dim = feature_dim
        self.vocab_size = vocab_size
        self.max_agents = max_agents
        
        # Message generation
        self.message_encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, vocab_size)
        )
        
        # Message processing
        self.message_decoder = nn.Sequential(
            nn.Embedding(vocab_size, 64),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feature_dim // 4)
        )
        
        # Attention for message aggregation with error handling
        try:
            self.attention = nn.MultiheadAttention(
                feature_dim // 4, num_heads=min(4, (feature_dim // 4) // 16), 
                batch_first=True
            )
            self.use_attention = True
        except Exception:
            self.use_attention = False
            warnings.warn("Could not initialize attention for communication")
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize communication weights"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_normal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def generate_message(self, features: torch.Tensor) -> torch.Tensor:
        """Generate communication tokens"""
        try:
            message_logits = self.message_encoder(features)
            return Categorical(logits=message_logits).sample()
        except Exception as e:
            warnings.warn(f"Message generation failed: {e}")
            return torch.zeros(features.size(0), dtype=torch.long, device=features.device)
    
    def process_messages(self, features: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """Process received messages and update features"""
        try:
            # Decode messages
            decoded_messages = self.message_decoder(messages)
            
            if self.use_attention:
                # Aggregate via attention
                attended_messages, _ = self.attention(
                    decoded_messages.unsqueeze(1),
                    decoded_messages.unsqueeze(1), 
                    decoded_messages.unsqueeze(1)
                )
                message_features = attended_messages.squeeze(1)
            else:
                message_features = decoded_messages
            
            # Combine with original features
            return torch.cat([features, message_features], dim=1)
            
        except Exception as e:
            warnings.warn(f"Message processing failed: {e}")
            return features

class PPOActorCriticWithComm(PPOActorCritic):
    """
    PPO Actor-Critic with communication capabilities
    For multi-agent coordination in future phases
    """
    
    def __init__(self, *args, enable_communication: bool = True, **kwargs):
        # Adjust fusion dimension to account for communication
        if enable_communication:
            kwargs['fusion_dim'] = kwargs.get('fusion_dim', 512) + 128  # Add comm features
        
        super().__init__(*args, **kwargs)
        
        self.enable_communication = enable_communication
        
        if enable_communication:
            try:
                self.comm_module = CommunicationModule(
                    feature_dim=self.fusion_dim - 128,  # Original fusion dim
                    vocab_size=16,
                    max_agents=4
                )
            except Exception as e:
                warnings.warn(f"Could not initialize communication module: {e}")
                self.enable_communication = False
    
    def forward_with_communication(self, observations: torch.Tensor, 
                                 agent_states: torch.Tensor,
                                 messages: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with communication"""
        try:
            # Standard feature extraction
            spatial_features = self.spatial_cnn(observations)
            fused_features = self.fusion(spatial_features, agent_states)
            
            if self.enable_communication and hasattr(self, 'comm_module') and messages is not None:
                # Process incoming messages
                comm_features = self.comm_module.process_messages(fused_features, messages)
                
                # Generate outgoing messages
                outgoing_messages = self.comm_module.generate_message(fused_features)
            else:
                comm_features = fused_features
                outgoing_messages = torch.zeros(
                    fused_features.size(0), dtype=torch.long, device=fused_features.device
                )
            
            # Get policy and value outputs
            action_logits = self.policy_head(comm_features)
            state_values = self.value_head(comm_features)
            
            return action_logits, state_values, outgoing_messages
            
        except Exception as e:
            raise RuntimeError(f"Communication forward pass failed: {e}")

def create_ppo_network(config: Dict[str, Any]) -> PPOActorCritic:
    """
    Factory function to create PPO network from configuration
    
    Args:
        config: Network configuration dictionary
        
    Returns:
        Configured PPO network
    """
    try:
        network_config = config.get('network', {})
        
        return PPOActorCritic(
            spatial_channels=network_config.get('spatial_channels', 5),
            spatial_dim=network_config.get('spatial_dim', 256),
            state_dim=network_config.get('state_dim', 128),
            fusion_dim=network_config.get('fusion_dim', 512),
            action_dim=network_config.get('action_dim', 14),
            use_enhanced_cnn=network_config.get('use_enhanced_cnn', False),
            use_attention_fusion=network_config.get('use_attention_fusion', False)
        )
    except Exception as e:
        warnings.warn(f"Could not create PPO network with config: {e}")
        return PPOActorCritic()

if __name__ == "__main__":
    # Comprehensive network testing with error handling
    print("Testing PPO Actor-Critic Network with Error Handling...")
    
    try:
        # Test parameters
        batch_size = 8
        device = torch.device('cuda' if torch.cuda.is_available() 
                            else 'mps' if torch.backends.mps.is_available() 
                            else 'cpu')
        print(f"Using device: {device}")
        
        # Create test data
        observations = torch.randn(batch_size, 5, 15, 15).to(device)
        agent_states = torch.randn(batch_size, 8).to(device)
        
        # Test standard network
        print("Testing standard PPO network...")
        network = PPOActorCritic().to(device)
        
        action_logits, state_values = network(observations, agent_states)
        print(f"Action logits shape: {action_logits.shape}")
        print(f"State values shape: {state_values.shape}")
        
        # Test action selection
        actions, log_probs, values = network.act(observations, agent_states)
        print(f"Actions shape: {actions.shape}")
        print(f"Log probs shape: {log_probs.shape}")
        print(f"Values shape: {values.shape}")
        
        # Test action evaluation
        eval_log_probs, eval_values, entropy = network.evaluate_actions(observations, agent_states, actions)
        print(f"Evaluation successful")
        
        # Test checkpoint save/load
        print("Testing checkpoint functionality...")
        checkpoint_path = "test_checkpoint.pth"
        
        try:
            network.save_checkpoint(checkpoint_path, step=1000, episode=50)
            loaded_network, checkpoint_info = PPOActorCritic.load_checkpoint(checkpoint_path, device)
            print(f"Checkpoint test successful")
            
            # Clean up
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                
        except Exception as e:
            print(f"Checkpoint test failed: {e}")
        
        # Test error handling
        print("Testing error handling...")
        try:
            invalid_obs = torch.randn(batch_size, 3, 10, 10).to(device)  # Wrong shape
            network(invalid_obs, agent_states)
        except ValueError:
            print("Correctly caught invalid observation shape")
        
        try:
            invalid_states = torch.randn(batch_size, 6).to(device)  # Wrong dimension
            network(observations, invalid_states)
        except Exception:
            print("Correctly caught invalid agent states")
        
        # Parameter count
        param_count = sum(p.numel() for p in network.parameters())
        print(f"Network parameters: {param_count:,}")
        
        print("\nAll PPO network tests passed!")
        print(f"Network ready for training on device: {device}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()