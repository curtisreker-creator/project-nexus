# File: agents/networks/multimodal_fusion.py
"""
Multi-Modal Fusion Network for Project NEXUS
Combines spatial CNN features with agent state information
COMPLETELY FIXED VERSION - All syntax errors resolved
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import warnings

class AgentStateEncoder(nn.Module):
    """
    Encodes agent state information (inventory, health, energy) into feature vector
    """
    
    def __init__(self, state_dim: int = 128):
        super(AgentStateEncoder, self).__init__()
        
        self.state_dim = state_dim
        
        # Agent state components (based on GridWorld agent structure)
        self.inventory_size = 4  # wood, stone, food, tool
        self.status_size = 2     # health, energy
        self.position_size = 2   # x, y coordinates
        
        # Total input size
        self.input_size = self.inventory_size + self.status_size + self.position_size
        
        # State encoding layers with better normalization
        self.state_encoder = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(inplace=True),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            
            nn.Linear(128, state_dim),
            nn.ReLU(inplace=True)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, agent_states: torch.Tensor) -> torch.Tensor:
        """
        Encode agent state information
        
        Args:
            agent_states: Tensor of shape (batch_size, input_size)
                         Contains [inventory(4), health(1), energy(1), position(2)]
        
        Returns:
            State features of shape (batch_size, state_dim)
        """
        # Input validation
        if not isinstance(agent_states, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(agent_states)}")
        
        if agent_states.dim() != 2:
            raise ValueError(f"Expected 2D tensor (batch, features), got {agent_states.dim()}D")
        
        if agent_states.shape[1] != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {agent_states.shape[1]}")
        
        # Check for invalid values
        if torch.isnan(agent_states).any():
            raise ValueError("Agent states contain NaN values")
        
        if torch.isinf(agent_states).any():
            raise ValueError("Agent states contain infinite values")
        
        try:
            return self.state_encoder(agent_states)
        except Exception as e:
            raise RuntimeError(f"Agent state encoding failed: {e}")
    
    def encode_from_dict(self, agent_dict: Dict[str, Any], 
                        grid_size: Tuple[int, int] = (15, 15),
                        device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Encode agent state from dictionary format (as used in GridWorld)
        
        Args:
            agent_dict: Agent dictionary with keys: 'inventory', 'health', 'energy', 'pos'
            grid_size: Grid dimensions for position normalization
            device: Target device for tensor
            
        Returns:
            Encoded state tensor of shape (1, state_dim)
        """
        try:
            # Extract inventory values with error handling
            inventory = agent_dict.get('inventory', {})
            inv_values = [
                float(inventory.get('wood', 0)),
                float(inventory.get('stone', 0)),
                float(inventory.get('food', 0)),
                float(inventory.get('tool', 0))
            ]
            
            # Normalize health and energy to [0, 1] with bounds checking
            health = float(agent_dict.get('health', 100))
            energy = float(agent_dict.get('energy', 100))
            
            health = max(0.0, min(100.0, health)) / 100.0
            energy = max(0.0, min(100.0, energy)) / 100.0
            
            # Normalize position to [0, 1] with bounds checking
            pos = agent_dict.get('pos', (0, 0))
            pos_x = max(0.0, min(float(grid_size[0] - 1), float(pos[0]))) / float(grid_size[0])
            pos_y = max(0.0, min(float(grid_size[1] - 1), float(pos[1]))) / float(grid_size[1])
            
            # Combine all features
            state_vector = inv_values + [health, energy, pos_x, pos_y]
            
            # Create tensor
            tensor = torch.tensor(state_vector, dtype=torch.float32)
            
            if device is not None:
                tensor = tensor.to(device)
                
            return tensor.unsqueeze(0)
            
        except Exception as e:
            raise ValueError(f"Could not encode agent dictionary: {e}")

class MultiModalFusion(nn.Module):
    """
    Fusion network combining spatial CNN features with agent state
    """
    
    def __init__(self, spatial_dim: int = 256, state_dim: int = 128, output_dim: int = 512):
        super(MultiModalFusion, self).__init__()
        
        self.spatial_dim = spatial_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        
        # Agent state encoder
        self.state_encoder = AgentStateEncoder(state_dim)
        
        # Fusion layers with better architecture
        fusion_input_dim = spatial_dim + state_dim
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            
            nn.Linear(512, 384),
            nn.ReLU(inplace=True),
            nn.LayerNorm(384),
            nn.Dropout(0.1),
            
            nn.Linear(384, output_dim),
            nn.ReLU(inplace=True)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize fusion layer weights"""
        for module in self.fusion_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, spatial_features: torch.Tensor, agent_states: torch.Tensor) -> torch.Tensor:
        """
        Fuse spatial and state features
        
        Args:
            spatial_features: Spatial CNN features (batch_size, spatial_dim)
            agent_states: Agent state tensor (batch_size, 8)
            
        Returns:
            Fused features (batch_size, output_dim)
        """
        # Input validation
        if not isinstance(spatial_features, torch.Tensor) or not isinstance(agent_states, torch.Tensor):
            raise TypeError("Both inputs must be torch.Tensors")
        
        if spatial_features.dim() != 2 or agent_states.dim() != 2:
            raise ValueError("Both inputs must be 2D tensors")
        
        if spatial_features.shape[0] != agent_states.shape[0]:
            raise ValueError(f"Batch size mismatch: spatial {spatial_features.shape[0]} vs states {agent_states.shape[0]}")
        
        if spatial_features.shape[1] != self.spatial_dim:
            raise ValueError(f"Expected spatial dim {self.spatial_dim}, got {spatial_features.shape[1]}")
        
        # Device compatibility check
        if spatial_features.device != agent_states.device:
            agent_states = agent_states.to(spatial_features.device)
        
        try:
            # Encode agent states
            state_features = self.state_encoder(agent_states)
            
            # Concatenate features
            combined_features = torch.cat([spatial_features, state_features], dim=1)
            
            # Apply fusion layers
            fused_features = self.fusion_layers(combined_features)
            
            return fused_features
            
        except Exception as e:
            raise RuntimeError(f"Multi-modal fusion failed: {e}")

class AttentionalFusion(nn.Module):
    """
    Advanced fusion with cross-modal attention
    Enhanced error handling for production use
    """
    
    def __init__(self, spatial_dim: int = 256, state_dim: int = 128, output_dim: int = 512):
        super(AttentionalFusion, self).__init__()
        
        self.spatial_dim = spatial_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        
        self.state_encoder = AgentStateEncoder(state_dim)
        
        # Cross-attention layers with error handling
        try:
            self.spatial_to_state_attn = nn.MultiheadAttention(
                spatial_dim, num_heads=min(8, spatial_dim // 32), 
                batch_first=True, dropout=0.1
            )
            self.state_to_spatial_attn = nn.MultiheadAttention(
                state_dim, num_heads=min(4, state_dim // 32),
                batch_first=True, dropout=0.1
            )
            self.use_attention = True
        except Exception as e:
            warnings.warn(f"Could not initialize attention layers: {e}")
            self.use_attention = False
        
        # Final fusion
        self.fusion_layers = nn.Sequential(
            nn.Linear(spatial_dim + state_dim, 512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
            nn.ReLU(inplace=True)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for module in self.fusion_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, spatial_features: torch.Tensor, agent_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with cross-modal attention and fallback"""
        # Input validation
        if not isinstance(spatial_features, torch.Tensor) or not isinstance(agent_states, torch.Tensor):
            raise TypeError("Both inputs must be torch.Tensors")
        
        # Device compatibility
        if spatial_features.device != agent_states.device:
            agent_states = agent_states.to(spatial_features.device)
        
        try:
            # Encode states
            state_features = self.state_encoder(agent_states)
            
            if self.use_attention:
                try:
                    # Reshape for attention (add sequence dimension)
                    spatial_attn_input = spatial_features.unsqueeze(1)  # (batch, 1, spatial_dim)
                    state_attn_input = state_features.unsqueeze(1)      # (batch, 1, state_dim)
                    
                    # Cross attention
                    spatial_attended, _ = self.spatial_to_state_attn(
                        spatial_attn_input, state_attn_input, state_attn_input
                    )
                    state_attended, _ = self.state_to_spatial_attn(
                        state_attn_input, spatial_attn_input, spatial_attn_input
                    )
                    
                    # Squeeze back to (batch, dim)
                    spatial_attended = spatial_attended.squeeze(1)
                    state_attended = state_attended.squeeze(1)
                    
                    # Final fusion
                    combined = torch.cat([spatial_attended, state_attended], dim=1)
                    
                except Exception as e:
                    warnings.warn(f"Attention failed, using simple concatenation: {e}")
                    combined = torch.cat([spatial_features, state_features], dim=1)
            else:
                combined = torch.cat([spatial_features, state_features], dim=1)
            
            return self.fusion_layers(combined)
            
        except Exception as e:
            raise RuntimeError(f"Attentional fusion failed: {e}")

def prepare_agent_state_batch(agent_dicts: Union[List[Dict[str, Any]], Dict[str, Any]], 
                             grid_size: Tuple[int, int] = (15, 15),
                             device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convert batch of agent dictionaries to state tensor with robust error handling
    
    Args:
        agent_dicts: List of agent dictionaries or single agent dictionary
        grid_size: Grid dimensions for normalization
        device: Target device for tensor
        
    Returns:
        Batch tensor of shape (batch_size, 8)
    """
    # Handle single agent case
    if isinstance(agent_dicts, dict):
        agent_dicts = [agent_dicts]
    
    if not agent_dicts:
        raise ValueError("Empty agent_dicts provided")
    
    batch_states = []
    
    for i, agent_dict in enumerate(agent_dicts):
        try:
            # Extract and normalize features with error handling
            inventory = agent_dict.get('inventory', {})
            inv_values = [
                float(inventory.get('wood', 0)),
                float(inventory.get('stone', 0)),
                float(inventory.get('food', 0)),
                float(inventory.get('tool', 0))
            ]
            
            # Normalize health and energy with bounds checking
            health = float(agent_dict.get('health', 100))
            energy = float(agent_dict.get('energy', 100))
            
            health = max(0.0, min(100.0, health)) / 100.0
            energy = max(0.0, min(100.0, energy)) / 100.0
            
            # Normalize position with bounds checking
            pos = agent_dict.get('pos', (0, 0))
            pos_x = max(0.0, min(float(grid_size[0] - 1), float(pos[0]))) / float(grid_size[0])
            pos_y = max(0.0, min(float(grid_size[1] - 1), float(pos[1]))) / float(grid_size[1])
            
            state_vector = inv_values + [health, energy, pos_x, pos_y]
            batch_states.append(state_vector)
            
        except Exception as e:
            raise ValueError(f"Error processing agent {i}: {e}")
    
    try:
        tensor = torch.tensor(batch_states, dtype=torch.float32)
        
        if device is not None:
            tensor = tensor.to(device)
            
        return tensor
        
    except Exception as e:
        raise RuntimeError(f"Could not create batch tensor: {e}")

def validate_agent_dict(agent_dict: Dict[str, Any]) -> bool:
    """
    Validate that an agent dictionary has required fields
    
    Args:
        agent_dict: Agent dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['inventory', 'health', 'energy', 'pos']
    
    for field in required_fields:
        if field not in agent_dict:
            return False
    
    # Validate inventory structure
    inventory = agent_dict['inventory']
    required_items = ['wood', 'stone', 'food', 'tool']
    
    for item in required_items:
        if item not in inventory:
            return False
        
        try:
            float(inventory[item])
        except (ValueError, TypeError):
            return False
    
    # Validate health and energy
    try:
        health = float(agent_dict['health'])
        energy = float(agent_dict['energy'])
        
        if health < 0 or energy < 0:
            return False
            
    except (ValueError, TypeError):
        return False
    
    # Validate position
    pos = agent_dict['pos']
    if not isinstance(pos, (tuple, list)) or len(pos) != 2:
        return False
    
    try:
        float(pos[0])
        float(pos[1])
    except (ValueError, TypeError):
        return False
    
    return True

def create_multimodal_fusion(config: dict) -> nn.Module:
    """
    Factory function to create fusion network from configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured fusion network
    """
    try:
        if config.get('use_attention_fusion', False):
            return AttentionalFusion(
                spatial_dim=config.get('spatial_dim', 256),
                state_dim=config.get('state_dim', 128),
                output_dim=config.get('fusion_dim', 512)
            )
        else:
            return MultiModalFusion(
                spatial_dim=config.get('spatial_dim', 256),
                state_dim=config.get('state_dim', 128),
                output_dim=config.get('fusion_dim', 512)
            )
    except Exception as e:
        warnings.warn(f"Could not create fusion network with config, using defaults: {e}")
        return MultiModalFusion()

if __name__ == "__main__":
    # Test multi-modal fusion with error handling
    print("Testing Multi-Modal Fusion with Error Handling...")
    
    try:
        batch_size = 4
        
        # Test spatial features
        spatial_features = torch.randn(batch_size, 256)
        
        # Test agent states (realistic values)
        agent_states = torch.tensor([
            [2.0, 1.0, 3.0, 0.0, 0.8, 0.9, 0.5, 0.3],  # Agent 1
            [0.0, 5.0, 1.0, 1.0, 1.0, 0.7, 0.2, 0.8],  # Agent 2
            [1.0, 0.0, 0.0, 0.0, 0.6, 0.5, 0.9, 0.1],  # Agent 3
            [3.0, 2.0, 2.0, 0.0, 0.9, 0.8, 0.1, 0.9],  # Agent 4
        ], dtype=torch.float32)
        
        # Test basic fusion
        print("Testing basic fusion...")
        fusion_net = MultiModalFusion(spatial_dim=256, state_dim=128, output_dim=512)
        fused_output = fusion_net(spatial_features, agent_states)
        
        print(f"✅ Spatial features shape: {spatial_features.shape}")
        print(f"✅ Agent states shape: {agent_states.shape}")
        print(f"✅ Fused output shape: {fused_output.shape}")
        print(f"✅ Parameters: {sum(p.numel() for p in fusion_net.parameters()):,}")
        
        # Test attentional fusion
        print("\nTesting attentional fusion...")
        attn_fusion = AttentionalFusion(spatial_dim=256, state_dim=128, output_dim=512)
        attn_output = attn_fusion(spatial_features, agent_states)
        
        print(f"✅ Attentional output shape: {attn_output.shape}")
        print(f"✅ Attentional parameters: {sum(p.numel() for p in attn_fusion.parameters()):,}")
        
        # Test agent dictionary processing
        print("\nTesting agent dictionary processing...")
        test_agent_dict = {
            'inventory': {'wood': 2, 'stone': 1, 'food': 3, 'tool': 0},
            'health': 80,
            'energy': 90,
            'pos': (7, 5)
        }
        
        # Validate agent dict
        is_valid = validate_agent_dict(test_agent_dict)
        print(f"✅ Agent dict validation: {is_valid}")
        
        # Test batch preparation
        agent_dicts = [test_agent_dict for _ in range(batch_size)]
        batch_tensor = prepare_agent_state_batch(agent_dicts)
        
        print(f"✅ Batch preparation successful: {batch_tensor.shape}")
        
        # Test error handling with invalid input
        print("\nTesting error handling...")
        try:
            invalid_states = torch.randn(batch_size, 6)  # Wrong dimension
            fusion_net(spatial_features, invalid_states)
        except ValueError as e:
            print(f"✅ Correctly caught invalid input: {e}")
        
        # Test device compatibility
        if torch.backends.mps.is_available():
            print("\nTesting MPS compatibility...")
            device = torch.device('mps')
            mps_fusion = MultiModalFusion().to(device)
            mps_spatial = spatial_features.to(device)
            mps_states = agent_states.to(device)
            mps_output = mps_fusion(mps_spatial, mps_states)
            print(f"✅ MPS output shape: {mps_output.shape}")
        
        print("\n✅ All multi-modal fusion tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()