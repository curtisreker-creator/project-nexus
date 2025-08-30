# Technical Architecture - Project NEXUS

This document provides a comprehensive technical overview of Project NEXUS's multi-agent reinforcement learning architecture, designed for scalable and efficient training of cooperative intelligent agents.

## 🏗️ System Overview

Project NEXUS implements a hierarchical architecture with three main layers:

1. **Environment Layer**: Multi-agent grid world simulation
2. **Neural Architecture Layer**: Deep learning models for agent decision-making
3. **Training Layer**: Distributed PPO training pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Experience  │  │ PPO Trainer │  │ Distributed Envs    │  │
│  │ Buffer      │←→│ (Actor-     │←→│ (8+ Parallel)       │  │
│  │ (GAE)       │  │  Critic)    │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                 Neural Architecture                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Spatial CNN │→ │ Multi-Modal │→ │ Policy/Value Heads  │  │
│  │ (5→256)     │  │ Fusion      │  │ (π/V)               │  │
│  └─────────────┘  │ (256+128→   │  └─────────────────────┘  │
│         ↑         │  512)       │           ↓               │
│  ┌─────────────┐  └─────────────┘  ┌─────────────────────┐  │
│  │ Grid Obs    │                   │ Actions & Values    │  │
│  │ (5,15,15)   │                   │                     │  │
│  └─────────────┘                   └─────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                 Environment Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ GridWorld   │→ │ Multi-Agent │→ │ Resource & Building │  │
│  │ (15×15)     │  │ System      │  │ Management          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 🌍 Environment Layer

### GridWorld Core Architecture

The `GridWorld` environment implements a fully observable 15×15 grid supporting up to 4 agents with real-time coordination.

#### State Representation
```python
# 5-channel observation tensor
observation_shape = (5, 15, 15)

channels = {
    0: "topology",     # Empty spaces, walls, obstacles
    1: "resources",    # Wood (1), Stone (2), Food (3) 
    2: "agents",       # Agent positions (agent_id + 1)
    3: "buildings",    # Constructed buildings
    4: "activity"      # Recent action traces
}
```

#### Action Space Design
```python
action_space = spaces.Discrete(14)

action_mapping = {
    # Movement actions (8-directional)
    0: "move_NW", 1: "move_N",  2: "move_NE",
    3: "move_W",               4: "move_E", 
    5: "move_SW", 6: "move_S",  7: "move_SE",
    
    # Interaction actions
    8:  "gather_resource",     # Collect resources at current position
    9:  "build_shelter",       # Construct basic shelter
    10: "build_storage",       # Construct resource storage
    11: "build_workshop",      # Construct crafting facility
    12: "craft_tool",          # Create tools from resources
    13: "communicate"          # Send communication token
}
```

#### Agent State Components
```python
agent_state = {
    'id': int,                          # Unique agent identifier
    'pos': Tuple[int, int],            # Grid coordinates (x, y)
    'inventory': {                     # Resource inventory
        'wood': int,
        'stone': int, 
        'food': int,
        'tool': int
    },
    'health': float,                   # Health points (0-100)
    'energy': float,                   # Energy level (0-100)
    'communication_tokens': List[int]  # Recent messages received
}
```

### Environment Dynamics

#### Resource System
- **Spawn Rate**: 8 resources per episode (configurable)
- **Types**: Wood (building), Stone (durability), Food (health restoration)
- **Respawn**: Dynamic respawning based on agent progress
- **Depletion**: Resources have finite amounts (1-3 units)

#### Physics & Collision
```python
def _move_agent(self, agent_id: int, direction: int) -> float:
    """Movement with collision detection and boundary constraints"""
    agent = self.agents[agent_id]
    old_pos = agent['pos']
    dx, dy = self.directions[direction]
    new_x, new_y = old_pos[0] + dx, old_pos[1] + dy
    
    # Boundary checking
    if not (0 <= new_x < self.size[0] and 0 <= new_y < self.size[1]):
        return -0.1  # Boundary penalty
    
    # Collision detection
    if self.grid[new_x, new_y] > 0:  # Occupied by another agent/building
        return -0.1  # Collision penalty
    
    # Execute movement
    self.grid[old_pos] = 0
    agent['pos'] = (new_x, new_y)
    self.grid[new_x, new_y] = agent_id + 1
    return 0.0  # Neutral reward for movement
```

## 🧠 Neural Architecture Layer

### Spatial CNN Module

The spatial CNN processes 5-channel grid observations to extract spatial features and relationships.

#### Architecture Specification
```python
class SpatialCNN(nn.Module):
    def __init__(self, input_channels=5, feature_dim=256):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv block: 5 → 32 channels
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            
            # Second conv block: 32 → 64 channels with pooling
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),  # 15×15 → 7×7
            
            # Third conv block: 64 → 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            # Fourth conv block: 128 → 128 channels with pooling  
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2),  # 7×7 → 3×3
        )
        
        # Feature compression
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True)
        )
```

#### Enhanced CNN with Attention
```python
class EnhancedSpatialCNN(SpatialCNN):
    """CNN with spatial attention mechanisms"""
    
    def __init__(self, *args, use_attention=True, **kwargs):
        super().__init__(*args, **kwargs)
        
        if use_attention:
            self.attention = SpatialAttention(128)
    
    def forward(self, x):
        # Standard convolution processing
        spatial_features = self.conv_layers[:-1](x)  # Stop before final layer
        
        # Apply spatial attention
        if hasattr(self, 'attention'):
            spatial_features = self.attention(spatial_features)
            
        # Final processing
        spatial_features = self.conv_layers[-1](spatial_features)
        
        # Flatten and compress
        batch_size = x.size(0)
        flattened = spatial_features.view(batch_size, -1)
        return self.fc_layers(flattened)
```

### Multi-Modal Fusion Architecture

The fusion layer combines spatial CNN features with agent state information for comprehensive decision-making.

#### Agent State Encoder
```python
class AgentStateEncoder(nn.Module):
    """Encodes agent state into dense feature representation"""
    
    def __init__(self, state_dim=128):
        super().__init__()
        
        # Input: [inventory(4), health(1), energy(1), position(2)] = 8 features
        self.state_encoder = nn.Sequential(
            nn.Linear(8, 64),
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
    
    def encode_from_dict(self, agent_dict, grid_size=(15, 15)):
        """Convert GridWorld agent dict to normalized tensor"""
        # Extract and normalize features
        inventory = agent_dict['inventory']
        inv_values = [
            float(inventory['wood']),
            float(inventory['stone']), 
            float(inventory['food']),
            float(inventory['tool'])
        ]
        
        # Normalize health/energy to [0,1]
        health = max(0, min(100, agent_dict['health'])) / 100.0
        energy = max(0, min(100, agent_dict['energy'])) / 100.0
        
        # Normalize position to [0,1]
        pos = agent_dict['pos']
        pos_x = pos[0] / float(grid_size[0])
        pos_y = pos[1] / float(grid_size[1])
        
        state_vector = inv_values + [health, energy, pos_x, pos_y]
        return torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)
```

#### Fusion Mechanisms
```python
class MultiModalFusion(nn.Module):
    """Standard concatenation-based fusion"""
    
    def __init__(self, spatial_dim=256, state_dim=128, output_dim=512):
        super().__init__()
        
        self.state_encoder = AgentStateEncoder(state_dim)
        
        # Fusion layers
        fusion_input_dim = spatial_dim + state_dim  # 256 + 128 = 384
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
    
    def forward(self, spatial_features, agent_states):
        # Encode agent states
        state_features = self.state_encoder(agent_states)
        
        # Concatenate and fuse
        combined_features = torch.cat([spatial_features, state_features], dim=1)
        return self.fusion_layers(combined_features)


class AttentionalFusion(MultiModalFusion):
    """Advanced fusion with cross-modal attention"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Cross-attention mechanisms
        self.spatial_to_state_attn = nn.MultiheadAttention(
            self.spatial_dim, num_heads=8, batch_first=True
        )
        self.state_to_spatial_attn = nn.MultiheadAttention(
            self.state_dim, num_heads=4, batch_first=True
        )
    
    def forward(self, spatial_features, agent_states):
        state_features = self.state_encoder(agent_states)
        
        # Reshape for attention (add sequence dimension)
        spatial_seq = spatial_features.unsqueeze(1)  # (batch, 1, spatial_dim)
        state_seq = state_features.unsqueeze(1)      # (batch, 1, state_dim)
        
        # Cross-attention
        spatial_attended, _ = self.spatial_to_state_attn(
            spatial_seq, state_seq, state_seq
        )
        state_attended, _ = self.state_to_spatial_attn(
            state_seq, spatial_seq, spatial_seq  
        )
        
        # Combine attended features
        combined = torch.cat([
            spatial_attended.squeeze(1), 
            state_attended.squeeze(1)
        ], dim=1)
        
        return self.fusion_layers(combined)
```

### Policy-Value Architecture

The actor-critic architecture separates policy (action selection) and value (state evaluation) learning with shared feature representations.

#### Policy Head Design
```python
class PolicyHead(nn.Module):
    """Action policy network with entropy regularization"""
    
    def __init__(self, input_dim=512, action_dim=14, hidden_dim=256):
        super().__init__()
        
        self.policy_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            
            nn.Linear(hidden_dim, action_dim)  # No activation - raw logits
        )
        
        # Initialize with smaller variance for stable training
        self._initialize_weights(gain=0.01)
    
    def forward(self, x):
        return self.policy_layers(x)
    
    def get_action_distribution(self, x):
        logits = self.forward(x)
        return Categorical(logits=logits)
```

#### Value Head Design
```python
class ValueHead(nn.Module):
    """State value estimation network"""
    
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        
        self.value_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim), 
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            
            nn.Linear(hidden_dim, 1)  # Single value output
        )
        
        self._initialize_weights(gain=1.0)
    
    def forward(self, x):
        return self.value_layers(x)
```

### Complete PPO Network Integration

```python
class PPOActorCritic(nn.Module):
    """Complete PPO Actor-Critic Network"""
    
    def __init__(self, 
                 spatial_channels=5,
                 spatial_dim=256,
                 state_dim=128, 
                 fusion_dim=512,
                 action_dim=14,
                 use_enhanced_cnn=False,
                 use_attention_fusion=False):
        super().__init__()
        
        # Feature extraction
        if use_enhanced_cnn:
            self.spatial_cnn = EnhancedSpatialCNN(spatial_channels, spatial_dim)
        else:
            self.spatial_cnn = SpatialCNN(spatial_channels, spatial_dim)
            
        # Multi-modal fusion
        if use_attention_fusion:
            self.fusion = AttentionalFusion(spatial_dim, state_dim, fusion_dim)
        else:
            self.fusion = MultiModalFusion(spatial_dim, state_dim, fusion_dim)
            
        # Policy and value heads
        self.policy_head = PolicyHead(fusion_dim, action_dim)
        self.value_head = ValueHead(fusion_dim)
    
    def forward(self, observations, agent_states):
        """Forward pass returning action logits and state values"""
        # Extract spatial features
        spatial_features = self.spatial_cnn(observations)
        
        # Fuse with agent states
        fused_features = self.fusion(spatial_features, agent_states)
        
        # Get policy and value outputs
        action_logits = self.policy_head(fused_features)
        state_values = self.value_head(fused_features)
        
        return action_logits, state_values
    
    def act(self, observations, agent_states, deterministic=False):
        """Select actions for environment interaction"""
        action_logits, state_values = self.forward(observations, agent_states)
        
        action_dist = Categorical(logits=action_logits)
        
        if deterministic:
            actions = torch.argmax(action_logits, dim=1)
        else:
            actions = action_dist.sample()
            
        log_probs = action_dist.log_prob(actions)
        
        return actions, log_probs, state_values.squeeze(-1)
    
    def evaluate_actions(self, observations, agent_states, actions):
        """Evaluate actions for PPO training"""
        action_logits, state_values = self.forward(observations, agent_states)
        
        action_dist = Categorical(logits=action_logits)
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        
        return log_probs, state_values.squeeze(-1), entropy
```

## 🎓 Training Architecture

### PPO Training Pipeline

The training system implements Proximal Policy Optimization with Generalized Advantage Estimation for stable multi-agent learning.

#### Core PPO Algorithm
```python
def ppo_update(policy_net, optimizer, states, actions, rewards, 
               old_log_probs, values, returns, advantages, 
               clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01):
    """Single PPO update step"""
    
    # Forward pass
    new_log_probs, new_values, entropy = policy_net.evaluate_actions(
        states['observations'], states['agent_states'], actions
    )
    
    # Policy loss with clipping
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value loss
    value_pred_clipped = values + (new_values - values).clamp(-clip_ratio, clip_ratio)
    value_losses = (new_values - returns).pow(2)
    value_losses_clipped = (value_pred_clipped - returns).pow(2)
    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
    
    # Entropy bonus
    entropy_loss = entropy.mean()
    
    # Combined loss
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
    optimizer.step()
    
    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(), 
        'entropy': entropy_loss.item(),
        'total_loss': loss.item()
    }
```

#### Advantage Estimation (GAE)
```python
def compute_gae(rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation"""
    advantages = torch.zeros_like(rewards)
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = next_values
        else:
            next_value = values[t + 1]
            
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae
    
    returns = advantages + values
    return advantages, returns
```

### Distributed Training Architecture

#### Parallel Environment Management
```python
class ParallelEnvironments:
    """Manages multiple environment instances for data collection"""
    
    def __init__(self, env_factory, n_envs=8):
        self.n_envs = n_envs
        self.envs = [env_factory() for _ in range(n_envs)]
        self.current_obs = [env.reset()[0] for env in self.envs]
        self.current_info = [env.reset()[1] for env in self.envs]
    
    def step(self, actions):
        """Execute actions in all environments"""
        results = []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            results.append((obs, reward, terminated, truncated, info))
            
            # Auto-reset if episode ends
            if terminated or truncated:
                obs, info = env.reset()
                results[-1] = (obs, reward, True, truncated, info)
                
        return list(zip(*results))  # Transpose results
    
    def reset_all(self):
        """Reset all environments"""
        results = [env.reset() for env in self.envs]
        self.current_obs, self.current_info = zip(*results)
        return self.current_obs, self.current_info
```

## 🔧 Configuration System

### Network Configuration Presets
```yaml
# configs/network.yaml
network:
  # Architecture dimensions
  spatial_channels: 5
  spatial_dim: 256
  state_dim: 128
  fusion_dim: 512
  action_dim: 14
  
  # Advanced features
  use_enhanced_cnn: false
  use_attention_fusion: false
  enable_communication: false

presets:
  lightweight:
    spatial_dim: 128
    fusion_dim: 256
    # ~150K parameters
    
  standard:
    spatial_dim: 256  
    fusion_dim: 512
    # ~800K parameters
    
  advanced:
    spatial_dim: 384
    fusion_dim: 768
    use_enhanced_cnn: true
    use_attention_fusion: true
    # ~1.5M parameters
    
  performance:
    spatial_dim: 512
    fusion_dim: 1024
    use_enhanced_cnn: true
    use_attention_fusion: true
    enable_communication: true
    # ~3M parameters
```

### Training Configuration
```yaml
# configs/training.yaml  
training:
  # PPO hyperparameters
  batch_size: 64
  buffer_size: 2048
  learning_rate: 3e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio: 0.2
  value_loss_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 0.5
  n_epochs: 4
  n_parallel_envs: 8
  
  # Training schedule
  total_timesteps: 1000000
  eval_frequency: 10000
  save_frequency: 50000
  
  # Curriculum learning
  curriculum_stages:
    stage1:
      max_agents: 1
      max_resources: 4
      timesteps: 200000
    stage2:
      max_agents: 2
      max_resources: 6  
      timesteps: 400000
    stage3:
      max_agents: 4
      max_resources: 8
      timesteps: 400000
```

## 📊 Performance Characteristics

### Computational Complexity

| Component | Time Complexity | Space Complexity | Parameters |
|-----------|----------------|------------------|------------|
| Spatial CNN | O(n²) | O(n²) | ~400K |
| State Encoder | O(1) | O(1) | ~50K |
| Fusion Layer | O(d) | O(d) | ~300K |
| Policy Head | O(d) | O(d) | ~130K |
| Value Head | O(d) | O(d) | ~65K |
| **Total** | **O(n²)** | **O(n²)** | **~945K** |

*Where n = grid size (15), d = feature dimension (512)*

### Memory Usage Analysis
```python
# Memory profiling for standard network
def profile_memory_usage():
    network = create_standard_network()
    
    # Model parameters
    param_memory = sum(p.numel() * p.element_size() for p in network.parameters())
    
    # Forward pass memory (batch_size=32)
    batch_size = 32
    obs = torch.randn(batch_size, 5, 15, 15)
    states = torch.randn(batch_size, 8)
    
    # Estimate forward pass memory
    forward_memory = (
        obs.numel() * obs.element_size() +      # Input observations
        states.numel() * states.element_size() + # Agent states
        batch_size * 512 * 4 +                  # Intermediate features
        batch_size * 14 * 4 +                   # Action logits
        batch_size * 1 * 4                      # State values
    )
    
    return {
        'parameter_memory_mb': param_memory / (1024 * 1024),
        'forward_memory_mb': forward_memory / (1024 * 1024),
        'total_memory_estimate_mb': (param_memory + forward_memory) / (1024 * 1024)
    }

# Typical results on Apple Silicon:
# parameter_memory_mb: ~3.6 MB
# forward_memory_mb: ~1.2 MB  
# total_memory_estimate_mb: ~4.8 MB
```

### Training Performance Benchmarks

| Device | Network | Throughput | Memory | Training Time |
|--------|---------|-----------|---------|---------------|
| M1 Pro | Lightweight | 1200 steps/sec | 2.1 GB | 2.5 hours |
| M1 Pro | Standard | 800 steps/sec | 4.3 GB | 4.2 hours |
| M1 Max | Advanced | 650 steps/sec | 7.8 GB | 6.1 hours |
| M2 Pro | Performance | 400 steps/sec | 12.1 GB | 9.3 hours |

*For 1M timesteps with curriculum learning*

## 🔮 Future Architecture Extensions

### Communication Architecture (Phase 3)
```python
class CommunicationModule(nn.Module):
    """Discrete token-based inter-agent communication"""
    
    def __init__(self, feature_dim=512, vocab_size=16, max_agents=4):
        super().__init__()
        
        # Message generation
        self.message_encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, vocab_size)
        )
        
        # Message processing with attention
        self.message_attention = nn.MultiheadAttention(
            feature_dim // 4, num_heads=4, batch_first=True
        )
```

### Hierarchical Architecture (Phase 4)
- **High-level Planner**: Long-term goal setting and task allocation
- **Mid-level Coordinator**: Resource allocation and role assignment  
- **Low-level Controller**: Immediate action execution

### Meta-Learning Integration (Future)
- **MAML Integration**: Few-shot adaptation to new environments
- **Neural Architecture Search**: Automated network design optimization
- **Evolutionary Strategies**: Population-based hyperparameter optimization

---

This architecture provides the foundation for scalable multi-agent reinforcement learning research while maintaining modularity and extensibility for future enhancements.