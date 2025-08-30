# Quick Start Guide - Project NEXUS

Get up and running with Project NEXUS in under 5 minutes! This guide covers the essential steps to start experimenting with multi-agent reinforcement learning.

## ⚡ 5-Minute Setup

### Prerequisites Check
```bash
# Verify you have the required tools
python --version  # Should be 3.9+
conda --version   # Any recent version
git --version     # Any recent version
```

### Installation
```bash
# 1. Clone and enter directory
git clone https://github.com/curtisreker-creator/project-nexus.git
cd project-nexus

# 2. Create environment (2-3 minutes)
conda env create -f environment-macos.yml
conda activate nexus

# 3. Verify installation (30 seconds)
python test_basic.py
```

Expected output:
```
✅ Environment created successfully!
✅ Basic environment test passed!
```

## 🎮 Your First Simulation

### Single Agent Demo
Create your first simulation in under 10 lines of code:

```python
from environment.grid_world import GridWorld
from agents.networks import create_lightweight_network

# Create environment with 1 agent
env = GridWorld(n_agents=1, max_resources=5)
obs, info = env.reset(seed=42)

# Create a neural network
network = create_lightweight_network()

# Run 10 simulation steps
for step in range(10):
    # Convert observation for network
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
    
    # Prepare agent state
    agent_dict = info['agents'][0]
    agent_state = [
        agent_dict['inventory']['wood'],
        agent_dict['inventory']['stone'], 
        agent_dict['inventory']['food'],
        agent_dict['inventory']['tool'],
        agent_dict['health'] / 100.0,
        agent_dict['energy'] / 100.0,
        agent_dict['pos'][0] / 15.0,
        agent_dict['pos'][1] / 15.0
    ]
    agent_states = torch.tensor([agent_state], dtype=torch.float32)
    
    # Get action from network
    actions, log_probs, values = network.act(obs_tensor, agent_states)
    
    # Execute action in environment
    obs, reward, terminated, truncated, info = env.step(actions.item())
    
    print(f"Step {step+1}: Action={actions.item()}, Reward={reward:.3f}")
    
    if terminated or truncated:
        print("Episode finished!")
        break
```

### Quick Demo Script
Run the provided demo for instant gratification:

```bash
# Run single agent demo
python scripts/demo.py

# Expected output:
# === SINGLE AGENT DEMO ===
# Environment size: (15, 15)
# Observation shape: (5, 15, 15)
# Action space: 14 actions (Discrete)
# ...
```

## 🤖 Multi-Agent Coordination

### Two-Agent Collaboration
```python
from environment.grid_world import GridWorld
from agents.networks import create_standard_network

# Create multi-agent environment
env = GridWorld(n_agents=2, max_resources=8)
network = create_standard_network()

obs, info = env.reset(seed=123)
print(f"Agents: {len(info['agents'])}")
print(f"Resources: {info['resources_remaining']}")

# Run coordinated simulation
total_reward = 0
for step in range(20):
    # Network handles multi-agent observation processing
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
    
    # For demo, we'll control agent 0 (current limitation)
    agent_state = prepare_agent_state(info['agents'][0])
    
    actions, _, values = network.act(obs_tensor, agent_state)
    obs, reward, terminated, truncated, info = env.step(actions.item())
    
    total_reward += reward
    
    if (step + 1) % 5 == 0:
        print(f"Step {step+1}: Total reward = {total_reward:.2f}")
        print(f"Resources remaining: {info['resources_remaining']}")

print(f"Final reward: {total_reward:.2f}")
```

## 🧠 Neural Network Playground

### Experiment with Different Network Sizes

```python
from agents.networks import (
    create_lightweight_network,
    create_standard_network, 
    create_advanced_network
)

# Compare network sizes
networks = {
    'lightweight': create_lightweight_network(),
    'standard': create_standard_network(), 
    'advanced': create_advanced_network()
}

for name, network in networks.items():
    if network:
        params = sum(p.numel() for p in network.parameters())
        print(f"{name}: {params:,} parameters")

# Output:
# lightweight: ~150,000 parameters
# standard: ~800,000 parameters  
# advanced: ~1,500,000 parameters
```

### Test Network Performance
```python
import time
import torch

# Create test data
batch_size = 32
obs = torch.randn(batch_size, 5, 15, 15)
states = torch.randn(batch_size, 8)

network = create_standard_network()

# Benchmark forward pass
start_time = time.time()
for _ in range(100):
    with torch.no_grad():
        logits, values = network(obs, states)

elapsed = time.time() - start_time
throughput = (100 * batch_size) / elapsed

print(f"Throughput: {throughput:.1f} samples/second")
```

## 🎯 Configuration & Customization

### Custom Environment Settings
```python
# Create environment with custom parameters
env = GridWorld(
    size=(20, 20),        # Larger grid
    n_agents=3,           # More agents
    max_resources=12,     # More resources
    max_steps=2000        # Longer episodes
)

obs, info = env.reset(seed=42)
print(f"Custom environment: {env.size}, {env.n_agents} agents")
```

### Network Configuration
```python
from agents.networks.network_factory import NetworkFactory

# Create custom network configuration
factory = NetworkFactory()

custom_network = factory.create_network(
    override_config={
        'spatial_dim': 128,      # Smaller spatial features
        'fusion_dim': 256,       # Smaller fusion layer
        'use_enhanced_cnn': True # Enable attention mechanisms
    }
)

print(f"Custom network created with enhanced features")
```

## 📊 Monitoring & Visualization

### Environment Visualization (Requires Display)
```python
# Enable rendering mode
env = GridWorld(n_agents=2, render_mode='human')
obs, info = env.reset(seed=42)

# Render current state
env.render()

# Run a few steps with visualization
for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(1)  # Pause to see the action
```

### Training Progress Tracking
```python
# Simple training metrics tracking
class SimpleTracker:
    def __init__(self):
        self.episode_rewards = []
        self.step_count = 0
    
    def log_step(self, reward):
        self.step_count += 1
        if hasattr(self, 'current_episode_reward'):
            self.current_episode_reward += reward
        else:
            self.current_episode_reward = reward
    
    def log_episode_end(self):
        self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0
        
        if len(self.episode_rewards) % 10 == 0:
            avg_reward = np.mean(self.episode_rewards[-10:])
            print(f"Episode {len(self.episode_rewards)}: Avg reward = {avg_reward:.3f}")

# Usage in simulation loop
tracker = SimpleTracker()
# ... integrate with your simulation
```

## 🔧 Common Use Cases

### Scenario 1: Resource Collection Challenge
```python
# Setup: Multiple agents competing for limited resources
env = GridWorld(n_agents=3, max_resources=5)
network = create_standard_network()

# Goal: Maximize total resource collection
def resource_collection_challenge():
    obs, info = env.reset(seed=42)
    initial_resources = info['resources_remaining']
    
    for step in range(50):
        # Agent decision making
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
        agent_state = prepare_agent_state(info['agents'][0])
        
        actions, _, _ = network.act(obs_tensor, agent_state)
        obs, reward, terminated, truncated, info = env.step(actions.item())
        
        if terminated or truncated:
            break
    
    resources_collected = initial_resources - info['resources_remaining']
    total_inventory = sum(sum(agent['inventory'].values()) for agent in info['agents'])
    
    print(f"Resources collected: {resources_collected}")
    print(f"Total inventory: {total_inventory}")
    
    return resources_collected

score = resource_collection_challenge()
```

### Scenario 2: Cooperative Building
```python
# Setup: Agents must coordinate to build structures
def cooperative_building_demo():
    env = GridWorld(n_agents=2, max_resources=10)
    obs, info = env.reset(seed=123)
    
    print("Starting cooperative building challenge...")
    print(f"Initial resources: {info['resources_remaining']}")
    
    building_actions = [9, 10, 11]  # Building-related actions
    
    for step in range(100):
        # Simulate cooperative behavior
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
        agent_state = prepare_agent_state(info['agents'][0])
        
        actions, _, _ = network.act(obs_tensor, agent_state)
        obs, reward, terminated, truncated, info = env.step(actions.item())
        
        if step % 20 == 0:
            print(f"Step {step}: Buildings = {len(env.buildings)}, Resources = {info['resources_remaining']}")
        
        if terminated or truncated:
            break
    
    print(f"Final buildings constructed: {len(env.buildings)}")

cooperative_building_demo()
```

## 🚀 Next Steps

### Immediate Exploration
1. **Experiment with Parameters**: Try different agent counts, grid sizes, resource amounts
2. **Test Network Architectures**: Compare lightweight vs. advanced networks
3. **Custom Scenarios**: Create your own challenges and objectives

### Advanced Features  
1. **Read Architecture Guide**: `docs/ARCHITECTURE.md` for deep technical details
2. **Training Pipeline**: Explore `scripts/train.py` for RL training
3. **Custom Networks**: Study `agents/networks/` for neural architecture insights

### Development Path
1. **Contribute**: Check `CONTRIBUTING.md` for contribution guidelines
2. **Research**: Explore multi-agent RL papers and implement new ideas
3. **Optimization**: Profile and optimize network performance

## 📚 Learning Resources

### Understanding the Code
- **Environment**: Start with `environment/grid_world.py`
- **Networks**: Study `agents/networks/ppo_networks.py`
- **Integration**: Review `test_integration.py` for complete examples

### Multi-Agent RL Concepts
- **Observation Spaces**: How agents perceive their world
- **Action Coordination**: How agents choose complementary actions
- **Reward Shaping**: Designing rewards for cooperative behavior

### Troubleshooting
- **Run Tests**: `python test_integration.py` for comprehensive validation
- **Check Imports**: Verify all modules load correctly
- **Monitor Resources**: Watch memory and CPU usage during experiments

---

**Ready to dive deeper?** Check out the [Architecture Documentation](docs/ARCHITECTURE.md) or start building your first custom agent! 

**Need help?** Visit our [GitHub Issues](https://github.com/curtisreker-creator/project-nexus/issues) or [Discussions](https://github.com/curtisreker-creator/project-nexus/discussions).