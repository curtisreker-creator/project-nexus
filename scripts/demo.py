# File: scripts/demo.py
"""
NEXUS Environment Demonstration
Shows environment capabilities and basic agent interactions
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from environment import create_environment, load_config
from environment.grid_world import GridWorld
import numpy as np

def demo_single_agent():
    """Demonstrate single agent environment"""
    print("\n=== SINGLE AGENT DEMO ===")
    
    env = GridWorld(n_agents=1, seed=42)
    obs = env.reset()
    
    print(f"Environment size: {env.size}")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space.n} actions")
    print(f"Initial resources: {len(env.resources)}")
    print(f"Agent starting position: {env.agents[0]['pos']}")
    
    total_reward = 0
    resources_gathered = 0
    
    for step in range(20):
        # Simple policy: try to gather if on resource, otherwise move randomly
        agent_pos = env.agents[0]['pos']
        on_resource = any(r['pos'] == agent_pos for r in env.resources)
        
        if on_resource:
            action = 8  # Gather action
        else:
            action = np.random.randint(0, 8)  # Random movement
        
        obs, reward, done, info = env.step([action])
        total_reward += reward[0]
        
        current_inventory = sum(env.agents[0]['inventory'].values())
        if current_inventory > resources_gathered:
            resources_gathered = current_inventory
            print(f"Step {step}: Gathered resource! Total inventory: {current_inventory}")
        
        if step % 5 == 0:
            print(f"Step {step}: Reward={reward[0]:.3f}, Resources remaining={len(env.resources)}")
        
        if done:
            print(f"Episode finished at step {step}")
            break
    
    print(f"Final total reward: {total_reward:.2f}")
    print(f"Resources gathered: {resources_gathered}")
    return env

def demo_multi_agent():
    """Demonstrate multi-agent environment"""
    print("\n=== MULTI-AGENT DEMO ===")
    
    env = GridWorld(n_agents=3, seed=123)
    obs = env.reset()
    
    print(f"Number of agents: {env.n_agents}")
    for i, agent in enumerate(env.agents):
        print(f"Agent {i} starting position: {agent['pos']}")
    
    total_rewards = [0] * env.n_agents
    
    for step in range(15):
        # Each agent acts independently
        actions = []
        for i in range(env.n_agents):
            agent_pos = env.agents[i]['pos']
            on_resource = any(r['pos'] == agent_pos for r in env.resources)
            
            if on_resource:
                actions.append(8)  # Gather
            else:
                actions.append(np.random.randint(0, 8))  # Move
        
        obs, rewards, done, info = env.step(actions)
        
        for i, reward in enumerate(rewards):
            total_rewards[i] += reward
        
        if step % 5 == 0:
            print(f"Step {step}: Rewards={[f'{r:.2f}' for r in rewards]}, "
                  f"Resources remaining={len(env.resources)}")
        
        if done:
            print(f"Episode finished at step {step}")
            break
    
    print(f"Final total rewards: {[f'{r:.2f}' for r in total_rewards]}")
    for i, agent in enumerate(env.agents):
        inventory_total = sum(agent['inventory'].values())
        print(f"Agent {i} inventory: {inventory_total} items")
    
    return env

def demo_with_config():
    """Demonstrate environment creation from config file"""
    print("\n=== CONFIG-BASED DEMO ===")
    
    try:
        config = load_config()
        print("Configuration loaded successfully:")
        print(f"Grid size: {config['environment']['grid_size']}")
        print(f"Max agents: {config['environment']['max_agents']}")
        print(f"Max resources: {config['environment']['max_resources']}")
        
        env = create_environment()
        obs = env.reset()
        print(f"Environment created with observation shape: {obs.shape}")
        
        return env
    except FileNotFoundError:
        print("Config file not found, using default parameters")
        return GridWorld(seed=42)

def main():
    """Run all demonstrations"""
    print("🚀 NEXUS GRID WORLD DEMONSTRATION")
    print("=" * 50)
    
    # Demo 1: Single agent
    env1 = demo_single_agent()
    
    # Demo 2: Multi-agent
    env2 = demo_multi_agent()
    
    # Demo 3: Config-based
    env3 = demo_with_config()
    
    print("\n" + "=" * 50)
    print("✅ All demonstrations completed successfully!")
    print("\nEnvironment is ready for neural network integration (Session 2)")
    print("Next: Implement CNN architecture and PPO training")
    
    # Optionally show rendering (uncomment if you want to see the grid)
    # print("\nShowing final environment state...")
    # env3.render()

if __name__ == "__main__":
    main()