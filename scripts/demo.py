# File: scripts/demo.py
"""
Ares Prime Environment Demonstration
Shows environment capabilities and basic agent interactions
"""
import sys
import os
import numpy as np
import gymnasium as gym

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environment import create_environment, load_config
from environment.grid_world import GridWorld


def demo_single_agent():
    """Demonstrate single agent environment"""
    print("\n=== SINGLE AGENT DEMO ===")
    
    env = GridWorld(n_agents=1)
    obs, info = env.reset(seed=42)
    
    print(f"Environment size: {env.size}")
    print(f"Observation shape: {obs.shape}")

    if isinstance(env.action_space, gym.spaces.Discrete):
        print(f"Action space: {env.action_space.n} actions (Discrete)")
    else:
        print(f"Action space: {env.action_space}")

    print(f"Initial resources: {info['resources_remaining']}")
    print(f"Agent starting position: {info['agents'][0]['pos']}")
    
    total_reward = 0.0
    resources_gathered = 0
    
    for step in range(20):
        agent_pos = env.agents[0]['pos']
        on_resource = any(r['pos'] == agent_pos for r in env.resources)
        
        # FIX: Explicitly cast the sampled action to a standard Python int
        action = 8 if on_resource else int(env.action_space.sample())
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        current_inventory = sum(env.agents[0]['inventory'].values())
        if current_inventory > resources_gathered:
            resources_gathered = current_inventory
            print(f"Step {step+1}: Gathered resource! Total inventory: {current_inventory}")
        
        if (step + 1) % 5 == 0:
            print(f"Step {step+1}: Reward={reward:.3f}, Resources remaining={info['resources_remaining']}")
        
        if terminated or truncated:
            print(f"Episode finished at step {step+1}")
            break
    
    print(f"Final total reward: {total_reward:.2f}")
    print(f"Resources gathered: {resources_gathered}")
    return env

def demo_multi_agent():
    """Demonstrate multi-agent environment by stepping each agent individually"""
    print("\n=== MULTI-AGENT DEMO ===")
    
    env = GridWorld(n_agents=3)
    obs, info = env.reset(seed=123)
    
    print(f"Number of agents: {env.n_agents}")
    for i, agent in enumerate(env.agents):
        print(f"Agent {i} starting position: {agent['pos']}")
    
    total_rewards = [0.0] * env.n_agents
    
    for step in range(15):
        # NOTE: This is a simplified demo. The underlying environment currently only moves
        # one agent (agent_id=0) per step(). A true multi-agent setup would require a
        # different environment API like PettingZoo.
        agent_to_move = 0
        agent_pos = env.agents[agent_to_move]['pos']
        on_resource = any(r['pos'] == agent_pos for r in env.resources)
        
        # FIX: Explicitly cast the sampled action to a standard Python int
        action = 8 if on_resource else int(env.action_space.sample())

        obs, reward, terminated, truncated, info = env.step(action)
        total_rewards[agent_to_move] += reward
        
        if (step + 1) % 5 == 0:
            print(f"Step {step+1}: Agent {agent_to_move} took action. Reward={reward:.2f}, "
                  f"Resources remaining={info['resources_remaining']}")
        
        if terminated or truncated:
            print(f"Episode finished at step {step+1}")
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
        print("Configuration loaded successfully.")
        
        env = create_environment()
        seed = config.get('environment', {}).get('seed')
        obs, info = env.reset(seed=seed)
        
        print(f"Environment created with observation shape: {obs.shape}")
        
        return env
    except FileNotFoundError:
        print("Config file 'configs/default.yaml' not found, using default parameters.")
        env = GridWorld()
        env.reset(seed=42)
        return env

def main():
    """Run all demonstrations"""
    print("🚀 Ares Prime Simulation Demo")
    print("=" * 50)
    
    demo_single_agent()
    demo_multi_agent()
    demo_with_config()
    
    print("\n" + "=" * 50)
    print("✅ All demonstrations completed successfully!")

if __name__ == "__main__":
    main()