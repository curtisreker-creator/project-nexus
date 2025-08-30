# File: test_basic.py (temporary test file)
import gymnasium as gym
from environment.grid_world import GridWorld

# Test basic functionality
env = GridWorld(n_agents=2)

obs, info = env.reset(seed=42)

print("✅ Environment created successfully!")
print(f"Observation shape: {obs.shape}")
print(f"Number of agents: {len(env.agents)}")
print(f"Number of resources: {info.get('resources_remaining', 0)}")

# Corrected typo from action-space to action_space
if isinstance(env.action_space, gym.spaces.Discrete):
    print(f"Action space size: {env.action_space.n}")

# Test a few steps
for i in range(3):
    # The environment's step function now takes a single integer action.
    # We will sample a random action for this test.
    # Corrected typo from action-space to action_space
    action = int(env.action_space.sample())
    
    # The step function now returns 5 values.
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 'done' is now the combination of 'terminated' or 'truncated'.
    done = terminated or truncated
    
    # 'reward' is now a single float, not a list.
    print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Done={done}")

print("\n✅ Basic environment test passed!")

# Uncomment to see visualization (requires display)
# print("\nRendering final state...")
# env.render()