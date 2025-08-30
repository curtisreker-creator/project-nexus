# File: test_basic.py (temporary test file)
from environment.grid_world import GridWorld

# Test basic functionality
env = GridWorld(n_agents=2, seed=42)
obs = env.reset()

print("✅ Environment created successfully!")
print(f"Observation shape: {obs.shape}")
print(f"Number of agents: {len(env.agents)}")
print(f"Number of resources: {len(env.resources)}")
print(f"Action space size: {env.action_space.n}")

# Test a few steps
for i in range(3):
    actions = [0, 1]  # Simple movement actions
    obs, rewards, done, info = env.step(actions)
    print(f"Step {i}: Rewards={rewards}, Done={done}")

print("✅ Basic environment test passed!")

# Uncomment to see visualization (requires display)
# env.render()