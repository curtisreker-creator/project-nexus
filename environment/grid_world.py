# File: environment/grid_world.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Dict, Any, List, Optional

class GridWorld(gym.Env):
    """15x15 Grid World Environment for Reinforcement Learning"""
    
    def __init__(self, size: Tuple[int, int] = (15, 15), n_agents: int = 1, 
                 max_resources: int = 8, max_steps: int = 1000, 
                 render_mode: Optional[str] = None):
        super().__init__()
        
        self.size = size
        self.n_agents = n_agents
        self.max_resources = max_resources
        self.max_steps = max_steps
        self.render_mode = render_mode # For visualization
        
        # Environment state
        self.grid = np.zeros(size, dtype=np.int32)
        self.agents = []
        self.resources = []
        self.buildings = []
        self.step_count = 0
        
        # Action space: [movement(8), interactions(6)]
        self.action_space = spaces.Discrete(14)
        
        # Observation space: [channels=5, height=15, width=15]
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(5, *size), dtype=np.float32
        )
        
        self.directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),   
            (1, -1),  (1, 0),  (1, 1)
        ]
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state"""
        super().reset(seed=seed) # Handles seeding for you
        
        # Clear environment
        self.grid.fill(0)
        self.agents.clear()
        self.resources.clear()
        self.buildings.clear()
        self.step_count = 0
        
        self._spawn_agents()
        self._spawn_resources()
        
        observation = self._create_observation()
        info = self._get_info()
        
        return observation, info
    
    def _spawn_agents(self):
        """Spawn agents at random empty locations"""
        for i in range(self.n_agents):
            while True:
                # Use the environment's random number generator for reproducibility
                pos = tuple(self.np_random.integers(0, self.size[0], size=2))
                if self.grid[pos] == 0:
                    agent = {
                        'id': i,
                        'pos': pos,
                        'inventory': {'wood': 0, 'stone': 0, 'food': 0, 'tool': 0},
                        'health': 100,
                        'energy': 100
                    }
                    self.agents.append(agent)
                    self.grid[pos] = i + 1
                    break
    
    def _spawn_resources(self):
        """Spawn resources at random empty locations"""
        resource_types = ['wood', 'stone', 'food']
        
        for _ in range(self.max_resources):
            while True:
                pos = tuple(self.np_random.integers(0, self.size[0], size=2))
                if self.grid[pos] == 0:
                    resource_type = self.np_random.choice(resource_types)
                    resource = {
                        'type': resource_type,
                        'pos': pos,
                        'amount': self.np_random.integers(1, 4)
                    }
                    self.resources.append(resource)
                    resource_id = {'wood': -1, 'stone': -2, 'food': -3}[resource_type]
                    self.grid[pos] = resource_id
                    break
    
    def _create_observation(self) -> np.ndarray:
        """Create multi-channel observation tensor"""
        obs = np.zeros((5, *self.size), dtype=np.float32)
        obs[0] = (self.grid == 0).astype(np.float32)
        for resource in self.resources:
            x, y = resource['pos']
            resource_value = {'wood': 1, 'stone': 2, 'food': 3}[resource['type']]
            obs[1, x, y] = resource_value
        for agent in self.agents:
            x, y = agent['pos']
            obs[2, x, y] = agent['id'] + 1
        for building in self.buildings:
            x, y = building['pos']
            obs[3, x, y] = 1
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action for the agent"""
        self.step_count += 1
        
        # Assuming single agent for now, as per action_space
        agent_id = 0 
        reward = self._execute_action(agent_id, action)
        
        terminated = len(self.resources) == 0
        truncated = self.step_count >= self.max_steps
        
        observation = self._create_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def _execute_action(self, agent_id: int, action: int) -> float:
        """Execute single agent action"""
        reward = -0.01
        
        if action < 8:
            reward += self._move_agent(agent_id, action)
        elif action == 8:
            reward += self._gather_resource(agent_id)
        
        return reward
    
    def _move_agent(self, agent_id: int, direction: int) -> float:
        """Move agent in specified direction"""
        agent = self.agents[agent_id]
        old_pos = agent['pos']
        dx, dy = self.directions[direction]
        new_x, new_y = old_pos[0] + dx, old_pos[1] + dy
        
        if (0 <= new_x < self.size[0] and 0 <= new_y < self.size[1] and 
            self.grid[new_x, new_y] <= 0):
            
            self.grid[old_pos] = 0
            agent['pos'] = (new_x, new_y)
            if self.grid[new_x, new_y] == 0:
                self.grid[new_x, new_y] = agent_id + 1
            return 0.0
        
        return -0.1
    
    def _gather_resource(self, agent_id: int) -> float:
        """Gather resource at agent position"""
        agent = self.agents[agent_id]
        x, y = agent['pos']

        target_resource_idx = -1
        for i, resource in enumerate(self.resources):
            if resource['pos'] == (x,y):
                target_resource_idx = i
                break

        if target_resource_idx != -1:
            resource = self.resources.pop(target_resource_idx)
            agent['inventory'][resource['type']] += resource['amount']
            self.grid[x, y] = agent_id + 1
            return 1.0
        
        return -0.05
        
    def _get_info(self) -> Dict[str, Any]:
        """Returns the info dictionary for the current step"""
        return {
            'step_count': self.step_count,
            'agents': self.agents,
            'resources_remaining': len(self.resources)
        }

    def render(self):
        """Render the environment using Matplotlib"""
        if self.render_mode is None:
            gym.logger.warn("You are calling render method without specifying any render mode.")
            return

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        
        ax.set_xticks(np.arange(self.size[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.size[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="lightgray", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)
        ax.tick_params(which="major", bottom=False, left=False, labelbottom=False, labelleft=False)

        resource_colors = {'wood': 'saddlebrown', 'stone': 'dimgray', 'food': 'forestgreen'}
        for resource in self.resources:
            y, x = resource['pos']
            rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=resource_colors[resource['type']], alpha=0.7)
            ax.add_patch(rect)
        
        agent_colors = ['crimson', 'royalblue', 'gold', 'darkviolet']
        for agent in self.agents:
            y, x = agent['pos']
            color = agent_colors[agent['id'] % len(agent_colors)]
            circle = patches.Circle((x, y), 0.3, facecolor=color)
            ax.add_patch(circle)
        
        ax.set_xlim(-0.5, self.size[1] - 0.5)
        ax.set_ylim(-0.5, self.size[0] - 0.5)
        ax.invert_yaxis()
        ax.set_title(f"Ares Prime Simulation - Step {self.step_count}")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Quick test
    env = GridWorld(n_agents=1, render_mode='human')
    obs, info = env.reset(seed=42)
    
    print(f"Environment created! Observation shape: {obs.shape}")
    print(f"Initial Info: {info}")
    env.render()
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\n--- After 1 Step ---")
    print(f"Action taken: {action}")
    print(f"Reward received: {reward}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")
    env.render()