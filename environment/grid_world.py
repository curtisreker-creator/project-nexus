# File: environment/grid_world.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Dict, Any, List

class GridWorld(gym.Env):
    """15x15 Grid World Environment for Multi-Agent RL"""
    
    def __init__(self, size: Tuple[int, int] = (15, 15), n_agents: int = 1, 
             max_resources: int = 8, max_steps: int = 1000, seed: int = None):
        super().__init__()
        
        self.size = size
        self.n_agents = n_agents
        self.max_resources = max_resources
        self.seed_value = seed
        
        # Environment state
        self.grid = np.zeros(size, dtype=np.int32)
        self.agents = []
        self.resources = []
        self.buildings = []
        self.step_count = 0
        self.max_steps = 1000
        
        # Action space: [movement(8), interactions(6)]
        self.action_space = spaces.Discrete(14)
        
        # Observation space: [channels=5, height=15, width=15]
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(5, *size), dtype=np.float32
        )
        
        # Movement directions (8-directional)
        self.directions = [
            (-1, -1), (-1, 0), (-1, 1),  # Up-left, Up, Up-right
            (0, -1),           (0, 1),   # Left, Right  
            (1, -1),  (1, 0),  (1, 1)    # Down-left, Down, Down-right
        ]
        
        self.reset()
    
    def reset(self, seed=None) -> np.ndarray:
        """Reset environment to initial state"""
        if seed is not None:
            self.seed_value = seed
        
        if self.seed_value is not None:
            np.random.seed(self.seed_value)
        
        # Clear environment
        self.grid.fill(0)
        self.agents.clear()
        self.resources.clear()
        self.buildings.clear()
        self.step_count = 0
        
        # Spawn agents and resources
        self._spawn_agents()
        self._spawn_resources()
        
        return self._create_observation()
    
    def _spawn_agents(self):
        """Spawn agents at random empty locations"""
        for i in range(self.n_agents):
            while True:
                x, y = np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1])
                if self.grid[x, y] == 0:  # Empty cell
                    agent = {
                        'id': i,
                        'pos': (x, y),
                        'inventory': {'wood': 0, 'stone': 0, 'food': 0, 'tool': 0},
                        'health': 100,
                        'energy': 100
                    }
                    self.agents.append(agent)
                    self.grid[x, y] = i + 1  # Agent ID in grid
                    break
    
    def _spawn_resources(self):
        """Spawn resources at random empty locations"""
        resource_types = ['wood', 'stone', 'food']
        
        for _ in range(self.max_resources):
            while True:
                x, y = np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1])
                if self.grid[x, y] == 0:  # Empty cell
                    resource_type = np.random.choice(resource_types)
                    resource = {
                        'type': resource_type,
                        'pos': (x, y),
                        'amount': np.random.randint(1, 4)
                    }
                    self.resources.append(resource)
                    # Mark in grid (negative values for resources)
                    resource_id = {'wood': -1, 'stone': -2, 'food': -3}[resource_type]
                    self.grid[x, y] = resource_id
                    break
    
    def _create_observation(self) -> np.ndarray:
        """Create multi-channel observation tensor"""
        obs = np.zeros((5, *self.size), dtype=np.float32)
        
        # Channel 0: Environment topology
        obs[0] = (self.grid == 0).astype(np.float32)
        
        # Channel 1: Resources
        for resource in self.resources:
            x, y = resource['pos']
            resource_value = {'wood': 1, 'stone': 2, 'food': 3}[resource['type']]
            obs[1, x, y] = resource_value
        
        # Channel 2: Agents
        for agent in self.agents:
            x, y = agent['pos']
            obs[2, x, y] = agent['id'] + 1
        
        # Channel 3: Buildings (future)
        for building in self.buildings:
            x, y = building['pos']
            obs[3, x, y] = 1
            
        # Channel 4: Activity history (future)
        
        return obs
    
    def step(self, actions: List[int]) -> Tuple[np.ndarray, List[float], bool, Dict]:
        """Execute actions for all agents"""
        if not isinstance(actions, list):
            actions = [actions]
        
        rewards = []
        
        for i, action in enumerate(actions):
            if i < len(self.agents):
                reward = self._execute_action(i, action)
                rewards.append(reward)
        
        self.step_count += 1
        done = self.step_count >= self.max_steps or len(self.resources) == 0
        
        observation = self._create_observation()
        info = {
            'step_count': self.step_count,
            'agents': self.agents,
            'resources_remaining': len(self.resources)
        }
        
        return observation, rewards, done, info
    
    def _execute_action(self, agent_id: int, action: int) -> float:
        """Execute single agent action"""
        reward = -0.01  # Time penalty
        
        if action < 8:  # Movement
            reward += self._move_agent(agent_id, action)
        elif action == 8:  # Gather
            reward += self._gather_resource(agent_id)
        # Other actions (build, use tool, etc.) can be added later
        
        return reward
    
    def _move_agent(self, agent_id: int, direction: int) -> float:
        """Move agent in specified direction"""
        agent = self.agents[agent_id]
        old_pos = agent['pos']
        dx, dy = self.directions[direction]
        new_x, new_y = old_pos[0] + dx, old_pos[1] + dy
        
        # Check bounds and collision
        if (0 <= new_x < self.size[0] and 0 <= new_y < self.size[1] and 
            self.grid[new_x, new_y] <= 0):  # Empty or resource
            
            self.grid[old_pos] = 0
            agent['pos'] = (new_x, new_y)
            if self.grid[new_x, new_y] == 0:
                self.grid[new_x, new_y] = agent_id + 1
            return 0.0
        
        return -0.1  # Invalid move penalty
    
    def _gather_resource(self, agent_id: int) -> float:
        """Gather resource at agent position"""
        agent = self.agents[agent_id]
        pos = agent['pos']
        
        for i, resource in enumerate(self.resources):
            if resource['pos'] == pos:
                agent['inventory'][resource['type']] += resource['amount']
                self.resources.pop(i)
                self.grid[pos] = agent_id + 1
                return 1.0  # Gathering reward
        
        return -0.05  # Failed gathering
    
    def render(self, mode='human'):
        """Render the environment"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw grid lines
        for i in range(self.size[0] + 1):
            ax.axhline(y=i, color='lightgray', linewidth=0.5)
        for i in range(self.size[1] + 1):
            ax.axvline(x=i, color='lightgray', linewidth=0.5)
        
        # Draw resources
        resource_colors = {'wood': 'brown', 'stone': 'gray', 'food': 'green'}
        for resource in self.resources:
            x, y = resource['pos']
            color = resource_colors[resource['type']]
            rect = patches.Rectangle((y, self.size[0] - x - 1), 1, 1, 
                                   facecolor=color, alpha=0.7)
            ax.add_patch(rect)
        
        # Draw agents
        agent_colors = ['red', 'blue', 'yellow', 'purple']
        for agent in self.agents:
            x, y = agent['pos']
            color = agent_colors[agent['id'] % len(agent_colors)]
            circle = patches.Circle((y + 0.5, self.size[0] - x - 0.5), 0.3, 
                                  facecolor=color)
            ax.add_patch(circle)
        
        ax.set_xlim(0, self.size[1])
        ax.set_ylim(0, self.size[0])
        ax.set_aspect('equal')
        ax.set_title(f'NEXUS Grid World - Step {self.step_count}')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Quick test
    env = GridWorld(n_agents=2, seed=42)
    obs = env.reset()
    print(f"Environment created! Observation shape: {obs.shape}")
    env.render()