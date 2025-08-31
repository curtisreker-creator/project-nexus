# File: environment/enhanced_grid_world.py
"""
Enhanced GridWorld Environment - Phase 2B Implementation
Configurable map size with scalable resource systems and fog of war foundation
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class EnhancedGridWorld(gym.Env):
    """
    Enhanced multi-agent grid environment with configurable size and advanced mechanics
    
    Key Enhancements:
    - Configurable map size (15x15 → 75x75)  
    - Multi-resource system with unit quantities
    - Field of view system (foundation for fog of war)
    - Inventory limitations and storage buildings
    - Shared mapping system preparation
    """
    
    def __init__(self, 
                 size: Tuple[int, int] = (50, 50),
                 n_agents: int = 2,
                 max_resources: int = 15,
                 max_steps: int = 1000,
                 vision_range: int = 7,
                 inventory_limit: int = 12,
                 render_mode: Optional[str] = None):
        """
        Initialize Enhanced GridWorld Environment
        
        Args:
            size: Grid dimensions (width, height)
            n_agents: Number of agents (1-4)
            max_resources: Maximum resources on map
            max_steps: Episode length limit
            vision_range: Agent field of view radius
            inventory_limit: Maximum items per agent
            render_mode: Rendering mode ('human', 'rgb_array', None)
        """
        super().__init__()
        
        # Core environment parameters
        self.size = size
        self.n_agents = min(max(n_agents, 1), 4)  # Clamp 1-4 agents
        self.max_resources = max_resources
        self.max_steps = max_steps
        self.vision_range = vision_range
        self.inventory_limit = inventory_limit
        self.render_mode = render_mode
        
        # Environment state
        self.grid = np.zeros(size, dtype=np.int32)
        self.step_count = 0
        self.agents = []
        self.resources = []
        self.buildings = []
        
        # Enhanced resource system
        self.resource_types = {
            'wood': {'spawn_weight': 0.25, 'max_units': 8, 'color': 'saddlebrown'},
            'stone': {'spawn_weight': 0.20, 'max_units': 12, 'color': 'dimgray'}, 
            'coal': {'spawn_weight': 0.15, 'max_units': 6, 'color': 'black'},
            'metal_ore': {'spawn_weight': 0.15, 'max_units': 4, 'color': 'silver'},
            'water': {'spawn_weight': 0.15, 'max_units': 10, 'color': 'blue'},
            'food': {'spawn_weight': 0.10, 'max_units': 5, 'color': 'forestgreen'}
        }
        
        # Action space - expanded for future mapping/communication
        self.action_space = spaces.Discrete(22)  # Prepared for mapping actions
        
        # Observation space - prepared for enhanced observations
        if vision_range > 0:
            # Field of view observations
            obs_shape = (10, vision_range, vision_range)  # 10 channels for mapping
        else:
            # Full observability fallback
            obs_shape = (10, size[0], size[1])
            
        self.observation_space = spaces.Box(
            low=-1.0, high=10.0, shape=obs_shape, dtype=np.float32
        )
        
        # Movement directions (8-directional)
        self.directions = [
            (-1, -1), (-1, 0), (-1, 1),  # NW, N, NE
            (0, -1),           (0, 1),   # W,    E  
            (1, -1),  (1, 0),  (1, 1)    # SW, S, SE
        ]
        
        # Action mapping for current implementation
        self.action_mappings = {
            # Movement (0-7)
            0: "move_NW", 1: "move_N", 2: "move_NE",
            3: "move_W",              4: "move_E",
            5: "move_SW", 6: "move_S", 7: "move_SE",
            
            # Current interactions (8-13)
            8: "gather_resource", 9: "build_shelter", 10: "build_storage",
            11: "build_workshop", 12: "craft_tool", 13: "communicate",
            
            # Future mapping/communication actions (14-21) - prepared but not active
            14: "request_help", 15: "share_discovery", 16: "warn_danger", 17: "share_map",
            18: "request_map", 19: "mark_exploration", 20: "claim_territory", 21: "set_waypoint"
        }
        
        # Initialize random number generator
        self.np_random = None
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed, options=options)
        
        # Clear environment state
        self.grid.fill(0)
        self.agents.clear()
        self.resources.clear()
        self.buildings.clear()
        self.step_count = 0
        
        # Initialize environment
        self._spawn_agents()
        self._spawn_resources()
        
        observation = self._create_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step
        
        Args:
            action: Action to execute (0-21)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        self.step_count += 1
        reward = 0.0
        
        # Execute action (currently only agent 0 - multi-agent in future phases)
        if 0 <= action <= 7:
            reward = self._move_agent(0, action)
        elif action == 8:
            reward = self._gather_resource(0)
        elif action == 9:
            reward = self._build_shelter(0)
        elif action == 10:
            reward = self._build_storage(0)
        elif action == 11:
            reward = self._build_workshop(0)
        elif action == 12:
            reward = self._craft_tool(0)
        elif action == 13:
            reward = self._communicate(0)
        elif 14 <= action <= 21:
            # Future mapping/communication actions - placeholder
            reward = self._placeholder_action(0, action)
        
        # Check termination conditions
        terminated = len(self.resources) == 0 or self._check_survival_failure()
        truncated = self.step_count >= self.max_steps
        
        observation = self._create_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _spawn_agents(self):
        """Spawn agents at random empty locations with enhanced state tracking"""
        for i in range(self.n_agents):
            attempts = 0
            while attempts < 100:  # Prevent infinite loops on crowded maps
                pos = tuple(self.np_random.integers(0, self.size[0], size=2))
                if self.grid[pos] == 0:
                    agent = {
                        'id': i,
                        'pos': pos,
                        'inventory': {rtype: 0 for rtype in self.resource_types.keys()},
                        'inventory_total': 0,  # Track total for limit enforcement
                        'health': 100,
                        'energy': 100,
                        'personal_map': np.full(self.size + (6,), -1.0),  # Personal map memory
                        'map_confidence': np.zeros(self.size),  # Confidence in map knowledge
                        'last_communication': None,  # Last message sent/received
                        'power_contribution': 0  # Contribution to group power generation
                    }
                    self.agents.append(agent)
                    self.grid[pos] = i + 1
                    break
                attempts += 1
            
            if attempts >= 100:
                raise RuntimeError(f"Could not place agent {i} after 100 attempts")
    
    def _spawn_resources(self):
        """Spawn diverse resources with unit quantities"""
        resource_list = list(self.resource_types.keys())
        weights = [self.resource_types[rtype]['spawn_weight'] for rtype in resource_list]
        
        for _ in range(self.max_resources):
            attempts = 0
            while attempts < 100:
                pos = tuple(self.np_random.integers(0, self.size[0], size=2))
                if self.grid[pos] == 0:
                    # Select resource type based on spawn weights
                    resource_type = self.np_random.choice(resource_list, p=weights)
                    max_units = self.resource_types[resource_type]['max_units']
                    
                    resource = {
                        'type': resource_type,
                        'pos': pos,
                        'total_units': self.np_random.integers(3, max_units + 1),
                        'remaining_units': 0  # Will be set to total_units
                    }
                    resource['remaining_units'] = resource['total_units']
                    
                    self.resources.append(resource)
                    # Mark grid with resource type index (1-6)
                    resource_index = resource_list.index(resource_type) + 1
                    self.grid[pos] = -(resource_index)  # Negative for resources
                    break
                attempts += 1
            
            if attempts >= 100:
                print(f"Warning: Could not place resource after 100 attempts")
    
    def _create_observation(self, agent_id: int = 0) -> np.ndarray:
        """
        Create observation for specified agent (with field of view support)
        
        Args:
            agent_id: ID of agent to create observation for
            
        Returns:
            Observation tensor of shape (10, vision_range, vision_range) or (10, height, width)
        """
        if self.vision_range > 0:
            return self._create_fov_observation(agent_id)
        else:
            return self._create_full_observation()
    
    def _create_fov_observation(self, agent_id: int) -> np.ndarray:
        """Create field-of-view limited observation"""
        agent_pos = self.agents[agent_id]['pos']
        half_vision = self.vision_range // 2
        
        # Initialize observation with unknown values (-1)
        obs = np.full((10, self.vision_range, self.vision_range), -1.0)
        
        for dy in range(-half_vision, half_vision + 1):
            for dx in range(-half_vision, half_vision + 1):
                world_x = agent_pos[0] + dx
                world_y = agent_pos[1] + dy
                local_x = dx + half_vision
                local_y = dy + half_vision
                
                # Check bounds
                if (0 <= world_x < self.size[0] and 0 <= world_y < self.size[1]):
                    grid_value = self.grid[world_x, world_y]
                    
                    # Channel 0: Terrain/Topology (passable/impassable)
                    obs[0, local_y, local_x] = 1.0 if grid_value == 0 else 0.0
                    
                    # Channels 1-6: Resource types and quantities
                    if grid_value < 0:  # Resource present
                        resource_idx = abs(grid_value) - 1
                        if resource_idx < 6:
                            obs[resource_idx + 1, local_y, local_x] = 1.0
                            # Find resource and add quantity info
                            for resource in self.resources:
                                if resource['pos'] == (world_x, world_y):
                                    obs[resource_idx + 1, local_y, local_x] = resource['remaining_units'] / 10.0
                                    break
                    
                    # Channel 7: Other agents
                    if grid_value > 0:  # Agent present
                        obs[7, local_y, local_x] = grid_value / 10.0
                    
                    # Channel 8: Buildings (placeholder)
                    obs[8, local_y, local_x] = 0.0  # No buildings yet
                    
                    # Channel 9: Activity traces (placeholder)
                    obs[9, local_y, local_x] = 0.0  # No activity tracking yet
        
        return obs
    
    def _create_full_observation(self) -> np.ndarray:
        """Create full observability observation (fallback)"""
        obs = np.zeros((10, self.size[0], self.size[1]))
        
        # Channel 0: Terrain (all passable for now)
        obs[0] = 1.0
        
        # Channels 1-6: Resources by type
        resource_types_list = list(self.resource_types.keys())
        for resource in self.resources:
            x, y = resource['pos']
            resource_idx = resource_types_list.index(resource['type']) + 1
            obs[resource_idx, x, y] = resource['remaining_units'] / 10.0
        
        # Channel 7: Agents
        for agent in self.agents:
            x, y = agent['pos']
            obs[7, x, y] = (agent['id'] + 1) / 10.0
        
        # Channels 8-9: Buildings and activity (placeholder)
        # Will be implemented in subsequent phases
        
        return obs
    
    def _move_agent(self, agent_id: int, direction: int) -> float:
        """Move agent with boundary and collision checking"""
        agent = self.agents[agent_id]
        old_pos = agent['pos']
        dx, dy = self.directions[direction]
        new_x, new_y = old_pos[0] + dx, old_pos[1] + dy
        
        # Boundary checking for any map size
        if not (0 <= new_x < self.size[0] and 0 <= new_y < self.size[1]):
            agent['energy'] = max(0, agent['energy'] - 1)  # Energy penalty for hitting walls
            return -0.1  # Boundary penalty
        
        # Collision detection - can't move to occupied spaces
        if self.grid[new_x, new_y] > 0:  # Another agent present
            return -0.05  # Collision penalty
        
        # Execute movement
        self.grid[old_pos] = 0
        agent['pos'] = (new_x, new_y)
        
        # Update grid - handle resources vs empty space
        if self.grid[new_x, new_y] <= 0:  # Empty or resource
            original_value = self.grid[new_x, new_y]
            self.grid[new_x, new_y] = agent_id + 1
            # If there was a resource, restore it as overlay
            if original_value < 0:
                self.grid[new_x, new_y] = original_value  # Keep resource marker
        
        # Energy cost for movement (larger maps = more energy needed)
        movement_cost = 0.5 if max(self.size) > 30 else 0.2
        agent['energy'] = max(0, agent['energy'] - movement_cost)
        
        return 0.0  # Neutral reward for successful movement
    
    def _gather_resource(self, agent_id: int) -> float:
        """Enhanced resource gathering with unit quantities and inventory limits"""
        agent = self.agents[agent_id]
        x, y = agent['pos']
        
        # Check inventory space
        if agent['inventory_total'] >= self.inventory_limit:
            return -0.1  # Inventory full penalty
        
        # Find resource at current position
        target_resource_idx = -1
        for i, resource in enumerate(self.resources):
            if resource['pos'] == (x, y):
                target_resource_idx = i
                break
        
        if target_resource_idx == -1:
            return -0.05  # No resource here penalty
        
        resource = self.resources[target_resource_idx]
        resource_type = resource['type']
        
        # Gather one unit (or remaining units if less than 1)
        units_to_gather = min(1, resource['remaining_units'])
        
        if units_to_gather > 0:
            # Add to inventory
            agent['inventory'][resource_type] += units_to_gather
            agent['inventory_total'] += units_to_gather
            resource['remaining_units'] -= units_to_gather
            
            # Remove resource if depleted
            if resource['remaining_units'] <= 0:
                self.resources.pop(target_resource_idx)
                self.grid[x, y] = agent_id + 1  # Agent now occupies space
            
            # Reward based on resource rarity and remaining inventory space
            rarity_bonus = 1.0 / self.resource_types[resource_type]['spawn_weight']
            inventory_efficiency = 1.0 - (agent['inventory_total'] / self.inventory_limit)
            
            reward = 1.0 + (rarity_bonus * 0.2) + (inventory_efficiency * 0.3)
            return reward
        
        return -0.02  # Resource depleted penalty
    
    def _build_shelter(self, agent_id: int) -> float:
        """Build shelter (placeholder for future building system)"""
        # Placeholder - will be enhanced in Week 2
        return 0.0
    
    def _build_storage(self, agent_id: int) -> float:
        """Build storage depot (placeholder for future storage system)"""
        # Placeholder - will be enhanced in Week 2
        return 0.0
    
    def _build_workshop(self, agent_id: int) -> float:
        """Build workshop (placeholder for future crafting system)"""
        # Placeholder - will be enhanced in Week 2
        return 0.0
    
    def _craft_tool(self, agent_id: int) -> float:
        """Craft tools from resources (placeholder)"""
        # Placeholder - will be enhanced in Week 2
        return 0.0
    
    def _communicate(self, agent_id: int) -> float:
        """Basic communication (placeholder for mapping system)"""
        # Placeholder - will be enhanced in Week 3
        return 0.0
    
    def _placeholder_action(self, agent_id: int, action: int) -> float:
        """Placeholder for future mapping/communication actions"""
        # These will be implemented in Week 3 (mapping system)
        return 0.0
    
    def _check_survival_failure(self) -> bool:
        """Check if agents have failed survival objectives"""
        # Placeholder for power generation failure conditions
        # Will be implemented in Week 2
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current environment information"""
        return {
            'step_count': self.step_count,
            'agents': self.agents,
            'resources_remaining': len(self.resources),
            'total_resource_units': sum(r['remaining_units'] for r in self.resources),
            'environment_size': self.size,
            'vision_range': self.vision_range,
            'inventory_limit': self.inventory_limit,
            'power_status': 'not_implemented',  # Placeholder
            'shared_map_data': 'not_implemented'  # Placeholder
        }
    
    def render(self):
        """Render environment with enhanced resource visualization"""
        if self.render_mode is None:
            return
        
        # Scale figure size based on map size
        fig_size = min(12, max(6, self.size[0] / 5))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        ax.set_aspect('equal')
        
        # Grid lines
        ax.set_xticks(np.arange(self.size[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.size[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="lightgray", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)
        ax.tick_params(which="major", bottom=False, left=False, labelbottom=False, labelleft=False)
        
        # Render resources with unit indicators
        for resource in self.resources:
            y, x = resource['pos']
            color = self.resource_types[resource['type']]['color']
            
            # Resource square
            rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                   facecolor=color, alpha=0.7, edgecolor='black')
            ax.add_patch(rect)
            
            # Unit quantity text
            ax.text(x, y, str(resource['remaining_units']), 
                   ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        
        # Render agents with inventory indicators
        agent_colors = ['crimson', 'royalblue', 'gold', 'darkviolet']
        for agent in self.agents:
            y, x = agent['pos']
            color = agent_colors[agent['id'] % len(agent_colors)]
            
            # Agent circle
            circle = patches.Circle((x, y), 0.3, facecolor=color, edgecolor='black')
            ax.add_patch(circle)
            
            # Inventory total indicator
            if agent['inventory_total'] > 0:
                ax.text(x, y - 0.6, f"{agent['inventory_total']}/{self.inventory_limit}", 
                       ha='center', va='center', fontsize=6, 
                       bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))
        
        # Set limits and title
        ax.set_xlim(-0.5, self.size[1] - 0.5)
        ax.set_ylim(-0.5, self.size[0] - 0.5)
        ax.invert_yaxis()
        
        title = f"NEXUS Enhanced Environment - Step {self.step_count}"
        if self.vision_range > 0:
            title += f" | Vision: {self.vision_range}x{self.vision_range}"
        title += f" | Size: {self.size[0]}x{self.size[1]}"
        
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
    
    def get_size_info(self) -> Dict[str, Any]:
        """Get detailed environment size information for benchmarking"""
        return {
            'grid_dimensions': self.size,
            'total_cells': self.size[0] * self.size[1],
            'observation_shape': self.observation_space.shape,
            'memory_estimate_mb': self._estimate_memory_usage(),
            'complexity_score': self._calculate_complexity_score()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage for current configuration"""
        # Grid memory
        grid_memory = np.prod(self.size) * 4  # int32
        
        # Observation memory per agent
        obs_memory = np.prod(self.observation_space.shape) * 4  # float32
        total_obs_memory = obs_memory * self.n_agents
        
        # Agent state memory
        agent_memory = (len(self.resource_types) + 10) * 4 * self.n_agents  # Rough estimate
        
        # Personal maps memory
        personal_maps_memory = np.prod(self.size) * 6 * 4 * self.n_agents  # 6 channels per agent
        
        total_bytes = grid_memory + total_obs_memory + agent_memory + personal_maps_memory
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def _calculate_complexity_score(self) -> float:
        """Calculate environment complexity score for research metrics"""
        size_factor = np.prod(self.size) / (15 * 15)  # Relative to original
        resource_factor = self.max_resources / 8  # Relative to original
        agent_factor = self.n_agents
        vision_factor = (15 * 15) / (self.vision_range ** 2) if self.vision_range > 0 else 1.0
        
        return size_factor * resource_factor * agent_factor * vision_factor


# Backward compatibility wrapper
class GridWorld(EnhancedGridWorld):
    """Backward compatibility wrapper for existing code"""
    
    def __init__(self, n_agents: int = 1, max_resources: int = 8, 
                 max_steps: int = 500, render_mode: Optional[str] = None):
        """
        Original GridWorld interface for backward compatibility
        
        Args:
            n_agents: Number of agents
            max_resources: Maximum resources
            max_steps: Episode length
            render_mode: Rendering mode
        """
        super().__init__(
            size=(15, 15),  # Original size
            n_agents=n_agents,
            max_resources=max_resources, 
            max_steps=max_steps,
            vision_range=0,  # Full observability (original behavior)
            inventory_limit=999,  # Unlimited (original behavior)
            render_mode=render_mode
        )


if __name__ == "__main__":
    # Test enhanced environment with different configurations
    print("🧪 Testing Enhanced GridWorld Configurations...")
    
    configs_to_test = [
        {'name': 'Small', 'size': (30, 30), 'n_agents': 2},
        {'name': 'Medium', 'size': (50, 50), 'n_agents': 3}, 
        {'name': 'Large', 'size': (75, 75), 'n_agents': 4}
    ]
    
    for config in configs_to_test:
        print(f"\n--- Testing {config['name']} Configuration ---")
        env = EnhancedGridWorld(size=config['size'], n_agents=config['n_agents'])
        obs, info = env.reset(seed=42)
        
        size_info = env.get_size_info()
        print(f"✅ {config['name']} Environment Created Successfully!")
        print(f"   Grid: {size_info['grid_dimensions']} ({size_info['total_cells']:,} cells)")
        print(f"   Observation: {size_info['observation_shape']}")
        print(f"   Memory Estimate: {size_info['memory_estimate_mb']:.1f} MB")
        print(f"   Complexity Score: {size_info['complexity_score']:.2f}x")
        
        # Test a few steps
        for step in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   Step {step+1}: Action={action}, Reward={reward:.3f}")
        
        print(f"   Final Resources: {info['total_resource_units']} units across {info['resources_remaining']} nodes")
    
    # Test backward compatibility
    print(f"\n--- Testing Backward Compatibility ---")
    old_env = GridWorld(n_agents=1, max_resources=5)
    obs, info = old_env.reset(seed=42)
    print(f"✅ Backward Compatible GridWorld: {old_env.size}, Obs: {obs.shape}")
    
    print(f"\n🎉 All Enhanced GridWorld tests passed!")
    print(f"🚀 Ready for Week 1 Day 1-2 implementation!")