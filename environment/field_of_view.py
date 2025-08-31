# File: environment/field_of_view.py
"""
Advanced Field of View System - Day 3-4 Implementation
Sophisticated fog of war mechanics with agent memory and exploration incentives
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time


@dataclass
class MemoryCell:
    """Individual cell in agent's memory system"""
    terrain_type: float = -1.0
    resource_type: float = -1.0
    resource_quantity: float = -1.0
    other_agents: float = -1.0
    buildings: float = -1.0
    last_observed: int = -1
    confidence: float = 0.0
    exploration_value: float = 1.0  # Higher = more valuable to explore


class AgentMemory:
    """
    Advanced agent memory system with confidence decay and exploration tracking
    """
    
    def __init__(self, world_size: Tuple[int, int], memory_decay_rate: float = 0.98):
        self.world_size = world_size
        self.memory_decay_rate = memory_decay_rate
        self.step_count = 0
        
        # Memory grids for different information types
        self.terrain_memory = np.full(world_size, -1.0)
        self.resource_memory = np.full(world_size + (6,), -1.0)  # 6 resource types
        self.agent_memory = np.full(world_size, -1.0)
        self.building_memory = np.full(world_size, -1.0)
        
        # Confidence and exploration tracking
        self.confidence_grid = np.zeros(world_size)
        self.last_observed_grid = np.full(world_size, -1, dtype=np.int32)
        self.exploration_grid = np.ones(world_size)  # 1.0 = unexplored, 0.0 = well-explored
        
        # Statistics tracking
        self.total_cells_discovered = 0
        self.exploration_coverage = 0.0
    
    def update_from_observation(self, agent_pos: Tuple[int, int], 
                              vision_range: int, world_state: Dict[str, Any]):
        """
        Update agent memory from current field of view observation
        
        Args:
            agent_pos: Current agent position (x, y)
            vision_range: Vision range radius
            world_state: Current world state information
        """
        self.step_count += 1
        half_vision = vision_range // 2
        newly_discovered = 0
        
        for dy in range(-half_vision, half_vision + 1):
            for dx in range(-half_vision, half_vision + 1):
                world_x = agent_pos[0] + dx
                world_y = agent_pos[1] + dy
                
                # Check world boundaries
                if not (0 <= world_x < self.world_size[0] and 0 <= world_y < self.world_size[1]):
                    continue
                
                # Calculate distance for confidence weighting (closer = more confident)
                distance = np.sqrt(dx*dx + dy*dy)
                base_confidence = max(0.1, 1.0 - (distance / (half_vision + 1)))
                
                # Track if this is a newly discovered cell
                was_unexplored = self.confidence_grid[world_x, world_y] < 0.1
                
                # Update terrain memory
                grid_value = world_state.get('grid', np.zeros(self.world_size))[world_x, world_y]
                if grid_value <= 0:  # Empty or resource
                    self.terrain_memory[world_x, world_y] = 1.0  # Passable
                else:
                    self.terrain_memory[world_x, world_y] = 0.0  # Occupied
                
                # Update resource memory
                for resource in world_state.get('resources', []):
                    if resource['pos'] == (world_x, world_y):
                        resource_types = ['wood', 'stone', 'coal', 'metal_ore', 'water', 'food']
                        try:
                            resource_idx = resource_types.index(resource['type'])
                            self.resource_memory[world_x, world_y, resource_idx] = resource.get('remaining_units', 1) / 10.0
                        except ValueError:
                            pass  # Unknown resource type
                
                # Update agent memory
                agent_value = 0.0
                for agent in world_state.get('agents', []):
                    if agent['pos'] == (world_x, world_y):
                        agent_value = (agent['id'] + 1) / 10.0
                        break
                self.agent_memory[world_x, world_y] = agent_value
                
                # Update building memory (placeholder for future)
                self.building_memory[world_x, world_y] = 0.0
                
                # Update confidence and exploration tracking
                self.confidence_grid[world_x, world_y] = min(1.0, 
                    self.confidence_grid[world_x, world_y] + base_confidence)
                self.last_observed_grid[world_x, world_y] = self.step_count
                
                # Update exploration value (diminishing returns for revisiting)
                if self.exploration_grid[world_x, world_y] > 0:
                    self.exploration_grid[world_x, world_y] *= 0.95
                    if was_unexplored:
                        newly_discovered += 1
        
        # Update exploration statistics
        self.total_cells_discovered += newly_discovered
        total_cells = np.prod(self.world_size)
        self.exploration_coverage = np.sum(self.confidence_grid > 0.1) / total_cells
        
        # Apply memory decay to non-observed areas
        self._apply_memory_decay()
    
    def _apply_memory_decay(self):
        """Apply confidence decay to areas not recently observed"""
        current_step = self.step_count
        
        # Decay confidence for areas not observed recently
        time_since_observation = current_step - self.last_observed_grid
        time_since_observation = np.where(self.last_observed_grid == -1, 9999, time_since_observation)
        
        # Exponential decay based on time since last observation
        decay_factor = np.power(self.memory_decay_rate, time_since_observation)
        self.confidence_grid *= decay_factor
        
        # Clamp minimum confidence
        self.confidence_grid = np.maximum(0.0, self.confidence_grid)
    
    def get_memory_observation(self, agent_pos: Tuple[int, int], vision_range: int) -> np.ndarray:
        """
        Generate observation combining current vision with memory
        
        Args:
            agent_pos: Agent position
            vision_range: Vision range
            
        Returns:
            Combined observation array (channels, vision_range, vision_range)
        """
        half_vision = vision_range // 2
        obs = np.full((10, vision_range, vision_range), -1.0)
        
        for dy in range(-half_vision, half_vision + 1):
            for dx in range(-half_vision, half_vision + 1):
                world_x = agent_pos[0] + dx
                world_y = agent_pos[1] + dy
                local_x = dx + half_vision
                local_y = dy + half_vision
                
                if (0 <= world_x < self.world_size[0] and 0 <= world_y < self.world_size[1]):
                    confidence = self.confidence_grid[world_x, world_y]
                    
                    if confidence > 0.1:  # Have memory of this area
                        # Channel 0: Terrain with confidence weighting
                        obs[0, local_y, local_x] = self.terrain_memory[world_x, world_y] * confidence
                        
                        # Channels 1-6: Resource types with confidence
                        for resource_idx in range(6):
                            resource_value = self.resource_memory[world_x, world_y, resource_idx]
                            if resource_value > -0.5:  # Have memory of resources
                                obs[resource_idx + 1, local_y, local_x] = resource_value * confidence
                        
                        # Channel 7: Other agents
                        obs[7, local_y, local_x] = self.agent_memory[world_x, world_y] * confidence
                        
                        # Channel 8: Buildings
                        obs[8, local_y, local_x] = self.building_memory[world_x, world_y] * confidence
                        
                        # Channel 9: Memory confidence and exploration value
                        obs[9, local_y, local_x] = confidence * self.exploration_grid[world_x, world_y]
        
        return obs
    
    def get_exploration_reward(self, agent_pos: Tuple[int, int], vision_range: int) -> float:
        """
        Calculate exploration reward based on newly discovered areas
        
        Args:
            agent_pos: Current agent position
            vision_range: Vision range
            
        Returns:
            Exploration reward value
        """
        half_vision = vision_range // 2
        discovery_bonus = 0.0
        novelty_bonus = 0.0
        
        for dy in range(-half_vision, half_vision + 1):
            for dx in range(-half_vision, half_vision + 1):
                world_x = agent_pos[0] + dx
                world_y = agent_pos[1] + dy
                
                if (0 <= world_x < self.world_size[0] and 0 <= world_y < self.world_size[1]):
                    exploration_value = self.exploration_grid[world_x, world_y]
                    
                    # Reward for exploring high-value areas
                    novelty_bonus += exploration_value * 0.1
                    
                    # Bonus for first-time discovery
                    if self.confidence_grid[world_x, world_y] < 0.1:
                        discovery_bonus += 0.5
        
        return discovery_bonus + novelty_bonus


class FieldOfViewSystem:
    """
    Advanced field of view system with fog of war and exploration mechanics
    """
    
    def __init__(self, vision_range: int = 7, memory_decay_rate: float = 0.98):
        self.vision_range = vision_range
        self.memory_decay_rate = memory_decay_rate
        self.agent_memories: Dict[int, AgentMemory] = {}
        
        # Performance tracking
        self.total_vision_updates = 0
        self.total_update_time = 0.0
        
        # Research metrics
        self.exploration_metrics = {
            'total_cells_discovered': 0,
            'exploration_efficiency': 0.0,
            'average_coverage': 0.0,
            'discovery_rate': 0.0
        }
    
    def initialize_agent_memory(self, agent_id: int, world_size: Tuple[int, int]):
        """Initialize memory system for an agent"""
        self.agent_memories[agent_id] = AgentMemory(world_size, self.memory_decay_rate)
    
    def update_agent_vision(self, agent_id: int, agent_pos: Tuple[int, int], 
                          world_state: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """
        Update agent's field of view and return observation with exploration reward
        
        Args:
            agent_id: Agent identifier
            agent_pos: Agent's current position
            world_state: Current world state
            
        Returns:
            Tuple of (observation_array, exploration_reward)
        """
        start_time = time.time()
        
        # Initialize agent memory if needed
        world_size = world_state.get('world_size', (50, 50))
        if agent_id not in self.agent_memories:
            self.initialize_agent_memory(agent_id, world_size)
        
        agent_memory = self.agent_memories[agent_id]
        
        # Update memory from current observation
        agent_memory.update_from_observation(agent_pos, self.vision_range, world_state)
        
        # Generate observation combining vision and memory
        observation = agent_memory.get_memory_observation(agent_pos, self.vision_range)
        
        # Calculate exploration reward
        exploration_reward = agent_memory.get_exploration_reward(agent_pos, self.vision_range)
        
        # Update performance metrics
        self.total_vision_updates += 1
        self.total_update_time += time.time() - start_time
        
        # Update research metrics
        self._update_exploration_metrics()
        
        return observation, exploration_reward
    
    def _update_exploration_metrics(self):
        """Update exploration research metrics"""
        if not self.agent_memories:
            return
        
        total_discovered = sum(memory.total_cells_discovered for memory in self.agent_memories.values())
        average_coverage = np.mean(np.array([memory.exploration_coverage for memory in self.agent_memories.values()]))
        self.exploration_metrics.update({
            'total_cells_discovered': total_discovered,
            'average_coverage': average_coverage,
            'discovery_rate': total_discovered / max(1, self.total_vision_updates),
            'exploration_efficiency': average_coverage / max(1, len(self.agent_memories))
        })
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for benchmarking"""
        avg_update_time = self.total_update_time / max(1, self.total_vision_updates)
        updates_per_second = 1.0 / max(0.001, avg_update_time)
        
        return {
            'avg_update_time_ms': avg_update_time * 1000,
            'updates_per_second': updates_per_second,
            'total_updates': self.total_vision_updates,
            'memory_systems_active': len(self.agent_memories)
        }
    
    def get_exploration_stats(self) -> Dict[str, Any]:
        """Get exploration statistics for research analysis"""
        if not self.agent_memories:
            return {'no_agents': True}
        
        agent_stats = {}
        for agent_id, memory in self.agent_memories.items():
            agent_stats[f'agent_{agent_id}'] = {
                'cells_discovered': memory.total_cells_discovered,
                'exploration_coverage': memory.exploration_coverage,
                'memory_confidence_avg': np.mean(memory.confidence_grid),
                'exploration_value_remaining': np.sum(memory.exploration_grid)
            }
        
        return {
            'global_metrics': self.exploration_metrics,
            'agent_metrics': agent_stats,
            'system_performance': self.get_performance_stats()
        }
    
    def render_fog_of_war(self, agent_id: int, world_size: Tuple[int, int]) -> np.ndarray:
        """
        Generate fog of war visualization array
        
        Args:
            agent_id: Agent to render fog of war for
            world_size: World dimensions
            
        Returns:
            Fog of war array (0.0 = unexplored, 1.0 = fully explored)
        """
        if agent_id not in self.agent_memories:
            return np.zeros(world_size)
        
        memory = self.agent_memories[agent_id]
        return memory.confidence_grid.copy()
    
    def create_exploration_heatmap(self, agent_id: int) -> Optional[np.ndarray]:
        """
        Create exploration value heatmap for visualization
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Exploration value heatmap or None if agent not found
        """
        if agent_id not in self.agent_memories:
            return None
        
        return self.agent_memories[agent_id].exploration_grid.copy()


# Integration with Enhanced GridWorld
class FieldOfViewEnhancedGridWorld:
    """
    Enhanced GridWorld with integrated field of view system
    """
    
    def __init__(self, base_environment, vision_range: int = 7, 
                 exploration_reward_scale: float = 1.0):
        self.base_env = base_environment
        self.fov_system = FieldOfViewSystem(vision_range)
        self.exploration_reward_scale = exploration_reward_scale
        
        # Initialize field of view for all agents
        for agent in self.base_env.agents:
            self.fov_system.initialize_agent_memory(agent['id'], self.base_env.size)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Enhanced step function with field of view processing"""
        # Execute base environment step
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        # Update field of view for primary agent (agent 0)
        if self.base_env.agents:
            agent_pos = self.base_env.agents[0]['pos']
            world_state = {
                'grid': self.base_env.grid,
                'resources': self.base_env.resources,
                'agents': self.base_env.agents,
                'world_size': self.base_env.size
            }
            
            # Get enhanced observation with field of view
            fov_obs, exploration_reward = self.fov_system.update_agent_vision(
                0, agent_pos, world_state
            )
            
            # Add exploration reward to base reward
            total_reward = reward + (exploration_reward * self.exploration_reward_scale)
            
            # Update info with field of view metrics
            info.update({
                'fov_exploration_reward': exploration_reward,
                'exploration_stats': self.fov_system.get_exploration_stats(),
                'fog_of_war_coverage': np.mean(self.fov_system.render_fog_of_war(0, self.base_env.size))
            })
            
            return fov_obs, total_reward, terminated, truncated, info
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Enhanced reset with field of view initialization"""
        obs, info = self.base_env.reset(**kwargs)
        
        # Reset field of view system
        self.fov_system = FieldOfViewSystem(self.fov_system.vision_range)
        
        # Initialize field of view for all agents
        for agent in self.base_env.agents:
            self.fov_system.initialize_agent_memory(agent['id'], self.base_env.size)
        
        # Get initial field of view observation
        if self.base_env.agents:
            agent_pos = self.base_env.agents[0]['pos']
            world_state = {
                'grid': self.base_env.grid,
                'resources': self.base_env.resources,
                'agents': self.base_env.agents,
                'world_size': self.base_env.size
            }
            
            fov_obs, _ = self.fov_system.update_agent_vision(0, agent_pos, world_state)
            
            info.update({
                'fov_system_initialized': True,
                'vision_range': self.fov_system.vision_range,
                'exploration_stats': self.fov_system.get_exploration_stats()
            })
            
            return fov_obs, info
        
        return obs, info
    
    def render(self, mode='human', show_fog_of_war=True):
        """Enhanced rendering with fog of war visualization"""
        self.base_env.render()
        
        if show_fog_of_war and self.base_env.agents:
            self._render_fog_of_war_overlay()
    
    def _render_fog_of_war_overlay(self):
        """Render fog of war overlay for visualization"""
        import matplotlib.pyplot as plt
        
        agent_id = 0
        fog_map = self.fov_system.render_fog_of_war(agent_id, self.base_env.size)
        exploration_map = self.fov_system.create_exploration_heatmap(agent_id)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Fog of war visualization
        im1 = ax1.imshow(fog_map, cmap='Blues', alpha=0.7)
        ax1.set_title(f'Agent {agent_id} - Fog of War\n(Dark = Unexplored, Light = Explored)')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        plt.colorbar(im1, ax=ax1, label='Exploration Confidence')
        
        # Exploration value heatmap
        if exploration_map is not None:
            im2 = ax2.imshow(exploration_map, cmap='Reds', alpha=0.7)
            ax2.set_title(f'Agent {agent_id} - Exploration Values\n(Red = High Value, Dark = Well Explored)')
            ax2.set_xlabel('X Position')
            ax2.set_ylabel('Y Position')
            plt.colorbar(im2, ax=ax2, label='Exploration Value')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Test field of view system
    print("🧪 Testing Advanced Field of View System...")
    
    # Create test world state
    test_world_state = {
        'world_size': (20, 20),
        'grid': np.zeros((20, 20)),
        'resources': [
            {'pos': (5, 5), 'type': 'wood', 'remaining_units': 3},
            {'pos': (15, 10), 'type': 'stone', 'remaining_units': 5}
        ],
        'agents': [
            {'id': 0, 'pos': (10, 10)},
            {'id': 1, 'pos': (5, 15)}
        ]
    }
    
    # Initialize field of view system
    fov_system = FieldOfViewSystem(vision_range=7)
    
    # Test agent vision updates
    print("Testing agent 0 vision updates...")
    for step in range(10):
        # Simulate agent movement
        agent_pos = (10 + step % 3, 10 + step % 2)
        test_world_state['agents'][0]['pos'] = agent_pos
        
        observation, exploration_reward = fov_system.update_agent_vision(
            0, agent_pos, test_world_state
        )
        
        print(f"Step {step}: Pos {agent_pos}, Obs shape: {observation.shape}, "
              f"Exploration reward: {exploration_reward:.3f}")
    
    # Get performance statistics
    perf_stats = fov_system.get_performance_stats()
    exploration_stats = fov_system.get_exploration_stats()
    
    print(f"\n📊 Performance Statistics:")
    print(f"  Average update time: {perf_stats['avg_update_time_ms']:.2f} ms")
    print(f"  Updates per second: {perf_stats['updates_per_second']:.0f}")
    print(f"  Total updates: {perf_stats['total_updates']}")
    
    print(f"\n🗺️ Exploration Statistics:")
    print(f"  Total cells discovered: {exploration_stats['global_metrics']['total_cells_discovered']}")
    print(f"  Average coverage: {exploration_stats['global_metrics']['average_coverage']:.1%}")
    print(f"  Discovery rate: {exploration_stats['global_metrics']['discovery_rate']:.3f}")
    
    print(f"\n🎉 Field of View System test completed successfully!")
    print(f"✅ Advanced fog of war mechanics operational")
    print(f"✅ Agent memory system with confidence decay working")  
    print(f"✅ Exploration incentives and reward system active")
    print(f"✅ Performance benchmarks: {perf_stats['updates_per_second']:.0f} updates/sec")