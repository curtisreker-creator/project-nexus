# File: tests/test_field_of_view_system.py
"""
Comprehensive test suite for Day 3-4 Field of View System
Tests fog of war mechanics, agent memory, exploration incentives, and performance
"""

import pytest
import numpy as np
import torch
import time
from typing import Dict, Tuple, Any
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# FIX: Update imports to use the new V2 class and correct, case-sensitive filename
try:
    from environment.field_of_view import FieldOfViewSystem, AgentMemory, FieldOfViewEnhancedGridWorld
    from environment.EnhancedGridWorldV2 import EnhancedGridWorldV2
    FOV_AVAILABLE = True
except ImportError:
    FOV_AVAILABLE = False
    print("⚠️ Field of View System not available - tests will be skipped")


class TestFieldOfViewSystem:
    """Test suite for field of view system functionality"""
    
    @pytest.mark.skipif(not FOV_AVAILABLE, reason="Field of View System not available")
    def test_agent_memory_initialization(self):
        """Test agent memory system initialization and basic operations"""
        print("\n🧠 Testing agent memory initialization...")
        
        world_size = (30, 30)
        agent_memory = AgentMemory(world_size, memory_decay_rate=0.95)
        
        # Verify initialization
        assert agent_memory.world_size == world_size
        assert agent_memory.terrain_memory.shape == world_size
        assert agent_memory.resource_memory.shape == world_size + (6,)
        assert agent_memory.confidence_grid.shape == world_size
        
        # Test that memory starts empty/unknown
        assert np.all(agent_memory.terrain_memory == -1.0)
        assert np.all(agent_memory.confidence_grid == 0.0)
        assert agent_memory.total_cells_discovered == 0
        
        print(f"   ✅ Memory initialized: {world_size} world, decay rate 0.95")
        
    @pytest.mark.skipif(not FOV_AVAILABLE, reason="Field of View System not available")
    def test_field_of_view_observation_generation(self):
        """Test field of view observation generation with limited vision"""
        print("\n👁️ Testing field of view observation generation...")
        
        vision_range = 7
        fov_system = FieldOfViewSystem(vision_range=vision_range)
        
        # Create test world state
        world_size = (40, 40)
        test_world_state = {
            'world_size': world_size,
            'grid': np.zeros(world_size),
            'resources': [
                {'pos': (10, 10), 'type': 'wood', 'remaining_units': 5},
                {'pos': (15, 15), 'type': 'stone', 'remaining_units': 3}
            ],
            'agents': [{'id': 0, 'pos': (12, 12)}]
        }
        
        # Test observation generation
        agent_pos = (12, 12)
        observation, exploration_reward = fov_system.update_agent_vision(
            0, agent_pos, test_world_state
        )
        
        # Verify observation properties
        expected_shape = (10, vision_range, vision_range)
        assert observation.shape == expected_shape, f"Expected {expected_shape}, got {observation.shape}"
        
        # Check that observation contains some visible information
        visible_cells = np.sum(observation != -1.0)
        assert visible_cells > 0, "No visible information in observation"
        
        # Check exploration reward is positive for first visit
        assert exploration_reward > 0, f"Expected positive exploration reward, got {exploration_reward}"
        
        print(f"   ✅ Observation generated: {observation.shape}")
        print(f"   ✅ Visible cells: {visible_cells}, Exploration reward: {exploration_reward:.3f}")
        
    @pytest.mark.skipif(not FOV_AVAILABLE, reason="Field of View System not available")
    def test_fog_of_war_mechanics(self):
        """Test fog of war mechanics and memory persistence"""
        print("\n🌫️ Testing fog of war mechanics...")
        
        vision_range = 5
        fov_system = FieldOfViewSystem(vision_range=vision_range)
        world_size = (20, 20)
        
        # Create simple world state
        world_state = {
            'world_size': world_size,
            'grid': np.zeros(world_size),
            'resources': [{'pos': (10, 10), 'type': 'wood', 'remaining_units': 3}],
            'agents': [{'id': 0, 'pos': (5, 5)}]
        }
        
        # Agent explores area around (5, 5)
        observation1, reward1 = fov_system.update_agent_vision(0, (5, 5), world_state)
        
        # Move agent to area around (15, 15) - far from initial position
        world_state['agents'][0]['pos'] = (15, 15)
        fov_system.update_agent_vision(0, (15, 15), world_state)
        
        # Move back to original area (5, 5) - should have memory
        world_state['agents'][0]['pos'] = (5, 5)
        observation3, reward3 = fov_system.update_agent_vision(0, (5, 5), world_state)
        
        # Check that memory persists (less fog of war on return)
        fog_coverage = fov_system.render_fog_of_war(0, world_size)
        explored_percentage = np.sum(fog_coverage > 0.1) / np.prod(world_size)
        
        assert explored_percentage > 0.05, f"Expected >5% explored, got {explored_percentage:.1%}"
        assert reward3 < reward1, f"Expected lower reward on revisit, got {reward3} vs {reward1}"
        
        print(f"   ✅ Fog of war working: {explored_percentage:.1%} explored")
        print(f"   ✅ Memory persistence: Initial reward {reward1:.3f}, Revisit reward {reward3:.3f}")
        
    @pytest.mark.skipif(not FOV_AVAILABLE, reason="Field of View System not available")
    def test_exploration_incentives(self):
        """Test exploration incentive system and reward mechanics"""
        print("\n🗺️ Testing exploration incentives...")
        
        vision_range = 7
        fov_system = FieldOfViewSystem(vision_range=vision_range)
        world_size = (25, 25)
        
        world_state = {
            'world_size': world_size,
            'grid': np.zeros(world_size),
            'resources': [],
            'agents': [{'id': 0, 'pos': (12, 12)}]
        }
        
        exploration_path = [
            (12, 12), (12, 13), (12, 14), (13, 14), (14, 14),
            (12, 12), (12, 13)
        ]
        
        rewards = []
        for pos in exploration_path:
            world_state['agents'][0]['pos'] = pos
            _, reward = fov_system.update_agent_vision(0, pos, world_state)
            rewards.append(reward)
            
        new_area_rewards = rewards[:5]
        revisit_rewards = rewards[5:]
        
        avg_new_reward = np.mean(np.array(new_area_rewards))
        avg_revisit_reward = np.mean(np.array(revisit_rewards))
        
        assert avg_new_reward > avg_revisit_reward, "New areas should reward more than revisits"
        
        print(f"   ✅ Exploration incentives working: New {avg_new_reward:.3f} vs Revisit {avg_revisit_reward:.3f}")
        
    @pytest.mark.skipif(not FOV_AVAILABLE, reason="Field of View System not available")
    def test_memory_decay_mechanics(self):
        """Test memory confidence decay over time"""
        print("\n⏰ Testing memory decay mechanics...")
        
        vision_range = 5
        memory_decay_rate = 0.9
        fov_system = FieldOfViewSystem(vision_range=vision_range, memory_decay_rate=memory_decay_rate)
        world_size = (15, 15)
        
        world_state = {
            'world_size': world_size, 'grid': np.zeros(world_size),
            'resources': [], 'agents': [{'id': 0, 'pos': (7, 7)}]
        }
        
        fov_system.update_agent_vision(0, (7, 7), world_state)
        initial_confidence = fov_system.agent_memories[0].confidence_grid[7, 7]
        
        for _ in range(10):
            world_state['agents'][0]['pos'] = (1, 1)
            fov_system.update_agent_vision(0, (1, 1), world_state)
        
        decayed_confidence = fov_system.agent_memories[0].confidence_grid[7, 7]
        
        assert decayed_confidence < initial_confidence, "Memory should decay"
        
        print(f"   ✅ Memory decay working: {initial_confidence:.3f} → {decayed_confidence:.3f}")
        
    @pytest.mark.skipif(not FOV_AVAILABLE, reason="Field of View System not available")
    def test_performance_benchmarks(self):
        """Test field of view system performance across different scales"""
        print("\n⚡ Testing performance benchmarks...")
        
        config = {'world_size': (40, 40), 'vision_range': 7}
        fov_system = FieldOfViewSystem(vision_range=config['vision_range'])
        world_size = config['world_size']
        
        world_state = {
            'world_size': world_size, 'grid': np.zeros(world_size),
            'resources': [], 'agents': [{'id': 0, 'pos': (world_size[0]//2, world_size[1]//2)}]
        }
        
        start_time = time.time()
        for i in range(50):
            agent_pos = (world_size[0]//2 + i % 5, world_size[1]//2 + i % 3)
            world_state['agents'][0]['pos'] = agent_pos
            fov_system.update_agent_vision(0, agent_pos, world_state)
        
        elapsed_time = time.time() - start_time
        updates_per_second = 50 / elapsed_time
        
        assert updates_per_second > 50, f"Performance too slow: {updates_per_second:.1f} updates/sec"
        print(f"   ✅ Performance acceptable: {updates_per_second:.0f} updates/sec")
        
    @pytest.mark.skipif(not FOV_AVAILABLE, reason="Field of View System not available")
    def test_integration_with_enhanced_environment(self):
        """Test integration between field of view system and enhanced environment"""
        print("\n🔗 Testing integration with enhanced environment...")
        
        # FIX: Instantiate the correct V2 class
        base_env = EnhancedGridWorldV2(size=(30, 30), n_agents=2, vision_range=7, max_resources=10)
        
        fov_env = FieldOfViewEnhancedGridWorld(base_env, vision_range=7, exploration_reward_scale=1.0)
        
        obs, info = fov_env.reset(seed=42)
        
        assert obs.shape == (10, 7, 7)
        assert 'fov_system_initialized' in info
        
        for _ in range(10):
            action = fov_env.base_env.action_space.sample()
            obs, reward, terminated, truncated, info = fov_env.step(action)
            if terminated or truncated:
                break
        
        assert 'fov_exploration_reward' in info
        print(f"   ✅ Integration successful")
        
    @pytest.mark.skipif(not FOV_AVAILABLE, reason="Field of View System not available")
    def test_neural_network_compatibility(self):
        """Test that field of view observations work with neural networks"""
        print("\n🧠 Testing neural network compatibility...")
        
        vision_range = 7
        fov_system = FieldOfViewSystem(vision_range=vision_range)
        world_state = {
            'world_size': (40, 40), 'grid': np.zeros((40, 40)),
            'resources': [], 'agents': [{'id': 0, 'pos': (20, 20)}]
        }
        
        observation, _ = fov_system.update_agent_vision(0, (20, 20), world_state)
        
        obs_tensor = torch.from_numpy(observation).unsqueeze(0).float()
        assert obs_tensor.shape == (1, 10, vision_range, vision_range)
        
        obs_tensor.requires_grad_(True)
        loss = torch.sum(obs_tensor ** 2)
        loss.backward()
        assert obs_tensor.grad is not None
        
        print(f"   ✅ Neural network compatibility verified")


def run_day_3_4_validation():
    """Run complete Day 3-4 validation suite"""
    print("\n🚀 NEXUS Day 3-4 Validation: Field of View System")
    print("=" * 65)
    
    if not FOV_AVAILABLE:
        print("❌ Field of View System not available - skipping tests")
        return False
    
    test_suite = TestFieldOfViewSystem()
    
    try:
        # Run all Day 3-4 tests
        test_suite.test_agent_memory_initialization()
        test_suite.test_field_of_view_observation_generation()
        test_suite.test_fog_of_war_mechanics()
        test_suite.test_exploration_incentives()
        test_suite.test_memory_decay_mechanics()
        test_suite.test_performance_benchmarks()
        test_suite.test_integration_with_enhanced_environment()
        test_suite.test_neural_network_compatibility()
        
        print("=" * 65)
        print("🎉 ALL DAY 3-4 TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Day 3-4 validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_day_3_4_validation()
    exit(0 if success else 1)