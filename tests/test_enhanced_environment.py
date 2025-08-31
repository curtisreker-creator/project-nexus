# File: tests/test_enhanced_environment.py
"""
Comprehensive test suite for Enhanced GridWorld - Day 1-2 Implementation
Tests map size expansion, resource system enhancements, and backward compatibility
"""

import pytest
import numpy as np
import torch
from typing import Tuple, Dict, Any
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from environment.enhanced_grid_world import EnhancedGridWorld, GridWorld
    ENHANCED_GRID_AVAILABLE = True
except ImportError:
    ENHANCED_GRID_AVAILABLE = False
    print("⚠️ Enhanced GridWorld not available - tests will be skipped")


class TestEnhancedGridWorld:
    """Test suite for enhanced environment features"""
    
    @pytest.mark.skipif(not ENHANCED_GRID_AVAILABLE, reason="Enhanced GridWorld not available")
    def test_configurable_map_sizes(self):
        """Test environment creation with different map sizes"""
        print("\n🧪 Testing configurable map sizes...")
        
        test_sizes = [(30, 30), (50, 50), (75, 75), (25, 40)]  # Including non-square
        
        for size in test_sizes:
            env = EnhancedGridWorld(size=size, n_agents=2)
            obs, info = env.reset(seed=42)
            
            # Verify size configuration
            assert env.size == size, f"Expected size {size}, got {env.size}"
            assert env.grid.shape == size, f"Grid shape mismatch: expected {size}, got {env.grid.shape}"
            
            # Verify observation space scales correctly
            if env.vision_range > 0:
                expected_obs_shape = (10, env.vision_range, env.vision_range)
            else:
                expected_obs_shape = (10, size[0], size[1])
            
            assert obs.shape == expected_obs_shape, f"Observation shape mismatch: expected {expected_obs_shape}, got {obs.shape}"
            
            # Verify agents can be placed
            assert len(env.agents) == 2, f"Expected 2 agents, got {len(env.agents)}"
            for agent in env.agents:
                x, y = agent['pos']
                assert 0 <= x < size[0] and 0 <= y < size[1], f"Agent position {(x,y)} outside bounds {size}"
            
            print(f"   ✅ Size {size}: Grid {env.grid.shape}, Obs {obs.shape}, Agents {len(env.agents)}")
    
    @pytest.mark.skipif(not ENHANCED_GRID_AVAILABLE, reason="Enhanced GridWorld not available")
    def test_memory_scaling_benchmarks(self):
        """Test memory usage across different environment sizes"""
        print("\n📊 Testing memory scaling benchmarks...")
        
        size_configs = [
            {'size': (15, 15), 'name': 'Original'},
            {'size': (30, 30), 'name': 'Small Enhanced'},
            {'size': (50, 50), 'name': 'Medium Enhanced'},
            {'size': (75, 75), 'name': 'Large Enhanced'}
        ]
        
        memory_results = []
        
        for config in size_configs:
            env = EnhancedGridWorld(size=config['size'], n_agents=4, max_resources=20)
            obs, info = env.reset(seed=42)
            
            size_info = env.get_size_info()
            memory_results.append({
                'name': config['name'],
                'size': config['size'],
                'total_cells': size_info['total_cells'],
                'memory_mb': size_info['memory_estimate_mb'],
                'complexity_score': size_info['complexity_score']
            })
            
            print(f"   {config['name']} ({config['size']}): {size_info['memory_estimate_mb']:.1f} MB, " +
                  f"Complexity: {size_info['complexity_score']:.2f}x")
        
        # Verify memory scaling is reasonable (should be roughly quadratic)
        small_memory = memory_results[0]['memory_mb']  
        large_memory = memory_results[-1]['memory_mb']
        
        # 75x75 should use more memory than 15x15, but not excessive
        scaling_factor = large_memory / small_memory
        assert 10 <= scaling_factor <= 100, f"Memory scaling factor {scaling_factor:.1f}x seems unreasonable"
        
        return memory_results
    
    @pytest.mark.skipif(not ENHANCED_GRID_AVAILABLE, reason="Enhanced GridWorld not available")
    def test_multi_resource_system(self):
        """Test enhanced multi-resource system with unit quantities"""
        print("\n🔄 Testing multi-resource system...")
        
        env = EnhancedGridWorld(size=(40, 40), n_agents=1, max_resources=15)
        obs, info = env.reset(seed=42)
        
        # Verify all resource types can spawn
        resource_types_found = set()
        total_units = 0
        
        for resource in env.resources:
            resource_types_found.add(resource['type'])
            assert resource['remaining_units'] > 0, f"Resource {resource['type']} has no units"
            assert resource['total_units'] >= resource['remaining_units'], "Remaining > total units"
            total_units += resource['remaining_units']
        
        print(f"   Resource types found: {sorted(resource_types_found)}")
        print(f"   Total resource units on map: {total_units}")
        
        # Test resource gathering with unit depletion
        if env.resources:
            agent = env.agents[0]
            target_resource = env.resources[0]
            
            # Move agent to resource
            agent['pos'] = target_resource['pos']
            original_units = target_resource['remaining_units']
            original_count = len(env.resources)
            
            # Gather resource
            obs, reward, terminated, truncated, info = env.step(8)  # Gather action
            
            # Verify unit depletion
            if original_units > 1:
                assert len(env.resources) == original_count, "Resource removed prematurely"
                assert target_resource['remaining_units'] == original_units - 1, "Unit not decremented"
            else:
                assert len(env.resources) == original_count - 1, "Resource not removed when depleted"
            
            # Verify inventory update
            assert sum(agent['inventory'].values()) > 0, "Nothing added to inventory"
            
            print(f"   ✅ Resource gathering: {original_units} → {target_resource['remaining_units'] if target_resource in env.resources else 0} units")
    
    @pytest.mark.skipif(not ENHANCED_GRID_AVAILABLE, reason="Enhanced GridWorld not available")
    def test_inventory_limitations(self):
        """Test inventory limit enforcement"""
        print("\n🎒 Testing inventory limitations...")
        
        env = EnhancedGridWorld(size=(20, 20), n_agents=1, inventory_limit=5)
        obs, info = env.reset(seed=42)
        
        agent = env.agents[0]
        
        # Fill inventory to limit
        for resource_type in ['wood', 'stone', 'coal']:
            agent['inventory'][resource_type] = 2
        agent['inventory_total'] = 6  # Over limit
        
        # Try to gather more - should be penalized
        if env.resources:
            agent['pos'] = env.resources[0]['pos']
            obs, reward, terminated, truncated, info = env.step(8)  # Gather action
            
            assert reward < 0, f"Expected negative reward for full inventory, got {reward}"
            print(f"   ✅ Inventory limit enforced: reward={reward:.3f} for over-limit gathering")
    
    @pytest.mark.skipif(not ENHANCED_GRID_AVAILABLE, reason="Enhanced GridWorld not available")
    def test_field_of_view_observations(self):
        """Test field of view observation system"""
        print("\n👁️ Testing field of view observations...")
        
        vision_range = 7
        env = EnhancedGridWorld(size=(50, 50), n_agents=1, vision_range=vision_range)
        obs, info = env.reset(seed=42)
        
        # Verify observation shape
        expected_shape = (10, vision_range, vision_range)
        assert obs.shape == expected_shape, f"Expected observation shape {expected_shape}, got {obs.shape}"
        
        # Verify fog of war (unknown areas should be -1)
        unknown_count = np.sum(obs == -1.0)
        total_values = np.prod(obs.shape)
        fog_percentage = unknown_count / total_values
        
        print(f"   ✅ Field of view: {obs.shape}, {fog_percentage:.1%} fog of war")
        
        # Test that observation contains some visible information
        visible_count = np.sum(obs != -1.0)
        assert visible_count > 0, "No visible information in observation"
    
    @pytest.mark.skipif(not ENHANCED_GRID_AVAILABLE, reason="Enhanced GridWorld not available")
    def test_backward_compatibility(self):
        """Test that original GridWorld interface still works"""
        print("\n🔄 Testing backward compatibility...")
        
        # Create environment using old interface
        old_env = GridWorld(n_agents=2, max_resources=8)
        obs, info = old_env.reset(seed=42)
        
        # Verify original behavior is preserved
        assert old_env.size == (15, 15), f"Expected original size (15,15), got {old_env.size}"
        assert old_env.vision_range == 0, "Expected full observability for backward compatibility"
        assert old_env.inventory_limit == 999, "Expected unlimited inventory for backward compatibility"
        
        # Test that old code patterns still work
        action = old_env.action_space.sample()
        obs, reward, terminated, truncated, info = old_env.step(action)
        
        print(f"   ✅ Backward compatibility verified: size {old_env.size}, unlimited inventory")
    
    @pytest.mark.skipif(not ENHANCED_GRID_AVAILABLE, reason="Enhanced GridWorld not available")
    def test_performance_benchmarks(self):
        """Benchmark performance across different environment sizes"""
        print("\n⚡ Testing performance benchmarks...")
        
        import time
        
        benchmark_configs = [
            {'size': (15, 15), 'steps': 100},
            {'size': (30, 30), 'steps': 100},
            {'size': (50, 50), 'steps': 50},
            {'size': (75, 75), 'steps': 25}
        ]
        
        for config in benchmark_configs:
            env = EnhancedGridWorld(size=config['size'], n_agents=2)
            obs, info = env.reset(seed=42)
            
            # Benchmark step performance
            start_time = time.time()
            for _ in range(config['steps']):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    obs, info = env.reset(seed=42)
            
            elapsed_time = time.time() - start_time
            steps_per_second = config['steps'] / elapsed_time
            
            print(f"   {config['size']}: {steps_per_second:.1f} steps/sec")
            
            # Performance should be reasonable even for large environments
            assert steps_per_second > 10, f"Performance too slow: {steps_per_second:.1f} steps/sec"
    
    @pytest.mark.skipif(not ENHANCED_GRID_AVAILABLE, reason="Enhanced GridWorld not available")
    def test_neural_network_compatibility(self):
        """Test that enhanced observations work with neural networks"""
        print("\n🧠 Testing neural network compatibility...")
        
        env = EnhancedGridWorld(size=(30, 30), n_agents=1, vision_range=7)
        obs, info = env.reset(seed=42)
        
        # Test tensor conversion
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
        expected_tensor_shape = (1, 10, 7, 7)  # (batch, channels, height, width)
        
        assert obs_tensor.shape == expected_tensor_shape, f"Expected {expected_tensor_shape}, got {obs_tensor.shape}"
        
        # Test that observations are in valid range
        assert torch.all(obs_tensor >= -1.0), "Observation values below -1.0"
        assert torch.all(obs_tensor <= 10.0), "Observation values above 10.0"
        
        # Test MPS compatibility (Apple Silicon)
        if torch.backends.mps.is_available():
            mps_tensor = obs_tensor.to('mps')
            assert mps_tensor.device.type == 'mps', "Failed to move to MPS device"
            print(f"   ✅ MPS compatibility verified")
        
        print(f"   ✅ Neural network compatibility: {obs_tensor.shape}, range [{obs_tensor.min():.1f}, {obs_tensor.max():.1f}]")


def run_day_1_2_validation():
    """Run complete Day 1-2 validation suite"""
    print("🚀 NEXUS Day 1-2 Validation: Map Size Expansion")
    print("=" * 60)
    
    if not ENHANCED_GRID_AVAILABLE:
        print("❌ Enhanced GridWorld not available - skipping tests")
        return False
    
    test_suite = TestEnhancedGridWorld()
    
    try:
        # Run all Day 1-2 tests
        test_suite.test_configurable_map_sizes()
        test_suite.test_memory_scaling_benchmarks()
        test_suite.test_multi_resource_system()
        test_suite.test_inventory_limitations()
        test_suite.test_field_of_view_observations()
        test_suite.test_backward_compatibility()
        test_suite.test_performance_benchmarks()
        test_suite.test_neural_network_compatibility()
        
        print("=" * 60)
        print("🎉 ALL DAY 1-2 TESTS PASSED!")
        print("✅ Map size expansion implementation validated")
        print("✅ Multi-resource system operational")
        print("✅ Field of view foundation established")
        print("✅ Backward compatibility maintained")
        print("✅ Performance benchmarks acceptable")
        print("✅ Neural network compatibility verified")
        print("🚀 Ready to proceed to Day 3-4: Field of View System!")
        
        return True
        
    except Exception as e:
        print(f"❌ Day 1-2 validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_day_1_2_validation()
    exit(0 if success else 1)