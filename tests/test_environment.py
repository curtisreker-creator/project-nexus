# File: tests/test_environment.py
import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from environment.grid_world import GridWorld

class TestGridWorld:
    
    def test_environment_creation(self):
        """Test basic environment creation"""
        env = GridWorld(size=(15, 15), n_agents=1)
        obs, info = env.reset(seed=42)
        
        assert env.size == (15, 15)
        assert env.n_agents == 1
        assert len(env.agents) == 1
        print("✅ Environment creation test passed")
    
    def test_deterministic_seeding(self):
        """Test deterministic behavior with same seed"""
        env1 = GridWorld()
        env2 = GridWorld()
        
        obs1, info1 = env1.reset(seed=42)
        obs2, info2 = env2.reset(seed=42)
        
        np.testing.assert_array_equal(obs1, obs2)
        assert env1.agents[0]['pos'] == env2.agents[0]['pos']
        print("✅ Deterministic seeding test passed")
    
    def test_observation_structure(self):
        """Test observation tensor structure"""
        env = GridWorld(n_agents=2)
        obs, info = env.reset(seed=42)
        
        assert obs.shape == (5, 15, 15)
        assert obs.dtype == np.float32
        
        assert obs[0].sum() > 0  # Empty spaces
        assert obs[2].sum() >= 2  # At least 2 agents visible
        
        print("✅ Observation structure test passed")
    
    def test_movement_mechanics(self):
        """Test agent movement in all directions"""
        env = GridWorld(n_agents=1)
        obs, info = env.reset(seed=42)
        
        initial_pos = env.agents[0]['pos']
        
        for action in range(8):
            env.reset(seed=42)
            obs, reward, terminated, truncated, info = env.step(action)
            
            current_pos = env.agents[0]['pos']
            distance = abs(current_pos[0] - initial_pos[0]) + abs(current_pos[1] - initial_pos[1])
            assert distance <= 2
        
        print("✅ Movement mechanics test passed")
    
    def test_resource_gathering(self):
        """Test resource gathering functionality"""
        env = GridWorld(n_agents=1)
        obs, info = env.reset(seed=42)
        
        if len(env.resources) > 0:
            resource_pos = env.resources[0]['pos']
            resource_type = env.resources[0]['type']
            initial_count = len(env.resources)
            
            env.agents[0]['pos'] = resource_pos
            env.grid[resource_pos] = 1
            
            obs, reward, terminated, truncated, info = env.step(8) # Gather action
            
            assert len(env.resources) < initial_count
            assert env.agents[0]['inventory'][resource_type] > 0
            assert reward > 0.0
            
        print("✅ Resource gathering test passed")
    
    def test_multi_agent_initialization(self):
        """Test multiple agents can be initialized"""
        # TODO: Refactor this for a true multi-agent (PettingZoo) env in Phase 2.
        env = GridWorld(n_agents=3)
        obs, info = env.reset(seed=42)

        assert env.n_agents == 3
        assert len(info['agents']) == 3

        initial_pos_agent0 = info['agents'][0]['pos']
        obs, reward, terminated, truncated, info = env.step(0) # Move agent 0
        new_pos_agent0 = info['agents'][0]['pos']
        
        assert initial_pos_agent0 != new_pos_agent0
        
        print("✅ Multi-agent initialization test passed")
    
    def test_episode_termination(self):
        """Test episode termination conditions"""
        env = GridWorld(n_agents=1, max_steps=5)
        obs, info = env.reset(seed=42)
        
        done = False
        step_count = 0
        truncated = False # Pre-initialize to fix 'possibly unbound' warning
        
        while not done and step_count < 10:
            obs, reward, terminated, truncated, info = env.step(0)
            done = terminated or truncated
            step_count += 1
        
        assert truncated == True
        assert step_count == env.max_steps
        print("✅ Episode termination test passed")

def run_all_tests():
    """Run all environment tests"""
    test = TestGridWorld()
    
    print("Running comprehensive environment test suite...")
    print("=" * 50)
    
    test.test_environment_creation()
    test.test_deterministic_seeding()
    test.test_observation_structure()
    test.test_movement_mechanics()
    test.test_resource_gathering()
    test.test_multi_agent_initialization()
    test.test_episode_termination()
    
    print("=" * 50)
    print("🎉 ALL TESTS PASSED! Environment is ready for training.")

if __name__ == "__main__":
    run_all_tests()