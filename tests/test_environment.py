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
        env = GridWorld(size=(15, 15), n_agents=1, seed=42)
        assert env.size == (15, 15)
        assert env.n_agents == 1
        assert len(env.agents) == 1
        print("✅ Environment creation test passed")
    
    def test_deterministic_seeding(self):
        """Test deterministic behavior with same seed"""
        env1 = GridWorld(seed=42)
        env2 = GridWorld(seed=42)
        
        obs1 = env1.reset()
        obs2 = env2.reset()
        
        np.testing.assert_array_equal(obs1, obs2)
        assert env1.agents[0]['pos'] == env2.agents[0]['pos']
        print("✅ Deterministic seeding test passed")
    
    def test_observation_structure(self):
        """Test observation tensor structure"""
        env = GridWorld(n_agents=2, seed=42)
        obs = env.reset()
        
        # Check shape
        assert obs.shape == (5, 15, 15)
        assert obs.dtype == np.float32
        
        # Check channels have appropriate content
        assert obs[0].sum() > 0  # Empty spaces
        assert obs[2].sum() >= 2  # At least 2 agents visible
        
        print("✅ Observation structure test passed")
    
    def test_movement_mechanics(self):
        """Test agent movement in all directions"""
        env = GridWorld(n_agents=1, seed=42)
        obs = env.reset()
        
        initial_pos = env.agents[0]['pos']
        
        # Test each movement direction
        for action in range(8):
            env.reset(seed=42)  # Reset to same state
            obs, reward, done, info = env.step([action])
            
            # Agent should either move or stay in place (if blocked)
            current_pos = env.agents[0]['pos']
            distance = abs(current_pos[0] - initial_pos[0]) + abs(current_pos[1] - initial_pos[1])
            assert distance <= 2  # Maximum distance for diagonal move
        
        print("✅ Movement mechanics test passed")
    
    def test_resource_gathering(self):
        """Test resource gathering functionality"""
        env = GridWorld(n_agents=1, seed=42)
        obs = env.reset()
        
        # Find a resource and move agent there
        if len(env.resources) > 0:
            resource_pos = env.resources[0]['pos']
            resource_type = env.resources[0]['type']
            initial_count = len(env.resources)
            
            # Move agent to resource location
            env.agents[0]['pos'] = resource_pos
            env.grid[resource_pos] = 1  # Agent ID
            
            # Try to gather
            obs, reward, done, info = env.step([8])  # Gather action
            
            # Check if resource was gathered
            assert len(env.resources) <= initial_count
            # Check inventory increased
            assert env.agents[0]['inventory'][resource_type] > 0
            
        print("✅ Resource gathering test passed")
    
    def test_multi_agent_coordination(self):
        """Test multiple agents can act simultaneously"""
        env = GridWorld(n_agents=3, seed=42)
        obs = env.reset()
        
        # Test multi-agent step
        actions = [0, 1, 2]  # Different actions for each agent
        obs, rewards, done, info = env.step(actions)
        
        assert len(rewards) == 3
        assert all(isinstance(r, (int, float)) for r in rewards)
        
        print("✅ Multi-agent coordination test passed")
    
    def test_episode_termination(self):
        """Test episode termination conditions"""
        env = GridWorld(n_agents=1, max_steps=5, seed=42)
        obs = env.reset()
        
        # Run until termination
        step_count = 0
        done = False
        while not done and step_count < 10:
            obs, reward, done, info = env.step([0])
            step_count += 1
        
        # Should terminate within max_steps
        assert step_count <= env.max_steps
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
    test.test_multi_agent_coordination()
    test.test_episode_termination()
    
    print("=" * 50)
    print("🎉 ALL TESTS PASSED! Environment is ready for training.")

if __name__ == "__main__":
    run_all_tests()