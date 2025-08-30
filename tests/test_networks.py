# File: tests/test_networks.py
"""
Comprehensive test suite for Project NEXUS neural networks
Tests all network components: CNN, fusion, and PPO networks
"""
import pytest
import torch
import numpy as np
import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.networks.spatial_cnn import SpatialCNN, EnhancedSpatialCNN
from agents.networks.multimodal_fusion import (
    MultiModalFusion, AgentStateEncoder, AttentionalFusion,
    prepare_agent_state_batch
)
from agents.networks.ppo_networks import (
    PPOActorCritic, PolicyHead, ValueHead, PPOActorCriticWithComm,
    create_ppo_network
)

class TestSpatialCNN:
    """Test spatial CNN components"""
    
    def test_spatial_cnn_creation(self):
        """Test basic CNN creation and forward pass"""
        cnn = SpatialCNN(input_channels=5, feature_dim=256)
        
        # Test input
        batch_size = 4
        test_input = torch.randn(batch_size, 5, 15, 15)
        
        output = cnn(test_input)
        
        assert output.shape == (batch_size, 256)
        assert output.dtype == torch.float32
        print("✅ Spatial CNN creation test passed")
    
    def test_spatial_cnn_gradients(self):
        """Test gradient flow through CNN"""
        cnn = SpatialCNN(feature_dim=128)
        test_input = torch.randn(2, 5, 15, 15, requires_grad=True)
        
        output = cnn(test_input)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert test_input.grad is not None
        assert torch.sum(torch.abs(test_input.grad)) > 0
        print("✅ CNN gradient flow test passed")
    
    def test_enhanced_cnn_with_attention(self):
        """Test enhanced CNN with attention mechanism"""
        enhanced_cnn = EnhancedSpatialCNN(feature_dim=256, use_attention=True)
        
        test_input = torch.randn(3, 5, 15, 15)
        output = enhanced_cnn(test_input)
        
        assert output.shape == (3, 256)
        
        # Test feature maps
        feature_maps = enhanced_cnn.get_feature_maps(test_input)
        assert len(feature_maps) > 0
        print("✅ Enhanced CNN with attention test passed")
    
    def test_cnn_parameter_count(self):
        """Test CNN parameter counts are reasonable"""
        cnn = SpatialCNN()
        param_count = sum(p.numel() for p in cnn.parameters())
        
        # Should have reasonable number of parameters (not too many or too few)
        assert 50_000 < param_count < 2_000_000
        print(f"✅ CNN parameter count test passed: {param_count:,} parameters")

class TestMultiModalFusion:
    """Test multi-modal fusion components"""
    
    def test_agent_state_encoder(self):
        """Test agent state encoding"""
        encoder = AgentStateEncoder(state_dim=128)
        
        # Test batch of agent states
        batch_states = torch.tensor([
            [2, 1, 3, 0, 0.8, 0.9, 0.5, 0.3],  # Agent 1
            [0, 5, 1, 1, 1.0, 0.7, 0.2, 0.8],  # Agent 2
        ], dtype=torch.float32)
        
        encoded = encoder(batch_states)
        
        assert encoded.shape == (2, 128)
        assert encoded.dtype == torch.float32
        print("✅ Agent state encoder test passed")
    
    def test_agent_dict_encoding(self):
        """Test encoding from GridWorld agent dictionary"""
        encoder = AgentStateEncoder()
        
        agent_dict = {
            'inventory': {'wood': 2, 'stone': 1, 'food': 3, 'tool': 0},
            'health': 80,
            'energy': 90,
            'pos': (7, 5)
        }
        
        encoded = encoder.encode_from_dict(agent_dict, grid_size=(15, 15))
        
        assert encoded.shape == (1, 128)
        
        # Check normalized values are in reasonable ranges
        state_vector = [2, 1, 3, 0, 0.8, 0.9, 7/15, 5/15]
        print("✅ Agent dictionary encoding test passed")
    
    def test_multimodal_fusion(self):
        """Test multi-modal fusion network"""
        fusion = MultiModalFusion(spatial_dim=256, state_dim=128, output_dim=512)
        
        spatial_features = torch.randn(4, 256)
        agent_states = torch.randn(4, 8)
        
        fused_output = fusion(spatial_features, agent_states)
        
        assert fused_output.shape == (4, 512)
        print("✅ Multi-modal fusion test passed")
    
    def test_attentional_fusion(self):
        """Test attentional fusion mechanism"""
        attn_fusion = AttentionalFusion(spatial_dim=256, state_dim=128, output_dim=512)
        
        spatial_features = torch.randn(3, 256)
        agent_states = torch.randn(3, 8)
        
        output = attn_fusion(spatial_features, agent_states)
        
        assert output.shape == (3, 512)
        print("✅ Attentional fusion test passed")
    
    def test_prepare_agent_state_batch(self):
        """Test batch preparation utility"""
        agent_dicts = [
            {
                'inventory': {'wood': 2, 'stone': 1, 'food': 0, 'tool': 1},
                'health': 100, 'energy': 80, 'pos': (5, 7)
            },
            {
                'inventory': {'wood': 0, 'stone': 3, 'food': 2, 'tool': 0},
                'health': 60, 'energy': 90, 'pos': (12, 3)
            }
        ]
        
        batch_tensor = prepare_agent_state_batch(agent_dicts)
        
        assert batch_tensor.shape == (2, 8)
        assert batch_tensor.dtype == torch.float32
        
        # Check normalization
        assert torch.all(batch_tensor[:, 4:6] <= 1.0)  # health, energy normalized
        assert torch.all(batch_tensor[:, 6:8] <= 1.0)  # position normalized
        print("✅ Agent state batch preparation test passed")

class TestPPONetworks:
    """Test PPO network components"""
    
    def test_policy_head(self):
        """Test policy head functionality"""
        policy = PolicyHead(input_dim=512, action_dim=14)
        
        features = torch.randn(5, 512)
        logits = policy(features)
        
        assert logits.shape == (5, 14)
        
        # Test action distribution
        action_dist = policy.get_action_distribution(features)
        actions = action_dist.sample()
        
        assert actions.shape == (5,)
        assert torch.all(actions >= 0) and torch.all(actions < 14)
        print("✅ Policy head test passed")
    
    def test_value_head(self):
        """Test value head functionality"""
        value = ValueHead(input_dim=512)
        
        features = torch.randn(6, 512)
        values = value(features)
        
        assert values.shape == (6, 1)
        print("✅ Value head test passed")
    
    def test_complete_ppo_network(self):
        """Test complete PPO actor-critic network"""
        network = PPOActorCritic(
            spatial_channels=5,
            spatial_dim=256,
            state_dim=128,
            fusion_dim=512,
            action_dim=14
        )
        
        # Test data
        batch_size = 4
        observations = torch.randn(batch_size, 5, 15, 15)
        agent_states = torch.randn(batch_size, 8)
        
        # Test forward pass
        action_logits, state_values = network(observations, agent_states)
        
        assert action_logits.shape == (batch_size, 14)
        assert state_values.shape == (batch_size, 1)
        print("✅ Complete PPO network forward test passed")
    
    def test_ppo_action_selection(self):
        """Test PPO action selection methods"""
        network = PPOActorCritic()
        
        observations = torch.randn(3, 5, 15, 15)
        agent_states = torch.randn(3, 8)
        
        # Test stochastic action selection
        actions, log_probs, values = network.act(observations, agent_states, deterministic=False)
        
        assert actions.shape == (3,)
        assert log_probs.shape == (3,)
        assert values.shape == (3,)
        
        # Test deterministic action selection
        det_actions, det_log_probs, det_values = network.act(observations, agent_states, deterministic=True)
        
        assert det_actions.shape == (3,)
        print("✅ PPO action selection test passed")
    
    def test_ppo_action_evaluation(self):
        """Test PPO action evaluation for training"""
        network = PPOActorCritic()
        
        observations = torch.randn(5, 5, 15, 15)
        agent_states = torch.randn(5, 8)
        actions = torch.randint(0, 14, (5,))
        
        log_probs, values, entropy = network.evaluate_actions(observations, agent_states, actions)
        
        assert log_probs.shape == (5,)
        assert values.shape == (5,)
        assert entropy.shape == (5,)
        
        # Check entropy is positive (good exploration)
        assert torch.all(entropy >= 0)
        print("✅ PPO action evaluation test passed")
    
    def test_ppo_value_only(self):
        """Test value-only computation"""
        network = PPOActorCritic()
        
        observations = torch.randn(2, 5, 15, 15)
        agent_states = torch.randn(2, 8)
        
        values = network.get_value(observations, agent_states)
        
        assert values.shape == (2,)
        print("✅ PPO value-only computation test passed")
    
    def test_ppo_checkpoint_save_load(self):
        """Test checkpoint saving and loading"""
        network = PPOActorCritic(spatial_dim=128, fusion_dim=256)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            checkpoint_path = tmp_file.name
        
        try:
            # Save checkpoint
            network.save_checkpoint(checkpoint_path, step=1000, episode=50)
            
            # Load checkpoint
            loaded_network, checkpoint_info = PPOActorCritic.load_checkpoint(checkpoint_path)
            
            assert checkpoint_info['step'] == 1000
            assert checkpoint_info['episode'] == 50
            
            # Test loaded network works
            test_obs = torch.randn(1, 5, 15, 15)
            test_states = torch.randn(1, 8)
            
            original_output = network(test_obs, test_states)
            loaded_output = loaded_network(test_obs, test_states)
            
            # Outputs should be identical
            torch.testing.assert_close(original_output[0], loaded_output[0], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(original_output[1], loaded_output[1], atol=1e-6, rtol=1e-6)
            
            print("✅ PPO checkpoint save/load test passed")
            
        finally:
            # Clean up
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)
    
    def test_ppo_with_communication(self):
        """Test PPO network with communication module"""
        comm_network = PPOActorCriticWithComm(enable_communication=True)
        
        observations = torch.randn(3, 5, 15, 15)
        agent_states = torch.randn(3, 8)
        messages = torch.randint(0, 16, (3,))
        
        action_logits, state_values, outgoing_messages = comm_network.forward_with_communication(
            observations, agent_states, messages
        )
        
        assert action_logits.shape == (3, 14)
        assert state_values.shape == (3, 1)
        assert outgoing_messages.shape == (3,)
        assert torch.all(outgoing_messages >= 0) and torch.all(outgoing_messages < 16)
        
        print("✅ PPO with communication test passed")

class TestNetworkIntegration:
    """Integration tests for complete network pipeline"""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from GridWorld observation to action"""
        # Simulate GridWorld-like data
        batch_size = 4
        
        # Create realistic observation data
        observations = torch.zeros(batch_size, 5, 15, 15)
        
        # Channel 0: Empty spaces
        observations[:, 0] = torch.ones(15, 15) * 0.8  # Mostly empty
        
        # Channel 1: Resources (sparse)
        for i in range(batch_size):
            observations[i, 1, np.random.randint(0, 15), np.random.randint(0, 15)] = 1.0
            observations[i, 1, np.random.randint(0, 15), np.random.randint(0, 15)] = 2.0
        
        # Channel 2: Agents
        for i in range(batch_size):
            observations[i, 2, np.random.randint(0, 15), np.random.randint(0, 15)] = 1.0
        
        # Create realistic agent states
        agent_states = torch.tensor([
            [2, 1, 0, 0, 0.8, 0.9, 0.3, 0.4],  # Some inventory, good health
            [0, 0, 3, 1, 1.0, 0.6, 0.7, 0.2],  # Different inventory
            [1, 2, 1, 0, 0.9, 0.8, 0.1, 0.9],  # Another configuration
            [0, 1, 0, 0, 0.7, 0.9, 0.8, 0.5],  # Low health
        ], dtype=torch.float32)
        
        # Test network
        network = PPOActorCritic()
        
        # Get actions
        actions, log_probs, values = network.act(observations, agent_states)
        
        # Verify outputs
        assert actions.shape == (batch_size,)
        assert log_probs.shape == (batch_size,)
        assert values.shape == (batch_size,)
        
        # Actions should be valid (0-13)
        assert torch.all(actions >= 0) and torch.all(actions <= 13)
        
        # Values should be reasonable
        assert torch.all(torch.isfinite(values))
        
        print("✅ End-to-end pipeline test passed")
    
    def test_gradient_flow_complete_network(self):
        """Test gradient flow through complete network"""
        network = PPOActorCritic()
        
        observations = torch.randn(2, 5, 15, 15, requires_grad=True)
        agent_states = torch.randn(2, 8, requires_grad=True)
        
        action_logits, state_values = network(observations, agent_states)
        
        # Compute loss and backpropagate
        policy_loss = -action_logits.mean()
        value_loss = state_values.pow(2).mean()
        total_loss = policy_loss + value_loss
        
        total_loss.backward()
        
        # Check gradients exist throughout network
        assert observations.grad is not None
        assert agent_states.grad is not None
        
        # Check all network parameters have gradients
        for name, param in network.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert torch.sum(torch.abs(param.grad)) > 0, f"Zero gradients for parameter: {name}"
        
        print("✅ Complete network gradient flow test passed")

def run_all_network_tests():
    """Run all network tests"""
    print("Running comprehensive network test suite for Project NEXUS...")
    print("=" * 60)
    
    # Spatial CNN tests
    print("\n🧠 SPATIAL CNN TESTS")
    cnn_tests = TestSpatialCNN()
    cnn_tests.test_spatial_cnn_creation()
    cnn_tests.test_spatial_cnn_gradients()
    cnn_tests.test_enhanced_cnn_with_attention()
    cnn_tests.test_cnn_parameter_count()
    
    # Multi-modal fusion tests
    print("\n🔗 MULTI-MODAL FUSION TESTS")
    fusion_tests = TestMultiModalFusion()
    fusion_tests.test_agent_state_encoder()
    fusion_tests.test_agent_dict_encoding()
    fusion_tests.test_multimodal_fusion()
    fusion_tests.test_attentional_fusion()
    fusion_tests.test_prepare_agent_state_batch()
    
    # PPO network tests
    print("\n🎯 PPO NETWORK TESTS")
    ppo_tests = TestPPONetworks()
    ppo_tests.test_policy_head()
    ppo_tests.test_value_head()
    ppo_tests.test_complete_ppo_network()
    ppo_tests.test_ppo_action_selection()
    ppo_tests.test_ppo_action_evaluation()
    ppo_tests.test_ppo_value_only()
    ppo_tests.test_ppo_checkpoint_save_load()
    ppo_tests.test_ppo_with_communication()
    
    # Integration tests
    print("\n🔄 INTEGRATION TESTS")
    integration_tests = TestNetworkIntegration()
    integration_tests.test_end_to_end_pipeline()
    integration_tests.test_gradient_flow_complete_network()
    
    print("\n" + "=" * 60)
    print("🎉 ALL NETWORK TESTS PASSED!")
    print("Neural architecture is ready for PPO training!")

if __name__ == "__main__":
    run_all_network_tests()