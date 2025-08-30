# File: agents/training/rollout_buffer.py
"""
Rollout Buffer for PPO Experience Collection
Efficient storage and batch sampling for reinforcement learning
"""
import numpy as np
import torch
from typing import Tuple, Optional, Union, Any
import logging

class RolloutBuffer:
    """
    Buffer for storing and managing rollout data for PPO training
    Supports multi-agent scenarios with efficient memory management
    """
    
    def __init__(self, 
                 buffer_size: int,
                 observation_shape: Tuple[int, ...],
                 agent_state_dim: int,
                 n_agents: int = 1,
                 device: Optional[torch.device] = None):
        """
        Initialize rollout buffer
        
        Args:
            buffer_size: Maximum number of transitions to store
            observation_shape: Shape of environment observations (e.g., (5, 15, 15))
            agent_state_dim: Dimension of agent state vector (e.g., 8)
            n_agents: Number of agents (for multi-agent support)
            device: Device for tensor operations
        """
        self.buffer_size = buffer_size
        self.observation_shape = observation_shape
        self.agent_state_dim = agent_state_dim
        self.n_agents = n_agents
        self.device = device or torch.device('cpu')
        
        # Initialize storage arrays
        self.observations = np.zeros((buffer_size,) + observation_shape, dtype=np.float32)
        self.agent_states = np.zeros((buffer_size, agent_state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        
        # GAE computation results (computed externally)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        # Buffer state
        self.ptr = 0
        self.size = 0
        self.full = False
        
        self.logger = logging.getLogger('RolloutBuffer')
        self.logger.info(f"RolloutBuffer initialized: size={buffer_size}, obs_shape={observation_shape}")
    
    def store(self, 
              observation: np.ndarray,
              agent_state: np.ndarray,
              action: int,
              reward: float,
              value: float,
              log_prob: float,
              done: bool):
        """
        Store a single transition in the buffer
        
        Args:
            observation: Environment observation
            agent_state: Agent state vector
            action: Action taken
            reward: Reward received
            value: Value estimate from critic
            log_prob: Log probability of action
            done: Episode termination flag
        """
        if self.ptr >= self.buffer_size:
            raise RuntimeError(f"Buffer overflow: ptr={self.ptr}, size={self.buffer_size}")
        
        # Store transition data
        self.observations[self.ptr] = observation
        self.agent_states[self.ptr] = agent_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        # Update buffer state
        self.ptr += 1
        self.size = min(self.size + 1, self.buffer_size)
        
        if self.ptr == self.buffer_size:
            self.full = True
    
    def get_batch(self, batch_indices: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, ...]:
        """
        Get a batch of transitions for training
        
        Args:
            batch_indices: Specific indices to sample (if None, returns all stored data)
            
        Returns:
            Tuple of tensors: (observations, agent_states, actions, old_log_probs, 
                              advantages, returns, values)
        """
        if not self.full and self.ptr == 0:
            raise RuntimeError("Buffer is empty")
        
        # Determine valid indices
        valid_size = self.ptr if not self.full else self.buffer_size
        
        if batch_indices is None:
            batch_indices = np.arange(valid_size)
        else:
            # Ensure indices are within valid range
            batch_indices = batch_indices[batch_indices < valid_size]
        
        # Convert to tensors and move to device
        observations = torch.from_numpy(self.observations[batch_indices]).to(self.device)
        agent_states = torch.from_numpy(self.agent_states[batch_indices]).to(self.device)
        actions = torch.from_numpy(self.actions[batch_indices]).to(self.device)
        old_log_probs = torch.from_numpy(self.log_probs[batch_indices]).to(self.device)
        advantages = torch.from_numpy(self.advantages[batch_indices]).to(self.device)
        returns = torch.from_numpy(self.returns[batch_indices]).to(self.device)
        values = torch.from_numpy(self.values[batch_indices]).to(self.device)
        
        return observations, agent_states, actions, old_log_probs, advantages, returns, values
    
    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random batch of transitions
        
        Args:
            batch_size: Size of batch to sample
            
        Returns:
            Tuple of tensors for training
        """
        valid_size = self.ptr if not self.full else self.buffer_size
        
        if batch_size > valid_size:
            batch_size = valid_size
            self.logger.warning(f"Requested batch size {batch_size} > buffer size {valid_size}")
        
        batch_indices = np.random.choice(valid_size, size=batch_size, replace=False)
        return self.get_batch(batch_indices)
    
    def get_all_data(self) -> Tuple[torch.Tensor, ...]:
        """Get all stored data as tensors"""
        return self.get_batch()
    
    def clear(self):
        """Clear the buffer and reset to initial state"""
        self.ptr = 0
        self.size = 0
        self.full = False
        
        # Optional: Zero out arrays for memory efficiency
        # This is not strictly necessary but can help with debugging
        self.observations.fill(0)
        self.agent_states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.values.fill(0)
        self.log_probs.fill(0)
        self.dones.fill(False)
        self.advantages.fill(0)
        self.returns.fill(0)
        
        self.logger.debug("Buffer cleared")
    
    def compute_statistics(self) -> dict:
        """
        Compute statistics for stored data
        
        Returns:
            Dictionary with buffer statistics
        """
        valid_size = self.ptr if not self.full else self.buffer_size
        
        if valid_size == 0:
            return {'empty': True}
        
        valid_rewards = self.rewards[:valid_size]
        valid_values = self.values[:valid_size]
        valid_advantages = self.advantages[:valid_size]
        valid_dones = self.dones[:valid_size]
        
        stats = {
            'size': valid_size,
            'mean_reward': float(np.mean(valid_rewards)),
            'std_reward': float(np.std(valid_rewards)),
            'mean_value': float(np.mean(valid_values)),
            'std_value': float(np.std(valid_values)),
            'mean_advantage': float(np.mean(valid_advantages)),
            'std_advantage': float(np.std(valid_advantages)),
            'episode_endings': int(np.sum(valid_dones)),
            'max_reward': float(np.max(valid_rewards)),
            'min_reward': float(np.min(valid_rewards)),
        }
        
        return stats
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return self.full
    
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return self.size == 0
    
    def __len__(self) -> int:
        """Return current buffer size"""
        return self.size
    
    def __repr__(self) -> str:
        return (f"RolloutBuffer(size={self.size}/{self.buffer_size}, "
                f"obs_shape={self.observation_shape}, "
                f"agent_state_dim={self.agent_state_dim}, "
                f"device={self.device})")


class MultiAgentRolloutBuffer:
    """
    Extended rollout buffer for true multi-agent scenarios
    Manages separate buffers for each agent while enabling shared training
    """
    
    def __init__(self,
                 buffer_size: int,
                 observation_shape: Tuple[int, ...],
                 agent_state_dim: int,
                 n_agents: int,
                 device: Optional[torch.device] = None):
        """
        Initialize multi-agent rollout buffer
        
        Args:
            buffer_size: Buffer size per agent
            observation_shape: Shape of observations
            agent_state_dim: Agent state dimension
            n_agents: Number of agents
            device: Device for tensors
        """
        self.n_agents = n_agents
        self.buffer_size = buffer_size
        self.device = device or torch.device('cpu')
        
        # Create individual buffers for each agent
        self.agent_buffers = [
            RolloutBuffer(
                buffer_size=buffer_size,
                observation_shape=observation_shape,
                agent_state_dim=agent_state_dim,
                device=device
            )
            for _ in range(n_agents)
        ]
        
        self.logger = logging.getLogger('MultiAgentRolloutBuffer')
        self.logger.info(f"MultiAgent buffer initialized: {n_agents} agents, {buffer_size} size each")
    
    def store(self, agent_id: int, **kwargs):
        """Store transition for specific agent"""
        if agent_id >= self.n_agents:
            raise ValueError(f"Agent ID {agent_id} >= number of agents {self.n_agents}")
        
        self.agent_buffers[agent_id].store(**kwargs)
    
    def store_all_agents(self, transitions: list):
        """
        Store transitions for all agents simultaneously
        
        Args:
            transitions: List of transition dictionaries, one per agent
        """
        if len(transitions) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} transitions, got {len(transitions)}")
        
        for agent_id, transition in enumerate(transitions):
            self.agent_buffers[agent_id].store(**transition)
    
    def get_combined_batch(self, batch_indices: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, ...]:
        """
        Get combined batch from all agents
        
        Args:
            batch_indices: Indices to sample (applied to each agent buffer)
            
        Returns:
            Combined tensors from all agents
        """
        agent_batches = []
        
        for agent_buffer in self.agent_buffers:
            if not agent_buffer.is_empty():
                batch = agent_buffer.get_batch(batch_indices)
                agent_batches.append(batch)
        
        if not agent_batches:
            raise RuntimeError("All agent buffers are empty")
        
        # Concatenate batches from all agents
        combined_batch = []
        for i in range(len(agent_batches[0])):  # Number of tensor types
            tensors = [batch[i] for batch in agent_batches]
            combined_tensor = torch.cat(tensors, dim=0)
            combined_batch.append(combined_tensor)
        
        return tuple(combined_batch)
    
    def clear_all(self):
        """Clear all agent buffers"""
        for agent_buffer in self.agent_buffers:
            agent_buffer.clear()
    
    def get_statistics(self) -> dict:
        """Get statistics for all agents"""
        stats = {}
        
        for agent_id, agent_buffer in enumerate(self.agent_buffers):
            agent_stats = agent_buffer.compute_statistics()
            stats[f'agent_{agent_id}'] = agent_stats
        
        # Compute combined statistics
        if not all(stats[f'agent_{i}'].get('empty', False) for i in range(self.n_agents)):
            all_rewards = []
            all_values = []
            total_size = 0
            
            for agent_id in range(self.n_agents):
                agent_stats = stats[f'agent_{agent_id}']
                if not agent_stats.get('empty', False):
                    buffer = self.agent_buffers[agent_id]
                    valid_size = buffer.ptr if not buffer.full else buffer.buffer_size
                    all_rewards.extend(buffer.rewards[:valid_size])
                    all_values.extend(buffer.values[:valid_size])
                    total_size += valid_size
            
            if all_rewards:
                stats['combined'] = {
                    'total_size': total_size,
                    'mean_reward': float(np.mean(all_rewards)),
                    'std_reward': float(np.std(all_rewards)),
                    'mean_value': float(np.mean(all_values)),
                    'std_value': float(np.std(all_values)),
                }
        
        return stats
    
    def __len__(self) -> int:
        """Return total size across all agents"""
        return sum(len(buffer) for buffer in self.agent_buffers)


# Utility functions for buffer management
def create_buffer_from_config(config: dict, 
                            observation_shape: Tuple[int, ...],
                            agent_state_dim: int,
                            device: Optional[torch.device] = None) -> Union[RolloutBuffer, MultiAgentRolloutBuffer]:
    """
    Create appropriate buffer from configuration
    
    Args:
        config: Configuration dictionary
        observation_shape: Environment observation shape
        agent_state_dim: Agent state dimension
        device: Device for tensors
        
    Returns:
        Configured rollout buffer
    """
    training_config = config.get('training', {})
    env_config = config.get('environment', {})
    
    buffer_size = training_config.get('buffer_size', 2048)
    n_agents = env_config.get('max_agents', 1)
    
    if n_agents > 1:
        return MultiAgentRolloutBuffer(
            buffer_size=buffer_size,
            observation_shape=observation_shape,
            agent_state_dim=agent_state_dim,
            n_agents=n_agents,
            device=device
        )
    else:
        return RolloutBuffer(
            buffer_size=buffer_size,
            observation_shape=observation_shape,
            agent_state_dim=agent_state_dim,
            device=device
        )


if __name__ == "__main__":
    # Test rollout buffer functionality
    print("Testing Rollout Buffer...")
    
    try:
        # Test single-agent buffer
        buffer = RolloutBuffer(
            buffer_size=10,
            observation_shape=(5, 15, 15),
            agent_state_dim=8,
            n_agents=1
        )
        
        # Store some test data
        for i in range(5):
            buffer.store(
                observation=np.random.randn(5, 15, 15),
                agent_state=np.random.randn(8),
                action=i % 14,
                reward=np.random.randn(),
                value=np.random.randn(),
                log_prob=np.random.randn(),
                done=False
            )
        
        print(f"✅ Buffer size: {len(buffer)}")
        
        # Test statistics
        stats = buffer.compute_statistics()
        print(f"✅ Buffer stats: {stats['size']} transitions")
        
        # Test batch sampling (need to set advantages/returns first)
        buffer.advantages = np.random.randn(buffer.buffer_size)
        buffer.returns = np.random.randn(buffer.buffer_size)
        
        batch = buffer.get_batch()
        print(f"✅ Batch shapes: obs={batch[0].shape}, actions={batch[2].shape}")
        
        # Test multi-agent buffer
        multi_buffer = MultiAgentRolloutBuffer(
            buffer_size=5,
            observation_shape=(5, 15, 15),
            agent_state_dim=8,
            n_agents=2
        )
        
        print(f"✅ Multi-agent buffer created with {multi_buffer.n_agents} agents")
        
        print("✅ All rollout buffer tests passed!")
        
    except Exception as e:
        print(f"❌ Buffer test failed: {e}")
        import traceback
        traceback.print_exc()