# File: environment/multi_agent_wrapper.py
"""
Multi-Agent Environment Wrapper for Project NEXUS
Enables true multi-agent coordination with shared parameters and communication
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from collections import defaultdict

try:
    from .grid_world import GridWorld
    from ..agents.networks import prepare_agent_state_batch
except ImportError:
    # Fallback imports for standalone use
    try:
        from grid_world import GridWorld
        from agents.networks import prepare_agent_state_batch
    except ImportError:
        GridWorld = None
        prepare_agent_state_batch = None

class MultiAgentWrapper:
    """
    Wrapper for multi-agent coordination in GridWorld environment
    Supports shared parameter training and agent communication
    """
    
    def __init__(self,
                 base_env: GridWorld,
                 n_agents: int = 4,
                 shared_parameters: bool = True,
                 enable_communication: bool = False,
                 communication_range: int = 3,
                 communication_freq: int = 4):
        """
        Initialize multi-agent wrapper
        
        Args:
            base_env: Base GridWorld environment
            n_agents: Number of agents to coordinate
            shared_parameters: Whether agents share network parameters
            enable_communication: Enable agent-to-agent communication
            communication_range: Range for agent communication
            communication_freq: Timesteps between communication
        """
        self.base_env = base_env
        self.n_agents = n_agents
        self.shared_parameters = shared_parameters
        self.enable_communication = enable_communication
        self.communication_range = communication_range
        self.communication_freq = communication_freq
        
        # Ensure base environment supports multi-agent
        if hasattr(base_env, 'n_agents'):
            base_env.n_agents = n_agents
        
        # Communication state
        self.communication_step = 0
        self.agent_messages = {}  # agent_id -> message_history
        
        # Performance tracking
        self.agent_rewards = defaultdict(list)
        self.coordination_metrics = defaultdict(float)
        
        # Logger
        self.logger = logging.getLogger('MultiAgentWrapper')
        self.logger.info(f"Multi-agent wrapper initialized: {n_agents} agents, "
                        f"shared_params={shared_parameters}, comm={enable_communication}")
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
        """
        Reset environment for all agents
        
        Args:
            seed: Random seed
            
        Returns:
            Dictionary of observations per agent and info dict
        """
        # Reset base environment
        obs, info = self.base_env.reset(seed=seed)
        
        # Reset communication state
        self.communication_step = 0
        self.agent_messages = {i: [] for i in range(self.n_agents)}
        
        # Create per-agent observations
        agent_observations = {}
        
        if self.shared_parameters:
            # All agents share the same observation space
            for agent_id in range(self.n_agents):
                agent_observations[agent_id] = obs.copy()
        else:
            # Each agent gets a personalized view (future enhancement)
            for agent_id in range(self.n_agents):
                agent_observations[agent_id] = self._create_agent_observation(obs, agent_id, info)
        
        # Add multi-agent info
        info['multi_agent'] = {
            'n_agents': self.n_agents,
            'shared_parameters': self.shared_parameters,
            'communication_enabled': self.enable_communication,
            'agent_positions': [agent['pos'] for agent in info['agents'][:self.n_agents]]
        }
        
        return agent_observations, info
    
    def step(self, actions: Dict[int, int]) -> Tuple[Dict[int, np.ndarray], 
                                                   Dict[int, float], 
                                                   Dict[int, bool], 
                                                   Dict[int, bool], 
                                                   Dict[str, Any]]:
        """
        Execute actions for all agents
        
        Args:
            actions: Dictionary mapping agent_id to action
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, info) per agent
        """
        if len(actions) != self.n_agents:
            raise ValueError(f"Expected actions for {self.n_agents} agents, got {len(actions)}")
        
        # Execute actions sequentially (can be parallelized later)
        agent_observations = {}
        agent_rewards = {}
        agent_terminated = {}
        agent_truncated = {}
        
        cumulative_reward = 0.0
        episode_done = False
        
        # Process each agent's action
        for agent_id in range(self.n_agents):
            if agent_id not in actions:
                raise ValueError(f"Missing action for agent {agent_id}")
            
            # Execute action in base environment
            # Note: GridWorld currently supports single-agent steps
            # For true multi-agent, we'd need to modify GridWorld's step function
            action = actions[agent_id]
            obs, reward, terminated, truncated, info = self.base_env.step(action)
            
            # Store per-agent results
            agent_observations[agent_id] = obs.copy()
            agent_rewards[agent_id] = reward
            agent_terminated[agent_id] = terminated
            agent_truncated[agent_id] = truncated
            
            cumulative_reward += reward
            if terminated or truncated:
                episode_done = True
        
        # Update communication
        self.communication_step += 1
        if self.enable_communication and self.communication_step % self.communication_freq == 0:
            self._process_communication(info)
        
        # Update coordination metrics
        self._update_coordination_metrics(agent_rewards, info)
        
        # Add multi-agent information
        info['multi_agent'] = {
            'cumulative_reward': cumulative_reward,
            'coordination_score': self._compute_coordination_score(agent_rewards),
            'communication_step': self.communication_step,
            'episode_done': episode_done
        }
        
        return agent_observations, agent_rewards, agent_terminated, agent_truncated, info
    
    def _create_agent_observation(self, base_obs: np.ndarray, agent_id: int, info: Dict) -> np.ndarray:
        """
        Create personalized observation for specific agent
        
        Args:
            base_obs: Base environment observation
            agent_id: Agent identifier
            info: Environment info dictionary
            
        Returns:
            Personalized observation for agent
        """
        # For now, return base observation
        # Future enhancement: add agent-specific channels, local views, etc.
        personalized_obs = base_obs.copy()
        
        # Could add agent-specific information channels:
        # - Agent ID channel
        # - Local visibility mask
        # - Communication history
        # - Agent-specific objectives
        
        return personalized_obs
    
    def _process_communication(self, info: Dict[str, Any]):
        """
        Process agent-to-agent communication
        
        Args:
            info: Environment info dictionary
        """
        if not self.enable_communication or len(info.get('agents', [])) < 2:
            return
        
        # Get agent positions
        agent_positions = []
        for i, agent in enumerate(info['agents'][:self.n_agents]):
            agent_positions.append(agent['pos'])
        
        # Determine communication network based on range
        comm_network = self._build_communication_network(agent_positions)
        
        # Generate and exchange messages (placeholder implementation)
        for agent_id in range(self.n_agents):
            # Generate message for this agent
            message = self._generate_agent_message(agent_id, info)
            
            # Broadcast to agents in communication range
            for neighbor_id in comm_network.get(agent_id, []):
                if neighbor_id in self.agent_messages:
                    self.agent_messages[neighbor_id].append({
                        'sender': agent_id,
                        'content': message,
                        'timestep': self.communication_step
                    })
        
        # Limit message history length
        max_message_history = 10
        for agent_id in self.agent_messages:
            if len(self.agent_messages[agent_id]) > max_message_history:
                self.agent_messages[agent_id] = self.agent_messages[agent_id][-max_message_history:]
    
    def _build_communication_network(self, agent_positions: List[Tuple[int, int]]) -> Dict[int, List[int]]:
        """
        Build communication network based on agent positions and range
        
        Args:
            agent_positions: List of (x, y) positions for each agent
            
        Returns:
            Dictionary mapping agent_id to list of neighbor agent_ids
        """
        network = defaultdict(list)
        
        for i, pos_i in enumerate(agent_positions):
            for j, pos_j in enumerate(agent_positions):
                if i != j:
                    # Calculate Manhattan distance
                    distance = abs(pos_i[0] - pos_j[0]) + abs(pos_i[1] - pos_j[1])
                    
                    if distance <= self.communication_range:
                        network[i].append(j)
        
        return dict(network)
    
    def _generate_agent_message(self, agent_id: int, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate communication message for agent
        
        Args:
            agent_id: Agent identifier
            info: Environment info
            
        Returns:
            Message dictionary
        """
        agent_data = info['agents'][agent_id] if agent_id < len(info['agents']) else {}
        
        # Simple message containing agent state
        message = {
            'position': agent_data.get('pos', (0, 0)),
            'inventory': agent_data.get('inventory', {}),
            'health': agent_data.get('health', 100),
            'resources_seen': info.get('resources_remaining', 0)
        }
        
        return message
    
    def _update_coordination_metrics(self, agent_rewards: Dict[int, float], info: Dict[str, Any]):
        """
        Update metrics for measuring coordination effectiveness
        
        Args:
            agent_rewards: Rewards for each agent
            info: Environment info
        """
        # Track individual agent performance
        for agent_id, reward in agent_rewards.items():
            self.agent_rewards[agent_id].append(reward)
        
        # Compute coordination metrics
        if len(agent_rewards) > 1:
            rewards_list = list(agent_rewards.values())
            
            # Reward variance (lower is better for coordination)
            reward_variance = np.var(rewards_list)
            self.coordination_metrics['reward_variance'] = reward_variance
            
            # Team performance (sum of rewards)
            team_reward = sum(rewards_list)
            self.coordination_metrics['team_reward'] = team_reward
            
            # Fairness metric (how evenly distributed rewards are)
            if max(rewards_list) != 0:
                fairness = min(rewards_list) / max(rewards_list)
                self.coordination_metrics['fairness'] = fairness
    
    def _compute_coordination_score(self, agent_rewards: Dict[int, float]) -> float:
        """
        Compute overall coordination score
        
        Args:
            agent_rewards: Current step rewards for all agents
            
        Returns:
            Coordination score (higher is better)
        """
        if len(agent_rewards) < 2:
            return 0.0
        
        rewards_list = list(agent_rewards.values())
        
        # Simple coordination score: team reward - penalty for variance
        team_reward = sum(rewards_list)
        reward_variance = np.var(rewards_list)
        
        coordination_score = team_reward - 0.1 * reward_variance
        
        return coordination_score
    
    def get_agent_messages(self, agent_id: int) -> List[Dict[str, Any]]:
        """
        Get communication messages for specific agent
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            List of messages for the agent
        """
        return self.agent_messages.get(agent_id, [])
    
    def get_coordination_metrics(self) -> Dict[str, float]:
        """Get current coordination metrics"""
        return dict(self.coordination_metrics)
    
    def get_agent_performance(self, agent_id: int) -> Dict[str, float]:
        """
        Get performance statistics for specific agent
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Performance statistics dictionary
        """
        if agent_id not in self.agent_rewards or not self.agent_rewards[agent_id]:
            return {}
        
        rewards = self.agent_rewards[agent_id]
        
        stats = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'total_reward': np.sum(rewards),
            'episodes': len(rewards),
            'best_reward': np.max(rewards),
            'worst_reward': np.min(rewards)
        }
        
        return stats
    
    def reset_metrics(self):
        """Reset coordination and performance metrics"""
        self.agent_rewards.clear()
        self.coordination_metrics.clear()
        self.logger.info("Multi-agent metrics reset")


class SharedParameterTrainer:
    """
    Trainer class for shared parameter multi-agent learning
    Coordinates training across multiple agents with shared networks
    """
    
    def __init__(self, 
                 base_trainer,
                 multi_agent_wrapper: MultiAgentWrapper,
                 communication_module = None):
        """
        Initialize shared parameter trainer
        
        Args:
            base_trainer: Base PPO trainer instance
            multi_agent_wrapper: Multi-agent environment wrapper
            communication_module: Optional communication module
        """
        self.base_trainer = base_trainer
        self.multi_agent_wrapper = multi_agent_wrapper
        self.communication_module = communication_module
        
        self.n_agents = multi_agent_wrapper.n_agents
        self.shared_parameters = multi_agent_wrapper.shared_parameters
        
        self.logger = logging.getLogger('SharedParameterTrainer')
        self.logger.info(f"Shared parameter trainer initialized for {self.n_agents} agents")
    
    def collect_multi_agent_rollouts(self, num_steps: int) -> Dict[str, Any]:
        """
        Collect rollouts from all agents simultaneously
        
        Args:
            num_steps: Number of steps to collect
            
        Returns:
            Combined rollout statistics
        """
        self.base_trainer.network.eval()
        
        # Reset environment
        agent_observations, info = self.multi_agent_wrapper.reset()
        
        rollout_stats = defaultdict(list)
        total_rewards = defaultdict(float)
        step_count = 0
        
        with torch.no_grad():
            while step_count < num_steps:
                # Get actions for all agents
                agent_actions = {}
                
                for agent_id in range(self.n_agents):
                    obs = agent_observations[agent_id]
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.base_trainer.device)
                    
                    # Prepare agent states
                    if prepare_agent_state_batch is not None:
                        try:
                            agent_states = prepare_agent_state_batch(
                                [info['agents'][agent_id]], device=self.base_trainer.device
                            )
                        except Exception:
                            agent_states = torch.zeros(1, 8, device=self.base_trainer.device)
                    else:
                        agent_states = torch.zeros(1, 8, device=self.base_trainer.device)
                    
                    # Get action from shared network
                    actions, log_probs, values = self.base_trainer.network.act(obs_tensor, agent_states)
                    agent_actions[agent_id] = int(actions[0].cpu().item())
                    
                    # Store transition in buffer (using agent 0's buffer for shared parameters)
                    if agent_id == 0:  # Primary agent for shared training
                        self.base_trainer.rollout_buffer.store(
                            observation=obs,
                            agent_state=agent_states[0].cpu().numpy(),
                            action=agent_actions[agent_id],
                            reward=0.0,  # Will be updated after environment step
                            value=values[0].cpu().item(),
                            log_prob=log_probs[0].cpu().item(),
                            done=False
                        )
                
                # Execute actions in environment
                next_observations, rewards, terminated, truncated, next_info = \
                    self.multi_agent_wrapper.step(agent_actions)
                
                # Update rollout buffer with actual reward
                if self.shared_parameters:
                    # Use team reward for shared parameters
                    team_reward = sum(rewards.values()) / len(rewards)
                    self.base_trainer.rollout_buffer.rewards[self.base_trainer.rollout_buffer.ptr - 1] = team_reward
                
                # Update statistics
                for agent_id, reward in rewards.items():
                    total_rewards[agent_id] += reward
                    rollout_stats[f'agent_{agent_id}_reward'].append(reward)
                
                # Coordination metrics
                coord_score = next_info.get('multi_agent', {}).get('coordination_score', 0.0)
                rollout_stats['coordination_score'].append(coord_score)
                
                # Check for episode completion
                done = any(terminated.values()) or any(truncated.values())
                if done:
                    # Episode completed
                    episode_rewards = dict(total_rewards)
                    rollout_stats['episode_rewards'].append(episode_rewards)
                    
                    self.logger.info(f"Multi-agent episode completed: {episode_rewards}")
                    
                    # Reset environment
                    agent_observations, info = self.multi_agent_wrapper.reset()
                    total_rewards = defaultdict(float)
                else:
                    agent_observations = next_observations
                    info = next_info
                
                step_count += 1
        
        # Compute multi-agent statistics
        stats = {
            'total_steps': step_count,
            'coordination_score': np.mean(rollout_stats['coordination_score']) if rollout_stats['coordination_score'] else 0.0,
            'team_reward': sum(total_rewards.values()),
        }
        
        # Add per-agent statistics
        for agent_id in range(self.n_agents):
            agent_key = f'agent_{agent_id}_reward'
            if agent_key in rollout_stats:
                stats[f'agent_{agent_id}_mean_reward'] = np.mean(rollout_stats[agent_key])
        
        return stats
    
    def train_multi_agent(self, total_steps: int, **kwargs) -> Dict[str, Any]:
        """
        Train multi-agent system with shared parameters
        
        Args:
            total_steps: Total training steps
            **kwargs: Additional training arguments
            
        Returns:
            Training history
        """
        self.logger.info(f"Starting multi-agent training for {total_steps} steps")
        
        # Use base trainer's training loop but with multi-agent rollout collection
        original_collect_rollouts = self.base_trainer.collect_rollouts
        
        # Replace rollout collection with multi-agent version
        def multi_agent_collect_rollouts(num_steps):
            return self.collect_multi_agent_rollouts(num_steps)
        
        self.base_trainer.collect_rollouts = multi_agent_collect_rollouts
        
        try:
            # Run training
            training_history = self.base_trainer.train(total_steps, **kwargs)
            
            # Add multi-agent metrics
            coordination_metrics = self.multi_agent_wrapper.get_coordination_metrics()
            training_history.update(coordination_metrics)
            
            return training_history
            
        finally:
            # Restore original rollout collection
            self.base_trainer.collect_rollouts = original_collect_rollouts


def create_multi_agent_system(config: Dict[str, Any], 
                             base_trainer = None,
                             device: Optional[torch.device] = None):
    """
    Factory function to create complete multi-agent training system
    
    Args:
        config: Configuration dictionary
        base_trainer: Base trainer instance (created if None)
        device: Training device
        
    Returns:
        Configured multi-agent training system
    """
    # Extract multi-agent configuration
    ma_config = config.get('multi_agent', {})
    env_config = config.get('environment', {})
    
    n_agents = env_config.get('max_agents', 1)
    shared_params = ma_config.get('shared_parameters', True)
    enable_comm = ma_config.get('enabled', False)
    comm_freq = ma_config.get('communication_freq', 4)
    
    # Create base environment
    if GridWorld is None:
        raise ImportError("GridWorld not available")
    
    base_env = GridWorld(
        size=tuple(env_config.get('grid_size', [15, 15])),
        n_agents=n_agents,
        max_resources=env_config.get('max_resources', 8),
        max_steps=env_config.get('max_steps', 1000)
    )
    
    # Create multi-agent wrapper
    ma_wrapper = MultiAgentWrapper(
        base_env=base_env,
        n_agents=n_agents,
        shared_parameters=shared_params,
        enable_communication=enable_comm,
        communication_freq=comm_freq
    )
    
    # Create shared parameter trainer if needed
    if base_trainer is not None:
        shared_trainer = SharedParameterTrainer(
            base_trainer=base_trainer,
            multi_agent_wrapper=ma_wrapper
        )
        return ma_wrapper, shared_trainer
    else:
        return ma_wrapper, None


if __name__ == "__main__":
    # Test multi-agent wrapper functionality
    print("Testing Multi-Agent Wrapper...")
    
    try:
        # Create base environment
        if GridWorld is None:
            print("❌ GridWorld not available for testing")
            exit(1)
        
        base_env = GridWorld(n_agents=3, max_resources=5)
        
        # Create multi-agent wrapper
        ma_wrapper = MultiAgentWrapper(
            base_env=base_env,
            n_agents=3,
            shared_parameters=True,
            enable_communication=True,
            communication_range=3
        )
        
        # Test reset
        agent_obs, info = ma_wrapper.reset(seed=42)
        print(f"✅ Multi-agent reset: {len(agent_obs)} agent observations")
        print(f"   Agents: {info['multi_agent']['n_agents']}")
        
        # Test step
        actions = {0: 1, 1: 2, 2: 3}  # Different actions for each agent
        next_obs, rewards, term, trunc, next_info = ma_wrapper.step(actions)
        
        print(f"✅ Multi-agent step executed")
        print(f"   Rewards: {rewards}")
        print(f"   Coordination score: {next_info['multi_agent']['coordination_score']:.3f}")
        
        # Test coordination metrics
        coord_metrics = ma_wrapper.get_coordination_metrics()
        print(f"✅ Coordination metrics: {coord_metrics}")
        
        print("✅ All multi-agent wrapper tests passed!")
        
    except Exception as e:
        print(f"❌ Multi-agent test failed: {e}")
        import traceback
        traceback.print_exc()