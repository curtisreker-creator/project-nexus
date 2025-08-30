# File: agents/training/distributed_trainer.py
# Performance & Optimization Engineering (POE) Subsystem
# Distributed Training Infrastructure for 10x Performance Improvement

import ray
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import yaml
import time
from collections import defaultdict
import logging
from pathlib import Path

from environment.grid_world import GridWorld
from agents.networks.ppo_networks import PPOActorCritic
from agents.training.rollout_buffer import RolloutBuffer
from agents.training.gae_computer import compute_gae


@ray.remote
class DistributedEnvironmentWorker:
    """Distributed environment worker for parallel experience collection"""
    
    def __init__(self, env_config: Dict, worker_id: int):
        self.worker_id = worker_id
        self.env_config = env_config
        self.env = None
        self.current_obs = None
        self.current_info = None
        self.total_steps = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
        
        # Initialize environment
        self._initialize_environment()
        
    def _initialize_environment(self):
        """Initialize environment with worker-specific seed"""
        self.env = GridWorld(**self.env_config)
        # Use worker_id for deterministic but different seeds
        seed = self.env_config.get('seed', 42) + self.worker_id
        self.current_obs, self.current_info = self.env.reset(seed=seed)
        
    def collect_rollout(self, 
                       network_weights: Dict, 
                       rollout_length: int,
                       device_type: str = 'cpu') -> Dict:
        """Collect experience rollout using provided network weights"""
        
        # Create local network and load weights
        network = self._create_local_network(network_weights, device_type)
        
        # Storage for rollout data
        observations = []
        agent_states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        obs = self.current_obs
        
        for step in range(rollout_length):
            # Convert observation to tensor
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
            agent_state = torch.zeros(1, 8)  # Dummy agent state for now
            
            # Get action from network
            with torch.no_grad():
                action, log_prob, value = network.act(obs_tensor, agent_state)
                
            # Store transition data
            observations.append(obs.copy())
            agent_states.append(agent_state.squeeze(0).numpy())
            actions.append(action.item())
            log_probs.append(log_prob.item())
            values.append(value.item())
            
            # Environment step
            obs, reward, terminated, truncated, info = self.env.step(action.item())
            done = terminated or truncated
            
            rewards.append(reward)
            dones.append(done)
            
            self.current_episode_reward += reward
            self.total_steps += 1
            
            # Reset if episode ends
            if done:
                obs, _ = self.env.reset()
                self.episode_rewards.append(self.current_episode_reward)
                self.current_episode_reward = 0
        
        # Update current observation
        self.current_obs = obs
        
        # Return collected rollout data
        return {
            'worker_id': self.worker_id,
            'observations': np.array(observations),
            'agent_states': np.array(agent_states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'values': np.array(values),
            'log_probs': np.array(log_probs),
            'dones': np.array(dones),
            'total_steps': self.total_steps,
            'episode_rewards': self.episode_rewards.copy()
        }
    
    def _create_local_network(self, weights: Dict, device_type: str) -> PPOActorCritic:
        """Create local network and load weights"""
        device = torch.device(device_type)
        
        network = PPOActorCritic(
            spatial_channels=5,
            spatial_dim=256,
            state_dim=128,
            fusion_dim=512,
            action_dim=14
        ).to(device)
        
        # Load weights
        network.load_state_dict(weights)
        network.eval()
        
        return network
    
    def get_stats(self) -> Dict:
        """Get worker statistics"""
        return {
            'worker_id': self.worker_id,
            'total_steps': self.total_steps,
            'total_episodes': len(self.episode_rewards),
            'mean_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'last_10_episode_reward': np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else 0.0
        }


class DistributedPPOTrainer:
    """High-performance distributed PPO trainer with Ray"""
    
    def __init__(self, 
                 network_config: Dict,
                 env_config: Dict,
                 training_config: Dict,
                 device: Optional[torch.device] = None,
                 num_workers: int = 8):
        
        self.network_config = network_config
        self.env_config = env_config
        self.training_config = training_config
        self.num_workers = num_workers
        
        # Device setup
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
            
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            
        # Create main network
        self.network = self._create_network()
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=training_config.get('learning_rate', 3e-4)
        )
        
        # Create distributed workers
        self.workers = self._create_workers()
        
        # Training state
        self.global_step = 0
        self.update_count = 0
        self.training_history = defaultdict(list)
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
    def _create_network(self) -> PPOActorCritic:
        """Create main training network"""
        network = PPOActorCritic(
            spatial_channels=5,
            spatial_dim=self.network_config.get('spatial_dim', 256),
            state_dim=self.network_config.get('state_dim', 128),
            fusion_dim=self.network_config.get('fusion_dim', 512),
            action_dim=14
        ).to(self.device)
        
        return network
        
    def _create_workers(self) -> List:
        """Create distributed environment workers"""
        workers = []
        
        for worker_id in range(self.num_workers):
            worker = DistributedEnvironmentWorker.remote(
                self.env_config, 
                worker_id
            )
            workers.append(worker)
            
        return workers
    
    def collect_distributed_experience(self, rollout_length: int) -> RolloutBuffer:
        """Collect experience from all distributed workers"""
        
        start_time = time.perf_counter()
        
        # Get current network weights
        network_weights = {k: v.cpu() for k, v in self.network.state_dict().items()}
        
        # Dispatch rollout collection to all workers
        rollout_futures = []
        for worker in self.workers:
            future = worker.collect_rollout.remote(
                network_weights, 
                rollout_length,
                'cpu'  # Workers use CPU for efficiency
            )
            rollout_futures.append(future)
        
        # Collect results from all workers
        rollout_results = ray.get(rollout_futures)
        
        # Aggregate experience into single buffer
        total_buffer_size = rollout_length * self.num_workers
        buffer = RolloutBuffer(
            buffer_size=total_buffer_size,
            observation_shape=(5, 15, 15),
            agent_state_dim=8,
            device=self.device
        )
        
        # Fill buffer with worker experiences
        buffer_index = 0
        total_reward = 0
        total_episodes = 0
        
        for result in rollout_results:
            length = len(result['observations'])
            
            for i in range(length):
                buffer.store(
                    observation=result['observations'][i],
                    agent_state=result['agent_states'][i],
                    action=result['actions'][i],
                    reward=result['rewards'][i],
                    value=result['values'][i],
                    log_prob=result['log_probs'][i],
                    done=result['dones'][i]
                )
                
                total_reward += result['rewards'][i]
                
            total_episodes += len(result['episode_rewards'])
        
        # Compute GAE advantages
        rewards = buffer.rewards[:len(buffer)]
        values = buffer.values[:len(buffer)]
        dones = buffer.dones[:len(buffer)]
        
        advantages, returns = compute_gae(
            rewards, values, dones,
            gamma=self.training_config.get('gamma', 0.99),
            gae_lambda=self.training_config.get('gae_lambda', 0.95)
        )
        
        buffer.advantages[:len(buffer)] = advantages
        buffer.returns[:len(buffer)] = returns
        
        # Performance tracking
        collection_time = time.perf_counter() - start_time
        steps_per_sec = total_buffer_size / collection_time
        
        self.performance_tracker.record_collection(
            steps_per_sec=steps_per_sec,
            total_reward=total_reward,
            total_episodes=total_episodes,
            collection_time=collection_time
        )
        
        return buffer
    
    def update_network(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Update network using collected experience"""
        
        update_start = time.perf_counter()
        
        batch_size = self.training_config.get('batch_size', 64)
        n_epochs = self.training_config.get('n_epochs', 4)
        clip_ratio = self.training_config.get('clip_ratio', 0.2)
        
        policy_losses = []
        value_losses = []
        entropies = []
        
        for epoch in range(n_epochs):
            # Sample random batches
            indices = torch.randperm(len(buffer), device=self.device)
            
            for start_idx in range(0, len(buffer), batch_size):
                end_idx = min(start_idx + batch_size, len(buffer))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_obs = buffer.observations[batch_indices]
                batch_states = buffer.agent_states[batch_indices]  
                batch_actions = buffer.actions[batch_indices]
                batch_old_log_probs = buffer.log_probs[batch_indices]
                batch_advantages = buffer.advantages[batch_indices]
                batch_returns = buffer.returns[batch_indices]
                
                # Network forward pass
                new_log_probs, values, entropy = self.network.evaluate_actions(
                    batch_obs, batch_states, batch_actions
                )
                
                # PPO loss computation
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Policy loss with clipping
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values, batch_returns)
                
                # Total loss
                entropy_coef = self.training_config.get('entropy_coef', 0.01)
                value_coef = self.training_config.get('value_loss_coef', 0.5)
                
                total_loss = (
                    policy_loss + 
                    value_coef * value_loss - 
                    entropy_coef * entropy.mean()
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                max_grad_norm = self.training_config.get('max_grad_norm', 0.5)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_grad_norm)
                
                self.optimizer.step()
                
                # Track metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
        
        # Performance tracking
        update_time = time.perf_counter() - update_start
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'update_time': update_time
        }
    
    def train(self, 
              total_steps: int,
              rollout_length: int = 256,
              log_interval: int = 1000,
              save_interval: int = 10000,
              checkpoint_dir: str = "checkpoints") -> Dict:
        """Main distributed training loop"""
        
        self.logger.info(f"🚀 Starting distributed training with {self.num_workers} workers")
        self.logger.info(f"Target: {total_steps} total steps, device: {self.device}")
        
        training_start = time.perf_counter()
        
        while self.global_step < total_steps:
            
            # Collect distributed experience
            buffer = self.collect_distributed_experience(rollout_length)
            
            # Update network
            update_stats = self.update_network(buffer)
            
            # Update counters
            self.global_step += len(buffer)
            self.update_count += 1
            
            # Record training history
            for key, value in update_stats.items():
                self.training_history[key].append(value)
            
            # Performance metrics
            perf_stats = self.performance_tracker.get_current_stats()
            
            # Logging
            if self.update_count % (log_interval // rollout_length) == 0:
                self.logger.info(
                    f"Step {self.global_step:>8d} | "
                    f"Policy Loss: {update_stats['policy_loss']:.6f} | "
                    f"Value Loss: {update_stats['value_loss']:.6f} | "
                    f"Speed: {perf_stats['avg_steps_per_sec']:.1f} steps/sec | "
                    f"Reward: {perf_stats['avg_reward']:.3f}"
                )
            
            # Checkpointing
            if self.global_step % save_interval == 0:
                self.save_checkpoint(checkpoint_dir)
        
        total_time = time.perf_counter() - training_start
        final_perf = self.performance_tracker.get_final_stats(total_time)
        
        self.logger.info("🎉 Distributed training completed!")
        self.logger.info(f"📊 Final performance: {final_perf['overall_steps_per_sec']:.1f} steps/sec")
        self.logger.info(f"⚡ Performance improvement: {final_perf['speedup_factor']:.1f}x")
        
        return {
            **self.training_history,
            'performance_stats': final_perf
        }
    
    def save_checkpoint(self, checkpoint_dir: str):
        """Save training checkpoint"""
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'update_count': self.update_count,
            'training_history': dict(self.training_history),
            'performance_stats': self.performance_tracker.get_current_stats()
        }
        
        checkpoint_path = Path(checkpoint_dir) / f"distributed_checkpoint_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def cleanup(self):
        """Cleanup distributed resources"""
        if ray.is_initialized():
            ray.shutdown()


class PerformanceTracker:
    """Track and analyze distributed training performance"""
    
    def __init__(self):
        self.collection_times = []
        self.steps_per_sec_history = []
        self.reward_history = []
        self.episode_history = []
        
        # Baseline performance (from single-threaded training)
        self.baseline_steps_per_sec = 7.0  # From previous sessions
        
    def record_collection(self, steps_per_sec: float, total_reward: float, 
                         total_episodes: int, collection_time: float):
        """Record performance metrics from experience collection"""
        self.steps_per_sec_history.append(steps_per_sec)
        self.reward_history.append(total_reward)
        self.episode_history.append(total_episodes)
        self.collection_times.append(collection_time)
    
    def get_current_stats(self) -> Dict:
        """Get current performance statistics"""
        if not self.steps_per_sec_history:
            return {'avg_steps_per_sec': 0.0, 'avg_reward': 0.0}
            
        return {
            'avg_steps_per_sec': np.mean(self.steps_per_sec_history[-10:]),  # Last 10 measurements
            'max_steps_per_sec': np.max(self.steps_per_sec_history),
            'avg_reward': np.mean(self.reward_history[-10:]) if self.reward_history else 0.0,
            'speedup_vs_baseline': np.mean(self.steps_per_sec_history[-10:]) / self.baseline_steps_per_sec
        }
    
    def get_final_stats(self, total_training_time: float) -> Dict:
        """Get comprehensive final performance statistics"""
        if not self.steps_per_sec_history:
            return {}
            
        overall_steps = sum(len(self.reward_history[i] for i in range(len(self.reward_history))))
        overall_steps_per_sec = overall_steps / total_training_time if total_training_time > 0 else 0.0
        
        return {
            'overall_steps_per_sec': overall_steps_per_sec,
            'peak_steps_per_sec': np.max(self.steps_per_sec_history),
            'avg_steps_per_sec': np.mean(self.steps_per_sec_history),
            'speedup_factor': overall_steps_per_sec / self.baseline_steps_per_sec,
            'total_training_time': total_training_time,
            'collection_efficiency': np.mean(self.collection_times)
        }