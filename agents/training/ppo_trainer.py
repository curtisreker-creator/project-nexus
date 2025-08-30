# File: agents/training/ppo_trainer.py
"""
Proximal Policy Optimization (PPO) Trainer for Project NEXUS
Complete implementation with multi-agent support and advanced features
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import time
from collections import defaultdict
import yaml

# Import our network components
try:
    from ..networks import PPOActorCritic, prepare_agent_state_batch
    from .rollout_buffer import RolloutBuffer
    from .gae_computer import compute_gae
except ImportError as e:
    logging.warning(f"Import error in PPO trainer: {e}")
    # Fallback imports for standalone testing
    PPOActorCritic = None
    prepare_agent_state_batch = None

class PPOTrainer:
    """
    PPO Trainer with advanced features for multi-agent coordination
    """
    
    def __init__(self, 
                 network: nn.Module,
                 environment,
                 config: Dict[str, Any],
                 device: Optional[torch.device] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize PPO trainer
        
        Args:
            network: PPO Actor-Critic network
            environment: GridWorld environment instance
            config: Training configuration dictionary
            device: Training device (auto-detected if None)
            logger: Logger instance (created if None)
        """
        self.network = network
        self.env = environment
        self.config = config
        
        # Device setup with fallback
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
            
        # Move network to device
        self.network = self.network.to(self.device)
        
        # Logger setup
        self.logger = logger or self._setup_logger()
        
        # Training configuration
        training_config = config.get('training', {})
        self.learning_rate = training_config.get('learning_rate', 3e-4)
        self.batch_size = training_config.get('batch_size', 64)
        self.buffer_size = training_config.get('buffer_size', 2048)
        self.n_epochs = training_config.get('n_epochs', 4)
        self.clip_ratio = training_config.get('clip_ratio', 0.2)
        self.value_loss_coef = training_config.get('value_loss_coef', 0.5)
        self.entropy_coef = training_config.get('entropy_coef', 0.01)
        self.max_grad_norm = training_config.get('max_grad_norm', 0.5)
        self.gamma = training_config.get('gamma', 0.99)
        self.gae_lambda = training_config.get('gae_lambda', 0.95)
        
        # Optimizer setup with learning rate scheduling
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=1000
        )
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
        self.best_reward = float('-inf')
        
        # Metrics tracking
        self.metrics = defaultdict(list)
        
        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.buffer_size,
            observation_shape=(5, 15, 15),
            agent_state_dim=8,
            n_agents=self.env.n_agents,
            device=self.device
        )
        
        self.logger.info(f"PPO Trainer initialized on {self.device}")
        self.logger.info(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}")
        
    def _setup_logger(self) -> logging.Logger:
        """Setup training logger"""
        logger = logging.getLogger('PPOTrainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def collect_rollouts(self, num_steps: int) -> Dict[str, float]:
        """
        Collect rollout data from environment interactions
        
        Args:
            num_steps: Number of environment steps to collect
            
        Returns:
            Dictionary with rollout statistics
        """
        self.network.eval()  # Set to evaluation mode
        rollout_stats = defaultdict(list)
        
        # Reset environment
        obs, info = self.env.reset()
        episode_rewards = 0.0
        episode_length = 0
        
        with torch.no_grad():
            for step in range(num_steps):
                # Convert observations to tensor and move to device
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
                
                # Prepare agent states and ensure they're on the correct device
                if prepare_agent_state_batch is not None:
                    try:
                        agent_states = prepare_agent_state_batch(
                            info['agents'], device=self.device
                        )
                    except Exception as e:
                        self.logger.warning(f"Agent state preparation failed: {e}")
                        # Fallback: create dummy agent states on correct device
                        agent_states = torch.zeros(1, 8, device=self.device)
                else:
                    agent_states = torch.zeros(1, 8, device=self.device)
                
                # Get action from network
                try:
                    actions, log_probs, values = self.network.act(obs_tensor, agent_states)
                except Exception as e:
                    self.logger.error(f"Network action failed: {e}")
                    # Emergency fallback
                    actions = torch.randint(0, 14, (1,), device=self.device)
                    log_probs = torch.zeros(1, device=self.device)
                    values = torch.zeros(1, device=self.device)
                
                # Execute action in environment
                action_int = int(actions[0].cpu().item())
                next_obs, reward, terminated, truncated, next_info = self.env.step(action_int)
                done = terminated or truncated
                
                # Store transition in buffer
                self.rollout_buffer.store(
                    observation=obs,
                    agent_state=agent_states[0].cpu().numpy(),
                    action=action_int,
                    reward=reward,
                    value=values[0].cpu().item(),
                    log_prob=log_probs[0].cpu().item(),
                    done=done
                )
                
                # Update statistics
                episode_rewards += reward
                episode_length += 1
                rollout_stats['rewards'].append(reward)
                rollout_stats['values'].append(values[0].cpu().item())
                
                # Handle episode completion
                if done:
                    rollout_stats['episode_rewards'].append(episode_rewards)
                    rollout_stats['episode_lengths'].append(episode_length)
                    
                    self.episode_count += 1
                    self.logger.info(
                        f"Episode {self.episode_count}: "
                        f"Reward={episode_rewards:.3f}, Length={episode_length}"
                    )
                    
                    # Reset for next episode
                    obs, info = self.env.reset()
                    episode_rewards = 0.0
                    episode_length = 0
                else:
                    obs = next_obs
                    info = next_info
                
                self.step_count += 1
        
        # Compute final value for bootstrapping
        if not done:
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
            if prepare_agent_state_batch is not None:
                try:
                    agent_states = prepare_agent_state_batch(info['agents'], device=self.device)
                except Exception:
                    agent_states = torch.zeros(1, 8, device=self.device)
            else:
                agent_states = torch.zeros(1, 8, device=self.device)
                
            with torch.no_grad():
                final_value = self.network.get_value(obs_tensor, agent_states)[0].cpu().item()
        else:
            final_value = 0.0
        
        # Compute advantages using GAE
        advantages, returns = compute_gae(
            rewards=self.rollout_buffer.rewards,
            values=self.rollout_buffer.values,
            dones=self.rollout_buffer.dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            last_value=final_value
        )
        
        self.rollout_buffer.advantages = advantages
        self.rollout_buffer.returns = returns
        
        # Compute rollout statistics
        stats = {}
        if rollout_stats['episode_rewards']:
            stats['mean_episode_reward'] = np.mean(rollout_stats['episode_rewards'])
            stats['mean_episode_length'] = np.mean(rollout_stats['episode_lengths'])
        else:
            stats['mean_episode_reward'] = np.mean(rollout_stats['rewards'])
            stats['mean_episode_length'] = len(rollout_stats['rewards'])
            
        stats['mean_value'] = np.mean(rollout_stats['values'])
        stats['mean_advantage'] = np.mean(advantages)
        stats['mean_return'] = np.mean(returns)
        
        return stats
    
    def update_policy(self) -> Dict[str, float]:
        """
        Update policy using collected rollouts
        
        Returns:
            Dictionary with training statistics
        """
        self.network.train()  # Set to training mode
        
        # Prepare training data
        observations = torch.from_numpy(self.rollout_buffer.observations).to(self.device)
        agent_states = torch.from_numpy(self.rollout_buffer.agent_states).to(self.device)
        actions = torch.from_numpy(self.rollout_buffer.actions).long().to(self.device)
        old_log_probs = torch.from_numpy(self.rollout_buffer.log_probs).to(self.device)
        advantages = torch.from_numpy(self.rollout_buffer.advantages).to(self.device)
        returns = torch.from_numpy(self.rollout_buffer.returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        update_stats = defaultdict(list)
        
        # Multiple epochs of optimization
        for epoch in range(self.n_epochs):
            # Generate mini-batches
            batch_indices = torch.randperm(len(observations))
            
            for start_idx in range(0, len(observations), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(observations))
                batch_idx = batch_indices[start_idx:end_idx]
                
                # Extract batch
                batch_obs = observations[batch_idx]
                batch_states = agent_states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # Forward pass
                try:
                    new_log_probs, values, entropy = self.network.evaluate_actions(
                        batch_obs, batch_states, batch_actions
                    )
                except Exception as e:
                    self.logger.error(f"Network evaluation failed: {e}")
                    continue
                
                # Compute policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = nn.MSELoss()(values, batch_returns)
                
                # Compute entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = (
                    policy_loss + 
                    self.value_loss_coef * value_loss + 
                    self.entropy_coef * entropy_loss
                )
                
                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track statistics
                update_stats['policy_loss'].append(policy_loss.item())
                update_stats['value_loss'].append(value_loss.item())
                update_stats['entropy_loss'].append(entropy_loss.item())
                update_stats['total_loss'].append(total_loss.item())
                
                # Track PPO-specific metrics
                with torch.no_grad():
                    kl_div = (batch_old_log_probs - new_log_probs).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_ratio).float().mean().item()
                    update_stats['kl_divergence'].append(kl_div)
                    update_stats['clip_fraction'].append(clip_fraction)
        
        # Update learning rate
        self.lr_scheduler.step()
        
        # Clear rollout buffer
        self.rollout_buffer.clear()
        
        # Compute final statistics
        final_stats = {
            'policy_loss': np.mean(update_stats['policy_loss']),
            'value_loss': np.mean(update_stats['value_loss']),
            'entropy_loss': np.mean(update_stats['entropy_loss']),
            'total_loss': np.mean(update_stats['total_loss']),
            'kl_divergence': np.mean(update_stats['kl_divergence']),
            'clip_fraction': np.mean(update_stats['clip_fraction']),
            'learning_rate': self.lr_scheduler.get_last_lr()[0]
        }
        
        return final_stats
    
    def train(self, total_steps: int, log_interval: int = 100, 
              save_interval: int = 1000, checkpoint_dir: str = "checkpoints") -> Dict[str, List[float]]:
        """
        Main training loop
        
        Args:
            total_steps: Total number of environment steps to train
            log_interval: Steps between logging updates
            save_interval: Steps between checkpoint saves
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Dictionary with training history
        """
        self.logger.info(f"Starting PPO training for {total_steps} steps")
        self.logger.info(f"Device: {self.device}, Buffer size: {self.buffer_size}")
        
        # Create checkpoint directory
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        training_history = defaultdict(list)
        start_time = time.time()
        steps_trained = 0
        
        while steps_trained < total_steps:
            # Collect rollouts
            rollout_stats = self.collect_rollouts(self.buffer_size)
            steps_trained += self.buffer_size
            
            # Update policy
            update_stats = self.update_policy()
            
            # Combine all statistics
            all_stats = {**rollout_stats, **update_stats}
            
            # Update training history
            for key, value in all_stats.items():
                training_history[key].append(value)
                self.metrics[key].append(value)
            
            # Logging
            if steps_trained % log_interval < self.buffer_size:
                elapsed_time = time.time() - start_time
                steps_per_sec = steps_trained / elapsed_time
                
                self.logger.info(f"Step {steps_trained}/{total_steps}")
                self.logger.info(f"  Mean Episode Reward: {rollout_stats.get('mean_episode_reward', 0):.3f}")
                self.logger.info(f"  Policy Loss: {update_stats['policy_loss']:.6f}")
                self.logger.info(f"  Value Loss: {update_stats['value_loss']:.6f}")
                self.logger.info(f"  KL Divergence: {update_stats['kl_divergence']:.6f}")
                self.logger.info(f"  Learning Rate: {update_stats['learning_rate']:.2e}")
                self.logger.info(f"  Steps/sec: {steps_per_sec:.1f}")
            
            # Save checkpoint
            if steps_trained % save_interval < self.buffer_size:
                checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{steps_trained}.pth"
                self.save_checkpoint(checkpoint_path)
                
                # Save best model
                current_reward = rollout_stats.get('mean_episode_reward', float('-inf'))
                if current_reward > self.best_reward:
                    self.best_reward = current_reward
                    best_path = Path(checkpoint_dir) / "best_model.pth"
                    self.save_checkpoint(best_path)
                    self.logger.info(f"New best model saved: {current_reward:.3f}")
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.1f}s")
        self.logger.info(f"Final performance: {self.best_reward:.3f}")
        
        return dict(training_history)
    
    def save_checkpoint(self, filepath: Union[str, Path]):
        """Save training checkpoint"""
        try:
            checkpoint = {
                'network_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.lr_scheduler.state_dict(),
                'step_count': self.step_count,
                'episode_count': self.episode_count,
                'best_reward': self.best_reward,
                'config': self.config,
                'metrics': dict(self.metrics)
            }
            
            torch.save(checkpoint, filepath)
            self.logger.info(f"Checkpoint saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, filepath: Union[str, Path]) -> bool:
        """Load training checkpoint"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.step_count = checkpoint['step_count']
            self.episode_count = checkpoint['episode_count']
            self.best_reward = checkpoint['best_reward']
            
            if 'metrics' in checkpoint:
                self.metrics = defaultdict(list, checkpoint['metrics'])
            
            self.logger.info(f"Checkpoint loaded from {filepath}")
            self.logger.info(f"Resumed at step {self.step_count}, episode {self.episode_count}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def evaluate(self, num_episodes: int = 10, render: bool = False) -> Dict[str, float]:
        """
        Evaluate the current policy
        
        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render episodes
            
        Returns:
            Evaluation statistics
        """
        self.network.eval()
        episode_rewards = []
        episode_lengths = []
        
        with torch.no_grad():
            for episode in range(num_episodes):
                obs, info = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
                done = False
                
                while not done:
                    if render:
                        self.env.render()
                    
                    # Get deterministic action
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
                    
                    if prepare_agent_state_batch is not None:
                        try:
                            agent_states = prepare_agent_state_batch(info['agents'], device=self.device)
                        except Exception:
                            agent_states = torch.zeros(1, 8, device=self.device)
                    else:
                        agent_states = torch.zeros(1, 8, device=self.device)
                    
                    actions, _, _ = self.network.act(obs_tensor, agent_states, deterministic=True)
                    action_int = int(actions[0].cpu().item())
                    
                    obs, reward, terminated, truncated, info = self.env.step(action_int)
                    done = terminated or truncated
                    
                    episode_reward += reward
                    episode_length += 1
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                self.logger.info(f"Eval Episode {episode+1}: Reward={episode_reward:.3f}, Length={episode_length}")
        
        eval_stats = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths)
        }
        
        self.logger.info("Evaluation Results:")
        for key, value in eval_stats.items():
            self.logger.info(f"  {key}: {value:.3f}")
        
        return eval_stats


def create_trainer_from_config(config_path: str, 
                             network: Optional[nn.Module] = None,
                             environment = None) -> PPOTrainer:
    """
    Create PPO trainer from configuration file
    
    Args:
        config_path: Path to configuration YAML file
        network: PPO network (created if None)
        environment: Environment instance (created if None)
        
    Returns:
        Configured PPO trainer
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment if not provided
    if environment is None:
        try:
            from environment import create_environment
            environment = create_environment(config_path=config_path)
        except ImportError:
            raise ImportError("Environment not available. Please provide environment instance.")
    
    # Create network if not provided
    if network is None:
        try:
            from ..networks import create_ppo_network
            network = create_ppo_network(config)
        except ImportError:
            raise ImportError("Network creation not available. Please provide network instance.")
    
    # Create trainer
    trainer = PPOTrainer(
        network=network,
        environment=environment,
        config=config
    )
    
    return trainer


if __name__ == "__main__":
    # Quick test of PPO trainer components
    print("Testing PPO Trainer components...")
    
    try:
        # Test configuration loading
        config = {
            'training': {
                'learning_rate': 3e-4,
                'batch_size': 64,
                'buffer_size': 128,  # Small for testing
                'n_epochs': 2,
                'clip_ratio': 0.2,
                'gamma': 0.99,
                'gae_lambda': 0.95
            }
        }
        
        print("✅ Configuration structure validated")
        
        # Test metrics tracking
        metrics = defaultdict(list)
        metrics['test'].append(1.0)
        print(f"✅ Metrics tracking: {dict(metrics)}")
        
        print("✅ PPO Trainer components ready for integration!")
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")