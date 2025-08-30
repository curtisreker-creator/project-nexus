# File: research/meta_learning/maml_framework.py
# Research & Innovation Laboratory (RIL) Subsystem
# Meta-Learning Framework for Few-Shot Adaptation and Transfer Learning

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from collections import defaultdict
import copy

from environment.grid_world import GridWorld
from agents.networks.ppo_networks import PPOActorCritic


@dataclass
class MetaTaskConfig:
    """Configuration for meta-learning tasks"""
    task_name: str
    environment_params: Dict[str, Any]
    success_criteria: Dict[str, float]
    adaptation_steps: int = 5
    evaluation_episodes: int = 10


class ModelAgnosticMetaLearning:
    """MAML implementation for multi-agent reinforcement learning"""
    
    def __init__(self,
                 base_network: PPOActorCritic,
                 meta_learning_rate: float = 1e-3,
                 inner_learning_rate: float = 1e-2,
                 inner_steps: int = 5,
                 device: Optional[torch.device] = None):
        
        self.base_network = base_network
        self.meta_lr = meta_learning_rate
        self.inner_lr = inner_learning_rate  
        self.inner_steps = inner_steps
        
        self.device = device or torch.device('cpu')
        self.base_network.to(self.device)
        
        # Meta-optimizer for outer loop updates
        self.meta_optimizer = torch.optim.Adam(
            self.base_network.parameters(),
            lr=self.meta_lr
        )
        
        # Meta-learning statistics
        self.meta_step = 0
        self.task_performance_history = defaultdict(list)
        self.adaptation_curves = defaultdict(list)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🧠 MAML Framework initialized for multi-agent RL")
        
    def create_task_environment(self, task_config: MetaTaskConfig) -> GridWorld:
        """Create environment for specific meta-learning task"""
        
        env = GridWorld(**task_config.environment_params)
        return env
    
    def collect_task_data(self, 
                         env: GridWorld, 
                         network: PPOActorCritic,
                         num_episodes: int = 10) -> List[Dict[str, Any]]:
        """Collect trajectory data from task environment"""
        
        trajectories = []
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            
            trajectory = {
                'observations': [],
                'agent_states': [],
                'actions': [],
                'rewards': [],
                'log_probs': [],
                'values': [],
                'dones': []
            }
            
            done = False
            total_reward = 0
            
            while not done:
                # Convert observation to tensor
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
                agent_state = torch.zeros(1, 8).to(self.device)  # Dummy state
                
                # Get action from network
                with torch.no_grad():
                    action, log_prob, value = network.act(obs_tensor, agent_state)
                
                # Store transition
                trajectory['observations'].append(obs.copy())
                trajectory['agent_states'].append(agent_state.squeeze(0).cpu().numpy())
                trajectory['actions'].append(action.item())
                trajectory['log_probs'].append(log_prob.item())
                trajectory['values'].append(value.item())
                trajectory['rewards'].append(0)  # Will be filled after step
                
                # Environment step
                obs, reward, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated
                
                trajectory['rewards'][-1] = reward
                trajectory['dones'].append(done)
                total_reward += reward
            
            # Convert to numpy arrays
            for key in trajectory:
                trajectory[key] = np.array(trajectory[key])
                
            trajectory['total_reward'] = total_reward
            trajectories.append(trajectory)
            
        return trajectories
    
    def inner_loop_update(self, 
                         network: PPOActorCritic,
                         trajectories: List[Dict[str, Any]]) -> PPOActorCritic:
        """Perform inner loop gradient updates for task adaptation"""
        
        # Create a copy of the network for inner updates
        adapted_network = copy.deepcopy(network)
        inner_optimizer = torch.optim.SGD(
            adapted_network.parameters(),
            lr=self.inner_lr
        )
        
        # Convert trajectories to training batch
        batch_obs, batch_states, batch_actions, batch_rewards, batch_values, batch_log_probs = [], [], [], [], [], []
        
        for traj in trajectories:
            batch_obs.extend(traj['observations'])
            batch_states.extend(traj['agent_states'])
            batch_actions.extend(traj['actions'])
            batch_rewards.extend(traj['rewards'])
            batch_values.extend(traj['values'])
            batch_log_probs.extend(traj['log_probs'])
        
        # Convert to tensors
        batch_obs = torch.from_numpy(np.array(batch_obs)).float().to(self.device)
        batch_states = torch.from_numpy(np.array(batch_states)).float().to(self.device)
        batch_actions = torch.from_numpy(np.array(batch_actions)).long().to(self.device)
        batch_rewards = torch.from_numpy(np.array(batch_rewards)).float().to(self.device)
        batch_old_log_probs = torch.from_numpy(np.array(batch_log_probs)).float().to(self.device)
        
        # Compute returns (simple Monte Carlo for now)
        batch_returns = self._compute_returns(batch_rewards, trajectories)
        
        # Inner loop updates
        adaptation_losses = []
        
        for step in range(self.inner_steps):
            inner_optimizer.zero_grad()
            
            # Forward pass through adapted network
            new_log_probs, values, entropy = adapted_network.evaluate_actions(
                batch_obs, batch_states, batch_actions
            )
            
            # Simple policy gradient loss for inner updates
            advantages = batch_returns - values.detach()  # Don't backprop through value baseline
            
            # Policy loss
            policy_loss = -(new_log_probs * advantages).mean()
            
            # Value loss
            value_loss = 0.5 * (values - batch_returns).pow(2).mean()
            
            # Combined inner loss
            inner_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
            
            inner_loss.backward()
            inner_optimizer.step()
            
            adaptation_losses.append(inner_loss.item())
        
        self.logger.debug(f"Inner adaptation: {adaptation_losses[-1]:.6f}")
        return adapted_network
    
    def meta_update(self, 
                   task_configs: List[MetaTaskConfig],
                   episodes_per_task: int = 10) -> Dict[str, float]:
        """Perform meta-learning update across multiple tasks"""
        
        self.logger.info(f"🎯 Meta-update step {self.meta_step}: {len(task_configs)} tasks")
        
        meta_loss = 0.0
        task_performances = {}
        
        # Zero meta-optimizer gradients
        self.meta_optimizer.zero_grad()
        
        for task_config in task_configs:
            
            # Create task environment
            env = self.create_task_environment(task_config)
            
            # Collect initial trajectories with base network
            support_trajectories = self.collect_task_data(
                env, self.base_network, episodes_per_task // 2
            )
            
            # Inner loop adaptation
            adapted_network = self.inner_loop_update(self.base_network, support_trajectories)
            
            # Collect query trajectories with adapted network
            query_trajectories = self.collect_task_data(
                env, adapted_network, episodes_per_task // 2
            )
            
            # Compute meta-loss on query set
            task_loss = self._compute_meta_loss(adapted_network, query_trajectories)
            
            # Accumulate meta-gradients
            task_loss.backward()
            meta_loss += task_loss.item()
            
            # Track task performance
            avg_reward = np.mean([traj['total_reward'] for traj in query_trajectories])
            task_performances[task_config.task_name] = avg_reward
            self.task_performance_history[task_config.task_name].append(avg_reward)
            
            self.logger.debug(f"Task {task_config.task_name}: {avg_reward:.3f} reward")
        
        # Meta-optimizer step
        self.meta_optimizer.step()
        
        # Update meta-step counter
        self.meta_step += 1
        
        # Compute average meta-loss
        avg_meta_loss = meta_loss / len(task_configs)
        avg_performance = np.mean(list(task_performances.values()))
        
        self.logger.info(f"✅ Meta-update complete: Loss {avg_meta_loss:.6f}, Avg Reward {avg_performance:.3f}")
        
        return {
            'meta_loss': avg_meta_loss,
            'avg_task_performance': avg_performance,
            'task_performances': task_performances,
            'meta_step': self.meta_step
        }
    
    def _compute_returns(self, rewards: torch.Tensor, trajectories: List[Dict]) -> torch.Tensor:
        """Compute Monte Carlo returns for trajectories"""
        returns = []
        
        start_idx = 0
        for traj in trajectories:
            traj_len = len(traj['rewards'])
            traj_rewards = rewards[start_idx:start_idx + traj_len]
            
            # Monte Carlo returns
            traj_returns = []
            discounted_return = 0
            
            for reward in reversed(traj_rewards):
                discounted_return = reward + 0.99 * discounted_return
                traj_returns.append(discounted_return)
            
            traj_returns.reverse()
            returns.extend(traj_returns)
            start_idx += traj_len
        
        return torch.tensor(returns, device=self.device)
    
    def _compute_meta_loss(self, adapted_network: PPOActorCritic, 
                          query_trajectories: List[Dict]) -> torch.Tensor:
        """Compute meta-learning loss on query set"""
        
        # Convert query trajectories to batch
        batch_obs, batch_states, batch_actions, batch_rewards = [], [], [], []
        
        for traj in query_trajectories:
            batch_obs.extend(traj['observations'])
            batch_states.extend(traj['agent_states'])
            batch_actions.extend(traj['actions'])
            batch_rewards.extend(traj['rewards'])
        
        batch_obs = torch.from_numpy(np.array(batch_obs)).float().to(self.device)
        batch_states = torch.from_numpy(np.array(batch_states)).float().to(self.device)
        batch_actions = torch.from_numpy(np.array(batch_actions)).long().to(self.device)
        batch_rewards = torch.from_numpy(np.array(batch_rewards)).float().to(self.device)
        
        # Compute returns
        batch_returns = self._compute_returns(batch_rewards, query_trajectories)
        
        # Forward pass through adapted network
        log_probs, values, entropy = adapted_network.evaluate_actions(
            batch_obs, batch_states, batch_actions
        )
        
        # Meta-loss computation
        advantages = batch_returns - values.detach()
        policy_loss = -(log_probs * advantages).mean()
        value_loss = 0.5 * (values - batch_returns).pow(2).mean()
        
        meta_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
        
        return meta_loss
    
    def evaluate_adaptation(self, 
                           task_config: MetaTaskConfig,
                           num_adaptation_episodes: int = 5,
                           num_evaluation_episodes: int = 20) -> Dict[str, Any]:
        """Evaluate few-shot adaptation capability"""
        
        self.logger.info(f"🔬 Evaluating adaptation on task: {task_config.task_name}")
        
        env = self.create_task_environment(task_config)
        
        # Baseline performance with base network
        baseline_trajectories = self.collect_task_data(env, self.base_network, num_evaluation_episodes)
        baseline_performance = np.mean([traj['total_reward'] for traj in baseline_trajectories])
        
        # Collect adaptation data
        adaptation_trajectories = self.collect_task_data(env, self.base_network, num_adaptation_episodes)
        
        # Adapt network
        adapted_network = self.inner_loop_update(self.base_network, adaptation_trajectories)
        
        # Evaluate adapted performance
        adapted_trajectories = self.collect_task_data(env, adapted_network, num_evaluation_episodes)
        adapted_performance = np.mean([traj['total_reward'] for traj in adapted_trajectories])
        
        # Improvement metrics
        improvement = adapted_performance - baseline_performance
        improvement_ratio = adapted_performance / baseline_performance if baseline_performance != 0 else float('inf')
        
        results = {
            'task_name': task_config.task_name,
            'baseline_performance': baseline_performance,
            'adapted_performance': adapted_performance,
            'improvement': improvement,
            'improvement_ratio': improvement_ratio,
            'adaptation_episodes': num_adaptation_episodes,
            'success': improvement > task_config.success_criteria.get('min_improvement', 0.1)
        }
        
        self.logger.info(f"📊 Adaptation results: {baseline_performance:.3f} → {adapted_performance:.3f} "
                        f"(+{improvement:.3f}, {improvement_ratio:.2f}x)")
        
        return results


class EmergentBehaviorAnalyzer:
    """Analyze and quantify emergent coordination behaviors in multi-agent systems"""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cpu')
        self.behavior_metrics = {
            'coordination_index': [],
            'communication_efficiency': [],
            'role_specialization': [],
            'collective_intelligence': []
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🌟 Emergent Behavior Analyzer initialized")
    
    def analyze_multi_agent_episode(self, 
                                   observations: List[np.ndarray],
                                   actions: List[List[int]], 
                                   rewards: List[float],
                                   agent_positions: List[List[Tuple[int, int]]]) -> Dict[str, float]:
        """Analyze emergent behaviors in multi-agent episode"""
        
        num_agents = len(actions[0]) if actions else 0
        episode_length = len(observations)
        
        if num_agents < 2:
            return {'coordination_index': 0.0, 'communication_efficiency': 0.0}
        
        # 1. Coordination Index - measure of synchronized behavior
        coordination_index = self._compute_coordination_index(actions, agent_positions)
        
        # 2. Communication Efficiency - effectiveness of implicit communication
        comm_efficiency = self._compute_communication_efficiency(
            observations, actions, agent_positions
        )
        
        # 3. Role Specialization - degree of behavioral differentiation
        role_specialization = self._compute_role_specialization(actions, agent_positions)
        
        # 4. Collective Intelligence - emergent problem-solving capability  
        collective_intelligence = self._compute_collective_intelligence(
            actions, rewards, agent_positions
        )
        
        # 5. Spatial Coordination - spatial organization patterns
        spatial_coordination = self._compute_spatial_coordination(agent_positions)
        
        # 6. Temporal Synchronization - timing of coordinated actions
        temporal_sync = self._compute_temporal_synchronization(actions)
        
        behavior_metrics = {
            'coordination_index': coordination_index,
            'communication_efficiency': comm_efficiency,
            'role_specialization': role_specialization,
            'collective_intelligence': collective_intelligence,
            'spatial_coordination': spatial_coordination,
            'temporal_synchronization': temporal_sync,
            'emergence_score': self._compute_emergence_score(
                coordination_index, role_specialization, collective_intelligence
            )
        }
        
        # Update historical metrics
        for key, value in behavior_metrics.items():
            if key in self.behavior_metrics:
                self.behavior_metrics[key].append(value)
        
        return behavior_metrics
    
    def _compute_coordination_index(self, 
                                   actions: List[List[int]],
                                   positions: List[List[Tuple[int, int]]]) -> float:
        """Compute coordination index based on action synchronization"""
        
        if len(actions) < 2:
            return 0.0
            
        coordination_scores = []
        
        for t in range(len(actions) - 1):
            current_actions = actions[t]
            current_positions = positions[t] if t < len(positions) else []
            
            # Measure action similarity considering spatial context
            coord_score = 0.0
            agent_pairs = 0
            
            for i in range(len(current_actions)):
                for j in range(i + 1, len(current_actions)):
                    
                    # Spatial proximity bonus
                    if len(current_positions) > max(i, j):
                        pos_i, pos_j = current_positions[i], current_positions[j]
                        distance = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                        proximity_bonus = max(0, 1.0 - distance / 10.0)  # Normalize by max grid distance
                    else:
                        proximity_bonus = 0.5
                    
                    # Action coordination score
                    action_i, action_j = current_actions[i], current_actions[j]
                    
                    # Same action type (exploration, building, etc.)
                    action_similarity = 1.0 if action_i == action_j else 0.0
                    
                    # Complementary actions (e.g., gathering + building)
                    complementary_bonus = self._compute_action_complementarity(action_i, action_j)
                    
                    pair_coord = (action_similarity + complementary_bonus) * proximity_bonus
                    coord_score += pair_coord
                    agent_pairs += 1
            
            if agent_pairs > 0:
                coordination_scores.append(coord_score / agent_pairs)
        
        return np.mean(coordination_scores) if coordination_scores else 0.0
    
    def _compute_communication_efficiency(self,
                                        observations: List[np.ndarray],
                                        actions: List[List[int]], 
                                        positions: List[List[Tuple[int, int]]]) -> float:
        """Measure implicit communication through environmental changes"""
        
        if len(observations) < 2 or len(actions) < 1:
            return 0.0
        
        communication_events = 0
        total_opportunities = 0
        
        for t in range(len(observations) - 1):
            current_obs = observations[t]
            next_obs = observations[t + 1]
            current_actions = actions[t]
            
            # Detect environmental changes caused by agents
            env_changes = self._detect_environment_changes(current_obs, next_obs)
            
            # Measure if other agents respond to these changes
            for agent_id, action in enumerate(current_actions):
                
                # Did this agent's action cause an environmental change?
                if self._action_causes_change(action, env_changes):
                    
                    # Do other agents respond in subsequent timesteps?
                    response_detected = self._detect_coordinated_response(
                        agent_id, t, actions, positions, env_changes
                    )
                    
                    if response_detected:
                        communication_events += 1
                    
                    total_opportunities += 1
        
        return communication_events / total_opportunities if total_opportunities > 0 else 0.0
    
    def _compute_role_specialization(self,
                                   actions: List[List[int]],
                                   positions: List[List[Tuple[int, int]]]) -> float:
        """Measure behavioral differentiation between agents"""
        
        if not actions or len(actions[0]) < 2:
            return 0.0
        
        num_agents = len(actions[0])
        
        # Compute action distributions for each agent
        agent_action_dists = []
        
        for agent_id in range(num_agents):
            agent_actions = [actions[t][agent_id] for t in range(len(actions))]
            action_counts = np.bincount(agent_actions, minlength=14)  # 14 possible actions
            action_dist = action_counts / len(agent_actions)
            agent_action_dists.append(action_dist)
        
        # Compute pairwise Jensen-Shannon divergence (specialization metric)
        specialization_scores = []
        
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                
                # Jensen-Shannon divergence between action distributions
                dist_i = agent_action_dists[i]
                dist_j = agent_action_dists[j]
                
                # Avoid log(0) by adding small epsilon
                eps = 1e-10
                dist_i = dist_i + eps
                dist_j = dist_j + eps
                
                # Normalize
                dist_i = dist_i / dist_i.sum()
                dist_j = dist_j / dist_j.sum()
                
                # Jensen-Shannon divergence
                m = 0.5 * (dist_i + dist_j)
                js_div = 0.5 * np.sum(dist_i * np.log(dist_i / m)) + 0.5 * np.sum(dist_j * np.log(dist_j / m))
                
                specialization_scores.append(js_div)
        
        return np.mean(specialization_scores) if specialization_scores else 0.0
    
    def _compute_collective_intelligence(self,
                                       actions: List[List[int]],
                                       rewards: List[float],
                                       positions: List[List[Tuple[int, int]]]) -> float:
        """Measure emergent problem-solving beyond individual capabilities"""
        
        if len(rewards) < 10:  # Need sufficient data
            return 0.0
        
        # 1. Task completion efficiency
        total_reward = sum(rewards)
        reward_trend = np.polyfit(range(len(rewards)), rewards, 1)[0]  # Slope of reward curve
        
        # 2. Coordination complexity (unique coordination patterns)
        coordination_patterns = set()
        
        for t in range(len(actions) - 1):
            if t < len(positions) and len(positions[t]) == len(actions[t]):
                # Create coordination signature
                pattern = []
                for i, (action, pos) in enumerate(zip(actions[t], positions[t])):
                    pattern.append((action, pos[0] // 3, pos[1] // 3))  # Coarse-grained position
                coordination_patterns.add(tuple(sorted(pattern)))
        
        pattern_diversity = len(coordination_patterns) / len(actions)
        
        # 3. Emergent efficiency (performance beyond sum of parts)
        # Estimate individual agent capability
        individual_baseline = 0.1  # Assumed single-agent performance
        num_agents = len(actions[0]) if actions else 1
        expected_combined = individual_baseline * num_agents
        
        emergent_bonus = max(0, (total_reward / len(rewards)) - expected_combined)
        
        # Combine metrics
        collective_score = (
            0.4 * min(1.0, max(0.0, reward_trend + 0.5)) +  # Normalize trend
            0.3 * min(1.0, pattern_diversity) +
            0.3 * min(1.0, emergent_bonus * 5.0)  # Scale emergent bonus
        )
        
        return collective_score
    
    def _compute_spatial_coordination(self, positions: List[List[Tuple[int, int]]]) -> float:
        """Analyze spatial organization patterns"""
        
        if not positions or len(positions[0]) < 2:
            return 0.0
        
        spatial_scores = []
        
        for t in range(len(positions)):
            agent_positions = positions[t]
            
            # Formation compactness
            if len(agent_positions) >= 2:
                center = np.mean(agent_positions, axis=0)
                distances = [np.linalg.norm(np.array(pos) - center) for pos in agent_positions]
                compactness = 1.0 / (1.0 + np.std(distances))  # Reward tight formations
            else:
                compactness = 0.0
            
            # Coverage efficiency (spread across important areas)
            unique_regions = len(set((pos[0]//3, pos[1]//3) for pos in agent_positions))
            max_regions = min(9, len(agent_positions))  # 3x3 grid regions
            coverage = unique_regions / max_regions if max_regions > 0 else 0.0
            
            spatial_score = 0.6 * compactness + 0.4 * coverage
            spatial_scores.append(spatial_score)
        
        return np.mean(spatial_scores)
    
    def _compute_temporal_synchronization(self, actions: List[List[int]]) -> float:
        """Measure timing synchronization in multi-agent actions"""
        
        if len(actions) < 5 or len(actions[0]) < 2:
            return 0.0
        
        sync_scores = []
        
        # Sliding window analysis
        window_size = 5
        for start in range(len(actions) - window_size + 1):
            window_actions = actions[start:start + window_size]
            
            # Measure cross-correlation between agent action sequences
            num_agents = len(window_actions[0])
            correlations = []
            
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    seq_i = [window_actions[t][i] for t in range(window_size)]
                    seq_j = [window_actions[t][j] for t in range(window_size)]
                    
                    # Simple correlation measure
                    correlation = np.corrcoef(seq_i, seq_j)[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(abs(correlation))
            
            if correlations:
                sync_scores.append(np.mean(correlations))
        
        return np.mean(sync_scores) if sync_scores else 0.0
    
    def _compute_emergence_score(self, coordination: float, specialization: float, 
                               intelligence: float) -> float:
        """Compute overall emergence score combining multiple metrics"""
        
        # Weighted combination emphasizing balance
        emergence = (
            0.4 * coordination +           # Coordination is fundamental
            0.3 * specialization +         # Specialization shows complexity
            0.3 * intelligence             # Intelligence shows effectiveness
        )
        
        # Bonus for balanced development (no single metric dominates)
        balance_bonus = 1.0 - np.std([coordination, specialization, intelligence])
        emergence *= (1.0 + 0.2 * max(0, balance_bonus))
        
        return min(1.0, emergence)
    
    # Helper methods
    def _compute_action_complementarity(self, action_i: int, action_j: int) -> float:
        """Compute how well two actions complement each other"""
        
        # Define action types (simplified)
        movement_actions = set(range(8))  # 0-7 are movement
        interaction_actions = set(range(8, 14))  # 8-13 are interactions
        
        # Complementarity rules
        if action_i in movement_actions and action_j in interaction_actions:
            return 0.5  # Movement + interaction is complementary
        elif action_i in interaction_actions and action_j in movement_actions:
            return 0.5
        else:
            return 0.0
    
    def _detect_environment_changes(self, obs1: np.ndarray, obs2: np.ndarray) -> np.ndarray:
        """Detect changes in environment state"""
        return np.abs(obs1 - obs2) > 0.1  # Simple change detection
    
    def _action_causes_change(self, action: int, env_changes: np.ndarray) -> bool:
        """Check if action type typically causes environmental changes"""
        interaction_actions = set(range(8, 14))
        return action in interaction_actions and np.any(env_changes)
    
    def _detect_coordinated_response(self, 
                                   acting_agent: int,
                                   timestep: int,
                                   all_actions: List[List[int]],
                                   positions: List[List[Tuple[int, int]]],
                                   env_changes: np.ndarray) -> bool:
        """Detect if other agents respond to environmental change"""
        
        look_ahead = 3  # Check next 3 timesteps
        
        for t in range(timestep + 1, min(timestep + look_ahead + 1, len(all_actions))):
            for agent_id, action in enumerate(all_actions[t]):
                if agent_id != acting_agent:
                    # Check if this agent's action is a plausible response
                    if action in range(8, 14):  # Interaction actions
                        return True
        
        return False
    
    def get_emergence_report(self) -> Dict[str, Any]:
        """Generate comprehensive emergent behavior report"""
        
        if not any(self.behavior_metrics.values()):
            return {'status': 'insufficient_data'}
        
        report = {
            'summary_statistics': {},
            'trends': {},
            'emergence_classification': '',
            'recommendations': []
        }
        
        # Summary statistics
        for metric_name, values in self.behavior_metrics.items():
            if values:
                report['summary_statistics'][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'recent_avg': np.mean(values[-10:]) if len(values) >= 10 else np.mean(values)
                }
        
        # Trend analysis
        for metric_name, values in self.behavior_metrics.items():
            if len(values) >= 5:
                trend_slope = np.polyfit(range(len(values)), values, 1)[0]
                report['trends'][metric_name] = {
                    'slope': trend_slope,
                    'direction': 'increasing' if trend_slope > 0.01 else 'decreasing' if trend_slope < -0.01 else 'stable'
                }
        
        # Emergence classification
        avg_emergence = report['summary_statistics'].get('emergence_score', {}).get('mean', 0.0)
        
        if avg_emergence > 0.7:
            classification = 'strong_emergence'
        elif avg_emergence > 0.4:
            classification = 'moderate_emergence'
        elif avg_emergence > 0.2:
            classification = 'weak_emergence'
        else:
            classification = 'minimal_emergence'
            
        report['emergence_classification'] = classification
        
        # Recommendations
        if avg_emergence < 0.3:
            report['recommendations'].append("Consider increasing environment complexity to encourage coordination")
        
        coord_avg = report['summary_statistics'].get('coordination_index', {}).get('mean', 0.0)
        if coord_avg < 0.4:
            report['recommendations'].append("Implement reward shaping to encourage coordinated behaviors")
        
        return report


class MetaLearningTaskGenerator:
    """Generate diverse meta-learning tasks for comprehensive evaluation"""
    
    def __init__(self):
        self.task_templates = {
            'resource_efficiency': self._create_resource_efficiency_tasks,
            'spatial_coordination': self._create_spatial_coordination_tasks,  
            'temporal_sequencing': self._create_temporal_sequencing_tasks,
            'role_specialization': self._create_role_specialization_tasks,
            'adaptive_cooperation': self._create_adaptive_cooperation_tasks
        }
        
        self.logger = logging.getLogger(__name__)
    
    def generate_task_suite(self, num_tasks_per_type: int = 3) -> List[MetaTaskConfig]:
        """Generate comprehensive suite of meta-learning tasks"""
        
        self.logger.info(f"🎯 Generating meta-learning task suite: {num_tasks_per_type} tasks per type")
        
        all_tasks = []
        
        for task_type, generator_func in self.task_templates.items():
            tasks = generator_func(num_tasks_per_type)
            all_tasks.extend(tasks)
            
            self.logger.info(f"Generated {len(tasks)} {task_type} tasks")
        
        self.logger.info(f"✅ Total tasks generated: {len(all_tasks)}")
        return all_tasks
    
    def _create_resource_efficiency_tasks(self, num_tasks: int) -> List[MetaTaskConfig]:
        """Create tasks focused on efficient resource gathering"""
        
        tasks = []
        
        for i in range(num_tasks):
            # Vary resource distribution and scarcity
            resource_density = 0.05 + (i * 0.05)  # 5%, 10%, 15%
            
            task = MetaTaskConfig(
                task_name=f'resource_efficiency_{i+1}',
                environment_params={
                    'grid_size': [15, 15],
                    'max_agents': 2,
                    'max_resources': max(4, int(225 * resource_density)),
                    'resource_respawn': True,
                    'resource_respawn_rate': 0.01
                },
                success_criteria={'min_improvement': 0.15},
                adaptation_steps=5
            )
            
            tasks.append(task)
        
        return tasks
    
    def _create_spatial_coordination_tasks(self, num_tasks: int) -> List[MetaTaskConfig]:
        """Create tasks requiring spatial coordination"""
        
        tasks = []
        
        for i in range(num_tasks):
            # Vary grid size and agent count
            grid_size = 10 + (i * 3)  # 10x10, 13x13, 16x16
            num_agents = 2 + i  # 2, 3, 4 agents
            
            task = MetaTaskConfig(
                task_name=f'spatial_coordination_{i+1}',
                environment_params={
                    'grid_size': [grid_size, grid_size],
                    'max_agents': num_agents,
                    'max_resources': 6,
                    'building_reward_bonus': 2.0  # Encourage building cooperation
                },
                success_criteria={'min_improvement': 0.20},
                adaptation_steps=7
            )
            
            tasks.append(task)
        
        return tasks
    
    def _create_temporal_sequencing_tasks(self, num_tasks: int) -> List[MetaTaskConfig]:
        """Create tasks requiring temporal coordination"""
        
        tasks = []
        
        for i in range(num_tasks):
            # Vary episode length and urgency
            max_steps = 200 + (i * 100)  # 200, 300, 400 steps
            
            task = MetaTaskConfig(
                task_name=f'temporal_sequencing_{i+1}',
                environment_params={
                    'grid_size': [12, 12],
                    'max_agents': 3,
                    'max_resources': 5,
                    'max_steps': max_steps,
                    'time_pressure': True
                },
                success_criteria={'min_improvement': 0.25},
                adaptation_steps=6
            )
            
            tasks.append(task)
        
        return tasks
    
    def _create_role_specialization_tasks(self, num_tasks: int) -> List[MetaTaskConfig]:
        """Create tasks encouraging role differentiation"""
        
        tasks = []
        
        for i in range(num_tasks):
            # Different specialization pressures
            specialization_bonus = 1.0 + (i * 0.5)  # 1.0, 1.5, 2.0
            
            task = MetaTaskConfig(
                task_name=f'role_specialization_{i+1}',
                environment_params={
                    'grid_size': [15, 15],
                    'max_agents': 4,
                    'max_resources': 8,
                    'role_specialization_bonus': specialization_bonus,
                    'diverse_action_reward': True
                },
                success_criteria={'min_improvement': 0.18},
                adaptation_steps=8
            )
            
            tasks.append(task)
        
        return tasks
    
    def _create_adaptive_cooperation_tasks(self, num_tasks: int) -> List[MetaTaskConfig]:
        """Create tasks requiring adaptive cooperation strategies"""
        
        tasks = []
        
        for i in range(num_tasks):
            # Dynamic environments that change during episodes
            change_frequency = 50 + (i * 25)  # Change every 50, 75, 100 steps
            
            task = MetaTaskConfig(
                task_name=f'adaptive_cooperation_{i+1}',
                environment_params={
                    'grid_size': [12, 12],
                    'max_agents': 3,
                    'max_resources': 6,
                    'dynamic_environment': True,
                    'change_frequency': change_frequency,
                    'cooperation_bonus': 1.5
                },
                success_criteria={'min_improvement': 0.30},
                adaptation_steps=10
            )
            
            tasks.append(task)
        
        return tasks