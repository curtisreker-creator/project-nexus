"""
Project NEXUS - Self-Play Trainer Implementation
Competitive Multi-Agent Training with ELO Rating System

File: agents/training/self_play_trainer.py
"""

import torch
import numpy as np
import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import time
from pathlib import Path

from .ppo_trainer import PPOTrainer
from ..networks.network_factory import create_standard_network


@dataclass
class MatchResult:
    """Result of a competitive match"""
    team_a_agents: List[int]
    team_b_agents: List[int]
    team_a_score: float
    team_b_score: float
    winner: str  # 'team_a', 'team_b', or 'draw'
    match_duration: int
    total_resources: Dict[str, int]
    buildings_built: Dict[str, int]
    timestamp: float


@dataclass
class AgentProfile:
    """Profile for self-play agent"""
    agent_id: str
    elo_rating: float
    matches_played: int
    wins: int
    losses: int
    draws: int
    total_score: float
    strategy_signature: Optional[str] = None
    last_updated: float = 0.0


class ELOSystem:
    """ELO rating system for agent skill assessment"""

    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1200.0):
        self.k_factor = k_factor
        self.initial_rating = initial_rating

    def calculate_expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected win probability for team A"""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update_ratings(self, team_a_rating: float, team_b_rating: float,
                       actual_score: float) -> Tuple[float, float]:
        """
        Update ELO ratings based on match result
        actual_score: 1.0 = team_a wins, 0.0 = team_b wins, 0.5 = draw
        """
        expected_a = self.calculate_expected_score(team_a_rating, team_b_rating)
        expected_b = 1.0 - expected_a

        new_rating_a = team_a_rating + self.k_factor * (actual_score - expected_a)
        new_rating_b = team_b_rating + self.k_factor * ((1.0 - actual_score) - expected_b)

        return new_rating_a, new_rating_b


class AgentPool:
    """Manage diverse population of trained agents"""

    def __init__(self, max_size: int = 100, diversity_threshold: float = 0.1):
        self.max_size = max_size
        self.diversity_threshold = diversity_threshold
        self.agents: Dict[str, AgentProfile] = {}
        self.agent_networks: Dict[str, torch.nn.Module] = {}
        self.elo_system = ELOSystem()

    def add_agent(self, agent_id: str, network: torch.nn.Module,
                  initial_rating: Optional[float] = None) -> None:
        """Add new agent to pool"""
        rating = initial_rating or self.elo_system.initial_rating

        profile = AgentProfile(
            agent_id=agent_id,
            elo_rating=rating,
            matches_played=0,
            wins=0,
            losses=0,
            draws=0,
            total_score=0.0,
            last_updated=time.time()
        )

        self.agents[agent_id] = profile
        self.agent_networks[agent_id] = network.clone() if hasattr(network, 'clone') else network

    def get_opponents(self, agent_id: str, count: int = 1) -> List[str]:
        """Get suitable opponents for agent based on ELO rating"""
        if agent_id not in self.agents:
            return []

        agent_rating = self.agents[agent_id].elo_rating

        # Find agents with similar ratings (within 200 ELO points)
        candidates = []
        for other_id, profile in self.agents.items():
            if other_id != agent_id and abs(profile.elo_rating - agent_rating) < 200:
                candidates.append((other_id, abs(profile.elo_rating - agent_rating)))

        # Sort by rating similarity and return top candidates
        candidates.sort(key=lambda x: x[1])
        return [agent_id for agent_id, _ in candidates[:count]]

    def update_after_match(self, match_result: MatchResult) -> None:
        """Update agent profiles after match"""
        # Calculate team ratings
        team_a_rating = np.mean([self.agents[aid].elo_rating
                                 for aid in match_result.team_a_agents if aid in self.agents])
        team_b_rating = np.mean([self.agents[aid].elo_rating
                                 for aid in match_result.team_b_agents if aid in self.agents])

        # Determine actual score
        if match_result.winner == 'team_a':
            actual_score = 1.0
        elif match_result.winner == 'team_b':
            actual_score = 0.0
        else:
            actual_score = 0.5

        # Update ratings
        new_rating_a, new_rating_b = self.elo_system.update_ratings(
            team_a_rating, team_b_rating, actual_score)

        # Apply rating changes to individual agents
        rating_change_a = new_rating_a - team_a_rating
        rating_change_b = new_rating_b - team_b_rating

        # Update team A agents
        for agent_id in match_result.team_a_agents:
            if agent_id in self.agents:
                profile = self.agents[agent_id]
                profile.elo_rating += rating_change_a
                profile.matches_played += 1
                profile.total_score += match_result.team_a_score

                if match_result.winner == 'team_a':
                    profile.wins += 1
                elif match_result.winner == 'team_b':
                    profile.losses += 1
                else:
                    profile.draws += 1

                profile.last_updated = time.time()

        # Update team B agents
        for agent_id in match_result.team_b_agents:
            if agent_id in self.agents:
                profile = self.agents[agent_id]
                profile.elo_rating += rating_change_b
                profile.matches_played += 1
                profile.total_score += match_result.team_b_score

                if match_result.winner == 'team_b':
                    profile.wins += 1
                elif match_result.winner == 'team_a':
                    profile.losses += 1
                else:
                    profile.draws += 1

                profile.last_updated = time.time()

    def get_leaderboard(self, top_k: int = 10) -> List[AgentProfile]:
        """Get top agents by ELO rating"""
        sorted_agents = sorted(self.agents.values(),
                               key=lambda x: x.elo_rating, reverse=True)
        return sorted_agents[:top_k]


class SelfPlayTrainer(PPOTrainer):
    """
    Self-Play Trainer for competitive multi-agent learning
    Extends PPO trainer with competitive matchmaking and ELO rating
    """

    def __init__(self, config: Dict[str, Any], device: torch.device,
                 visual_callback: Optional[callable] = None):
        super().__init__(config, device)

        # Self-play configuration
        self.competitive_mode = config.get('competitive_mode', 'resource_gathering')
        self.match_duration = config.get('match_duration', 1000)  # steps
        self.matches_per_update = config.get('matches_per_update', 10)
        self.population_size = config.get('population_size', 50)

        # Agent pool and matchmaking
        self.agent_pool = AgentPool(max_size=self.population_size)
        self.match_history: List[MatchResult] = []
        self.current_generation = 0

        # Visual analytics integration
        self.visual_callback = visual_callback
        self.metrics_buffer = deque(maxlen=1000)

        # Performance tracking
        self.match_stats = defaultdict(list)

        # Initialize with base agent
        self._initialize_base_agents()

        self.logger.info("🎪 SelfPlayTrainer initialized - Competitive MARL ready!")
        self.logger.info(f"🎯 Mode: {self.competitive_mode}")
        self.logger.info(f"⚔️  Population: {self.population_size} agents")

    def _initialize_base_agents(self) -> None:
        """Initialize base agents for self-play"""
        # Create diverse starting population
        for i in range(min(4, self.population_size)):
            agent_id = f"base_agent_{i}"
            network = create_standard_network()

            # Add slight variations to create initial diversity
            if i > 0:
                with torch.no_grad():
                    for param in network.parameters():
                        param.add_(torch.randn_like(param) * 0.01)

            self.agent_pool.add_agent(agent_id, network)

        self.logger.info(f"✅ Initialized {len(self.agent_pool.agents)} base agents")

    async def run_self_play_match(self, team_a_agents: List[str],
                                  team_b_agents: List[str]) -> MatchResult:
        """Run single competitive match between agent teams"""

        # Create competitive environment
        env_config = self.config.copy()
        env_config['n_agents'] = len(team_a_agents) + len(team_b_agents)
        env_config['competitive_mode'] = True
        env_config['max_steps'] = self.match_duration

        # Initialize environment with team assignments
        from environment.grid_world import GridWorld
        env = GridWorld(**env_config)
        obs, info = env.reset()

        # Get networks for teams
        team_a_networks = [self.agent_pool.agent_networks[aid] for aid in team_a_agents]
        team_b_networks = [self.agent_pool.agent_networks[aid] for aid in team_b_agents]

        # Match simulation
        team_a_total_reward = 0.0
        team_b_total_reward = 0.0
        steps = 0

        start_time = time.time()

        while steps < self.match_duration:
            # Get actions from both teams
            actions = []

            # Team A actions
            for i, network in enumerate(team_a_networks):
                obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0).to(self.device)
                action_probs, _ = network(obs_tensor)
                action = torch.multinomial(action_probs, 1).item()
                actions.append(action)

            # Team B actions
            for i, network in enumerate(team_b_networks):
                agent_idx = len(team_a_agents) + i
                obs_tensor = torch.FloatTensor(obs[agent_idx]).unsqueeze(0).to(self.device)
                action_probs, _ = network(obs_tensor)
                action = torch.multinomial(action_probs, 1).item()
                actions.append(action)

            # Environment step
            obs, rewards, terminated, truncated, info = env.step(actions)

            # Accumulate rewards by team
            team_a_total_reward += sum(rewards[:len(team_a_agents)])
            team_b_total_reward += sum(rewards[len(team_a_agents):])

            steps += 1

            # Visual callback for real-time display
            if self.visual_callback:
                await self.visual_callback({
                    'env_state': env.get_state(),
                    'team_a_agents': team_a_agents,
                    'team_b_agents': team_b_agents,
                    'team_a_score': team_a_total_reward,
                    'team_b_score': team_b_total_reward,
                    'step': steps
                })

            if terminated or truncated:
                break

        # Determine winner
        if team_a_total_reward > team_b_total_reward:
            winner = 'team_a'
        elif team_b_total_reward > team_a_total_reward:
            winner = 'team_b'
        else:
            winner = 'draw'

        # Create match result
        match_result = MatchResult(
            team_a_agents=team_a_agents,
            team_b_agents=team_b_agents,
            team_a_score=team_a_total_reward,
            team_b_score=team_b_total_reward,
            winner=winner,
            match_duration=steps,
            total_resources=info.get('total_resources', {}),
            buildings_built=info.get('buildings_built', {}),
            timestamp=time.time()
        )

        self.match_history.append(match_result)
        self.agent_pool.update_after_match(match_result)

        # Update metrics
        match_time = time.time() - start_time
        self.match_stats['match_duration'].append(match_time)
        self.match_stats['total_reward'].append(team_a_total_reward + team_b_total_reward)

        self.logger.info(f"🏆 Match complete: {winner} wins! "
                         f"({team_a_total_reward:.1f} vs {team_b_total_reward:.1f}) "
                         f"in {steps} steps")

        return match_result

    async def run_training_session(self, num_matches: int = 100) -> Dict[str, Any]:
        """Run full self-play training session"""

        self.logger.info(f"🚀 Starting self-play training: {num_matches} matches")

        session_start = time.time()
        matches_completed = 0

        # Training loop
        for match_num in range(num_matches):
            # Select teams for competition
            available_agents = list(self.agent_pool.agents.keys())

            if len(available_agents) >= 4:
                # Random team assignment for now
                np.random.shuffle(available_agents)
                team_a = available_agents[:2]
                team_b = available_agents[2:4]

                # Run competitive match
                match_result = await self.run_self_play_match(team_a, team_b)
                matches_completed += 1

                # Periodic logging
                if matches_completed % 10 == 0:
                    leaderboard = self.agent_pool.get_leaderboard(top_k=5)
                    top_agent = leaderboard[0] if leaderboard else None

                    if top_agent:
                        self.logger.info(f"📊 Progress: {matches_completed}/{num_matches} matches")
                        self.logger.info(f"🥇 Top agent: {top_agent.agent_id} "
                                         f"(ELO: {top_agent.elo_rating:.0f}, "
                                         f"Record: {top_agent.wins}-{top_agent.losses}-{top_agent.draws})")

                # Population management
                if matches_completed % 20 == 0:
                    await self._update_population()

            else:
                self.logger.warning("Not enough agents for competitive matches")
                break

        # Session summary
        session_duration = time.time() - session_start

        summary = {
            'matches_completed': matches_completed,
            'session_duration': session_duration,
            'matches_per_minute': matches_completed / (session_duration / 60.0),
            'final_leaderboard': [asdict(agent) for agent in self.agent_pool.get_leaderboard()],
            'average_match_duration': np.mean(self.match_stats['match_duration']),
            'total_agents': len(self.agent_pool.agents)
        }

        self.logger.info(f"🎪 Self-play session complete!")
        self.logger.info(f"⚡ Performance: {summary['matches_per_minute']:.1f} matches/min")
        self.logger.info(f"🏆 Total agents: {summary['total_agents']}")

        return summary

    async def _update_population(self) -> None:
        """Update agent population - add new agents, remove stagnant ones"""

        # Get current best agents
        leaderboard = self.agent_pool.get_leaderboard()

        if len(leaderboard) >= 2:
            # Create new agent by mixing top performers
            best_agent_1 = leaderboard[0]
            best_agent_2 = leaderboard[1]

            new_agent_id = f"gen_{self.current_generation}_hybrid"

            # Simple network mixing (average weights)
            network_1 = self.agent_pool.agent_networks[best_agent_1.agent_id]
            network_2 = self.agent_pool.agent_networks[best_agent_2.agent_id]

            new_network = create_standard_network()

            # Average parameters
            with torch.no_grad():
                for (param_new, param_1, param_2) in zip(new_network.parameters(),
                                                         network_1.parameters(),
                                                         network_2.parameters()):
                    param_new.copy_((param_1 + param_2) / 2.0)
                    # Add mutation
                    param_new.add_(torch.randn_like(param_new) * 0.01)

            # Add to population
            initial_rating = (best_agent_1.elo_rating + best_agent_2.elo_rating) / 2.0
            self.agent_pool.add_agent(new_agent_id, new_network, initial_rating)

            self.current_generation += 1

            self.logger.info(f"🧬 Created new hybrid agent: {new_agent_id} "
                             f"(ELO: {initial_rating:.0f})")

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics for dashboard"""

        if not self.match_history:
            return {}

        recent_matches = self.match_history[-50:]  # Last 50 matches

        # Calculate metrics
        win_rates = defaultdict(int)
        for match in recent_matches:
            win_rates[match.winner] += 1

        leaderboard = self.agent_pool.get_leaderboard(top_k=10)

        return {
            'timestamp': time.time(),
            'total_matches': len(self.match_history),
            'recent_win_rates': dict(win_rates),
            'active_agents': len(self.agent_pool.agents),
            'current_generation': self.current_generation,
            'leaderboard': [
                {
                    'agent_id': agent.agent_id,
                    'elo_rating': agent.elo_rating,
                    'matches_played': agent.matches_played,
                    'win_rate': (agent.wins / max(agent.matches_played, 1)) * 100
                }
                for agent in leaderboard
            ],
            'performance_stats': {
                'avg_match_duration': np.mean(self.match_stats['match_duration']) if self.match_stats[
                    'match_duration'] else 0,
                'matches_per_hour': len(recent_matches) / max(1, (
                            time.time() - recent_matches[0].timestamp) / 3600) if recent_matches else 0
            }
        }