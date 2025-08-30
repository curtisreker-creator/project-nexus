# File: research/analytics/insights_engine.py
# Analytics & Insights Engine (AIE) Subsystem
# Real-time Research Analytics and Competitive Intelligence

import asyncio
import logging
import numpy as np
import pandas as pd
import torch
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Project NEXUS imports
from environment.grid_world import GridWorld
from agents.base_agent import BaseAgent
from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentMetrics:
    """Container for experiment performance metrics"""
    experiment_id: str
    timestamp: datetime
    episode_rewards: List[float]
    episode_lengths: List[int]
    success_rate: float
    convergence_time: Optional[float]
    final_performance: float
    training_stability: float
    resource_utilization: Dict[str, float]
    agent_coordination_score: float
    communication_efficiency: float
    exploration_coverage: float


@dataclass
class PerformanceInsight:
    """Individual performance insight with actionable recommendations"""
    insight_type: str  # "performance", "stability", "efficiency", "coordination"
    severity: str      # "critical", "warning", "info", "success"
    title: str
    description: str
    evidence: Dict[str, Any]
    recommendations: List[str]
    confidence: float  # 0.0 to 1.0
    impact_score: float  # 0.0 to 10.0


@dataclass
class CompetitiveAnalysis:
    """Competitive intelligence analysis results"""
    benchmark_comparison: Dict[str, float]
    performance_ranking: int
    strengths: List[str]
    weaknesses: List[str]
    opportunities: List[str]
    threats: List[str]
    market_position: str  # "leader", "challenger", "follower", "niche"


class MetricsCollector:
    """Real-time metrics collection and aggregation"""
    
    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.real_time_stats = defaultdict(lambda: deque(maxlen=1000))
        
    def record_step_metrics(self, agent_id: int, step: int, metrics: Dict[str, float]):
        """Record metrics for a single training step"""
        timestamp = datetime.now()
        
        # Buffer for batch processing
        self.metrics_buffer.append({
            'timestamp': timestamp,
            'agent_id': agent_id,
            'step': step,
            'metrics': metrics
        })
        
        # Real-time stats for immediate insights
        for metric_name, value in metrics.items():
            self.real_time_stats[f"agent_{agent_id}_{metric_name}"].append(value)
    
    def get_recent_performance(self, lookback_steps: int = 100) -> Dict[str, float]:
        """Get recent performance statistics"""
        if not self.metrics_buffer:
            return {}
            
        recent_data = list(self.metrics_buffer)[-lookback_steps:]
        
        # Aggregate across agents and steps
        performance_stats = {}
        
        # Collect all metrics
        all_rewards = []
        all_lengths = []
        
        for entry in recent_data:
            metrics = entry['metrics']
            if 'reward' in metrics:
                all_rewards.append(metrics['reward'])
            if 'episode_length' in metrics:
                all_lengths.append(metrics['episode_length'])
        
        if all_rewards:
            performance_stats['mean_reward'] = np.mean(all_rewards)
            performance_stats['reward_std'] = np.std(all_rewards)
            performance_stats['reward_trend'] = self._calculate_trend(all_rewards)
        
        if all_lengths:
            performance_stats['mean_episode_length'] = np.mean(all_lengths)
            performance_stats['length_stability'] = 1.0 / (1.0 + np.std(all_lengths))
            
        return performance_stats


class PerformanceAnalyzer:
    """Advanced performance analysis and pattern detection"""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.benchmark_data = self._load_benchmarks()
        
    def analyze_experiment(self, metrics: ExperimentMetrics) -> List[PerformanceInsight]:
        """Comprehensive analysis of experiment performance"""
        insights = []
        
        # Performance analysis
        insights.extend(self._analyze_reward_performance(metrics))
        insights.extend(self._analyze_stability(metrics))
        insights.extend(self._analyze_efficiency(metrics))
        insights.extend(self._analyze_coordination(metrics))
        insights.extend(self._analyze_convergence(metrics))
        
        # Sort by impact score (highest first)
        insights.sort(key=lambda x: x.impact_score, reverse=True)
        
        return insights
    
    def _analyze_reward_performance(self, metrics: ExperimentMetrics) -> List[PerformanceInsight]:
        """Analyze reward performance patterns"""
        insights = []
        rewards = np.array(metrics.episode_rewards)
        
        if len(rewards) < 10:
            return insights
            
        # Performance level analysis
        mean_reward = np.mean(rewards)
        final_performance = np.mean(rewards[-10:]) if len(rewards) >= 10 else mean_reward
        
        # Compare against benchmarks
        benchmark_reward = self.benchmark_data.get('baseline_reward', 0.0)
        
        if final_performance > benchmark_reward * 1.2:
            insights.append(PerformanceInsight(
                insight_type="performance",
                severity="success",
                title="Exceptional Performance Achieved",
                description=f"Final performance ({final_performance:.3f}) exceeds benchmark by {((final_performance/benchmark_reward - 1) * 100):.1f}%",
                evidence={
                    "final_performance": final_performance,
                    "benchmark": benchmark_reward,
                    "improvement": final_performance / benchmark_reward
                },
                recommendations=[
                    "Document successful configuration for future experiments",
                    "Consider this as new performance baseline",
                    "Investigate which components contributed most to success"
                ],
                confidence=0.95,
                impact_score=9.0
            ))
        elif final_performance < benchmark_reward * 0.8:
            insights.append(PerformanceInsight(
                insight_type="performance",
                severity="warning",
                title="Underperforming Against Benchmark",
                description=f"Final performance ({final_performance:.3f}) is {((1 - final_performance/benchmark_reward) * 100):.1f}% below benchmark",
                evidence={
                    "final_performance": final_performance,
                    "benchmark": benchmark_reward,
                    "gap": benchmark_reward - final_performance
                },
                recommendations=[
                    "Review hyperparameter settings",
                    "Increase training duration or sample efficiency",
                    "Consider different neural architecture",
                    "Check for implementation bugs"
                ],
                confidence=0.90,
                impact_score=7.5
            ))
        
        # Learning progress analysis
        if len(rewards) >= 50:
            early_performance = np.mean(rewards[:10])
            late_performance = np.mean(rewards[-10:])
            learning_rate = (late_performance - early_performance) / len(rewards)
            
            if learning_rate < 0.001:  # Very slow learning
                insights.append(PerformanceInsight(
                    insight_type="performance",
                    severity="warning",
                    title="Slow Learning Progress",
                    description=f"Learning rate ({learning_rate:.6f}) indicates slow improvement",
                    evidence={
                        "learning_rate": learning_rate,
                        "early_performance": early_performance,
                        "late_performance": late_performance
                    },
                    recommendations=[
                        "Increase learning rate",
                        "Improve reward shaping",
                        "Add curriculum learning",
                        "Check exploration strategy"
                    ],
                    confidence=0.85,
                    impact_score=6.0
                ))
        
        return insights
    
    def _analyze_stability(self, metrics: ExperimentMetrics) -> List[PerformanceInsight]:
        """Analyze training stability"""
        insights = []
        rewards = np.array(metrics.episode_rewards)
        
        if len(rewards) < 20:
            return insights
            
        # Calculate stability metrics
        stability = metrics.training_stability
        reward_variance = np.var(rewards)
        
        # Rolling window stability
        window_size = min(50, len(rewards) // 4)
        rolling_means = [np.mean(rewards[i:i+window_size]) 
                        for i in range(len(rewards) - window_size + 1)]
        rolling_variance = np.var(rolling_means)
        
        if stability < 0.6:  # Low stability threshold
            insights.append(PerformanceInsight(
                insight_type="stability",
                severity="warning" if stability < 0.4 else "info",
                title="Training Instability Detected",
                description=f"Training stability score ({stability:.3f}) indicates high variance",
                evidence={
                    "stability_score": stability,
                    "reward_variance": reward_variance,
                    "rolling_variance": rolling_variance
                },
                recommendations=[
                    "Reduce learning rate for more stable updates",
                    "Increase batch size to reduce gradient noise",
                    "Add gradient clipping",
                    "Use learning rate scheduling",
                    "Consider different optimizer (e.g., Adam -> AdamW)"
                ],
                confidence=0.80,
                impact_score=7.0 if stability < 0.4 else 5.0
            ))
        elif stability > 0.85:
            insights.append(PerformanceInsight(
                insight_type="stability",
                severity="success",
                title="Excellent Training Stability",
                description=f"High stability score ({stability:.3f}) indicates consistent learning",
                evidence={
                    "stability_score": stability,
                    "reward_variance": reward_variance
                },
                recommendations=[
                    "Current settings provide good stability",
                    "Could potentially increase learning rate for faster convergence",
                    "Consider as baseline for future experiments"
                ],
                confidence=0.90,
                impact_score=6.5
            ))
        
        return insights
    
    def _analyze_efficiency(self, metrics: ExperimentMetrics) -> List[PerformanceInsight]:
        """Analyze computational and sample efficiency"""
        insights = []
        
        # Resource utilization analysis
        cpu_util = metrics.resource_utilization.get('cpu_percent', 0)
        memory_util = metrics.resource_utilization.get('memory_percent', 0)
        gpu_util = metrics.resource_utilization.get('gpu_percent', 0)
        
        if cpu_util < 30:
            insights.append(PerformanceInsight(
                insight_type="efficiency",
                severity="info",
                title="Low CPU Utilization",
                description=f"CPU utilization ({cpu_util:.1f}%) suggests potential for parallelization",
                evidence={"cpu_utilization": cpu_util},
                recommendations=[
                    "Increase number of parallel environments",
                    "Enable CPU-based data preprocessing",
                    "Consider vectorized environment implementation"
                ],
                confidence=0.75,
                impact_score=4.0
            ))
        
        if gpu_util > 0 and gpu_util < 50:
            insights.append(PerformanceInsight(
                insight_type="efficiency",
                severity="info",
                title="Suboptimal GPU Utilization",
                description=f"GPU utilization ({gpu_util:.1f}%) indicates room for improvement",
                evidence={"gpu_utilization": gpu_util},
                recommendations=[
                    "Increase batch size to better utilize GPU",
                    "Enable mixed precision training",
                    "Consider larger neural networks",
                    "Implement gradient accumulation"
                ],
                confidence=0.70,
                impact_score=5.5
            ))
        
        # Sample efficiency analysis
        if metrics.convergence_time and len(metrics.episode_rewards) > 0:
            sample_efficiency = metrics.final_performance / len(metrics.episode_rewards)
            
            if sample_efficiency < 0.001:  # Low sample efficiency
                insights.append(PerformanceInsight(
                    insight_type="efficiency",
                    severity="warning",
                    title="Low Sample Efficiency",
                    description="Agent requires many samples to achieve performance",
                    evidence={
                        "sample_efficiency": sample_efficiency,
                        "total_episodes": len(metrics.episode_rewards),
                        "final_performance": metrics.final_performance
                    },
                    recommendations=[
                        "Implement experience replay",
                        "Use more sophisticated exploration strategies",
                        "Add auxiliary learning objectives",
                        "Consider model-based approaches"
                    ],
                    confidence=0.80,
                    impact_score=6.5
                ))
        
        return insights
    
    def _analyze_coordination(self, metrics: ExperimentMetrics) -> List[PerformanceInsight]:
        """Analyze multi-agent coordination effectiveness"""
        insights = []
        
        coordination_score = metrics.agent_coordination_score
        communication_efficiency = metrics.communication_efficiency
        
        if coordination_score < 0.5:
            insights.append(PerformanceInsight(
                insight_type="coordination",
                severity="warning",
                title="Poor Agent Coordination",
                description=f"Coordination score ({coordination_score:.3f}) indicates agents are not working together effectively",
                evidence={
                    "coordination_score": coordination_score,
                    "communication_efficiency": communication_efficiency
                },
                recommendations=[
                    "Improve reward shaping to encourage cooperation",
                    "Add explicit coordination rewards",
                    "Enhance communication mechanisms",
                    "Implement centralized training with decentralized execution",
                    "Consider attention-based agent interactions"
                ],
                confidence=0.85,
                impact_score=8.0
            ))
        elif coordination_score > 0.8:
            insights.append(PerformanceInsight(
                insight_type="coordination",
                severity="success",
                title="Excellent Agent Coordination",
                description=f"High coordination score ({coordination_score:.3f}) shows effective teamwork",
                evidence={
                    "coordination_score": coordination_score,
                    "communication_efficiency": communication_efficiency
                },
                recommendations=[
                    "Current coordination mechanisms are working well",
                    "Consider reducing communication overhead if possible",
                    "Document successful coordination strategies"
                ],
                confidence=0.90,
                impact_score=7.5
            ))
        
        if communication_efficiency < 0.3:
            insights.append(PerformanceInsight(
                insight_type="coordination",
                severity="info",
                title="Low Communication Efficiency",
                description=f"Communication efficiency ({communication_efficiency:.3f}) suggests wasted communication",
                evidence={"communication_efficiency": communication_efficiency},
                recommendations=[
                    "Implement communication attention mechanisms",
                    "Add communication cost to encourage selective messaging",
                    "Prune unnecessary communication channels",
                    "Use information-theoretic communication losses"
                ],
                confidence=0.75,
                impact_score=5.0
            ))
        
        return insights
    
    def _analyze_convergence(self, metrics: ExperimentMetrics) -> List[PerformanceInsight]:
        """Analyze convergence characteristics"""
        insights = []
        
        if metrics.convergence_time is None:
            insights.append(PerformanceInsight(
                insight_type="performance",
                severity="warning",
                title="No Convergence Detected",
                description="Training has not converged within the allocated time",
                evidence={"total_episodes": len(metrics.episode_rewards)},
                recommendations=[
                    "Extend training duration",
                    "Adjust convergence criteria",
                    "Check for proper exploration-exploitation balance",
                    "Review learning rate scheduling"
                ],
                confidence=0.70,
                impact_score=6.0
            ))
        elif metrics.convergence_time < len(metrics.episode_rewards) * 0.3:
            insights.append(PerformanceInsight(
                insight_type="performance",
                severity="success",
                title="Fast Convergence Achieved",
                description=f"Convergence in {metrics.convergence_time:.1f} episodes (early convergence)",
                evidence={
                    "convergence_time": metrics.convergence_time,
                    "total_episodes": len(metrics.episode_rewards)
                },
                recommendations=[
                    "Excellent convergence speed",
                    "Consider reducing training time for efficiency",
                    "Use as baseline for hyperparameter optimization"
                ],
                confidence=0.90,
                impact_score=7.0
            ))
        
        return insights
    
    def _load_benchmarks(self) -> Dict[str, float]:
        """Load benchmark performance data"""
        # In a real implementation, this would load from a database or file
        return {
            'baseline_reward': 0.75,
            'human_performance': 0.95,
            'random_baseline': 0.1,
            'coordination_threshold': 0.6,
            'stability_threshold': 0.7
        }


class CompetitiveIntelligence:
    """Competitive analysis and market positioning"""
    
    def __init__(self):
        self.competitor_data = self._load_competitor_data()
        self.research_trends = self._load_research_trends()
        
    def analyze_competitive_position(self, metrics: ExperimentMetrics) -> CompetitiveAnalysis:
        """Analyze competitive position against state-of-the-art"""
        
        # Benchmark comparison
        benchmark_comparison = self._compare_benchmarks(metrics)
        
        # Calculate ranking
        ranking = self._calculate_ranking(metrics.final_performance)
        
        # SWOT analysis
        strengths, weaknesses = self._analyze_strengths_weaknesses(metrics)
        opportunities, threats = self._analyze_opportunities_threats(metrics)
        
        # Market position
        position = self._determine_market_position(ranking, metrics.final_performance)
        
        return CompetitiveAnalysis(
            benchmark_comparison=benchmark_comparison,
            performance_ranking=ranking,
            strengths=strengths,
            weaknesses=weaknesses,
            opportunities=opportunities,
            threats=threats,
            market_position=position
        )
    
    def _compare_benchmarks(self, metrics: ExperimentMetrics) -> Dict[str, float]:
        """Compare against known benchmarks"""
        return {
            'vs_random_baseline': metrics.final_performance / 0.1,
            'vs_human_performance': metrics.final_performance / 0.95,
            'vs_current_sota': metrics.final_performance / 0.88,
            'vs_coordination_threshold': metrics.agent_coordination_score / 0.6
        }
    
    def _calculate_ranking(self, performance: float) -> int:
        """Calculate ranking against competitive landscape"""
        # Simplified ranking based on performance percentiles
        if performance > 0.9:
            return 1  # Top tier
        elif performance > 0.8:
            return 2  # Strong performer
        elif performance > 0.6:
            return 3  # Competitive
        else:
            return 4  # Below competitive threshold
    
    def _analyze_strengths_weaknesses(self, metrics: ExperimentMetrics) -> Tuple[List[str], List[str]]:
        """Identify strengths and weaknesses"""
        strengths = []
        weaknesses = []
        
        # Analyze various aspects
        if metrics.agent_coordination_score > 0.8:
            strengths.append("Exceptional multi-agent coordination")
        elif metrics.agent_coordination_score < 0.5:
            weaknesses.append("Poor coordination between agents")
        
        if metrics.training_stability > 0.8:
            strengths.append("Highly stable training process")
        elif metrics.training_stability < 0.6:
            weaknesses.append("Unstable training dynamics")
        
        if metrics.communication_efficiency > 0.7:
            strengths.append("Efficient agent communication")
        elif metrics.communication_efficiency < 0.4:
            weaknesses.append("Inefficient communication overhead")
        
        if metrics.exploration_coverage > 0.8:
            strengths.append("Comprehensive environment exploration")
        elif metrics.exploration_coverage < 0.5:
            weaknesses.append("Limited exploration strategy")
        
        return strengths, weaknesses
    
    def _analyze_opportunities_threats(self, metrics: ExperimentMetrics) -> Tuple[List[str], List[str]]:
        """Identify market opportunities and threats"""
        opportunities = [
            "Growing demand for multi-agent AI systems",
            "Potential applications in robotics and autonomous systems",
            "Academic publication opportunities",
            "Open-source community adoption"
        ]
        
        threats = [
            "Rapid advancement by major tech companies",
            "New algorithmic breakthroughs making current approach obsolete",
            "Computational resource requirements limiting adoption",
            "Regulatory challenges for AI deployment"
        ]
        
        # Adjust based on performance
        if metrics.final_performance > 0.85:
            opportunities.append("Position as state-of-the-art solution")
        else:
            threats.append("Risk of being outpaced by competitors")
        
        return opportunities, threats
    
    def _determine_market_position(self, ranking: int, performance: float) -> str:
        """Determine market position"""
        if ranking == 1 and performance > 0.9:
            return "leader"
        elif ranking <= 2 and performance > 0.8:
            return "challenger"
        elif ranking <= 3:
            return "follower"
        else:
            return "niche"
    
    def _load_competitor_data(self) -> Dict[str, Any]:
        """Load competitive landscape data"""
        return {
            'openai_baselines': {'performance': 0.82, 'coordination': 0.7},
            'deepmind_marl': {'performance': 0.88, 'coordination': 0.85},
            'academic_benchmarks': {'performance': 0.75, 'coordination': 0.6}
        }
    
    def _load_research_trends(self) -> Dict[str, Any]:
        """Load current research trends"""
        return {
            'hot_topics': ['attention mechanisms', 'curriculum learning', 'meta-learning'],
            'emerging_applications': ['robotics', 'game_ai', 'autonomous_vehicles'],
            'funding_trends': {'multi_agent_rl': 'increasing', 'coordination': 'hot'}
        }


class InsightsEngine:
    """Main Analytics & Insights Engine class"""
    
    def __init__(self, config_path: str, output_dir: str = "research/analytics/outputs"):
        self.config = load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.performance_analyzer = PerformanceAnalyzer(config_path)
        self.competitive_intel = CompetitiveIntelligence()
        
        # Analytics state
        self.active_experiments = {}
        self.historical_data = []
        self.real_time_dashboard_data = {}
        
        logger.info("Analytics & Insights Engine initialized")
    
    async def start_experiment_monitoring(self, experiment_id: str, config: Dict[str, Any]):
        """Start monitoring a new experiment"""
        self.active_experiments[experiment_id] = {
            'start_time': datetime.now(),
            'config': config,
            'metrics': [],
            'status': 'running'
        }
        
        logger.info(f"Started monitoring experiment: {experiment_id}")
    
    def record_training_step(self, experiment_id: str, agent_id: int, step: int, metrics: Dict[str, float]):
        """Record metrics from a training step"""
        if experiment_id not in self.active_experiments:
            logger.warning(f"Unknown experiment ID: {experiment_id}")
            return
            
        # Record in metrics collector
        self.metrics_collector.record_step_metrics(agent_id, step, metrics)
        
        # Store experiment-specific data
        self.active_experiments[experiment_id]['metrics'].append({
            'timestamp': datetime.now(),
            'agent_id': agent_id,
            'step': step,
            'metrics': metrics
        })
        
        # Update real-time dashboard
        self._update_real_time_dashboard(experiment_id)
    
    def complete_experiment(self, experiment_id: str, final_metrics: ExperimentMetrics) -> Dict[str, Any]:
        """Complete an experiment and generate comprehensive analysis"""
        if experiment_id not in self.active_experiments:
            logger.error(f"Unknown experiment ID: {experiment_id}")
            return {}
        
        # Mark as completed
        self.active_experiments[experiment_id]['status'] = 'completed'
        self.active_experiments[experiment_id]['end_time'] = datetime.now()
        
        # Generate comprehensive insights
        performance_insights = self.performance_analyzer.analyze_experiment(final_metrics)
        competitive_analysis = self.competitive_intel.analyze_competitive_position(final_metrics)
        
        # Create analysis report
        report = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'experiment_duration': (
                self.active_experiments[experiment_id]['end_time'] - 
                self.active_experiments[experiment_id]['start_time']
            ).total_seconds(),
            'final_metrics': asdict(final_metrics),
            'performance_insights': [asdict(insight) for insight in performance_insights],
            'competitive_analysis': asdict(competitive_analysis),
            'recommendations': self._generate_actionable_recommendations(performance_insights),
            'dashboard_summary': self._generate_dashboard_summary(final_metrics, performance_insights)
        }
        
        # Save to historical data
        self.historical_data.append(report)
        
        # Export report
        self._export_analysis_report(report)
        
        # Generate visualizations
        self._generate_analysis_visualizations(final_metrics, performance_insights)
        
        logger.info(f"Completed analysis for experiment: {experiment_id}")
        return report
    
    def get_real_time_insights(self, experiment_id: str) -> Dict[str, Any]:
        """Get real-time insights for ongoing experiment"""
        if experiment_id not in self.active_experiments:
            return {}
            
        # Get recent performance data
        recent_stats = self.metrics_collector.get_recent_performance()
        
        # Generate quick insights
        quick_insights = []
        
        if 'reward_trend' in recent_stats:
            trend = recent_stats['reward_trend']
            if trend > 0.001:
                quick_insights.append({
                    'type': 'positive',
                    'message': f"Reward is trending upward ({trend:.4f}/episode)"
                })
            elif trend < -0.001:
                quick_insights.append({
                    'type': 'warning',
                    'message': f"Reward is trending downward ({trend:.4f}/episode)"
                })
        
        return {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'recent_performance': recent_stats,
            'quick_insights': quick_insights,
            'status': self.active_experiments[experiment_id]['status']
        }
    
    def generate_comparative_analysis(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments"""
        if not experiment_ids:
            return {}
            
        comparisons = {}
        
        for exp_id in experiment_ids:
            if exp_id in self.historical_data:
                exp_data = next(data for data in self.historical_data 
                               if data['experiment_id'] == exp_id)
                comparisons[exp_id] = {
                    'final_performance': exp_data['final_metrics']['final_performance'],
                    'convergence_time': exp_data['final_metrics']['convergence_time'],
                    'stability': exp_data['final_metrics']['training_stability'],
                    'coordination': exp_data['final_metrics']['agent_coordination_score']
                }
        
        # Statistical analysis
        if len(comparisons) > 1:
            performances = [data['final_performance'] for data in comparisons.values()]
            best_experiment = max(comparisons.keys(), 
                                key=lambda x: comparisons[x]['final_performance'])
            
            return {
                'comparison_data': comparisons,
                'statistical_summary': {
                    'mean_performance': np.mean(performances),
                    'std_performance': np.std(performances),
                    'best_experiment': best_experiment,
                    'performance_range': max(performances) - min(performances)
                },
                'recommendations': self._generate_comparative_recommendations(comparisons)
            }
        
        return {'comparison_data': comparisons}
    
    def _update_real_time_dashboard(self, experiment_id: str):
        """Update real-time dashboard data"""
        recent_stats = self.metrics_collector.get_recent_performance(lookback_steps=50)
        
        self.real_time_dashboard_data[experiment_id] = {
            'last_update': datetime.now().isoformat(),
            'performance_metrics': recent_stats,
            'training_progress': {
                'current_episode': len(self.active_experiments[experiment_id]['metrics']),
                'runtime_minutes': (datetime.now() - 
                                   self.active_experiments[experiment_id]['start_time']).total_seconds() / 60
            }
        }
    
    def _generate_actionable_recommendations(self, insights: List[PerformanceInsight]) -> List[str]:
        """Generate prioritized actionable recommendations"""
        recommendations = []
        
        # Group insights by severity and impact
        critical_insights = [i for i in insights if i.severity == "critical"]
        warning_insights = [i for i in insights if i.severity == "warning"]
        
        # Priority 1: Critical issues
        for insight in critical_insights:
            recommendations.extend(insight.recommendations[:2])  # Top 2 recommendations
        
        # Priority 2: High-impact warnings
        high_impact_warnings = [i for i in warning_insights if i.impact_score > 7.0]
        for insight in high_impact_warnings:
            recommendations.extend(insight.recommendations[:1])  # Top recommendation
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        seen = set()
        for rec in recommendations:
            if rec not in seen:
                unique_recommendations.append(rec)
                seen.add(rec)
        
        return unique_recommendations[:10]  # Top 10 recommendations
    
    def _generate_dashboard_summary(self, metrics: ExperimentMetrics, insights: List[PerformanceInsight]) -> Dict[str, Any]:
        """Generate summary for dashboard display"""
        critical_count = len([i for i in insights if i.severity == "critical"])
        warning_count = len([i for i in insights if i.severity == "warning"])
        success_count = len([i for i in insights if i.severity == "success"])
        
        # Overall health score
        health_score = max(0, min(100, 
            70 + success_count * 10 - warning_count * 5 - critical_count * 15
        ))
        
        return {
            'overall_health_score': health_score,
            'performance_grade': self._calculate_performance_grade(metrics.final_performance),
            'key_metrics': {
                'final_performance': round(metrics.final_performance, 3),
                'stability_score': round(metrics.training_stability, 3),
                'coordination_score': round(metrics.agent_coordination_score, 3),
                'success_rate': round(metrics.success_rate, 3)
            },
            'alert_counts': {
                'critical': critical_count,
                'warning': warning_count,
                'success': success_count
            },
            'top_insight': insights[0].title if insights else "No significant insights"
        }
    
    def _calculate_performance_grade(self, performance: float) -> str:
        """Calculate letter grade for performance"""
        if performance >= 0.9:
            return "A+"
        elif performance >= 0.85:
            return "A"
        elif performance >= 0.8:
            return "A-"
        elif performance >= 0.75:
            return "B+"
        elif performance >= 0.7:
            return "B"
        elif performance >= 0.65:
            return "B-"
        elif performance >= 0.6:
            return "C+"
        elif performance >= 0.55:
            return "C"
        else:
            return "C-"
    
    def _generate_comparative_recommendations(self, comparisons: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate recommendations from comparative analysis"""
        recommendations = []
        
        if len(comparisons) < 2:
            return recommendations
        
        # Find best performing experiment
        best_exp = max(comparisons.keys(), 
                      key=lambda x: comparisons[x]['final_performance'])
        best_performance = comparisons[best_exp]['final_performance']
        
        # Analyze what made it successful
        best_data = comparisons[best_exp]
        
        if best_data.get('stability', 0) > 0.8:
            recommendations.append(f"Adopt stability techniques from experiment {best_exp}")
        
        if best_data.get('coordination', 0) > 0.8:
            recommendations.append(f"Replicate coordination approach from experiment {best_exp}")
        
        # Find common failure patterns
        poor_performers = [exp_id for exp_id, data in comparisons.items() 
                          if data['final_performance'] < best_performance * 0.8]
        
        if len(poor_performers) > len(comparisons) * 0.5:
            recommendations.append("Review common hyperparameters across poor performers")
            recommendations.append("Consider systematic hyperparameter optimization")
        
        return recommendations
    
    def _export_analysis_report(self, report: Dict[str, Any]):
        """Export detailed analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_report_{report['experiment_id']}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Analysis report exported to: {filepath}")
    
    def _generate_analysis_visualizations(self, metrics: ExperimentMetrics, insights: List[PerformanceInsight]):
        """Generate visualization plots for analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Experiment Analysis: {metrics.experiment_id}', fontsize=16)
        
        # Reward progression
        axes[0, 0].plot(metrics.episode_rewards)
        axes[0, 0].set_title('Reward Progression')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Episode length distribution
        axes[0, 1].hist(metrics.episode_lengths, bins=20, alpha=0.7)
        axes[0, 1].set_title('Episode Length Distribution')
        axes[0, 1].set_xlabel('Episode Length')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Performance metrics radar
        radar_metrics = [
            metrics.final_performance,
            metrics.training_stability,
            metrics.agent_coordination_score,
            metrics.communication_efficiency,
            metrics.exploration_coverage
        ]
        radar_labels = ['Performance', 'Stability', 'Coordination', 'Communication', 'Exploration']
        
        angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False)
        radar_metrics.append(radar_metrics[0])  # Close the polygon
        angles = np.append(angles, angles[0])
        
        axes[1, 0] = plt.subplot(2, 2, 3, projection='polar')
        axes[1, 0].plot(angles, radar_metrics, 'o-', linewidth=2)
        axes[1, 0].fill(angles, radar_metrics, alpha=0.25)
        axes[1, 0].set_xticks(angles[:-1])
        axes[1, 0].set_xticklabels(radar_labels)
        axes[1, 0].set_title('Performance Radar')
        
        # Insights severity distribution
        severity_counts = defaultdict(int)
        for insight in insights:
            severity_counts[insight.severity] += 1
        
        if severity_counts:
            axes[1, 1].bar(severity_counts.keys(), severity_counts.values())
            axes[1, 1].set_title('Insights by Severity')
            axes[1, 1].set_xlabel('Severity Level')
            axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_filename = f"analysis_viz_{metrics.experiment_id}_{timestamp}.png"
        viz_filepath = self.output_dir / viz_filename
        plt.savefig(viz_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Analysis visualization saved to: {viz_filepath}")
    
    def generate_research_trends_report(self) -> Dict[str, Any]:
        """Generate comprehensive research trends analysis"""
        if not self.historical_data:
            return {"error": "No historical data available"}
        
        # Analyze performance trends over time
        performance_over_time = [
            {
                'timestamp': data['timestamp'],
                'performance': data['final_metrics']['final_performance'],
                'experiment_id': data['experiment_id']
            }
            for data in self.historical_data
        ]
        
        # Identify best practices
        top_performers = sorted(
            self.historical_data,
            key=lambda x: x['final_metrics']['final_performance'],
            reverse=True
        )[:3]
        
        best_practices = []
        if top_performers:
            for performer in top_performers:
                config = performer.get('experiment_config', {})
                best_practices.append({
                    'experiment_id': performer['experiment_id'],
                    'performance': performer['final_metrics']['final_performance'],
                    'key_settings': self._extract_key_settings(config)
                })
        
        # Research gaps analysis
        research_gaps = self._identify_research_gaps()
        
        # Future directions
        future_directions = self._recommend_future_directions()
        
        trends_report = {
            'report_date': datetime.now().isoformat(),
            'total_experiments': len(self.historical_data),
            'performance_trends': performance_over_time,
            'best_practices': best_practices,
            'research_gaps': research_gaps,
            'future_directions': future_directions,
            'success_factors': self._analyze_success_factors()
        }
        
        # Export trends report
        self._export_trends_report(trends_report)
        
        return trends_report
    
    def _extract_key_settings(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key configuration settings that impact performance"""
        key_settings = {}
        
        # Common important hyperparameters
        important_keys = [
            'learning_rate', 'batch_size', 'gamma', 'gae_lambda',
            'num_agents', 'communication_dim', 'network_architecture',
            'exploration_strategy', 'reward_shaping'
        ]
        
        for key in important_keys:
            if key in config:
                key_settings[key] = config[key]
        
        return key_settings
    
    def _identify_research_gaps(self) -> List[str]:
        """Identify potential research gaps based on historical data"""
        gaps = []
        
        # Analyze performance distribution
        performances = [data['final_metrics']['final_performance'] 
                       for data in self.historical_data]
        
        if performances:
            mean_perf = np.mean(performances)
            
            if mean_perf < 0.7:
                gaps.append("Need better baseline algorithms")
            
            if np.std(performances) > 0.2:
                gaps.append("High variance in results suggests need for better hyperparameter optimization")
        
        # Analyze coordination scores
        coord_scores = [data['final_metrics']['agent_coordination_score'] 
                       for data in self.historical_data]
        
        if coord_scores and np.mean(coord_scores) < 0.6:
            gaps.append("Multi-agent coordination mechanisms need improvement")
        
        # Check for common failure patterns
        common_warnings = defaultdict(int)
        for data in self.historical_data:
            for insight in data.get('performance_insights', []):
                if insight['severity'] in ['warning', 'critical']:
                    common_warnings[insight['title']] += 1
        
        if common_warnings:
            most_common = max(common_warnings.items(), key=lambda x: x[1])
            if most_common[1] > len(self.historical_data) * 0.3:
                gaps.append(f"Recurring issue: {most_common[0]}")
        
        return gaps
    
    def _recommend_future_directions(self) -> List[str]:
        """Recommend future research directions"""
        directions = []
        
        # Based on performance analysis
        performances = [data['final_metrics']['final_performance'] 
                       for data in self.historical_data]
        
        if performances:
            best_performance = max(performances)
            
            if best_performance < 0.9:
                directions.append("Investigate advanced neural architectures (Transformers, Graph Networks)")
            
            if best_performance > 0.85:
                directions.append("Focus on sample efficiency and transfer learning")
        
        # Based on coordination analysis
        coord_scores = [data['final_metrics']['agent_coordination_score'] 
                       for data in self.historical_data]
        
        if coord_scores and max(coord_scores) > 0.8:
            directions.append("Explore emergent communication and culture formation")
        elif coord_scores and np.mean(coord_scores) < 0.6:
            directions.append("Research improved coordination mechanisms")
        
        # Always relevant directions for MARL
        directions.extend([
            "Investigate curriculum learning for complex task hierarchies",
            "Explore meta-learning for rapid adaptation to new environments",
            "Research scalability to larger agent populations",
            "Study real-world transfer learning applications"
        ])
        
        return directions[:8]  # Top 8 directions
    
    def _analyze_success_factors(self) -> Dict[str, Any]:
        """Analyze factors that contribute to experimental success"""
        if len(self.historical_data) < 3:
            return {"error": "Insufficient data for success factor analysis"}
        
        # Separate high and low performers
        performances = [(data['experiment_id'], data['final_metrics']['final_performance']) 
                       for data in self.historical_data]
        performances.sort(key=lambda x: x[1], reverse=True)
        
        top_third = performances[:len(performances)//3]
        bottom_third = performances[-len(performances)//3:]
        
        # Extract configurations
        top_configs = []
        bottom_configs = []
        
        for exp_id, _ in top_third:
            data = next(d for d in self.historical_data if d['experiment_id'] == exp_id)
            top_configs.append(self._extract_key_settings(data.get('experiment_config', {})))
        
        for exp_id, _ in bottom_third:
            data = next(d for d in self.historical_data if d['experiment_id'] == exp_id)
            bottom_configs.append(self._extract_key_settings(data.get('experiment_config', {})))
        
        # Find differentiating factors
        success_factors = {}
        
        # Analyze numerical hyperparameters
        for key in ['learning_rate', 'batch_size', 'gamma']:
            top_values = [config.get(key) for config in top_configs if key in config]
            bottom_values = [config.get(key) for config in bottom_configs if key in config]
            
            if top_values and bottom_values:
                top_mean = np.mean(top_values)
                bottom_mean = np.mean(bottom_values)
                
                if abs(top_mean - bottom_mean) > np.std(top_values + bottom_values):
                    success_factors[key] = {
                        'top_performers_avg': top_mean,
                        'bottom_performers_avg': bottom_mean,
                        'difference': top_mean - bottom_mean
                    }
        
        return {
            'success_factors': success_factors,
            'top_performer_ids': [exp_id for exp_id, _ in top_third],
            'performance_threshold': top_third[-1][1] if top_third else 0.0
        }
    
    def _export_trends_report(self, trends_report: Dict[str, Any]):
        """Export research trends report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_trends_report_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(trends_report, f, indent=2, default=str)
        
        logger.info(f"Research trends report exported to: {filepath}")
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend slope using linear regression"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope


class RealTimeDashboard:
    """Real-time monitoring dashboard for experiments"""
    
    def __init__(self, insights_engine: InsightsEngine):
        self.engine = insights_engine
        self.update_interval = 10  # seconds
        self.is_running = False
    
    async def start_monitoring(self, experiment_ids: List[str]):
        """Start real-time monitoring for specified experiments"""
        self.is_running = True
        
        while self.is_running:
            try:
                dashboard_data = {}
                
                for exp_id in experiment_ids:
                    if exp_id in self.engine.active_experiments:
                        dashboard_data[exp_id] = self.engine.get_real_time_insights(exp_id)
                
                # Update dashboard display
                self._update_dashboard_display(dashboard_data)
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Dashboard monitoring error: {e}")
                await asyncio.sleep(self.update_interval)
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_running = False
        logger.info("Real-time monitoring stopped")
    
    def _update_dashboard_display(self, dashboard_data: Dict[str, Any]):
        """Update dashboard display (placeholder for actual UI)"""
        # In a real implementation, this would update a web dashboard
        # For now, we'll log the key metrics
        
        for exp_id, data in dashboard_data.items():
            recent_perf = data.get('recent_performance', {})
            mean_reward = recent_perf.get('mean_reward', 0)
            
            logger.info(f"Dashboard Update - {exp_id}: "
                       f"Recent Reward: {mean_reward:.3f}, "
                       f"Status: {data.get('status', 'unknown')}")


class AutomatedInsights:
    """Automated insight generation and alerting"""
    
    def __init__(self, insights_engine: InsightsEngine):
        self.engine = insights_engine
        self.alert_thresholds = {
            'performance_drop': 0.1,  # 10% performance drop triggers alert
            'instability_threshold': 0.4,
            'coordination_failure': 0.3
        }
    
    def check_for_alerts(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Check for automated alerts during training"""
        alerts = []
        
        if experiment_id not in self.engine.active_experiments:
            return alerts
        
        recent_stats = self.engine.metrics_collector.get_recent_performance(lookback_steps=20)
        
        # Performance drop alert
        if 'reward_trend' in recent_stats:
            trend = recent_stats['reward_trend']
            if trend < -self.alert_thresholds['performance_drop']:
                alerts.append({
                    'type': 'performance_drop',
                    'severity': 'warning',
                    'message': f"Significant performance drop detected (trend: {trend:.4f})",
                    'timestamp': datetime.now().isoformat(),
                    'recommended_action': 'Consider reducing learning rate or checking for bugs'
                })
        
        # Instability alert
        if 'reward_std' in recent_stats:
            recent_std = recent_stats['reward_std']
            recent_mean = recent_stats.get('mean_reward', 1.0)
            
            if recent_mean != 0 and recent_std / abs(recent_mean) > 2.0:  # High coefficient of variation
                alerts.append({
                    'type': 'training_instability',
                    'severity': 'warning',
                    'message': f"High training variance detected (CV: {recent_std/abs(recent_mean):.2f})",
                    'timestamp': datetime.now().isoformat(),
                    'recommended_action': 'Consider reducing learning rate or increasing batch size'
                })
        
        return alerts
    
    def generate_mid_training_recommendations(self, experiment_id: str) -> List[str]:
        """Generate recommendations during training"""
        recommendations = []
        
        # Check recent performance
        recent_stats = self.engine.metrics_collector.get_recent_performance(lookback_steps=50)
        
        if 'mean_reward' in recent_stats:
            mean_reward = recent_stats['mean_reward']
            
            # Compare to expected performance at this stage
            expected_performance = self._estimate_expected_performance(experiment_id)
            
            if mean_reward < expected_performance * 0.7:
                recommendations.append("Performance below expectations - consider early stopping and hyperparameter adjustment")
            elif mean_reward > expected_performance * 1.3:
                recommendations.append("Exceptional performance - document current settings for future use")
        
        # Stability recommendations
        if 'reward_std' in recent_stats and 'mean_reward' in recent_stats:
            stability = 1.0 / (1.0 + recent_stats['reward_std'] / abs(recent_stats['mean_reward']))
            
            if stability < 0.5:
                recommendations.append("High training variance - reduce learning rate or increase batch size")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _estimate_expected_performance(self, experiment_id: str) -> float:
        """Estimate expected performance based on training progress"""
        if experiment_id not in self.engine.active_experiments:
            return 0.5  # Default expectation
        
        exp_data = self.engine.active_experiments[experiment_id]
        current_episodes = len(exp_data['metrics'])
        
        # Simple heuristic based on episode count and baseline performance
        baseline_performance = 0.1  # Random baseline
        target_performance = 0.8    # Target performance
        
        # Assume sigmoid learning curve
        progress_ratio = current_episodes / 1000  # Assume 1000 episodes for convergence
        expected = baseline_performance + (target_performance - baseline_performance) * (
            1 / (1 + np.exp(-5 * (progress_ratio - 0.5)))
        )
        
        return expected


class ExperimentComparison:
    """Advanced experiment comparison and meta-analysis"""
    
    def __init__(self, insights_engine: InsightsEngine):
        self.engine = insights_engine
    
    def compare_hyperparameter_impact(self, hyperparameter: str) -> Dict[str, Any]:
        """Analyze impact of specific hyperparameter across experiments"""
        if not self.engine.historical_data:
            return {"error": "No historical data available"}
        
        # Extract hyperparameter values and corresponding performances
        param_performance_pairs = []
        
        for data in self.engine.historical_data:
            config = data.get('experiment_config', {})
            performance = data['final_metrics']['final_performance']
            
            if hyperparameter in config:
                param_performance_pairs.append((config[hyperparameter], performance))
        
        if len(param_performance_pairs) < 3:
            return {"error": f"Insufficient data for {hyperparameter} analysis"}
        
        # Statistical analysis
        param_values = [pair[0] for pair in param_performance_pairs]
        performances = [pair[1] for pair in param_performance_pairs]
        
        correlation, p_value = stats.pearsonr(param_values, performances)
        
        # Optimal value estimation
        if len(param_performance_pairs) >= 5:
            optimal_idx = np.argmax(performances)
            optimal_value = param_values[optimal_idx]
        else:
            optimal_value = None
        
        return {
            'hyperparameter': hyperparameter,
            'correlation_with_performance': correlation,
            'statistical_significance': p_value,
            'optimal_value': optimal_value,
            'value_range': (min(param_values), max(param_values)),
            'performance_range': (min(performances), max(performances)),
            'recommendation': self._generate_hyperparameter_recommendation(
                hyperparameter, correlation, optimal_value
            )
        }
    
    def _generate_hyperparameter_recommendation(self, param: str, correlation: float, optimal_value: Any) -> str:
        """Generate recommendation for hyperparameter tuning"""
        if abs(correlation) > 0.7:
            direction = "higher" if correlation > 0 else "lower"
            strength = "strong" if abs(correlation) > 0.8 else "moderate"
            
            rec = f"{strength.title()} correlation suggests {direction} {param} values improve performance"
            
            if optimal_value is not None:
                rec += f". Optimal observed value: {optimal_value}"
            
            return rec
        elif abs(correlation) > 0.3:
            return f"Weak correlation suggests {param} has limited impact on performance"
        else:
            return f"No clear correlation between {param} and performance"


# Main execution and testing functions
def create_sample_metrics(experiment_id: str = "test_exp_001") -> ExperimentMetrics:
    """Create sample metrics for testing"""
    np.random.seed(42)
    
    # Simulate learning curve
    episodes = 500
    base_reward = 0.1
    final_reward = 0.85
    noise_level = 0.1
    
    episode_rewards = []
    for i in range(episodes):
        # Sigmoid learning curve with noise
        progress = i / episodes
        expected_reward = base_reward + (final_reward - base_reward) * (
            1 / (1 + np.exp(-10 * (progress - 0.5)))
        )
        actual_reward = expected_reward + np.random.normal(0, noise_level)
        episode_rewards.append(actual_reward)
    
    episode_lengths = np.random.poisson(50, episodes).tolist()
    
    return ExperimentMetrics(
        experiment_id=experiment_id,
        timestamp=datetime.now(),
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        success_rate=0.78,
        convergence_time=350.0,
        final_performance=np.mean(episode_rewards[-20:]),
        training_stability=0.75,
        resource_utilization={
            'cpu_percent': 45.2,
            'memory_percent': 62.1,
            'gpu_percent': 78.5
        },
        agent_coordination_score=0.72,
        communication_efficiency=0.65,
        exploration_coverage=0.82
    )


async def main():
    """Demonstration of Analytics & Insights Engine"""
    print("🔍 Project NEXUS - Analytics & Insights Engine Demo")
    print("=" * 60)
    
    # Initialize the insights engine
    config_path = "configs/training/ppo_baseline.yaml"  # Placeholder config
    engine = InsightsEngine(config_path)
    
    # Create sample experiment
    experiment_id = "demo_experiment_001"
    await engine.start_experiment_monitoring(experiment_id, {
        'learning_rate': 0.0003,
        'batch_size': 32,
        'num_agents': 4,
        'gamma': 0.99
    })
    
    # Simulate some training steps
    print("\n📊 Simulating training steps...")
    for step in range(100):
        for agent_id in range(4):
            metrics = {
                'reward': np.random.normal(0.5 + step * 0.001, 0.1),
                'episode_length': np.random.poisson(45),
                'value_loss': np.random.exponential(0.1),
                'policy_loss': np.random.exponential(0.05)
            }
            engine.record_training_step(experiment_id, agent_id, step, metrics)
    
    # Get real-time insights
    print("\n⚡ Real-time insights:")
    real_time_insights = engine.get_real_time_insights(experiment_id)
    for insight in real_time_insights.get('quick_insights', []):
        print(f"  • {insight['message']}")
    
    # Complete experiment analysis
    print("\n🎯 Completing experiment analysis...")
    sample_metrics = create_sample_metrics(experiment_id)
    analysis_report = engine.complete_experiment(experiment_id, sample_metrics)
    
    # Display key insights
    print("\n💡 Key Performance Insights:")
    for insight in analysis_report['performance_insights'][:3]:
        print(f"  {insight['severity'].upper()}: {insight['title']}")
        print(f"    {insight['description']}")
        print()
    
    # Competitive analysis
    print("🏆 Competitive Analysis:")
    comp_analysis = analysis_report['competitive_analysis']
    print(f"  Market Position: {comp_analysis['market_position'].title()}")
    print(f"  Performance Ranking: #{comp_analysis['performance_ranking']}")
    print(f"  Key Strengths: {', '.join(comp_analysis['strengths'][:2])}")
    
    # Recommendations
    print("\n📋 Top Recommendations:")
    for i, rec in enumerate(analysis_report['recommendations'][:3], 1):
        print(f"  {i}. {rec}")
    
    print(f"\n✅ Analysis complete! Report saved to: {engine.output_dir}")
    print(f"📈 Dashboard Summary: {analysis_report['dashboard_summary']['overall_health_score']}/100 Health Score")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())