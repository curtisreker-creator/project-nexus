"""
Project NEXUS - Real-Time Training Dashboard
Web-based analytics dashboard with live metrics streaming

File: visual_analytics/live_dashboard.py
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading
from collections import deque
import numpy as np

# Web framework imports
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

# Project imports
from ..renderers.gridworld_renderer import GridWorldRenderer, create_single_match_renderer


class MetricsCollector:
    """Collect and manage real-time training metrics"""

    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size

        # Training metrics
        self.training_curves = {
            'timestamps': deque(maxlen=buffer_size),
            'rewards': deque(maxlen=buffer_size),
            'losses': deque(maxlen=buffer_size),
            'exploration_rates': deque(maxlen=buffer_size)
        }

        # Self-play metrics
        self.match_results = deque(maxlen=buffer_size)
        self.elo_history = {}  # agent_id -> deque of ratings
        self.win_rates = deque(maxlen=50)

        # Performance metrics
        self.system_metrics = {
            'gpu_utilization': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'training_speed': deque(maxlen=100),
            'fps': deque(maxlen=100)
        }

        # Real-time statistics
        self.current_stats = {
            'active_matches': 0,
            'total_matches_played': 0,
            'current_generation': 0,
            'top_agent_elo': 0,
            'average_match_duration': 0
        }

    def add_training_metrics(self, timestamp: float, reward: float,
                             loss: float, exploration_rate: float):
        """Add training step metrics"""
        self.training_curves['timestamps'].append(timestamp)
        self.training_curves['rewards'].append(reward)
        self.training_curves['losses'].append(loss)
        self.training_curves['exploration_rates'].append(exploration_rate)

    def add_match_result(self, match_data: Dict[str, Any]):
        """Add self-play match result"""
        self.match_results.append(match_data)

        # Update ELO history
        for agent_id in match_data.get('participating_agents', []):
            if agent_id not in self.elo_history:
                self.elo_history[agent_id] = deque(maxlen=100)

            new_elo = match_data.get('agent_elos', {}).get(agent_id, 1200)
            self.elo_history[agent_id].append({
                'timestamp': match_data['timestamp'],
                'elo': new_elo
            })

        # Update current stats
        self.current_stats['total_matches_played'] += 1

        if 'winner' in match_data:
            win_rate = len([m for m in list(self.match_results)[-20:]
                            if m.get('winner') != 'draw']) / max(20, len(self.match_results))
            self.win_rates.append(win_rate)

    def add_system_metrics(self, gpu_util: float, memory_usage: float,
                           training_speed: float, fps: float):
        """Add system performance metrics"""
        self.system_metrics['gpu_utilization'].append(gpu_util)
        self.system_metrics['memory_usage'].append(memory_usage)
        self.system_metrics['training_speed'].append(training_speed)
        self.system_metrics['fps'].append(fps)

    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get latest metrics for dashboard"""
        return {
            'training_curves': {
                key: list(values) for key, values in self.training_curves.items()
            },
            'elo_leaderboard': self._get_elo_leaderboard(),
            'match_statistics': self._get_match_statistics(),
            'system_performance': self._get_system_performance(),
            'current_stats': self.current_stats.copy()
        }

    def _get_elo_leaderboard(self) -> List[Dict[str, Any]]:
        """Get current ELO leaderboard"""
        leaderboard = []

        for agent_id, history in self.elo_history.items():
            if history:
                latest = history[-1]
                leaderboard.append({
                    'agent_id': agent_id,
                    'elo': latest['elo'],
                    'matches_played': len(history),
                    'trend': self._calculate_elo_trend(history)
                })

        return sorted(leaderboard, key=lambda x: x['elo'], reverse=True)[:10]

    def _get_match_statistics(self) -> Dict[str, Any]:
        """Get match statistics summary"""
        if not self.match_results:
            return {}

        recent_matches = list(self.match_results)[-50:]

        return {
            'total_matches': len(self.match_results),
            'recent_average_duration': np.mean([m.get('duration', 0) for m in recent_matches]),
            'win_distribution': self._calculate_win_distribution(recent_matches),
            'matches_per_hour': len(recent_matches) / max(1, (time.time() - recent_matches[0]['timestamp']) / 3600)
        }

    def _get_system_performance(self) -> Dict[str, Any]:
        """Get system performance summary"""
        return {
            'average_gpu_utilization': np.mean(list(self.system_metrics['gpu_utilization'])) if self.system_metrics[
                'gpu_utilization'] else 0,
            'average_memory_usage': np.mean(list(self.system_metrics['memory_usage'])) if self.system_metrics[
                'memory_usage'] else 0,
            'current_training_speed': list(self.system_metrics['training_speed'])[-1] if self.system_metrics[
                'training_speed'] else 0,
            'current_fps': list(self.system_metrics['fps'])[-1] if self.system_metrics['fps'] else 0
        }

    def _calculate_elo_trend(self, history: deque) -> str:
        """Calculate ELO rating trend"""
        if len(history) < 2:
            return 'stable'

        recent = [h['elo'] for h in list(history)[-10:]]
        if len(recent) >= 2:
            slope = np.polyfit(range(len(recent)), recent, 1)[0]
            if slope > 5:
                return 'rising'
            elif slope < -5:
                return 'falling'

        return 'stable'

    def _calculate_win_distribution(self, matches: List[Dict]) -> Dict[str, int]:
        """Calculate win distribution for recent matches"""
        distribution = {'team_a': 0, 'team_b': 0, 'draw': 0}

        for match in matches:
            winner = match.get('winner', 'draw')
            if winner in distribution:
                distribution[winner] += 1

        return distribution


class DashboardPlotter:
    """Generate real-time plots for dashboard"""

    @staticmethod
    def create_training_curves(metrics: Dict[str, List]) -> Dict[str, Any]:
        """Create training curves plot"""
        if not metrics.get('timestamps'):
            return {}

        fig = go.Figure()

        # Rewards curve
        fig.add_trace(go.Scatter(
            x=list(metrics['timestamps']),
            y=list(metrics['rewards']),
            mode='lines',
            name='Reward',
            line=dict(color='#00ff88', width=2)
        ))

        # Loss curve (secondary axis)
        fig.add_trace(go.Scatter(
            x=list(metrics['timestamps']),
            y=list(metrics['losses']),
            mode='lines',
            name='Loss',
            yaxis='y2',
            line=dict(color='#ff4444', width=2)
        ))

        fig.update_layout(
            title='Training Progress',
            xaxis_title='Time',
            yaxis_title='Reward',
            yaxis2=dict(
                title='Loss',
                overlaying='y',
                side='right'
            ),
            template='plotly_dark',
            height=400
        )

        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

    @staticmethod
    def create_elo_leaderboard(leaderboard: List[Dict]) -> Dict[str, Any]:
        """Create ELO leaderboard visualization"""
        if not leaderboard:
            return {}

        agents = [agent['agent_id'] for agent in leaderboard]
        elos = [agent['elo'] for agent in leaderboard]
        trends = [agent['trend'] for agent in leaderboard]

        # Color based on trend
        colors = []
        for trend in trends:
            if trend == 'rising':
                colors.append('#00ff88')
            elif trend == 'falling':
                colors.append('#ff4444')
            else:
                colors.append('#888888')

        fig = go.Figure(data=[
            go.Bar(
                x=agents,
                y=elos,
                marker_color=colors,
                text=[f"{elo:.0f}" for elo in elos],
                textposition='auto'
            )
        ])

        fig.update_layout(
            title='ELO Leaderboard',
            xaxis_title='Agent',
            yaxis_title='ELO Rating',
            template='plotly_dark',
            height=300
        )

        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

    @staticmethod
    def create_system_performance(performance: Dict[str, Any]) -> Dict[str, Any]:
        """Create system performance dashboard"""
        if not performance:
            return {}

        # Create gauge charts for key metrics
        fig = go.Figure()

        # GPU Utilization gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=performance['average_gpu_utilization'],
            domain={'x': [0, 0.5], 'y': [0.5, 1]},
            title={'text': "GPU Utilization (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#00ff88"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 85], 'color': "yellow"},
                    {'range': [85, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))

        # Training Speed gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=performance['current_training_speed'],
            domain={'x': [0.5, 1], 'y': [0.5, 1]},
            title={'text': "Training Speed (steps/sec)"},
            gauge={
                'axis': {'range': [0, 1000]},
                'bar': {'color': "#4488ff"}
            }
        ))

        # FPS gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=performance['current_fps'],
            domain={'x': [0, 0.5], 'y': [0, 0.5]},
            title={'text': "Rendering FPS"},
            gauge={
                'axis': {'range': [0, 60]},
                'bar': {'color': "#ff8844"}
            }
        ))

        # Memory Usage gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=performance['average_memory_usage'],
            domain={'x': [0.5, 1], 'y': [0, 0.5]},
            title={'text': "Memory Usage (GB)"},
            gauge={
                'axis': {'range': [0, 16]},
                'bar': {'color': "#8844ff"}
            }
        ))

        fig.update_layout(
            template='plotly_dark',
            height=500
        )

        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))


class LiveDashboard:
    """Main live dashboard controller"""

    def __init__(self, host: str = 'localhost', port: int = 8050):
        self.host = host
        self.port = port

        # Initialize Flask app
        self.app = Flask(__name__, template_folder='web_interface')
        self.app.config['SECRET_KEY'] = 'nexus_dashboard_key'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Components
        self.metrics_collector = MetricsCollector()
        self.plotter = DashboardPlotter()
        self.gridworld_renderer: Optional[GridWorldRenderer] = None

        # Dashboard state
        self.running = False
        self.update_thread = None
        self.update_interval = 2.0  # seconds

        # Setup routes
        self._setup_routes()
        self._setup_socketio_events()

        print(f"🎛️  Dashboard initialized at http://{host}:{port}")

    def _setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def dashboard_home():
            return render_template('dashboard.html')

        @self.app.route('/api/metrics')
        def get_metrics():
            return jsonify(self.metrics_collector.get_latest_metrics())

        @self.app.route('/api/plots/training')
        def get_training_plots():
            metrics = self.metrics_collector.get_latest_metrics()
            plots = {
                'training_curves': self.plotter.create_training_curves(metrics['training_curves']),
                'elo_leaderboard': self.plotter.create_elo_leaderboard(metrics['elo_leaderboard']),
                'system_performance': self.plotter.create_system_performance(metrics['system_performance'])
            }
            return jsonify(plots)

        @self.app.route('/api/status')
        def get_status():
            return jsonify({
                'status': 'running' if self.running else 'stopped',
                'uptime': time.time() - getattr(self, 'start_time', time.time()),
                'connected_clients': len(self.socketio.server.manager.get_participants('/', '/'))
            })

    def _setup_socketio_events(self):
        """Setup SocketIO events for real-time updates"""

        @self.socketio.on('connect')
        def handle_connect():
            print("🔌 Client connected to dashboard")
            emit('connected', {'status': 'Connected to NEXUS Dashboard'})

        @self.socketio.on('disconnect')
        def handle_disconnect():
            print("🔌 Client disconnected from dashboard")

        @self.socketio.on('request_update')
        def handle_update_request():
            metrics = self.metrics_collector.get_latest_metrics()
            emit('metrics_update', metrics)

    def integrate_with_trainer(self, trainer):
        """Integrate dashboard with self-play trainer"""

        def visual_callback(match_state):
            # Update GridWorld renderer
            if self.gridworld_renderer:
                self.gridworld_renderer.update_state(match_state)

            # Update metrics
            if 'team_a_score' in match_state and 'team_b_score' in match_state:
                self.metrics_collector.add_match_result({
                    'timestamp': time.time(),
                    'team_a_score': match_state['team_a_score'],
                    'team_b_score': match_state['team_b_score'],
                    'winner': self._determine_winner(match_state),
                    'duration': match_state.get('step', 0),
                    'participating_agents': match_state.get('team_a_agents', []) + match_state.get('team_b_agents', []),
                    'agent_elos': match_state.get('agent_elos', {})
                })

        # Set visual callback on trainer
        trainer.visual_callback = visual_callback

        print("🔗 Dashboard integrated with self-play trainer")

    def start_gridworld_visualization(self):
        """Start GridWorld real-time visualization"""
        if not self.gridworld_renderer:
            self.gridworld_renderer = create_single_match_renderer(rtx_optimized=True)
            self.gridworld_renderer.start_rendering()
            print("🎨 GridWorld visualization started")

    def start_dashboard(self, debug: bool = False):
        """Start the web dashboard server"""
        if self.running:
            return

        self.running = True
        self.start_time = time.time()

        # Start background update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

        print(f"🚀 Starting dashboard server at http://{self.host}:{self.port}")
        print("🎛️  Access the dashboard in your web browser")

        # Start Flask-SocketIO server
        self.socketio.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=debug,
            use_reloader=False  # Disable reloader to avoid threading issues
        )

    def stop_dashboard(self):
        """Stop the dashboard server"""
        self.running = False

        if self.gridworld_renderer:
            self.gridworld_renderer.stop_rendering()

        print("🛑 Dashboard stopped")

    def _update_loop(self):
        """Background loop for real-time updates"""
        while self.running:
            try:
                # Get latest metrics
                metrics = self.metrics_collector.get_latest_metrics()

                # Emit to all connected clients
                self.socketio.emit('metrics_update', metrics, namespace='/')

                # Sleep until next update
                time.sleep(self.update_interval)

            except Exception as e:
                print(f"⚠️  Dashboard update error: {e}")
                time.sleep(self.update_interval)

    def _determine_winner(self, match_state: Dict[str, Any]) -> str:
        """Determine match winner from state"""
        team_a_score = match_state.get('team_a_score', 0)
        team_b_score = match_state.get('team_b_score', 0)

        if team_a_score > team_b_score:
            return 'team_a'
        elif team_b_score > team_a_score:
            return 'team_b'
        else:
            return 'draw'

    # Public methods for external metric updates
    def update_training_metrics(self, reward: float, loss: float, exploration_rate: float):
        """Update training metrics from external source"""
        self.metrics_collector.add_training_metrics(
            timestamp=time.time(),
            reward=reward,
            loss=loss,
            exploration_rate=exploration_rate
        )

    def update_system_metrics(self, gpu_util: float, memory_usage: float,
                              training_speed: float, fps: float):
        """Update system performance metrics"""
        self.metrics_collector.add_system_metrics(
            gpu_util=gpu_util,
            memory_usage=memory_usage,
            training_speed=training_speed,
            fps=fps
        )

    def add_match_result(self, match_data: Dict[str, Any]):
        """Add match result from external source"""
        self.metrics_collector.add_match_result(match_data)