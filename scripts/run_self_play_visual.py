import asyncio
import threading
import time
import signal
import sys
from pathlib import Path
import yaml
import torch
import psutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from agents.training.self_play_trainer import SelfPlayTrainer
from visual_analytics.renderers.gridworld_renderer import create_single_match_renderer


class NexusSystemManager:
    """
    Main system manager for integrated self-play training with visual analytics
    """

    def __init__(self, config_path: str = "configs/self_play_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

        # System components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trainer = None
        self.gridworld_renderer = None

        # System monitoring
        self.running = False
        self.session_start_time = None
        self.matches_completed = 0

        print("🌟 NEXUS System Manager initialized")
        print(f"🎯 Device: {self.device}")
        print(f"⚙️  Config: {config_path}")

        if torch.cuda.is_available():
            print(f"🚀 GPU: {torch.cuda.get_device_name()}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"💾 GPU Memory: {gpu_memory:.1f}GB")

        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_config(self) -> dict:
        """Load system configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️  Config file not found: {self.config_path}")
            print("🔧 Using default configuration")
            config = self._get_default_config()

        return config

    def _get_default_config(self) -> dict:
        """Get default configuration for RTX 3060 system"""
        return {
            'training': {
                'batch_size': 64,
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_epsilon': 0.2,
                'entropy_coef': 0.01,
                'value_loss_coef': 0.5,
                'max_grad_norm': 0.5
            },
            'self_play': {
                'competitive_mode': 'resource_gathering',
                'match_duration': 1000,
                'matches_per_update': 10,
                'population_size': 50,
                'tournament_mode': True,
                'elo_k_factor': 32.0
            },
            'environment': {
                'grid_size': 15,
                'n_agents': 4,
                'max_resources': 20,
                'resource_respawn_rate': 0.02,
                'building_enabled': True,
                'fog_of_war': False
            },
            'visualization': {
                'enable_gridworld_renderer': True,
                'enable_web_dashboard': False,  # Disabled for now
                'dashboard_port': 8050,
                'update_frequency': 2.0,
                'rtx_optimization': True,
                'target_fps': 60
            },
            'system': {
                'gpu_memory_fraction': 0.8,
                'parallel_environments': 8,
                'async_rendering': True,
                'performance_monitoring': True
            }
        }

    async def initialize_system(self):
        """Initialize all system components"""
        print("🚀 Initializing NEXUS system components...")

        self.session_start_time = time.time()

        # 1. Initialize GridWorld renderer
        if self.config['visualization']['enable_gridworld_renderer']:
            print("🎨 Initializing GridWorld renderer...")
            self.gridworld_renderer = create_single_match_renderer(
                rtx_optimized=self.config['visualization']['rtx_optimization']
            )

        # 2. Initialize self-play trainer
        print("🤖 Initializing self-play trainer...")
        trainer_config = {
            **self.config['training'],
            **self.config['self_play'],
            **self.config['environment']
        }

        # Create visual callback for renderer integration
        async def visual_callback(match_state):
            if self.gridworld_renderer:
                self.gridworld_renderer.update_state(match_state)

        self.trainer = SelfPlayTrainer(trainer_config, self.device, visual_callback)

        print("✅ System initialization complete!")

    async def start_training_session(self, num_matches: int = 50):
        """Start complete training session with visual analytics"""
        print(f"🎪 Starting NEXUS training session: {num_matches} matches")

        self.running = True

        # Start visual components
        if self.gridworld_renderer:
            self.gridworld_renderer.start_rendering()
            print("🎨 GridWorld renderer started")
            print("🖼️  Visualization window should now be visible")

        # Run self-play training
        try:
            session_results = await self.trainer.run_training_session(num_matches)
            self.matches_completed = session_results['matches_completed']

            print("\n🎉 Training session completed successfully!")
            print(f"📊 Session summary:")
            print(f"   • Matches completed: {session_results['matches_completed']}")
            print(f"   • Session duration: {session_results['session_duration']:.1f}s")
            print(f"   • Performance: {session_results['matches_per_minute']:.1f} matches/min")
            print(f"   • Final population: {session_results['total_agents']} agents")

            # Show final leaderboard
            if session_results.get('final_leaderboard'):
                print("\n🏆 Final Agent Leaderboard:")
                for i, agent in enumerate(session_results['final_leaderboard'][:5], 1):
                    win_rate = (agent['wins'] / max(agent['matches_played'], 1)) * 100
                    print(f"   {i}. {agent['agent_id']}: {agent['elo_rating']:.0f} ELO "
                          f"({agent['wins']}-{agent['losses']}-{agent['draws']}, {win_rate:.1f}% WR)")

            return session_results

        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return None

        finally:
            await self._cleanup_session()

    def _get_gpu_stats(self):
        """Get GPU statistics using PyTorch"""
        if not torch.cuda.is_available():
            return 0.0, 0.0

        try:
            # Get memory usage
            memory_allocated = torch.cuda.memory_allocated() / 1024 ** 3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024 ** 3  # GB

            # Estimate utilization based on memory usage (rough approximation)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            utilization_estimate = min(100, (memory_allocated / total_memory) * 100)

            return utilization_estimate, memory_allocated
        except:
            return 0.0, 0.0

    def _log_system_status(self):
        """Log system performance"""
        gpu_util, gpu_memory = self._get_gpu_stats()

        if self.matches_completed > 0:
            runtime = time.time() - self.session_start_time
            matches_per_minute = (self.matches_completed / runtime) * 60

            print(f"📊 System Status:")
            print(f"   🎯 GPU: {gpu_util:.1f}% utilization, {gpu_memory:.2f}GB memory")
            print(f"   🏆 Matches: {self.matches_completed} completed")
            print(f"   ⚡ Performance: {matches_per_minute:.1f} matches/min")

            if self.gridworld_renderer:
                perf_stats = self.gridworld_renderer.get_performance_stats()
                print(f"   🎨 Rendering: {perf_stats.get('current_fps', 0):.1f} FPS")

    async def _cleanup_session(self):
        """Clean up system components"""
        print("🧹 Cleaning up system components...")

        self.running = False

        # Stop visual components
        if self.gridworld_renderer:
            self.gridworld_renderer.stop_rendering()

        print("✅ Cleanup complete")

    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        print(f"\n🛑 Received signal {signum} - initiating graceful shutdown...")
        self.running = False

        # Run cleanup
        try:
            if self.gridworld_renderer:
                self.gridworld_renderer.stop_rendering()
        except:
            pass

        sys.exit(0)

    def print_system_info(self):
        """Print comprehensive system information"""
        print("🌟 NEXUS System Information:")
        print("=" * 50)

        # Hardware info
        print("🖥️  Hardware:")
        print(f"   • Device: {self.device}")
        if torch.cuda.is_available():
            print(f"   • GPU: {torch.cuda.get_device_name()}")
            print(f"   • CUDA Version: {torch.version.cuda}")
            print(f"   • GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")

        try:
            print(f"   • CPU: {psutil.cpu_count()} cores")
            print(f"   • RAM: {psutil.virtual_memory().total / 1024 ** 3:.1f}GB")
        except:
            print(f"   • System info: Available")

        # Configuration
        print("\n⚙️  Configuration:")
        print(f"   • Self-play mode: {self.config['self_play']['competitive_mode']}")
        print(f"   • Population size: {self.config['self_play']['population_size']}")
        print(f"   • Match duration: {self.config['self_play']['match_duration']} steps")
        print(f"   • Grid size: {self.config['environment']['grid_size']}x{self.config['environment']['grid_size']}")

        # Performance estimates
        print("\n🚀 Performance Estimates:")
        if torch.cuda.is_available():
            print(f"   • Expected training speed: 300-500 steps/sec")
            print(f"   • Expected rendering FPS: 45-60 FPS")
            print(f"   • Expected matches/hour: 100-200")
            print(f"   • GPU utilization target: 40-60%")
        else:
            print(f"   • CPU-only mode - reduced performance expected")

        print("=" * 50)


async def main():
    """Main execution function"""
    print("🌟 Project NEXUS - Self-Play Training with Visual Analytics")
    print("🚀 RTX 3060 Accelerated Multi-Agent Reinforcement Learning")
    print()

    # Initialize system
    system_manager = NexusSystemManager()
    system_manager.print_system_info()

    # Initialize all components
    await system_manager.initialize_system()

    print("\n🎪 Starting training session...")
    print("🎨 GridWorld visualization will open in a separate window")
    print("⌨️  Press Ctrl+C to stop training gracefully")
    print()

    # Run training session
    try:
        session_results = await system_manager.start_training_session(num_matches=50)

        if session_results:
            print("\n🎉 NEXUS Training Session Complete!")

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n🌟 Thank you for using Project NEXUS!")


def create_default_config():
    """Create default configuration file"""
    config_path = "configs/self_play_config.yaml"

    # Create configs directory if it doesn't exist
    Path("configs").mkdir(exist_ok=True)

    if not Path(config_path).exists():
        system_manager = NexusSystemManager()
        default_config = system_manager._get_default_config()

        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)

        print(f"✅ Created default config: {config_path}")
    else:
        print(f"⚠️  Config already exists: {config_path}")


if __name__ == "__main__":
    # Check for config creation
    if len(sys.argv) > 1 and sys.argv[1] == "--create-config":
        create_default_config()
        sys.exit(0)

    # Run main system
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Shutdown complete")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


