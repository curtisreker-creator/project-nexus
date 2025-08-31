# File: scripts/day_3_4_field_of_view_demo.py
"""
Day 3-4 Integration Demo: Advanced Field of View System
Demonstrates fog of war mechanics, exploration incentives, and agent memory systems
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# FIX: Use the correct, case-sensitive filenames and V2 class names
try:
    from environment.EnhancedGridWorldV2 import EnhancedGridWorldV2
    from environment.field_of_view import FieldOfViewSystem, FieldOfViewEnhancedGridWorld
    FOV_AVAILABLE = True
except ImportError as e:
    FOV_AVAILABLE = False
    IMPORT_ERROR_MSG = str(e)


# Only define functions if the necessary modules were successfully imported
if FOV_AVAILABLE:
    def demo_basic_field_of_view():
        """Demonstrate basic field of view mechanics"""
        print("👁️ NEXUS Basic Field of View Demo")
        print("=" * 40)
        
        vision_range = 7
        fov_system = FieldOfViewSystem(vision_range=vision_range)
        
        # Create test environment
        world_size = (25, 25)
        world_state = {
            'world_size': world_size,
            'grid': np.zeros(world_size),
            'resources': [
                {'pos': (8, 8), 'type': 'wood', 'remaining_units': 5},
                {'pos': (15, 12), 'type': 'stone', 'remaining_units': 3},
                {'pos': (20, 20), 'type': 'coal', 'remaining_units': 2}
            ],
            'agents': [{'id': 0, 'pos': (10, 10)}]
        }
        
        print(f"World Size: {world_size}")
        print(f"Vision Range: {vision_range}x{vision_range}")
        print(f"Resources: {len(world_state['resources'])} nodes")
        
        # Test agent vision at different positions
        test_positions = [(10, 10), (8, 8), (15, 12), (20, 20), (5, 5)]
        
        for i, pos in enumerate(test_positions):
            world_state['agents'][0]['pos'] = pos
            observation, exploration_reward = fov_system.update_agent_vision(0, pos, world_state)
            
            visible_cells = np.sum(observation != -1.0)
            memory_coverage = np.sum(fov_system.render_fog_of_war(0, world_size) > 0.1)
            
            print(f"Position {i+1} {pos}: Visible cells: {visible_cells}, "
                  f"Exploration reward: {exploration_reward:.3f}, "
                  f"Memory coverage: {memory_coverage} cells")
        
        # Get final statistics
        exploration_stats = fov_system.get_exploration_stats()
        performance_stats = fov_system.get_performance_stats()
        
        print(f"\n📊 Final Statistics:")
        print(f"  Total cells discovered: {exploration_stats['global_metrics']['total_cells_discovered']}")
        print(f"  Exploration coverage: {exploration_stats['global_metrics']['average_coverage']:.1%}")
        print(f"  Average update time: {performance_stats['avg_update_time_ms']:.2f} ms")
        print(f"  Updates per second: {performance_stats['updates_per_second']:.0f}")


    def demo_exploration_incentives():
        """Demonstrate exploration incentive system"""
        print("\n🗺️ NEXUS Exploration Incentives Demo")
        print("=" * 40)
        
        vision_range = 5
        fov_system = FieldOfViewSystem(vision_range=vision_range)
        world_size = (20, 20)
        
        world_state = {
            'world_size': world_size,
            'grid': np.zeros(world_size),
            'resources': [
                {'pos': (5, 15), 'type': 'wood', 'remaining_units': 4},
                {'pos': (15, 5), 'type': 'stone', 'remaining_units': 6}
            ],
            'agents': [{'id': 0, 'pos': (10, 10)}]
        }
        
        # Simulate systematic exploration pattern
        exploration_pattern = [
            (10, 10), (11, 10), (12, 10), (13, 10), (14, 10),
            (14, 11), (14, 12), (14, 13), (14, 14), (14, 15),
            (13, 15), (12, 15), (11, 15), (10, 15), (9, 15),
            (10, 10), (11, 10), (12, 10)
        ]
        
        exploration_data = []
        
        print("Exploration Pattern Analysis:")
        for step, pos in enumerate(exploration_pattern):
            world_state['agents'][0]['pos'] = pos
            observation, exploration_reward = fov_system.update_agent_vision(0, pos, world_state)
            
            fog_map = fov_system.render_fog_of_war(0, world_size)
            coverage_percentage = np.sum(fog_map > 0.1) / np.prod(world_size)
            
            exploration_data.append({'reward': exploration_reward})
            
            phase = "NEW" if step < 15 else "REVISIT"
            print(f"  Step {step+1:2d} ({phase:7s}): {pos} → Reward: {exploration_reward:.3f}, Coverage: {coverage_percentage:.1%}")
        
        # Analysis
        new_area_rewards = [d['reward'] for d in exploration_data[:15]]
        revisit_rewards = [d['reward'] for d in exploration_data[15:]]
        
        avg_new_reward = np.mean(np.array(new_area_rewards))
        avg_revisit_reward = np.mean(np.array(revisit_rewards))
        
        print(f"\n📈 Exploration Analysis:")
        print(f"  Average NEW area reward: {avg_new_reward:.3f}")
        print(f"  Average REVISIT reward: {avg_revisit_reward:.3f}")


    def demo_memory_decay():
        """Demonstrate memory decay mechanics"""
        print("\n⏰ NEXUS Memory Decay Demo")
        print("=" * 40)
        
        vision_range = 7
        memory_decay_rate = 0.95
        fov_system = FieldOfViewSystem(vision_range=vision_range, memory_decay_rate=memory_decay_rate)
        
        world_size = (15, 15)
        world_state = {
            'world_size': world_size,
            'grid': np.zeros(world_size),
            'resources': [{'pos': (7, 7), 'type': 'wood', 'remaining_units': 3}],
            'agents': [{'id': 0, 'pos': (7, 7)}]
        }
        
        print(f"Memory decay rate: {memory_decay_rate} per step")
        
        observation, _ = fov_system.update_agent_vision(0, (7, 7), world_state)
        initial_confidence = fov_system.agent_memories[0].confidence_grid[7, 7]
        
        print(f"Initial confidence at (7, 7): {initial_confidence:.3f}")
        
        final_confidence = 0
        for step in range(1, 21):
            world_state['agents'][0]['pos'] = (1, 1)
            fov_system.update_agent_vision(0, (1, 1), world_state)
            current_confidence = fov_system.agent_memories[0].confidence_grid[7, 7]
            if step == 20:
                final_confidence = current_confidence

        world_state['agents'][0]['pos'] = (7, 7)
        fov_system.update_agent_vision(0, (7, 7), world_state)
        recovered_confidence = fov_system.agent_memories[0].confidence_grid[7, 7]
        
        print(f"\nReturn to (7, 7):")
        print(f"  Confidence before return: {final_confidence:.3f}")
        print(f"  Confidence after return: {recovered_confidence:.3f}")


    def demo_integration_with_environment():
        """Demonstrate full integration with enhanced environment"""
        print("\n🔗 NEXUS Full Integration Demo")
        print("=" * 40)
        
        # FIX: Instantiate the correct V2 class
        base_env = EnhancedGridWorldV2(size=(30, 30), n_agents=1, vision_range=7, max_resources=12)
        fov_env = FieldOfViewEnhancedGridWorld(base_env, vision_range=7, exploration_reward_scale=1.5)
        
        print(f"Environment: {base_env.size} with {len(base_env.agents)} agents")
        
        obs, info = fov_env.reset(seed=42)
        
        print(f"Initial observation shape: {obs.shape}")
        
        print(f"\nSimulation Results:")
        for step in range(20):
            action = fov_env.base_env.action_space.sample()
            obs, reward, terminated, truncated, info = fov_env.step(action)
            
            if step % 5 == 0:
                agent_pos = fov_env.base_env.agents[0]['pos']
                fog_coverage = info.get('fog_of_war_coverage', 0)
                print(f"  Step {step+1:2d}: Action {action:2d}, Pos {agent_pos}, "
                      f"Total Reward: {reward:+.2f}, Coverage: {fog_coverage:.1%}")
            
            if terminated or truncated:
                break


    def performance_comparison():
        """Compare performance across different field of view configurations"""
        print("\n⚡ NEXUS Performance Comparison")
        print("=" * 40)
        
        test_configs = [
            {'vision_range': 5, 'world_size': (25, 25), 'name': 'Small Vision'},
            {'vision_range': 7, 'world_size': (40, 40), 'name': 'Medium Vision'},
            {'vision_range': 9, 'world_size': (60, 60), 'name': 'Large Vision'},
            {'vision_range': 7, 'world_size': (75, 75), 'name': 'Max Scale'}
        ]
        
        for config in test_configs:
            fov_system = FieldOfViewSystem(vision_range=config['vision_range'])
            world_size = config['world_size']
            
            world_state = {
                'world_size': world_size,
                'grid': np.zeros(world_size),
                'resources': [],
                'agents': [{'id': 0, 'pos': (world_size[0]//2, world_size[1]//2)}]
            }
            
            start_time = time.time()
            update_count = 100
            
            for step in range(update_count):
                center_x, center_y = world_size[1]//2, world_size[0]//2
                offset_range = config['vision_range'] // 2
                agent_pos = (
                    center_y + (step % (2 * offset_range + 1)) - offset_range,
                    center_x + ((step // 3) % (2 * offset_range + 1)) - offset_range
                )
                world_state['agents'][0]['pos'] = agent_pos
                fov_system.update_agent_vision(0, agent_pos, world_state)
            
            elapsed_time = time.time() - start_time
            updates_per_second = update_count / elapsed_time
            perf_stats = fov_system.get_performance_stats()
            
            print(f"{config['name']:<12} | Vision: {config['vision_range']}x{config['vision_range']} | "
                  f"World: {world_size[0]}x{world_size[1]} | "
                  f"Speed: {updates_per_second:>6.0f}/sec | "
                  f"Avg: {perf_stats['avg_update_time_ms']:>5.2f}ms")


    def main():
        """Main execution function for Day 3-4"""
        print("🚀 PROJECT NEXUS - Day 3-4 Implementation Execution")
        print("Phase 2B: Advanced Field of View System with Fog of War")
        print("All Subsystems Engaged for Expert-Level Implementation")
        print("=" * 70)
        
        try:
            print("\n🎯 Step 1: Basic Field of View Demonstration")
            demo_basic_field_of_view()
            
            print("\n🎯 Step 2: Exploration Incentives Demonstration")
            demo_exploration_incentives()
            
            print("\n🎯 Step 3: Memory Decay Demonstration")
            demo_memory_decay()
            
            print("\n🎯 Step 4: Full Integration Demonstration")
            demo_integration_with_environment()
            
            print("\n🎯 Step 5: Performance Comparison Analysis")
            performance_comparison()
            
            print("\n🎯 Step 6: Validation Test Suite")
            from tests.test_field_of_view_system import run_day_3_4_validation
            validation_success = run_day_3_4_validation()
            
            if validation_success:
                print("\n🎉 DAY 3-4 IMPLEMENTATION COMPLETE!")
                return True
            else:
                print("\n❌ Day 3-4 validation failed")
                return False
                
        except Exception as e:
            print(f"\n❌ Day 3-4 implementation failed: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    if FOV_AVAILABLE:
        success = main()
        print(f"\n" + "=" * 70)
        if success:
            print("🚀 Day 3-4 EXECUTION SUCCESSFUL")
        else:
            print("❌ Day 3-4 EXECUTION FAILED - Review implementation and fix issues")
        exit(0 if success else 1)
    else:
        print("=" * 70)
        print("❌ CRITICAL ERROR: Required modules could not be imported.")
        print(f"   Reason: {IMPORT_ERROR_MSG}")
        print("=" * 70)
        exit(1)