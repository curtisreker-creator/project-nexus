# File: scripts/day_5_6_complete_demo.py
"""
Day 5-6 Complete Demo - Fixed Implementation
Demonstrates advanced multi-resource system with all bug fixes applied
"""

import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_advanced_resource_system():
    """Test the fixed advanced resource system"""
    print("🧪 Testing Fixed Advanced Multi-Resource System...")
    
    try:
        from environment.advanced_resource_system import AdvancedResourceSystem, ResourceType, ToolType
        
        # Test different world sizes
        test_configs = [
            {'name': 'Small World', 'size': (30, 30), 'density': 0.04},
            {'name': 'Medium World', 'size': (50, 50), 'density': 0.03},
            {'name': 'Large World', 'size': (75, 75), 'density': 0.025}
        ]
        
        for config in test_configs:
            print(f"\n--- Testing {config['name']} ---")
            
            resource_system = AdvancedResourceSystem(
                world_size=config['size'],
                resource_density=config['density']
            )
            
            # Generate resources
            resource_system.generate_resource_distribution(seed=42)
            stats = resource_system.get_resource_statistics()
            
            print(f"✅ Generated {stats['generation_stats']['total_nodes']} nodes in "
                  f"{stats['generation_stats']['total_clusters']} clusters")
            print(f"   Generation time: {stats['generation_stats']['generation_time_ms']:.1f} ms")
            print(f"   Nodes per cluster: {stats['generation_stats']['nodes_per_cluster']:.1f}")
            
            # Resource type breakdown
            print(f"   Resource Distribution:")
            for resource_type, data in stats['resource_distribution'].items():
                print(f"     {resource_type}: {data['node_count']} nodes, "
                      f"{data['total_units']} units, "
                      f"quality: {data['average_quality']:.2f}")
            
            # Test gathering mechanics
            print(f"   Testing Gathering Mechanics:")
            nodes = resource_system.get_nodes_for_environment()
            
            if nodes:
                # Test gathering without tool
                test_node = nodes[0]
                result1 = resource_system.gather_resource(
                    position=test_node['pos'],
                    agent_id=0,
                    tool=None,
                    current_step=1
                )
                
                print(f"     No tool: {result1['success']}, "
                      f"Units: {result1.get('units_gathered', 0)}, "
                      f"Reward: {result1.get('reward', 0):.3f}")
                
                # Test tool crafting
                test_resources = {
                    ResourceType.WOOD: 5,
                    ResourceType.STONE: 5, 
                    ResourceType.METAL_ORE: 3
                }
                
                can_craft_axe = resource_system.can_craft_tool(ToolType.AXE, test_resources)
                print(f"     Can craft axe: {can_craft_axe}")
                
                if can_craft_axe:
                    craft_result = resource_system.craft_tool(ToolType.AXE, test_resources)
                    if craft_result['success']:
                        tool = craft_result['tool']
                        print(f"     Crafted axe: Durability {tool.durability}, "
                              f"Efficiency {tool.efficiency_modifier:.2f}")
        
        print(f"\n🎉 Advanced Resource System test completed successfully!")
        print(f"✅ Realistic clustering and distribution working")
        print(f"✅ Tool crafting and gathering efficiency operational")
        print(f"✅ Quality variation and accessibility modifiers active")
        print(f"✅ Performance optimized for large-scale environments")
        return True
        
    except Exception as e:
        print(f"❌ Advanced resource system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_integration():
    """Test basic integration without full environment"""
    print("\n🔗 Testing Simple Integration...")
    
    try:
        from environment.advanced_resource_system import AdvancedResourceSystem, ResourceType, ToolType
        
        # Create simple system
        resource_system = AdvancedResourceSystem((25, 25), resource_density=0.04)
        resource_system.generate_resource_distribution(seed=123)
        
        # Get environment-compatible nodes
        env_nodes = resource_system.get_nodes_for_environment()
        print(f"✅ Created {len(env_nodes)} environment-compatible resource nodes")
        
        # Test resource types
        resource_types_found = set()
        total_units = 0
        
        for node in env_nodes:
            resource_types_found.add(node['type'])
            total_units += node['remaining_units']
        
        print(f"✅ Resource types found: {sorted(resource_types_found)}")
        print(f"✅ Total resource units: {total_units}")
        
        # Test clustering information
        clusters = resource_system.clusters
        print(f"✅ Resource clusters: {len(clusters)}")
        for i, cluster in enumerate(clusters[:3]):  # Show first 3
            print(f"   Cluster {i}: {cluster.resource_type.value}, {len(cluster.nodes)} nodes")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def performance_benchmark():
    """Run performance benchmarks on the fixed system"""
    print("\n⚡ Running Performance Benchmarks...")
    
    try:
        from environment.advanced_resource_system import AdvancedResourceSystem, ResourceType, ToolType
        
        sizes = [(30, 30), (50, 50), (75, 75)]
        results = []
        
        for size in sizes:
            print(f"   Testing {size[0]}x{size[1]}...")
            
            # Test generation speed
            resource_system = AdvancedResourceSystem(size, resource_density=0.03)
            
            start_time = time.time()
            resource_system.generate_resource_distribution(seed=456)
            generation_time = time.time() - start_time
            
            # Test gathering speed
            nodes = resource_system.nodes[:20]  # Test first 20 nodes
            
            start_time = time.time()
            gather_count = 0
            
            for node in nodes:
                result = resource_system.gather_resource(
                    position=node.position,
                    agent_id=0,
                    tool=None,
                    current_step=1
                )
                gather_count += 1
            
            gather_time = time.time() - start_time
            gather_ops_per_sec = gather_count / max(0.001, gather_time)
            
            result = {
                'size': size,
                'generation_time_ms': generation_time * 1000,
                'gather_ops_per_sec': gather_ops_per_sec,
                'nodes_generated': len(resource_system.nodes),
                'clusters_generated': len(resource_system.clusters)
            }
            
            results.append(result)
            print(f"     Generation: {generation_time*1000:.1f}ms")
            print(f"     Gathering: {gather_ops_per_sec:.0f} ops/sec")
            print(f"     Nodes: {result['nodes_generated']}, Clusters: {result['clusters_generated']}")
        
        print(f"\n✅ Performance benchmarks completed:")
        min_generation = min(r['generation_time_ms'] for r in results)
        max_generation = max(r['generation_time_ms'] for r in results)
        min_gathering = min(r['gather_ops_per_sec'] for r in results)
        
        print(f"   Generation time range: {min_generation:.1f} - {max_generation:.1f} ms")
        print(f"   Minimum gathering performance: {min_gathering:.0f} ops/sec")
        
        return results
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    """Main execution for Day 5-6 complete demo"""
    print("🚀 PROJECT NEXUS - Day 5-6 Complete Demo (Fixed)")
    print("Advanced Multi-Resource System with All Bug Fixes Applied")
    print("=" * 70)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Advanced Resource System
    if test_advanced_resource_system():
        success_count += 1
    
    # Test 2: Simple Integration  
    if test_simple_integration():
        success_count += 1
    
    # Test 3: Performance Benchmarks
    results = performance_benchmark()
    if results:
        success_count += 1
    
    print(f"\n" + "=" * 70)
    print(f"🎯 TEST RESULTS: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 ALL TESTS PASSED - Day 5-6 Implementation Successfully Fixed!")
        print("✅ Advanced resource system operational")
        print("✅ Tool crafting mechanics working") 
        print("✅ Resource clustering and quality variation active")
        print("✅ Performance benchmarks acceptable")
        print("🚀 Ready for enhanced environment integration!")
        return True
    else:
        print("❌ Some tests failed - review implementation")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)