# File: tests/test_multi_resource_system.py
"""
Comprehensive test suite for Day 5-6 Multi-Resource System
Tests resource clustering, tool crafting, gathering mechanics, and integration
"""

import pytest
import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Any
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from environment.advanced_resource_system import (
        AdvancedResourceSystem, ResourceType, ToolType, Tool, ResourceProperties
    )
    from environment.enhanced_grid_world_v3 import (
        EnhancedGridWorldV3, create_tool_crafting_scenario, create_exploration_scenario
    )
    MULTI_RESOURCE_AVAILABLE = True
except ImportError:
    MULTI_RESOURCE_AVAILABLE = False
    print("⚠️ Multi-Resource System not available - tests will be skipped")


class TestAdvancedResourceSystem:
    """Test suite for advanced resource system functionality"""
    
    @pytest.mark.skipif(not MULTI_RESOURCE_AVAILABLE, reason="Multi-Resource System not available")
    def test_resource_clustering_generation(self):
        """Test realistic resource clustering generation"""
        print("\n🗂️ Testing resource clustering generation...")
        
        world_sizes = [(30, 30), (50, 50), (75, 75)]
        
        for world_size in world_sizes:
            resource_system = AdvancedResourceSystem(world_size, resource_density=0.03)
            resource_system.generate_resource_distribution(seed=42)
            
            stats = resource_system.get_resource_statistics()
            
            # Verify clusters were created
            assert stats['generation_stats']['total_clusters'] > 0, "No clusters generated"
            assert stats['generation_stats']['total_nodes'] > 0, "No resource nodes generated"
            
            # Verify reasonable clustering
            nodes_per_cluster = stats['generation_stats']['nodes_per_cluster']
            assert 2.0 <= nodes_per_cluster <= 15.0, f"Unreasonable cluster size: {nodes_per_cluster}"
            
            # Verify all resource types are represented
            resource_types_found = list(stats['resource_distribution'].keys())
            assert len(resource_types_found) >= 4, f"Too few resource types: {resource_types_found}"
            
            print(f"   {world_size}: {stats['generation_stats']['total_clusters']} clusters, "
                  f"{stats['generation_stats']['total_nodes']} nodes, "
                  f"{nodes_per_cluster:.1f} nodes/cluster")
        
        print("   ✅ Resource clustering generation working correctly")
    
    @pytest.mark.skipif(not MULTI_RESOURCE_AVAILABLE, reason="Multi-Resource System not available")
    def test_tool_crafting_mechanics(self):
        """Test tool crafting system"""
        print("\n🔨 Testing tool crafting mechanics...")
        
        resource_system = AdvancedResourceSystem((20, 20), resource_density=0.05)
        
        # Test all tool types
        test_cases = [
            (ToolType.AXE, {'wood': 5, 'stone': 5, 'metal_ore': 3}),
            (ToolType.PICKAXE, {'wood': 5, 'stone': 5, 'metal_ore': 3}),
            (ToolType.BUCKET, {'wood': 5, 'metal_ore': 3}),
            (ToolType.SCYTHE, {'wood': 5, 'stone': 5, 'metal_ore': 5})
        ]
        
        for tool_type, available_resources in test_cases:
            # Convert to ResourceType format
            formatted_resources = {}
            for resource_name, amount in available_resources.items():
                resource_type = ResourceType(resource_name)
                formatted_resources[resource_type] = amount
            
            # Test crafting possibility check
            can_craft = resource_system.can_craft_tool(tool_type, formatted_resources)
            assert can_craft, f"Should be able to craft {tool_type.value}"
            
            # Test actual crafting
            craft_result = resource_system.craft_tool(tool_type, formatted_resources)
            assert craft_result['success'], f"Failed to craft {tool_type.value}"
            
            tool = craft_result['tool']
            assert tool.tool_type == tool_type, "Wrong tool type created"
            assert tool.durability > 0, "Tool has no durability"
            assert 0.7 <= tool.efficiency_modifier <= 1.3, "Tool efficiency out of range"
            
            print(f"   ✅ {tool_type.value}: Durability {tool.durability}, Efficiency {tool.efficiency_modifier:.2f}")
        
        # Test insufficient resources
        insufficient_resources = {ResourceType.WOOD: 1}
        can_craft_insufficient = resource_system.can_craft_tool(ToolType.AXE, insufficient_resources)
        assert not can_craft_insufficient, "Should not be able to craft with insufficient resources"
        
        print("   ✅ Tool crafting mechanics working correctly")
    
    @pytest.mark.skipif(not MULTI_RESOURCE_AVAILABLE, reason="Multi-Resource System not available")
    def test_gathering_efficiency_system(self):
        """Test gathering efficiency with different tools"""
        print("\n⚡ Testing gathering efficiency system...")
        
        resource_system = AdvancedResourceSystem((25, 25), resource_density=0.04)
        resource_system.generate_resource_distribution(seed=123)
        
        # Find different resource types for testing
        test_scenarios = []
        for node in resource_system.nodes[:4]:  # Test first 4 nodes
            test_scenarios.append({
                'position': node.position,
                'resource_type': node.resource_type,
                'expected_tool': resource_system.resource_properties[node.resource_type].required_tool
            })
        
        for scenario in test_scenarios:
            position = scenario['position']
            resource_type = scenario['resource_type']
            
            # Test gathering without tool
            result_no_tool = resource_system.gather_resource(
                position=position, agent_id=0, tool=None, current_step=1
            )
            
            # Test gathering with optimal tool
            optimal_tool_type = scenario['expected_tool']
            if optimal_tool_type != ToolType.NONE:
                # Create optimal tool
                tool_resources = {ResourceType.WOOD: 10, ResourceType.STONE: 10, ResourceType.METAL_ORE: 10}
                craft_result = resource_system.craft_tool(optimal_tool_type, tool_resources)
                
                if craft_result['success']:
                    optimal_tool = craft_result['tool']
                    
                    # Test gathering with tool (reset node first by regenerating)
                    resource_system.generate_resource_distribution(seed=123)
                    result_with_tool = resource_system.gather_resource(
                        position=position, agent_id=1, tool=optimal_tool, current_step=2
                    )
                    
                    if result_no_tool['success'] and result_with_tool['success']:
                        efficiency_improvement = result_with_tool['efficiency'] / result_no_tool['efficiency']
                        assert efficiency_improvement > 1.0, f"Tool should improve efficiency for {resource_type.value}"
                        
                        print(f"   {resource_type.value}: No tool {result_no_tool['efficiency']:.2f}, "
                              f"With {optimal_tool_type.value} {result_with_tool['efficiency']:.2f} "
                              f"({efficiency_improvement:.1f}x better)")
        
        print("   ✅ Gathering efficiency system working correctly")
    
    @pytest.mark.skipif(not MULTI_RESOURCE_AVAILABLE, reason="Multi-Resource System not available")
    def test_resource_quality_variation(self):
        """Test resource quality variation system"""
        print("\n💎 Testing resource quality variation...")
        
        resource_system = AdvancedResourceSystem((30, 30), resource_density=0.03)
        resource_system.generate_resource_distribution(seed=456)
        
        # Analyze quality distribution
        quality_values = [node.quality_modifier for node in resource_system.nodes]
        accessibility_values = [node.accessibility for node in resource_system.nodes]
        
        assert len(quality_values) > 0, "No resource nodes to test quality"
        
        # Quality should vary between 0.5 and 1.5
        min_quality = min(quality_values)
        max_quality = max(quality_values)
        avg_quality = np.mean(quality_values)
        
        assert 0.4 <= min_quality <= 0.6, f"Min quality out of range: {min_quality}"
        assert 1.4 <= max_quality <= 1.6, f"Max quality out of range: {max_quality}"
        assert 0.9 <= avg_quality <= 1.1, f"Average quality should be near 1.0: {avg_quality}"
        
        # Accessibility should be reasonable
        min_accessibility = min(accessibility_values)
        max_accessibility = max(accessibility_values)
        avg_accessibility = np.mean(accessibility_values)
        
        assert min_accessibility > 0.3, f"Accessibility too low: {min_accessibility}"
        assert max_accessibility < 2.0, f"Accessibility too high: {max_accessibility}"
        
        print(f"   Quality range: {min_quality:.2f} - {max_quality:.2f} (avg: {avg_quality:.2f})")
        print(f"   Accessibility range: {min_accessibility:.2f} - {max_accessibility:.2f} (avg: {avg_accessibility:.2f})")
        print("   ✅ Resource quality variation working correctly")
    
    @pytest.mark.skipif(not MULTI_RESOURCE_AVAILABLE, reason="Multi-Resource System not available")
    def test_performance_benchmarks(self):
        """Test performance across different scales"""
        print("\n⚡ Testing performance benchmarks...")
        
        test_configs = [
            {'size': (25, 25), 'density': 0.04, 'name': 'Small'},
            {'size': (50, 50), 'density': 0.03, 'name': 'Medium'},
            {'size': (75, 75), 'density': 0.025, 'name': 'Large'}
        ]
        
        performance_results = []
        
        for config in test_configs:
            resource_system = AdvancedResourceSystem(config['size'], config['density'])
            
            # Benchmark generation
            start_time = time.time()
            resource_system.generate_resource_distribution(seed=789)
            generation_time = time.time() - start_time
            
            # Benchmark gathering operations
            start_time = time.time()
            gather_operations = 0
            
            for node in resource_system.nodes[:10]:  # Test first 10 nodes
                result = resource_system.gather_resource(
                    position=node.position, agent_id=0, tool=None, current_step=1
                )
                gather_operations += 1
            
            gathering_time = time.time() - start_time
            gather_ops_per_sec = gather_operations / max(0.001, gathering_time)
            
            stats = resource_system.get_resource_statistics()
            
            result = {
                'name': config['name'],
                'size': config['size'],
                'generation_time_ms': generation_time * 1000,
                'nodes_generated': stats['generation_stats']['total_nodes'],
                'clusters_generated': stats['generation_stats']['total_clusters'],
                'gather_ops_per_sec': gather_ops_per_sec
            }
            
            performance_results.append(result)
            
            print(f"   {config['name']} ({config['size']}): "
                  f"Gen: {generation_time*1000:.1f}ms, "
                  f"Nodes: {result['nodes_generated']}, "
                  f"Gather: {gather_ops_per_sec:.0f} ops/sec")
            
            # Performance should be reasonable
            assert generation_time < 1.0, f"Generation too slow: {generation_time:.3f}s"
            assert gather_ops_per_sec > 100, f"Gathering too slow: {gather_ops_per_sec:.0f} ops/sec"
        
        print("   ✅ Performance benchmarks acceptable")
        return performance_results


class TestEnhancedGridWorldV3:
    """Test suite for enhanced environment integration"""
    
    @pytest.mark.skipif(not MULTI_RESOURCE_AVAILABLE, reason="Multi-Resource System not available")
    def test_environment_integration(self):
        """Test integration between all systems"""
        print("\n🔗 Testing environment integration...")
        
        env = create_tool_crafting_scenario((30, 30))
        obs, info = env.reset(seed=42)
        
        # Verify observation shape
        expected_shape = (12, env.vision_range, env.vision_range)
        assert obs.shape == expected_shape, f"Expected {expected_shape}, got {obs.shape}"
        
        # Verify resource integration
        assert info['resources_remaining'] > 0, "No resources in environment"
        assert len(env.resource_system.clusters) > 0, "No resource clusters"
        
        # Verify agents have enhanced state
        agent = env.agents[0]
        assert 'tools' in agent, "Agent missing tools list"
        assert 'skills' in agent, "Agent missing skills"
        assert 'experience' in agent, "Agent missing experience"
        
        print(f"   ✅ Environment: {env.size}, Obs: {obs.shape}")
        print(f"   ✅ Resources: {info['resources_remaining']} nodes, {len(env.resource_system.clusters)} clusters")
        print(f"   ✅ Agent state: {len(agent['inventory'])} resource types, {len(agent['tools'])} tools")
    
    @pytest.mark.skipif(not MULTI_RESOURCE_AVAILABLE, reason="Multi-Resource System not available")
    def test_tool_crafting_workflow(self):
        """Test complete tool crafting workflow in environment"""
        print("\n🔧 Testing tool crafting workflow...")
        
        env = create_tool_crafting_scenario((25, 25))
        obs, info = env.reset(seed=123)
        
        agent = env.agents[0]
        initial_tools = len(agent['tools'])
        
        # Phase 1: Gather resources for axe crafting
        resources_needed = {'wood': 3, 'stone': 2, 'metal_ore': 1}
        gathering_attempts = 0
        
        for step in range(50):
            # Try to gather resources
            obs, reward, terminated, truncated, info = env.step(8)  # Gather action
            gathering_attempts += 1
            
            # Check if we have enough resources
            has_enough = all(agent['inventory'].get(resource, 0) >= amount 
                           for resource, amount in resources_needed.items())
            
            if has_enough:
                print(f"   Gathered sufficient resources after {gathering_attempts} attempts")
                break
            
            # Move to find more resources if needed
            if step % 3 == 0:
                move_action = env.action_space.sample() % 8  # Random movement
                env.step(move_action)
            
            if terminated or truncated:
                break
        
        # Phase 2: Craft axe
        initial_inventory_total = agent['inventory_total']
        obs, reward, terminated, truncated, info = env.step(14)  # Craft axe action
        
        # Verify crafting result
        final_tools = len(agent['tools'])
        final_inventory_total = agent['inventory_total']
        
        if final_tools > initial_tools:
            print(f"   ✅ Tool crafted successfully: {initial_tools} → {final_tools} tools")
            print(f"   ✅ Resources consumed: {initial_inventory_total} → {final_inventory_total} inventory")
            
            # Phase 3: Test tool usage
            crafted_tool = agent['tools'][-1]  # Most recently crafted
            initial_durability = crafted_tool.durability
            
            # Equip tool
            obs, reward, terminated, truncated, info = env.step(12)  # Use tool action
            assert agent.get('active_tool') is not None, "Tool not equipped"
            
            # Use tool for gathering
            for _ in range(3):
                obs, reward, terminated, truncated,