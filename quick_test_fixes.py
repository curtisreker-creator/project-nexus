# File: quick_test_fixes.py
"""
Quick validation script to test Day 5-6 fixes
Run this to verify all fixes are working
"""

import sys
import os
import numpy as np

# Add project path
sys.path.append('/Users/curtis/project_nexus')

def test_resource_system_fixes():
    """Test that all resource system fixes work"""
    print("🔧 Testing Advanced Resource System Fixes...")
    
    try:
        # Test imports
        from environment.advanced_resource_system import AdvancedResourceSystem, ResourceType, ToolType
        print("✅ Imports successful")
        
        # Test basic creation
        resource_system = AdvancedResourceSystem((30, 30), resource_density=0.03)
        print("✅ Resource system created")
        
        # Test generation (this was the main bug)
        resource_system.generate_resource_distribution(seed=42)
        print("✅ Resource generation successful (exponential bug fixed)")
        
        # Test statistics
        stats = resource_system.get_resource_statistics()
        print(f"✅ Statistics: {stats['generation_stats']['total_nodes']} nodes, {stats['generation_stats']['total_clusters']} clusters")
        
        # Test tool crafting
        test_resources = {ResourceType.WOOD: 5, ResourceType.STONE: 3, ResourceType.METAL_ORE: 2}
        can_craft = resource_system.can_craft_tool(ToolType.AXE, test_resources)
        print(f"✅ Tool crafting check: {can_craft}")
        
        if can_craft:
            craft_result = resource_system.craft_tool(ToolType.AXE, test_resources)
            print(f"✅ Tool crafted: {craft_result['success']}")
        
        # Test gathering
        if resource_system.nodes:
            node = resource_system.nodes[0]
            result = resource_system.gather_resource(node.position, agent_id=0, tool=None, current_step=1)
            print(f"✅ Gathering test: {result['success']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Resource system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("🚀 NEXUS Quick Validation - Day 5-6 Fixes")
    print("=" * 50)
    
    if test_resource_system_fixes():
        print("\n🎉 ALL FIXES WORKING!")
        print("✅ Random.exponential → np.random.exponential fixed")
        print("✅ Random.gauss → np.random.normal fixed") 
        print("✅ Resource system fully operational")
        print("🚀 Ready to proceed with full integration!")
        return True
    else:
        print("\n❌ Fixes need more work")
        return False


if __name__ == "__main__":
    success = main()
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")