# File: emergency_test_fixes.py
"""
Emergency test to validate all syntax fixes
"""

def test_resource_system():
    """Test resource system standalone"""
    print("🔧 Testing Resource System...")
    
    try:
        import sys
        import os
        sys.path.append('/Users/curtis/project_nexus')
        
        # Test imports first
        from environment.advanced_resource_system import ResourceType, ToolType
        print("✅ Basic imports working")
        
        from environment.advanced_resource_system import AdvancedResourceSystem
        print("✅ AdvancedResourceSystem import working")
        
        # Test creation
        system = AdvancedResourceSystem((20, 20), 0.03)
        print("✅ System creation working")
        
        # Test generation - this is where the bug was
        system.generate_resource_distribution(seed=42)
        print("✅ Resource generation working (numpy fixes applied)")
        
        # Test basic stats
        stats = system.get_resource_statistics()
        nodes = stats['generation_stats']['total_nodes']
        clusters = stats['generation_stats']['total_clusters']
        print(f"✅ Generated {nodes} nodes in {clusters} clusters")
        
        # Test tool crafting
        resources = {ResourceType.WOOD: 5, ResourceType.STONE: 3, ResourceType.METAL_ORE: 2}
        can_craft = system.can_craft_tool(ToolType.AXE, resources)
        print(f"✅ Tool crafting check: {can_craft}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🚨 EMERGENCY VALIDATION - Day 5-6 Fixes")
    print("=" * 50)
    
    if test_resource_system():
        print("\n🎉 ALL CRITICAL FIXES WORKING!")
        print("✅ Syntax errors resolved")
        print("✅ Indentation fixed")
        print("✅ Random module calls corrected")
        print("🚀 System ready for full testing")
    else:
        print("\n❌ Fixes still need work")
    
if __name__ == "__main__":
    main()