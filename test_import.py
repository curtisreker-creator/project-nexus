# File: test_import.py

print("🧪 Starting granular import diagnostics...")
all_imports_successful = True

# --- Test 1: field_of_view ---
try:
    print(" -> Importing from 'environment.field_of_view'...")
    from environment.field_of_view import FieldOfViewSystem, FieldOfViewEnhancedGridWorld
    print("    ✅ Success.")
except Exception as e:
    print(f"    ❌ FAILED: {type(e).__name__} - {e}")
    all_imports_successful = False

# --- Test 2: enhanced_grid_world ---
try:
    print(" -> Importing from 'environment.enhanced_grid_world'...")
    from environment.EnhancedGridWorldV2 import EnhancedGridWorld
    print("    ✅ Success.")
except Exception as e:
    print(f"    ❌ FAILED: {type(e).__name__} - {e}")
    all_imports_successful = False

print("\n" + "---" * 10)
if all_imports_successful:
    print("✅ Both modules imported successfully in isolation.")
    print("   The problem is extremely unusual and may be related to circular dependencies.")
else:
    print("❌ The module marked 'FAILED' above is the source of the error.")
    print("   Please check that file for syntax errors or missing imports (like numpy, etc.).")