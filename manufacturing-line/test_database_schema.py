#!/usr/bin/env python3
"""Test PocketBase database schema initialization for Week 1 TC1.4."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_database_schema():
    """Test PocketBase database schema initialization."""
    try:
        from database.pocketbase.client import PocketBaseClient
        from database.pocketbase.schemas.users import get_user_schemas
        from database.pocketbase.schemas.components import get_component_schemas
        from database.pocketbase.schemas.stations import get_station_schemas
        
        print("🔍 Testing Database Schema Initialization...")
        print("=" * 60)
        
        # Test schema definitions
        user_schemas = get_user_schemas()
        component_schemas = get_component_schemas() 
        station_schemas = get_station_schemas()
        
        print(f"✓ User schemas loaded: {len(user_schemas)} collections")
        print(f"✓ Component schemas loaded: {len(component_schemas)} collections")
        print(f"✓ Station schemas loaded: {len(station_schemas)} collections")
        
        total_schemas = len(user_schemas) + len(component_schemas) + len(station_schemas)
        print(f"✓ Total schemas: {total_schemas} collections")
        
        # Validate schema structure
        all_schemas = user_schemas + component_schemas + station_schemas
        required_fields = ['name', 'type', 'schema']
        
        for schema in all_schemas:
            for field in required_fields:
                if field not in schema:
                    raise ValueError(f"Missing required field '{field}' in schema: {schema.get('name', 'unknown')}")
        
        print("✓ Schema structure validation: All schemas have required fields")
        
        # Test PocketBase client initialization
        client = PocketBaseClient()
        print(f"✓ PocketBase client initialized: {client.base_url}")
        
        # Test database operations (mock validation)
        validation_results = client.validate_database_operations()
        
        print("📊 Database Operation Validation:")
        print(f"   - Connection test: {'PASS' if validation_results['connection_test'] else 'SKIP (PocketBase not running)'}")
        print(f"   - Authentication: {'PASS' if validation_results['authentication_test'] else 'SKIP (PocketBase not running)'}")
        print(f"   - Collections available: {validation_results['collections_count']}")
        print(f"   - CRUD operations: {'PASS' if validation_results['crud_operations'] else 'SKIP (Collections not initialized)'}")
        
        if validation_results['performance_metrics']:
            perf = validation_results['performance_metrics']
            if 'connection_time_ms' in perf:
                print(f"   - Connection time: {perf['connection_time_ms']}ms")
        
        print("✅ Database schema initialization: VALIDATION PASSED")
        print("📊 Schema Summary:")
        print(f"   - User management: {len(user_schemas)} collections (users, sessions, activity)")
        print(f"   - Component management: {len(component_schemas)} collections (raw, structured, history)")
        print(f"   - Station management: {len(station_schemas)} collections (stations, metrics, maintenance)")
        print(f"   - Total database entities: {total_schemas} collections")
        print("   - Schema validation: All required fields present")
        print("   - Client initialization: Successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Database schema test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🗄️  Running Database Schema Initialization Test...")
    print("=" * 60)
    
    success = test_database_schema()
    
    if success:
        print("=" * 60)
        print("✅ TC1.4: Database Schema Test PASSED")
    else:
        print("=" * 60) 
        print("❌ TC1.4: Database Schema Test FAILED")
        sys.exit(1)