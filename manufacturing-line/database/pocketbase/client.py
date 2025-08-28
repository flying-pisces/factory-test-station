"""PocketBase client wrapper for manufacturing line system."""

import os
import json
import requests
from typing import Dict, Any, List, Optional
from dataclasses import asdict
import logging

from .schemas.users import get_user_schemas
from .schemas.components import get_component_schemas  
from .schemas.stations import get_station_schemas


class PocketBaseClient:
    """PocketBase client wrapper with authentication and schema management."""
    
    def __init__(self, base_url: str = None, admin_email: str = None, admin_password: str = None):
        self.base_url = base_url or os.getenv('POCKETBASE_URL', 'http://localhost:8090')
        self.admin_email = admin_email or os.getenv('POCKETBASE_ADMIN_EMAIL', 'admin@manufacturing.local')
        self.admin_password = admin_password or os.getenv('POCKETBASE_ADMIN_PASSWORD', 'adminpassword123')
        
        self.session = requests.Session()
        self.auth_token = None
        self.logger = logging.getLogger('PocketBaseClient')
        
        # Remove trailing slash if present
        self.base_url = self.base_url.rstrip('/')
        
    def authenticate_admin(self) -> bool:
        """Authenticate as admin user."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/admins/auth-with-password",
                json={
                    "identity": self.admin_email,
                    "password": self.admin_password
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.auth_token = data.get('token')
                self.session.headers.update({
                    'Authorization': f'Bearer {self.auth_token}'
                })
                self.logger.info("Admin authentication successful")
                return True
            else:
                self.logger.error(f"Admin authentication failed: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Admin authentication request failed: {e}")
            return False
    
    def create_collection(self, schema: Dict[str, Any]) -> bool:
        """Create a collection with the given schema."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/collections",
                json=schema,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                self.logger.info(f"Collection '{schema['name']}' created successfully")
                return True
            elif response.status_code == 400:
                # Collection might already exist
                error_data = response.json()
                if "already exists" in str(error_data):
                    self.logger.info(f"Collection '{schema['name']}' already exists")
                    return True
                else:
                    self.logger.error(f"Collection creation failed: {error_data}")
                    return False
            else:
                self.logger.error(f"Collection creation failed: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Collection creation request failed: {e}")
            return False
    
    def update_collection(self, collection_name: str, schema: Dict[str, Any]) -> bool:
        """Update an existing collection schema."""
        try:
            response = self.session.patch(
                f"{self.base_url}/api/collections/{collection_name}",
                json=schema,
                timeout=30
            )
            
            if response.status_code == 200:
                self.logger.info(f"Collection '{collection_name}' updated successfully")
                return True
            else:
                self.logger.error(f"Collection update failed: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Collection update request failed: {e}")
            return False
    
    def get_collection(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get collection information."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/collections/{collection_name}",
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Get collection request failed: {e}")
            return None
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/collections",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('items', [])
            else:
                return []
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"List collections request failed: {e}")
            return []
    
    def create_record(self, collection: str, data: Dict[str, Any]) -> Optional[str]:
        """Create a record in the specified collection."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/collections/{collection}/records",
                json=data,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                return result.get('id')
            else:
                self.logger.error(f"Record creation failed: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Record creation request failed: {e}")
            return None
    
    def get_record(self, collection: str, record_id: str) -> Optional[Dict[str, Any]]:
        """Get a record from the specified collection."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/collections/{collection}/records/{record_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Get record request failed: {e}")
            return None
    
    def list_records(self, collection: str, page: int = 1, per_page: int = 30, 
                    filter_query: str = None) -> Dict[str, Any]:
        """List records from the specified collection."""
        try:
            params = {
                'page': page,
                'perPage': per_page
            }
            
            if filter_query:
                params['filter'] = filter_query
            
            response = self.session.get(
                f"{self.base_url}/api/collections/{collection}/records",
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {'items': [], 'totalItems': 0, 'totalPages': 0}
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"List records request failed: {e}")
            return {'items': [], 'totalItems': 0, 'totalPages': 0}
    
    def initialize_database_schema(self) -> bool:
        """Initialize all database collections with their schemas."""
        self.logger.info("Starting database schema initialization...")
        
        # Authenticate as admin first
        if not self.authenticate_admin():
            self.logger.error("Cannot initialize schema - admin authentication failed")
            return False
        
        # Get all schema definitions
        all_schemas = []
        all_schemas.extend(get_user_schemas())
        all_schemas.extend(get_component_schemas())
        all_schemas.extend(get_station_schemas())
        
        success_count = 0
        total_count = len(all_schemas)
        
        for schema in all_schemas:
            collection_name = schema['name']
            
            # Check if collection already exists
            existing = self.get_collection(collection_name)
            
            if existing:
                # Update existing collection
                if self.update_collection(collection_name, schema):
                    success_count += 1
                else:
                    self.logger.warning(f"Failed to update collection: {collection_name}")
            else:
                # Create new collection
                if self.create_collection(schema):
                    success_count += 1
                else:
                    self.logger.error(f"Failed to create collection: {collection_name}")
        
        self.logger.info(f"Schema initialization complete: {success_count}/{total_count} collections processed")
        return success_count == total_count
    
    def validate_database_operations(self) -> Dict[str, Any]:
        """Validate database operations for testing."""
        results = {
            'connection_test': False,
            'authentication_test': False,
            'collections_count': 0,
            'crud_operations': False,
            'performance_metrics': {}
        }
        
        try:
            # Test connection
            import time
            start_time = time.time()
            
            response = self.session.get(f"{self.base_url}/api/health", timeout=5)
            if response.status_code == 200:
                results['connection_test'] = True
                connection_time = time.time() - start_time
                results['performance_metrics']['connection_time_ms'] = int(connection_time * 1000)
            
            # Test authentication
            if self.authenticate_admin():
                results['authentication_test'] = True
            
            # Test collections
            collections = self.list_collections()
            results['collections_count'] = len(collections)
            
            # Test CRUD operations with a test record
            if results['authentication_test']:
                test_data = {
                    'component_id': 'TEST_VALIDATION',
                    'component_type': 'resistor',
                    'processing_status': 'uploaded',
                    'upload_timestamp': '2024-01-01 00:00:00'
                }
                
                # Try to create and delete test record
                try:
                    record_id = self.create_record('raw_components', test_data)
                    if record_id:
                        # Try to read the record
                        record = self.get_record('raw_components', record_id)
                        if record and record.get('component_id') == 'TEST_VALIDATION':
                            results['crud_operations'] = True
                except Exception as e:
                    self.logger.debug(f"CRUD test failed (expected during development): {e}")
            
        except Exception as e:
            self.logger.error(f"Database validation failed: {e}")
        
        return results
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on PocketBase instance."""
        try:
            response = self.session.get(f"{self.base_url}/api/health", timeout=5)
            
            if response.status_code == 200:
                return {
                    'status': 'healthy',
                    'response_time_ms': response.elapsed.total_seconds() * 1000,
                    'version': response.headers.get('Server', 'Unknown')
                }
            else:
                return {
                    'status': 'unhealthy',
                    'error': f"HTTP {response.status_code}"
                }
        except requests.exceptions.RequestException as e:
            return {
                'status': 'error',
                'error': str(e)
            }


# Global client instance
pocketbase_client = PocketBaseClient()