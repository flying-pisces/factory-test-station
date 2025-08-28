"""User and authentication schema definitions for PocketBase."""

from typing import Dict, Any, List
from enum import Enum


class UserRole(Enum):
    """User role types in the manufacturing system."""
    SUPER_ADMIN = "super_admin"
    LINE_MANAGER = "line_manager" 
    STATION_ENGINEER = "station_engineer"
    COMPONENT_VENDOR = "component_vendor"


# PocketBase collection schema definitions
USER_SCHEMA = {
    "name": "users",
    "type": "auth",
    "system": False,
    "schema": [
        {
            "id": "users_name",
            "name": "name",
            "type": "text",
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "min": 2,
                "max": 100,
                "pattern": ""
            }
        },
        {
            "id": "users_avatar",
            "name": "avatar",
            "type": "file",
            "system": False,
            "required": False,
            "unique": False,
            "options": {
                "maxSelect": 1,
                "maxSize": 5242880,
                "mimeTypes": ["image/jpeg", "image/png", "image/svg+xml", "image/gif"],
                "thumbs": ["100x100"]
            }
        },
        {
            "id": "users_role",
            "name": "role",
            "type": "select",
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "maxSelect": 1,
                "values": [role.value for role in UserRole]
            }
        },
        {
            "id": "users_organization", 
            "name": "organization",
            "type": "text",
            "system": False,
            "required": False,
            "unique": False,
            "options": {
                "min": 2,
                "max": 200,
                "pattern": ""
            }
        },
        {
            "id": "users_department",
            "name": "department", 
            "type": "text",
            "system": False,
            "required": False,
            "unique": False,
            "options": {
                "min": 2,
                "max": 100,
                "pattern": ""
            }
        },
        {
            "id": "users_permissions",
            "name": "permissions",
            "type": "json",
            "system": False,
            "required": False,
            "unique": False,
            "options": {}
        },
        {
            "id": "users_preferences",
            "name": "preferences",
            "type": "json",
            "system": False,
            "required": False,
            "unique": False,
            "options": {}
        },
        {
            "id": "users_last_login",
            "name": "last_login",
            "type": "date",
            "system": False,
            "required": False,
            "unique": False,
            "options": {}
        },
        {
            "id": "users_active",
            "name": "active",
            "type": "bool",
            "system": False,
            "required": True,
            "unique": False,
            "options": {}
        }
    ],
    "listRule": "@request.auth.role = 'super_admin' || @request.auth.id = id",
    "viewRule": "@request.auth.id = id || @request.auth.role = 'super_admin'",
    "createRule": "@request.auth.role = 'super_admin'",
    "updateRule": "@request.auth.id = id || @request.auth.role = 'super_admin'",
    "deleteRule": "@request.auth.role = 'super_admin'",
    "options": {
        "allowEmailAuth": True,
        "allowOAuth2Auth": False,
        "allowUsernameAuth": True,
        "exceptEmailDomains": [],
        "manageRule": "@request.auth.role = 'super_admin'",
        "minPasswordLength": 8,
        "onlyEmailDomains": [],
        "requireEmail": True
    }
}

# User sessions for tracking active sessions
USER_SESSIONS_SCHEMA = {
    "name": "user_sessions",
    "type": "base",
    "system": False,
    "schema": [
        {
            "id": "sessions_user",
            "name": "user",
            "type": "relation", 
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "maxSelect": 1,
                "collectionId": "users"
            }
        },
        {
            "id": "sessions_token",
            "name": "session_token",
            "type": "text",
            "system": False,
            "required": True,
            "unique": True,
            "options": {
                "min": 32,
                "max": 128,
                "pattern": ""
            }
        },
        {
            "id": "sessions_ip_address",
            "name": "ip_address",
            "type": "text",
            "system": False,
            "required": False,
            "unique": False,
            "options": {
                "min": 7,
                "max": 45,
                "pattern": ""
            }
        },
        {
            "id": "sessions_user_agent",
            "name": "user_agent",
            "type": "text",
            "system": False,
            "required": False,
            "unique": False,
            "options": {
                "min": 0,
                "max": 500,
                "pattern": ""
            }
        },
        {
            "id": "sessions_expires_at",
            "name": "expires_at",
            "type": "date",
            "system": False,
            "required": True,
            "unique": False,
            "options": {}
        },
        {
            "id": "sessions_last_activity",
            "name": "last_activity",
            "type": "date",
            "system": False,
            "required": True,
            "unique": False,
            "options": {}
        }
    ],
    "listRule": "@request.auth.id = user",
    "viewRule": "@request.auth.id = user",
    "createRule": "@request.auth.id != ''",
    "updateRule": "@request.auth.id = user",
    "deleteRule": "@request.auth.id = user",
    "options": {}
}

# User activity log for audit trails
USER_ACTIVITY_SCHEMA = {
    "name": "user_activity",
    "type": "base",
    "system": False,
    "schema": [
        {
            "id": "activity_user",
            "name": "user",
            "type": "relation",
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "maxSelect": 1,
                "collectionId": "users"
            }
        },
        {
            "id": "activity_action",
            "name": "action",
            "type": "text",
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "min": 2,
                "max": 100,
                "pattern": ""
            }
        },
        {
            "id": "activity_resource",
            "name": "resource",
            "type": "text",
            "system": False,
            "required": False,
            "unique": False,
            "options": {
                "min": 0,
                "max": 200,
                "pattern": ""
            }
        },
        {
            "id": "activity_details",
            "name": "details",
            "type": "json",
            "system": False,
            "required": False,
            "unique": False,
            "options": {}
        },
        {
            "id": "activity_ip_address",
            "name": "ip_address",
            "type": "text",
            "system": False,
            "required": False,
            "unique": False,
            "options": {
                "min": 0,
                "max": 45,
                "pattern": ""
            }
        },
        {
            "id": "activity_timestamp",
            "name": "timestamp",
            "type": "date",
            "system": False,
            "required": True,
            "unique": False,
            "options": {}
        }
    ],
    "listRule": "@request.auth.role = 'super_admin' || @request.auth.id = user",
    "viewRule": "@request.auth.role = 'super_admin' || @request.auth.id = user",
    "createRule": "@request.auth.id != ''",
    "updateRule": "",
    "deleteRule": "@request.auth.role = 'super_admin'",
    "options": {}
}


def get_user_schemas() -> List[Dict[str, Any]]:
    """Get all user-related schema definitions."""
    return [
        USER_SCHEMA,
        USER_SESSIONS_SCHEMA,
        USER_ACTIVITY_SCHEMA
    ]


def get_user_permissions(role: UserRole) -> Dict[str, Any]:
    """Get default permissions for user role."""
    permissions = {
        UserRole.SUPER_ADMIN: {
            "system_admin": True,
            "user_management": True,
            "global_configuration": True,
            "all_lines_access": True,
            "all_stations_access": True,
            "analytics_access": True,
            "audit_access": True
        },
        UserRole.LINE_MANAGER: {
            "system_admin": False,
            "user_management": False,
            "global_configuration": False,
            "all_lines_access": False,  # Assigned specific lines
            "all_stations_access": False,  # Only stations in their lines
            "analytics_access": True,
            "audit_access": False,
            "line_management": True,
            "production_planning": True,
            "performance_monitoring": True
        },
        UserRole.STATION_ENGINEER: {
            "system_admin": False,
            "user_management": False,
            "global_configuration": False,
            "all_lines_access": False,
            "all_stations_access": False,  # Only assigned stations
            "analytics_access": True,
            "audit_access": False,
            "station_control": True,
            "test_configuration": True,
            "diagnostics_access": True,
            "maintenance_scheduling": True
        },
        UserRole.COMPONENT_VENDOR: {
            "system_admin": False,
            "user_management": False,
            "global_configuration": False,
            "all_lines_access": False,
            "all_stations_access": False,
            "analytics_access": True,  # Their component performance only
            "audit_access": False,
            "component_upload": True,
            "vendor_dashboard": True,
            "performance_metrics": True,
            "support_tickets": True
        }
    }
    
    return permissions.get(role, {})