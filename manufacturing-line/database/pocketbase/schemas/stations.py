"""Station data schema definitions for PocketBase."""

from typing import Dict, Any, List
from enum import Enum


class StationType(Enum):
    """Station types in the manufacturing system."""
    SMT = "smt"
    TEST = "test"
    ASSEMBLY = "assembly"
    QUALITY = "quality"
    INSPECTION = "inspection"
    PACKAGING = "packaging"
    REWORK = "rework"


class StationStatus(Enum):
    """Station operational status."""
    IDLE = "idle"
    RUNNING = "running"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    OFFLINE = "offline"


# Manufacturing stations configuration
STATIONS_SCHEMA = {
    "name": "stations",
    "type": "base", 
    "system": False,
    "schema": [
        {
            "id": "station_id",
            "name": "station_id",
            "type": "text",
            "system": False,
            "required": True,
            "unique": True,
            "options": {
                "min": 3,
                "max": 50,
                "pattern": "^[A-Z0-9_-]+$"
            }
        },
        {
            "id": "station_name",
            "name": "name",
            "type": "text",
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "min": 3,
                "max": 100,
                "pattern": ""
            }
        },
        {
            "id": "station_type",
            "name": "station_type",
            "type": "select",
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "maxSelect": 1,
                "values": [st.value for st in StationType]
            }
        },
        {
            "id": "station_line",
            "name": "line",
            "type": "relation",
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "maxSelect": 1,
                "collectionId": "manufacturing_lines"
            }
        },
        {
            "id": "station_position",
            "name": "position",
            "type": "number",
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "min": 0,
                "max": 100
            }
        },
        {
            "id": "station_configuration",
            "name": "configuration",
            "type": "json",
            "system": False,
            "required": False,
            "unique": False,
            "options": {}
        },
        {
            "id": "station_test_limits",
            "name": "test_limits",
            "type": "json",
            "system": False,
            "required": False,
            "unique": False,
            "options": {}
        },
        {
            "id": "station_components",
            "name": "components",
            "type": "relation",
            "system": False,
            "required": False,
            "unique": False,
            "options": {
                "maxSelect": 1000,
                "collectionId": "structured_components"
            }
        },
        {
            "id": "station_engineer",
            "name": "engineer",
            "type": "relation",
            "system": False,
            "required": False,
            "unique": False,
            "options": {
                "maxSelect": 1,
                "collectionId": "users"
            }
        },
        {
            "id": "station_status",
            "name": "status",
            "type": "select",
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "maxSelect": 1,
                "values": [ss.value for ss in StationStatus]
            }
        },
        {
            "id": "station_cost",
            "name": "cost_usd",
            "type": "number",
            "system": False,
            "required": False,
            "unique": False,
            "options": {
                "min": 0,
                "max": 10000000
            }
        },
        {
            "id": "station_uph",
            "name": "uph_capacity",
            "type": "number",
            "system": False,
            "required": False,
            "unique": False,
            "options": {
                "min": 1,
                "max": 10000
            }
        },
        {
            "id": "station_footprint",
            "name": "footprint_sqm",
            "type": "number",
            "system": False,
            "required": False,
            "unique": False,
            "options": {
                "min": 0.1,
                "max": 100
            }
        },
        {
            "id": "station_cycle_time",
            "name": "cycle_time_seconds",
            "type": "number",
            "system": False,
            "required": False,
            "unique": False,
            "options": {
                "min": 1,
                "max": 3600
            }
        },
        {
            "id": "station_discrete_events",
            "name": "discrete_event_profile",
            "type": "json",
            "system": False,
            "required": False,
            "unique": False,
            "options": {}
        }
    ],
    "listRule": "@request.auth.role = 'super_admin' || @request.auth.role = 'line_manager' || @request.auth.role = 'station_engineer'",
    "viewRule": "@request.auth.role = 'super_admin' || @request.auth.role = 'line_manager' || (@request.auth.role = 'station_engineer' && engineer = @request.auth.id)",
    "createRule": "@request.auth.role = 'super_admin' || @request.auth.role = 'line_manager'",
    "updateRule": "@request.auth.role = 'super_admin' || @request.auth.role = 'line_manager' || (@request.auth.role = 'station_engineer' && engineer = @request.auth.id)",
    "deleteRule": "@request.auth.role = 'super_admin'",
    "options": {}
}

# Station performance metrics
STATION_METRICS_SCHEMA = {
    "name": "station_metrics",
    "type": "base",
    "system": False,
    "schema": [
        {
            "id": "metrics_station",
            "name": "station",
            "type": "relation",
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "maxSelect": 1,
                "collectionId": "stations"
            }
        },
        {
            "id": "metrics_timestamp",
            "name": "timestamp",
            "type": "date",
            "system": False,
            "required": True,
            "unique": False,
            "options": {}
        },
        {
            "id": "metrics_uph_actual",
            "name": "uph_actual",
            "type": "number",
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "min": 0,
                "max": 10000
            }
        },
        {
            "id": "metrics_efficiency",
            "name": "efficiency_percent",
            "type": "number",
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "min": 0,
                "max": 100
            }
        },
        {
            "id": "metrics_cycle_time_actual",
            "name": "cycle_time_actual",
            "type": "number",
            "system": False,
            "required": False,
            "unique": False,
            "options": {
                "min": 0,
                "max": 3600
            }
        },
        {
            "id": "metrics_first_pass_yield",
            "name": "first_pass_yield",
            "type": "number",
            "system": False,
            "required": False,
            "unique": False,
            "options": {
                "min": 0,
                "max": 100
            }
        },
        {
            "id": "metrics_uptime",
            "name": "uptime_percent",
            "type": "number",
            "system": False,
            "required": False,
            "unique": False,
            "options": {
                "min": 0,
                "max": 100
            }
        },
        {
            "id": "metrics_errors",
            "name": "error_count",
            "type": "number",
            "system": False,
            "required": False,
            "unique": False,
            "options": {
                "min": 0,
                "max": 10000
            }
        },
        {
            "id": "metrics_warnings",
            "name": "warning_count",
            "type": "number",
            "system": False,
            "required": False,
            "unique": False,
            "options": {
                "min": 0,
                "max": 10000
            }
        },
        {
            "id": "metrics_additional_data",
            "name": "additional_data",
            "type": "json",
            "system": False,
            "required": False,
            "unique": False,
            "options": {}
        }
    ],
    "listRule": "@request.auth.role = 'super_admin' || @request.auth.role = 'line_manager' || @request.auth.role = 'station_engineer'",
    "viewRule": "@request.auth.role = 'super_admin' || @request.auth.role = 'line_manager' || @request.auth.role = 'station_engineer'",
    "createRule": "",  # Created by system monitoring
    "updateRule": "",
    "deleteRule": "@request.auth.role = 'super_admin'",
    "options": {}
}

# Station maintenance records
STATION_MAINTENANCE_SCHEMA = {
    "name": "station_maintenance",
    "type": "base",
    "system": False,
    "schema": [
        {
            "id": "maint_station",
            "name": "station",
            "type": "relation",
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "maxSelect": 1,
                "collectionId": "stations"
            }
        },
        {
            "id": "maint_type",
            "name": "maintenance_type",
            "type": "select",
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "maxSelect": 1,
                "values": ["preventive", "corrective", "calibration", "upgrade", "inspection"]
            }
        },
        {
            "id": "maint_scheduled_date",
            "name": "scheduled_date",
            "type": "date",
            "system": False,
            "required": True,
            "unique": False,
            "options": {}
        },
        {
            "id": "maint_completed_date",
            "name": "completed_date",
            "type": "date",
            "system": False,
            "required": False,
            "unique": False,
            "options": {}
        },
        {
            "id": "maint_technician",
            "name": "technician",
            "type": "relation",
            "system": False,
            "required": False,
            "unique": False,
            "options": {
                "maxSelect": 1,
                "collectionId": "users"
            }
        },
        {
            "id": "maint_description",
            "name": "description",
            "type": "editor",
            "system": False,
            "required": True,
            "unique": False,
            "options": {}
        },
        {
            "id": "maint_status",
            "name": "status",
            "type": "select",
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "maxSelect": 1,
                "values": ["scheduled", "in_progress", "completed", "cancelled", "overdue"]
            }
        },
        {
            "id": "maint_duration_hours",
            "name": "duration_hours",
            "type": "number",
            "system": False,
            "required": False,
            "unique": False,
            "options": {
                "min": 0.1,
                "max": 168
            }
        },
        {
            "id": "maint_cost",
            "name": "cost_usd",
            "type": "number",
            "system": False,
            "required": False,
            "unique": False,
            "options": {
                "min": 0,
                "max": 100000
            }
        },
        {
            "id": "maint_parts_used",
            "name": "parts_used",
            "type": "json",
            "system": False,
            "required": False,
            "unique": False,
            "options": {}
        },
        {
            "id": "maint_notes",
            "name": "notes",
            "type": "editor",
            "system": False,
            "required": False,
            "unique": False,
            "options": {}
        }
    ],
    "listRule": "@request.auth.role = 'super_admin' || @request.auth.role = 'line_manager' || @request.auth.role = 'station_engineer'",
    "viewRule": "@request.auth.role = 'super_admin' || @request.auth.role = 'line_manager' || @request.auth.role = 'station_engineer'",
    "createRule": "@request.auth.role = 'super_admin' || @request.auth.role = 'line_manager' || @request.auth.role = 'station_engineer'",
    "updateRule": "@request.auth.role = 'super_admin' || @request.auth.role = 'line_manager' || @request.auth.role = 'station_engineer'",
    "deleteRule": "@request.auth.role = 'super_admin'",
    "options": {}
}


def get_station_schemas() -> List[Dict[str, Any]]:
    """Get all station-related schema definitions."""
    return [
        STATIONS_SCHEMA,
        STATION_METRICS_SCHEMA,
        STATION_MAINTENANCE_SCHEMA
    ]


def get_station_default_configs() -> Dict[str, Dict[str, Any]]:
    """Get default configurations for different station types."""
    return {
        StationType.SMT.value: {
            "placement_accuracy": 0.05,  # mm
            "placement_speed": 7200,     # CPH
            "feeder_capacity": 24,
            "vision_inspection": True,
            "auto_calibration": True
        },
        StationType.TEST.value: {
            "test_channels": 8,
            "voltage_ranges": [0.1, 1, 10, 100],
            "current_ranges": [0.001, 0.01, 0.1, 1],
            "frequency_range": [0.1, 10000000],
            "accuracy": 0.1  # percent
        },
        StationType.ASSEMBLY.value: {
            "force_control": True,
            "torque_control": True,
            "position_accuracy": 0.01,  # mm
            "max_parts": 10,
            "vision_guidance": True
        },
        StationType.QUALITY.value: {
            "vision_resolution": "1920x1080",
            "measurement_accuracy": 0.005,  # mm
            "defect_classification": True,
            "statistical_analysis": True
        }
    }