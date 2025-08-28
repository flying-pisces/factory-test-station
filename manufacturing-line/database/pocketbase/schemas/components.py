"""Component data schema definitions for PocketBase."""

from typing import Dict, Any, List
from enum import Enum


class ComponentType(Enum):
    """Component types in the manufacturing system."""
    RESISTOR = "resistor"
    CAPACITOR = "capacitor"
    INDUCTOR = "inductor" 
    IC = "ic"
    DIODE = "diode"
    TRANSISTOR = "transistor"
    CONNECTOR = "connector"
    CRYSTAL = "crystal"
    OTHER = "other"


class ProcessingStatus(Enum):
    """Component data processing status."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    ERROR = "error"
    VALIDATED = "validated"


# Raw component data from vendors
RAW_COMPONENTS_SCHEMA = {
    "name": "raw_components",
    "type": "base",
    "system": False,
    "schema": [
        {
            "id": "raw_component_id",
            "name": "component_id",
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
            "id": "raw_component_type",
            "name": "component_type", 
            "type": "select",
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "maxSelect": 1,
                "values": [ct.value for ct in ComponentType]
            }
        },
        {
            "id": "raw_vendor",
            "name": "vendor",
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
            "id": "raw_cad_data",
            "name": "cad_data",
            "type": "json",
            "system": False,
            "required": False,
            "unique": False,
            "options": {}
        },
        {
            "id": "raw_api_data",
            "name": "api_data",
            "type": "json",
            "system": False,
            "required": False,
            "unique": False,
            "options": {}
        },
        {
            "id": "raw_ee_data",
            "name": "ee_data",
            "type": "json",
            "system": False,
            "required": False,
            "unique": False,
            "options": {}
        },
        {
            "id": "raw_files",
            "name": "files",
            "type": "file",
            "system": False,
            "required": False,
            "unique": False,
            "options": {
                "maxSelect": 10,
                "maxSize": 10485760,
                "mimeTypes": [
                    "application/pdf",
                    "image/jpeg",
                    "image/png",
                    "application/zip",
                    "text/csv",
                    "application/json"
                ],
                "thumbs": ["200x200"]
            }
        },
        {
            "id": "raw_processing_status",
            "name": "processing_status",
            "type": "select",
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "maxSelect": 1,
                "values": [ps.value for ps in ProcessingStatus]
            }
        },
        {
            "id": "raw_upload_timestamp",
            "name": "upload_timestamp",
            "type": "date",
            "system": False,
            "required": True,
            "unique": False,
            "options": {}
        },
        {
            "id": "raw_processing_notes",
            "name": "processing_notes",
            "type": "editor",
            "system": False,
            "required": False,
            "unique": False,
            "options": {}
        }
    ],
    "listRule": "@request.auth.role = 'super_admin' || @request.auth.role = 'line_manager' || vendor = @request.auth.id",
    "viewRule": "@request.auth.role = 'super_admin' || @request.auth.role = 'line_manager' || vendor = @request.auth.id",
    "createRule": "@request.auth.role = 'component_vendor' && vendor = @request.auth.id",
    "updateRule": "@request.auth.role = 'super_admin' || (@request.auth.role = 'component_vendor' && vendor = @request.auth.id)",
    "deleteRule": "@request.auth.role = 'super_admin' || (@request.auth.role = 'component_vendor' && vendor = @request.auth.id)",
    "options": {}
}

# Structured component data after MOS processing
STRUCTURED_COMPONENTS_SCHEMA = {
    "name": "structured_components",
    "type": "base",
    "system": False,
    "schema": [
        {
            "id": "struct_component_id",
            "name": "component_id",
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
            "id": "struct_raw_component",
            "name": "raw_component",
            "type": "relation",
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "maxSelect": 1,
                "collectionId": "raw_components"
            }
        },
        {
            "id": "struct_component_type",
            "name": "component_type",
            "type": "select",
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "maxSelect": 1,
                "values": [ct.value for ct in ComponentType]
            }
        },
        {
            "id": "struct_size",
            "name": "size",
            "type": "text",
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "min": 2,
                "max": 20,
                "pattern": ""
            }
        },
        {
            "id": "struct_price",
            "name": "price",
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
            "id": "struct_lead_time",
            "name": "lead_time",
            "type": "number",
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "min": 0,
                "max": 365
            }
        },
        {
            "id": "struct_discrete_event_profile",
            "name": "discrete_event_profile",
            "type": "json",
            "system": False,
            "required": True,
            "unique": False,
            "options": {}
        },
        {
            "id": "struct_physical_properties",
            "name": "physical_properties",
            "type": "json",
            "system": False,
            "required": False,
            "unique": False,
            "options": {}
        },
        {
            "id": "struct_electrical_properties",
            "name": "electrical_properties",
            "type": "json",
            "system": False,
            "required": False,
            "unique": False,
            "options": {}
        },
        {
            "id": "struct_availability",
            "name": "availability",
            "type": "json",
            "system": False,
            "required": False,
            "unique": False,
            "options": {}
        },
        {
            "id": "struct_processing_timestamp",
            "name": "processing_timestamp",
            "type": "date",
            "system": False,
            "required": True,
            "unique": False,
            "options": {}
        },
        {
            "id": "struct_validation_status",
            "name": "validation_status",
            "type": "bool",
            "system": False,
            "required": True,
            "unique": False,
            "options": {}
        }
    ],
    "listRule": "@request.auth.role = 'super_admin' || @request.auth.role = 'line_manager' || @request.auth.role = 'station_engineer'",
    "viewRule": "@request.auth.role = 'super_admin' || @request.auth.role = 'line_manager' || @request.auth.role = 'station_engineer'",
    "createRule": "",  # Created by system processing
    "updateRule": "@request.auth.role = 'super_admin'",
    "deleteRule": "@request.auth.role = 'super_admin'",
    "options": {}
}

# Component processing history
COMPONENT_PROCESSING_HISTORY_SCHEMA = {
    "name": "component_processing_history",
    "type": "base",
    "system": False,
    "schema": [
        {
            "id": "history_component",
            "name": "component",
            "type": "relation",
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "maxSelect": 1,
                "collectionId": "raw_components"
            }
        },
        {
            "id": "history_processing_step",
            "name": "processing_step",
            "type": "text",
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "min": 3,
                "max": 50,
                "pattern": ""
            }
        },
        {
            "id": "history_status",
            "name": "status",
            "type": "select",
            "system": False,
            "required": True,
            "unique": False,
            "options": {
                "maxSelect": 1,
                "values": ["started", "completed", "failed", "skipped"]
            }
        },
        {
            "id": "history_details",
            "name": "details",
            "type": "json",
            "system": False,
            "required": False,
            "unique": False,
            "options": {}
        },
        {
            "id": "history_processing_time",
            "name": "processing_time",
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
            "id": "history_timestamp",
            "name": "timestamp",
            "type": "date",
            "system": False,
            "required": True,
            "unique": False,
            "options": {}
        },
        {
            "id": "history_errors",
            "name": "errors",
            "type": "editor",
            "system": False,
            "required": False,
            "unique": False,
            "options": {}
        }
    ],
    "listRule": "@request.auth.role = 'super_admin' || @request.auth.role = 'line_manager'",
    "viewRule": "@request.auth.role = 'super_admin' || @request.auth.role = 'line_manager'",
    "createRule": "",  # Created by system processing
    "updateRule": "",
    "deleteRule": "@request.auth.role = 'super_admin'",
    "options": {}
}


def get_component_schemas() -> List[Dict[str, Any]]:
    """Get all component-related schema definitions."""
    return [
        RAW_COMPONENTS_SCHEMA,
        STRUCTURED_COMPONENTS_SCHEMA,
        COMPONENT_PROCESSING_HISTORY_SCHEMA
    ]


def get_component_validation_rules() -> Dict[str, Dict[str, Any]]:
    """Get validation rules for different component types."""
    return {
        ComponentType.RESISTOR.value: {
            "required_ee_data": ["resistance", "tolerance", "power_rating"],
            "optional_ee_data": ["temperature_coefficient", "voltage_rating"],
            "required_cad_data": ["package"],
            "size_patterns": ["0201", "0402", "0603", "0805", "1206", "1210", "2010", "2512"]
        },
        ComponentType.CAPACITOR.value: {
            "required_ee_data": ["capacitance", "voltage_rating", "dielectric_type"],
            "optional_ee_data": ["tolerance", "temperature_coefficient", "esr"],
            "required_cad_data": ["package"],
            "size_patterns": ["0201", "0402", "0603", "0805", "1206", "1210", "1812", "2220"]
        },
        ComponentType.IC.value: {
            "required_ee_data": ["pin_count", "operating_voltage"],
            "optional_ee_data": ["current_consumption", "operating_temperature"],
            "required_cad_data": ["package", "dimensions"],
            "size_patterns": ["QFN", "BGA", "TQFP", "SOIC", "SSOP", "MSOP", "DFN", "WLCSP"]
        },
        ComponentType.INDUCTOR.value: {
            "required_ee_data": ["inductance", "current_rating"],
            "optional_ee_data": ["tolerance", "dc_resistance", "saturation_current"],
            "required_cad_data": ["package"],
            "size_patterns": ["0402", "0603", "0805", "1008", "1206", "1812", "2520", "3225"]
        }
    }