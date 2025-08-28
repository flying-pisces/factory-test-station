# API Contracts - Manufacturing Line System

## Overview
This document defines the REST API contracts between different layers of the manufacturing line system.

## Base URL Structure
```
Production: https://factory.com/api/v1
Development: http://localhost:8000/api/v1
```

## Authentication
All API endpoints require Bearer token authentication except public health checks.

```http
Authorization: Bearer <jwt_token>
```

## 1. Line Level API

### 1.1 Line Management

#### Get Line Status
```http
GET /api/v1/line/{line_id}
```

Response:
```json
{
  "line_id": "SMT_LINE_01",
  "name": "SMT Production Line",
  "status": "running|stopped|error|maintenance",
  "stations": [
    {
      "station_id": "ICT_01",
      "status": "running",
      "uph_current": 120,
      "yield": 0.98
    }
  ],
  "conveyor": {
    "status": "running",
    "speed": 0.5,
    "dut_count": 5
  },
  "operators": [
    {
      "operator_id": "OP_001",
      "status": "idle",
      "assigned_station": "ICT_01"
    }
  ],
  "metrics": {
    "total_uph": 1800,
    "line_efficiency": 0.85,
    "total_yield": 0.96
  }
}
```

#### Start/Stop Line
```http
POST /api/v1/line/{line_id}/control
Content-Type: application/json

{
  "action": "start|stop|pause|resume",
  "reason": "maintenance|shift_change|emergency"
}
```

### 1.2 Production Management

#### Load Production Order
```http
POST /api/v1/line/{line_id}/production-order
Content-Type: application/json

{
  "order_id": "PO_2025_001",
  "product_type": "PCB_Rev_A",
  "quantity": 1000,
  "target_stations": ["ICT_01", "FCT_01", "FIRMWARE_01"],
  "priority": "high|normal|low"
}
```

## 2. Station Level API

### 2.1 Station Control

#### Get Station Status
```http
GET /api/v1/line/{line_id}/station/{station_id}
```

Response:
```json
{
  "station_id": "ICT_01",
  "name": "In-Circuit Test Station",
  "type": "test",
  "status": "idle|running|error|maintenance",
  "position": {"x": 10.0, "y": 5.0, "z": 2.0},
  "performance": {
    "takt_time": 30.0,
    "uph": 120,
    "retest_ratio": 0.05,
    "yield": 0.98,
    "uptime": 0.95
  },
  "current_dut": {
    "dut_id": "DUT_001234",
    "test_status": "in_progress",
    "start_time": "2025-01-28T10:30:00Z"
  },
  "fixture": {
    "fixture_id": "ICT_FIXTURE_01",
    "status": "ready",
    "cycle_count": 1250
  }
}
```

#### Execute Station Command
```http
POST /api/v1/line/{line_id}/station/{station_id}/command
Content-Type: application/json

{
  "command": "start_test|stop_test|reset|calibrate",
  "parameters": {
    "dut_id": "DUT_001234",
    "test_config": "standard"
  }
}
```

### 2.2 Test Results

#### Submit Test Results
```http
POST /api/v1/line/{line_id}/station/{station_id}/results
Content-Type: application/json

{
  "dut_id": "DUT_001234",
  "test_id": "ICT_TEST_001",
  "result": "pass|fail|retest",
  "measurements": [
    {
      "parameter": "voltage_5v",
      "value": 5.02,
      "limit_low": 4.8,
      "limit_high": 5.2,
      "unit": "V"
    }
  ],
  "duration": 25.5,
  "operator": "OP_001"
}
```

## 3. Component/Fixture Level API

### 3.1 Component Status

#### Get Component Status
```http
GET /api/v1/line/{line_id}/station/{station_id}/component/{component_id}
```

Response:
```json
{
  "component_id": "DMM_AGILENT_001",
  "name": "Agilent Digital Multimeter",
  "type": "measurement_equipment",
  "vendor": "Keysight",
  "status": "ready|busy|error|calibration_due",
  "visa_address": "TCPIP::192.168.1.100::INSTR",
  "capabilities": ["voltage_dc", "current_dc", "resistance"],
  "calibration": {
    "last_date": "2024-12-01",
    "next_due": "2025-12-01",
    "certificate": "CAL_2024_123"
  },
  "usage": {
    "power_on_hours": 8760,
    "measurement_count": 125000
  }
}
```

#### Upload Component Configuration
```http
POST /api/v1/line/{line_id}/station/{station_id}/component/{component_id}/config
Content-Type: multipart/form-data

{
  "config_file": <binary>,
  "cad_file": <binary>,
  "api_spec": <binary>,
  "version": "1.2.0",
  "description": "Updated measurement parameters"
}
```

## 4. Conveyor System API

### 4.1 Conveyor Control

#### Get Conveyor Status
```http
GET /api/v1/line/{line_id}/conveyor/{conveyor_id}
```

Response:
```json
{
  "conveyor_id": "BELT_001",
  "type": "belt",
  "status": "running|stopped|error",
  "speed_multiplier": 0.8,
  "segments": [
    {
      "id": "SEG_001",
      "from_station": "ICT_01",
      "to_station": "FCT_01",
      "length": 2.5,
      "speed": 0.4
    }
  ],
  "duts": [
    {
      "dut_id": "DUT_001234",
      "position": 0.6,
      "destination": "FCT_01"
    }
  ]
}
```

#### Control Conveyor
```http
POST /api/v1/line/{line_id}/conveyor/{conveyor_id}/control
Content-Type: application/json

{
  "action": "start|stop|emergency_stop",
  "speed": 0.8,
  "reason": "station_ready"
}
```

### 4.2 DUT Tracking

#### Load DUT
```http
POST /api/v1/line/{line_id}/conveyor/{conveyor_id}/dut
Content-Type: application/json

{
  "dut_id": "DUT_001234",
  "source_station": "ICT_01",
  "destination_station": "FCT_01",
  "priority": "normal"
}
```

## 5. Operator System API

### 5.1 Operator Management

#### Get Operator Status
```http
GET /api/v1/line/{line_id}/operator/{operator_id}
```

Response:
```json
{
  "operator_id": "DH_001",
  "name": "Digital Human Alpha",
  "status": "idle|busy|intervention_required|offline",
  "assigned_station": "ICT_01",
  "capabilities": ["button_press", "item_pickup", "visual_inspection"],
  "queue_size": 2,
  "performance": {
    "actions_completed": 156,
    "avg_response_time": 3.2,
    "success_rate": 0.94
  },
  "ai_metrics": {
    "skill_level": 0.85,
    "attention_level": 0.9,
    "fatigue_level": 0.2
  }
}
```

#### Assign Action
```http
POST /api/v1/line/{line_id}/operator/{operator_id}/action
Content-Type: application/json

{
  "action_type": "button_press|item_pickup|visual_inspection|issue_monitoring",
  "target_station": "ICT_01",
  "parameters": {
    "button": "start_test",
    "urgency": "high"
  },
  "priority": 8
}
```

## 6. Hook System API

### 6.1 External Integration Hooks

#### Register Webhook
```http
POST /api/v1/hooks/webhook
Content-Type: application/json

{
  "name": "MES_Integration",
  "url": "https://mes.factory.com/webhook",
  "events": ["test_completed", "line_stopped", "quality_alert"],
  "auth_type": "bearer",
  "auth_token": "abc123...",
  "retry_config": {
    "max_retries": 3,
    "retry_delay": 5
  }
}
```

#### Trigger Manual Event
```http
POST /api/v1/hooks/event
Content-Type: application/json

{
  "event_type": "manual_intervention",
  "source": "station_engineer",
  "data": {
    "station_id": "ICT_01",
    "reason": "calibration_check",
    "timestamp": "2025-01-28T10:30:00Z"
  }
}
```

## 7. Data Export API

### 7.1 Metrics Export

#### Export Production Data
```http
GET /api/v1/export/production
Query Parameters:
- start_date: ISO8601 date
- end_date: ISO8601 date
- stations: comma-separated station IDs
- format: json|csv|excel
- include_raw: boolean
```

#### Export Quality Data
```http
GET /api/v1/export/quality
Query Parameters:
- start_date: ISO8601 date
- end_date: ISO8601 date
- result_type: pass|fail|retest|all
- format: json|csv
```

## 8. Error Responses

### Standard Error Format
```json
{
  "error": {
    "code": "STATION_NOT_FOUND",
    "message": "Station ICT_02 not found in line SMT_LINE_01",
    "details": {
      "line_id": "SMT_LINE_01",
      "station_id": "ICT_02"
    },
    "timestamp": "2025-01-28T10:30:00Z",
    "request_id": "req_12345"
  }
}
```

### HTTP Status Codes
- 200: Success
- 201: Created
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 409: Conflict
- 422: Validation Error
- 500: Internal Server Error
- 503: Service Unavailable

## 9. WebSocket Events

### Real-time Updates
```javascript
// Connect to WebSocket
ws://factory.com/ws/line/{line_id}

// Event types:
{
  "event": "station_status_changed",
  "data": {
    "station_id": "ICT_01",
    "old_status": "idle",
    "new_status": "running"
  }
}

{
  "event": "dut_moved",
  "data": {
    "dut_id": "DUT_001234",
    "from_station": "ICT_01",
    "to_station": "FCT_01"
  }
}

{
  "event": "operator_action_completed",
  "data": {
    "operator_id": "DH_001",
    "action": "button_press",
    "success": true
  }
}
```

## 10. Rate Limits

- Production APIs: 1000 requests/minute per API key
- Control APIs: 100 requests/minute per API key  
- Export APIs: 10 requests/minute per API key
- WebSocket connections: 50 concurrent per line

## 11. Versioning

API versions follow semantic versioning (v1.0.0). Breaking changes require major version increment.

Current version: v1.0.0
Supported versions: v1.x.x