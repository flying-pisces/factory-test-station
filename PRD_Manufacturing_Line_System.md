# Product Requirements Document (PRD)
# Manufacturing Line Control System

## 1. Executive Summary

### 1.1 Product Vision
A cloud-deployed, multi-tier manufacturing line control system that orchestrates test stations, conveyor systems, and digital operators through a hierarchical web interface with role-based access control.

### 1.2 Key Objectives
- Unified control of 15-station manufacturing lines (SMT + FATP)
- Hierarchical access control with multi-tier user permissions
- Extensible architecture with clear integration hooks
- PocketBase-powered data persistence
- Modular design with separate repos for major components

## 2. System Architecture

### 2.1 Technology Stack
- **Backend**: Python 3.8+ with Flask/FastAPI
- **Database**: PocketBase (SQLite-based, real-time subscriptions)
- **Frontend**: React/Vue.js with WebSocket support
- **Simulation**: JAAMSIM discrete event simulation engine
- **AI Simulation**: NVIDIA Isaac Sim integration (future)
- **Deployment**: Docker containers on cloud (AWS/GCP/Azure)
- **Communication**: REST APIs + WebSocket for updates
- **Configuration**: JSON-based with schema validation

### 2.2 Repository Structure
```
manufacturing-line/              (Main orchestration repo)
├── line-controller/            
├── stations/                   → Submodule: factory-test-station
├── conveyors/                  → Submodule: conveyor-system
├── operators/                  → Submodule: digital-operator
├── simulation/                 # Discrete event simulation framework
│   ├── jaamsim_integration/    # JAAMSIM simulation configs
│   ├── isaac_sim_integration/  # Future NVIDIA Isaac Sim
│   ├── simulation_engine/      # Core simulation framework
│   └── scenario_configs/       # Simulation scenarios (1-up, 3-up, etc.)
├── common/                     
├── web-portal/                 
├── database/                   
├── config/                     
├── hooks/                      
└── docs/                       
```

### 2.3 Simulation-First Architecture

**Core Philosophy**: Every component (stations, conveyors, operators, line) has both:
1. **Physical Implementation**: Real hardware control and monitoring
2. **Digital Twin**: Discrete event simulation model for:
   - Design validation and optimization
   - Predictive analytics and bottleneck analysis
   - Scenario testing (what-if analysis)
   - Operator training and process validation
   - Real-time performance comparison

**Simulation Technologies**:
- **JAAMSIM**: Primary discrete event simulation engine (Java-based, cross-platform)
- **NVIDIA Isaac Sim**: Future integration for robotics and AI simulation
- **Bright Machines**: Integration patterns for intelligent manufacturing

### 2.4 Station Types

**SMT Line (3 stations):**
- ICT (In-Circuit Test)
- Firmware Programming
- FCT (Functional Circuit Test)

**FATP Line (12 stations):**
- IQC (Incoming Quality Control)
- Camera Test
- Display Test
- RF Test
- WiFi Test
- OTA (Over-The-Air) Update
- Battery Test
- Housing Assembly
- Burn-in Test
- OS Fusion
- MMI (Man-Machine Interface) Test
- Packaging

## 3. Core Components

### 3.1 Line Controller
**Responsibilities:**
- Orchestrate all stations, conveyors, and operators
- Manage production flow and routing
- Aggregate metrics and KPIs
- Handle inter-station communication
- **Simulation Orchestration**: Run digital twin simulations

**Key Features:**
- Production scheduling
- Line balancing algorithms
- Bottleneck detection
- Real-time dashboard
- **Simulation Features**:
  - Run JAAMSIM models for line optimization
  - Compare real-time performance vs. simulation predictions
  - What-if scenario analysis (e.g., station downtime, capacity changes)
  - Predictive bottleneck identification

### 3.2 Station Component
**Common Attributes:**
```json
{
  "station_id": "string",
  "name": "string",
  "type": "assembly|test",
  "line_position": "integer",
  "dimensions": {
    "x": "float",
    "y": "float", 
    "z": "float"
  },
  "performance": {
    "takt_time": "float",
    "uph": "integer",
    "retest_ratio": "float",
    "yield": "float"
  },
  "status": "idle|running|error|maintenance",
  "current_dut": "string|null",
  "simulation": {
    "jaamsim_config": "path/to/station.cfg",
    "fixture_type": "1-up|3-up-turntable|custom",
    "simulation_parameters": {
      "good_dut_percentage": 85,
      "relit_dut_percentage": 10,
      "nolit_dut_percentage": 5,
      "measurement_time": 9,
      "operator_load_time": 10,
      "ptb_litup_time": 5,
      "ptb_retry_count": 3
    },
    "digital_twin_active": true,
    "prediction_accuracy": 0.95
  }
}
```

### 3.3 Conveyor System
**Attributes:**
```json
{
  "conveyor_id": "string",
  "type": "belt|roller",
  "segments": [{
    "id": "string",
    "from_station": "string",
    "to_station": "string",
    "length": "float",
    "speed": "float"
  }],
  "dut_tracking": [{
    "dut_id": "string",
    "position": "float",
    "destination": "string"
  }],
  "status": "running|stopped|error",
  "simulation": {
    "transport_model": "EntityConveyor",
    "travel_time_distribution": "normal|exponential|uniform",
    "dut_flow_simulation": true,
    "bottleneck_prediction": true,
    "capacity_optimization": {
      "max_duts_in_transit": 10,
      "buffer_zones": ["ICT_01", "FCT_01"],
      "dynamic_speed_control": true
    }
  }
}
```

### 3.4 Digital Operator
**Attributes:**
```json
{
  "operator_id": "string",
  "name": "string",
  "type": "digital_human",
  "assigned_station": "string",
  "capabilities": [
    "button_press",
    "item_pickup",
    "visual_inspection",
    "issue_monitoring"
  ],
  "status": "idle|busy|intervention_required",
  "action_queue": [],
  "simulation": {
    "behavior_model": "DigitalHumanAgent",
    "skill_level": 0.9,
    "attention_level": 0.95,
    "fatigue_simulation": true,
    "learning_enabled": true,
    "reaction_time_distribution": "normal(2.0, 0.5)",
    "error_rate_base": 0.05,
    "performance_degradation": {
      "fatigue_factor": 0.1,
      "experience_factor": 0.02
    }
  }
}
```

## 4. User Access Model

### 4.1 User Roles and Permissions

| Role | Access Level | URL Pattern | Permissions |
|------|-------------|-------------|-------------|
| Super Admin | All | /* | Full system access, configuration, user management |
| Line Manager | Line + Station (read) | /line | Monitor line, view stations, manage production |
| Station Engineer | Station + Components | /line/station/* | Configure station, develop tests, view components |
| Component Vendor | Component only | /line/station/component/* | Upload CAD/API, maintain component specs |

### 4.2 Authentication Flow
- OAuth 2.0 / JWT tokens
- Role-based access control (RBAC)
- Session management via PocketBase

## 5. Data Management

### 5.1 PocketBase Collections
```
- lines (line configurations and status)
- stations (station registry and metrics)
- conveyors (conveyor configurations)
- operators (digital operator assignments)
- duts (device tracking through line)
- metrics (performance data)
- users (authentication and roles)
- audit_logs (system events)
- hooks (integration endpoints)
```

### 5.2 Data Flow
```
Component Data → Station Aggregation → Line Analytics → Dashboard
     ↑                    ↑                   ↑            ↑
   Vendor API      Station Engineer    Line Manager   Super Admin
```

## 6. Simulation Framework

### 6.1 JAAMSIM Integration

**Configuration Management**:
- Station-specific `.cfg` files based on fixture types:
  - `1-up-station-simulation.cfg`: Single DUT processing stations
  - `3-up-turntable-simulation.cfg`: Three-position turntable fixtures
  - Custom configurations for specialized equipment

**Simulation Entities**:
- **EntityGenerator**: DUT creation with configurable failure rates
- **EntityProcessor**: Station processing with realistic timing
- **EntityConveyor**: Material flow between stations
- **Branch/Assign**: DUT routing based on test results (pass/fail/retest)
- **Queue**: Buffer management and bottleneck modeling
- **Server**: Equipment and operator resource modeling

**Key Simulation Parameters**:
```json
{
  "simulation_config": {
    "good_dut_percentage": 85,
    "relit_dut_percentage": 10,
    "nolit_dut_percentage": 5,
    "total_dut_count": 1000,
    "measurement_time": "9s",
    "operator_load_time": "10s",
    "operator_unload_time": "5s",
    "ptb_litup_time": "5s",
    "ptb_retry_count": 3,
    "conveyor_travel_time": "3s",
    "real_time_factor": 16
  }
}
```

### 6.2 Real-Time Simulation Integration

**Digital Twin Synchronization**:
- Real station performance data feeds into simulation models
- Continuous model calibration based on actual performance
- Predictive analytics using simulation forecasting
- Anomaly detection: real vs. predicted performance comparison

**Simulation-Driven Operations**:
- **Predictive Maintenance**: Simulate equipment degradation
- **Capacity Planning**: Model demand variations and resource requirements
- **Process Optimization**: Test configuration changes before implementation
- **Training Scenarios**: Operator training with simulated failure modes

### 6.3 Multi-Level Simulation

**Station Level**:
- Individual station performance modeling
- Fixture-specific behavior simulation (1-up vs. 3-up turntable)
- Equipment wear and maintenance scheduling
- Quality prediction and yield optimization

**Line Level**:
- End-to-end flow simulation
- Bottleneck identification and resolution
- Buffer optimization and work-in-progress management
- Line balancing and takt time optimization

**Factory Level**:
- Multiple line coordination
- Resource sharing optimization
- Production schedule optimization
- Supply chain integration modeling

## 7. Integration Hooks

### 7.1 Inbound Hooks (Data Input)

#### Simulation Data Ingestion
```python
# Real-time performance data for simulation calibration
POST /api/hooks/simulation/calibration
{
  "station_id": "ICT_01",
  "actual_performance": {
    "cycle_time": 28.5,
    "yield": 0.97,
    "downtime_events": 2
  },
  "simulation_prediction": {
    "cycle_time": 30.0,
    "yield": 0.95,
    "confidence": 0.92
  }
}

# JAAMSIM simulation results
POST /api/hooks/simulation/results
{
  "simulation_id": "scenario_001",
  "configuration": "1-up-station",
  "results": {
    "total_processed": 1000,
    "average_cycle_time": 29.2,
    "line_efficiency": 0.88,
    "bottleneck_station": "RF_01",
    "predicted_uph": 122
  }
}
```
```python
# Station performance data
POST /api/hooks/station/{station_id}/metrics
{
  "timestamp": "ISO8601",
  "uph_actual": 120,
  "yield": 0.98,
  "retest_count": 2
}

# DUT tracking update
POST /api/hooks/dut/{dut_id}/location
{
  "station_id": "string",
  "status": "pass|fail|retest",
  "data": {}
}

# External system integration
POST /api/hooks/external/mes
{
  "event_type": "production_order",
  "payload": {}
}
```

### 7.2 Outbound Hooks (Data Output)

#### Simulation Triggers
```python
# Trigger predictive simulation
POST /api/simulation/predict
{
  "scenario": "station_downtime",
  "parameters": {
    "affected_station": "FCT_01",
    "downtime_duration": "2h",
    "alternative_routing": true
  }
}

# Export simulation configuration
GET /api/export/simulation?station=ICT_01&format=jaamsim
# Returns .cfg file for JAAMSIM execution
```
```python
# Line status webhook
webhook_config = {
  "url": "https://customer.com/webhook",
  "events": ["line_start", "line_stop", "station_error"],
  "auth": "Bearer token"
}

# Metrics export
GET /api/export/metrics?from=date&to=date&format=json|csv
```

## 7. Web Interface Requirements

### 7.1 URL Structure
```
https://factory.com/                          # Landing/login
https://factory.com/line                      # Line dashboard
https://factory.com/line/{line_id}            # Specific line view
https://factory.com/line/{line_id}/station/{station_id}  # Station detail
https://factory.com/line/{line_id}/station/{station_id}/fixture/{fixture_id}  # Component detail
```

### 7.2 Dashboard Components
- **Line Overview**: Visual representation of all stations
- **Production Metrics**: Real-time KPIs (UPH, yield, efficiency)
- **Station Status Grid**: Color-coded station states
- **DUT Tracker**: Current location of all DUTs in system
- **Alert Panel**: Active issues and interventions needed

## 8. Performance Requirements

### 8.1 System Metrics
- **Update Frequency**: 1-5 seconds for status updates
- **Concurrent Users**: 50+ per line
- **Data Retention**: 90 days hot, 2 years cold storage
- **Availability**: 99.5% uptime
- **Response Time**: <2s for dashboard load
- **Simulation Performance**: 
  - Model execution: <30s for 1000-DUT scenarios
  - Real-time factor: 16x (configurable)
  - Prediction accuracy: >95% for 24-hour forecasts

### 8.2 Scalability
- Horizontal scaling for multiple lines
- Microservice architecture for component isolation
- Message queue for async operations

## 9. Development Roadmap

### Phase 1: Foundation (Weeks 1-3)
- [ ] Create GitHub repos and link submodules
- [ ] Implement base classes for Line, Station, Conveyor, Operator
- [ ] Set up PocketBase with initial schema
- [ ] Create authentication system
- [ ] **Simulation Setup**:
  - [ ] Integrate existing JAAMSIM configurations (1-up, 3-up turntable)
  - [ ] Create simulation base classes and interfaces
  - [ ] Develop configuration management for .cfg files

### Phase 2: Core Integration (Weeks 4-6)
- [ ] Integrate factory-test-station as submodule
- [ ] Build REST API layer
- [ ] Implement station-to-line communication
- [ ] Create conveyor control system
- [ ] **Simulation Integration**:
  - [ ] Implement JAAMSIM execution engine
  - [ ] Create digital twin synchronization
  - [ ] Build simulation-to-reality data binding

### Phase 3: Web Portal (Weeks 7-9)
- [ ] Develop multi-tier web interface
- [ ] Implement real-time WebSocket updates
- [ ] Create role-based dashboards
- [ ] Add production monitoring views
- [ ] **Simulation Visualization**:
  - [ ] Embed JAAMSIM visualization in web portal
  - [ ] Create simulation control panels
  - [ ] Implement real-time vs. predicted performance dashboards

### Phase 4: Advanced Features (Weeks 10-12)
- [ ] Implement hook system for external integration
- [ ] Add analytics and reporting
- [ ] Create digital operator behaviors
- [ ] Performance optimization
- [ ] **Advanced Simulation**:
  - [ ] Predictive analytics and bottleneck detection
  - [ ] What-if scenario analysis tools
  - [ ] Machine learning-based model improvement
  - [ ] NVIDIA Isaac Sim integration planning

## 10. Success Metrics

### 10.1 Technical KPIs
- API response time <500ms (p95)
- Dashboard update latency <2s
- System availability >99.5%
- Zero data loss for production records

### 10.2 Business KPIs
- Reduce line setup time by 50%
- Increase visibility into production bottlenecks
- Enable remote monitoring and control
- Standardize station integration process

## 11. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Station integration complexity | High | Standardized adapter pattern, clear API specs |
| Network latency in cloud | Medium | Edge caching, local buffers, graceful degradation |
| Data synchronization | Medium | Event sourcing, conflict resolution strategies |
| Vendor API changes | Low | Version control, backward compatibility |

## 12. Appendices

### A. API Specification
Detailed OpenAPI 3.0 specification available at `/docs/api/`

### B. Database Schema
PocketBase collection schemas at `/database/schemas/`

### C. Integration Examples
Sample code for station/conveyor/operator integration at `/examples/`

### D. Deployment Guide
Cloud deployment instructions at `/docs/deployment/`

---

**Document Version**: 1.0.0  
**Last Updated**: 2025-01-28  
**Status**: Draft for Review