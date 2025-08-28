# Manufacturing Line Repository Reorganization Plan

## 🎯 **Complete Architecture-Driven Reorganization**

This plan reorganizes the entire repository to reflect the comprehensive manufacturing line control system with discrete event simulation backbone, standard data sockets, and AI-enabled optimization.

## 📁 **New Repository Structure**

```
manufacturing-line/
├── README.md                           # Project overview and quick start
├── CLAUDE.md                          # Claude Code development guidance  
├── requirements.txt                   # Project dependencies
├── setup.py                          # Package installation configuration
├── .gitignore                        # Git ignore patterns
├── docker-compose.yml                # Multi-service deployment
│
├── layers/                           # 🏗️ Manufacturing Layer Architecture
│   ├── __init__.py                   # Layer system initialization
│   ├── component_layer/              # Raw vendor data → Structured components
│   │   ├── __init__.py
│   │   ├── component_engine.py       # MOS Component Algo-Engine
│   │   ├── vendor_interfaces/        # CAD, API, EE data ingestion
│   │   │   ├── cad_processor.py
│   │   │   ├── api_processor.py
│   │   │   └── ee_processor.py
│   │   └── component_types/          # Specific component processing
│   │       ├── resistor_processor.py
│   │       ├── capacitor_processor.py
│   │       └── ic_processor.py
│   ├── station_layer/                # Component data → Station optimization
│   │   ├── __init__.py
│   │   ├── station_engine.py         # MOS Station Algo-Engine
│   │   ├── station_optimizer.py      # Station cost/UPH optimization
│   │   └── test_coverage/            # Test coverage analysis
│   ├── line_layer/                   # Station data → Line efficiency
│   │   ├── __init__.py
│   │   ├── line_engine.py            # MOS Line Algo-Engine
│   │   ├── line_optimizer.py         # Line efficiency calculation
│   │   └── retest_policies/          # AAB, ABA retest strategies
│   └── pm_layer/                     # 🧠 AI-Enabled Manufacturing Optimization
│       ├── __init__.py
│       ├── manufacturing_plan.py     # DUT flow simulation (10K+ DUTs)
│       ├── ai_optimizer.py           # Genetic algorithm optimization
│       ├── yield_optimization.py     # Yield vs MVA trade-off analysis
│       ├── line_visualizer.py        # Manufacturing plan visualization
│       └── pareto_analysis.py        # Multi-objective optimization
│
├── common/                           # 🔧 Shared Manufacturing Components
│   ├── __init__.py                   # Common package exports
│   ├── interfaces/                   # 🔌 Standard Data Sockets & Protocols
│   │   ├── __init__.py
│   │   ├── layer_interface.py        # MOS Algo-Engine base classes
│   │   ├── socket_manager.py         # Standard data socket manager
│   │   ├── manufacturing_interface.py # Core component interface
│   │   ├── communication_interface.py # Messaging protocols
│   │   └── data_interface.py         # Logging & metrics collection
│   ├── stations/                     # 🏭 Manufacturing Stations
│   │   ├── __init__.py
│   │   ├── base_station.py           # Abstract base station
│   │   ├── smt_station/              # SMT Station Module
│   │   │   ├── __init__.py
│   │   │   ├── smt_station.py        # Complete SMT implementation
│   │   │   ├── placement_engine.py   # Component placement logic
│   │   │   └── smt_optimization.py   # SMT-specific optimization
│   │   ├── test_station/             # Test Station Module
│   │   │   ├── __init__.py
│   │   │   ├── test_station.py       # Test station implementation
│   │   │   ├── test_sequences.py     # Test sequence definitions
│   │   │   └── measurement_engine.py # Measurement logic
│   │   ├── assembly_station/         # Assembly Station Module
│   │   ├── quality_station/          # Quality Control Station
│   │   └── station_manager.py        # Station orchestration
│   ├── operators/                    # 👤 Manufacturing Operators
│   │   ├── __init__.py
│   │   ├── base_operator.py          # Abstract base operator
│   │   ├── digital_human/            # Digital Human Module
│   │   │   ├── __init__.py
│   │   │   ├── digital_human.py      # Digital human implementation
│   │   │   ├── task_scheduler.py     # Task planning & execution
│   │   │   └── skill_library.py      # Operator skill definitions
│   │   ├── human_operator/           # Human Operator Interface
│   │   └── operator_scheduler.py     # Operator resource management
│   ├── conveyors/                    # 🔄 Transport Systems
│   │   ├── __init__.py
│   │   ├── base_conveyor.py          # Abstract base conveyor
│   │   ├── belt_conveyor/            # Belt Conveyor Module
│   │   │   ├── __init__.py
│   │   │   ├── belt_conveyor.py      # Belt conveyor implementation
│   │   │   └── belt_controller.py    # Speed & routing control
│   │   ├── indexing_conveyor/        # Indexing Conveyor Module
│   │   └── conveyor_manager.py       # Conveyor coordination
│   ├── equipment/                    # ⚙️ Test & Manufacturing Equipment
│   │   ├── __init__.py
│   │   ├── base_equipment.py         # Abstract base equipment
│   │   ├── test_equipment/           # Test Equipment Module
│   │   │   ├── __init__.py
│   │   │   ├── multimeter.py         # DMM implementation
│   │   │   ├── power_supply.py       # Power supply control
│   │   │   └── oscilloscope.py       # Scope measurement
│   │   ├── measurement_equipment/    # Precision Measurement
│   │   └── equipment_manager.py      # Equipment coordination
│   ├── fixtures/                     # 🔧 Manufacturing Fixtures
│   │   ├── __init__.py
│   │   ├── base_fixture.py           # Abstract base fixture
│   │   ├── test_fixture/             # Test Fixture Module
│   │   │   ├── __init__.py
│   │   │   ├── test_fixture.py       # Test fixture implementation
│   │   │   └── fixture_controller.py # Fixture automation
│   │   ├── assembly_fixture/         # Assembly Fixture Module
│   │   └── fixture_manager.py        # Fixture coordination
│   └── utils/                        # 🛠️ Utility Functions
│       ├── __init__.py
│       ├── timing_utils.py           # Timer & delay utilities
│       ├── data_utils.py             # Data validation & serialization
│       ├── math_utils.py             # Statistics & calculations
│       ├── config_manager.py         # Configuration management
│       └── logger.py                 # Centralized logging
│
├── simulation/                       # 🎮 Discrete Event Simulation Engine
│   ├── __init__.py                   # Simulation package initialization
│   ├── discrete_event_fsm/           # 🔄 FSM-Based Simulation Backbone
│   │   ├── __init__.py
│   │   ├── base_fsm.py              # DiscreteEventScheduler & BaseFiniteStateMachine
│   │   ├── dut_fsm.py               # Device Under Test FSM
│   │   ├── fixture_fsm.py           # Manufacturing fixture FSM
│   │   ├── equipment_fsm.py         # Test equipment FSM
│   │   ├── operator_fsm.py          # Operator behavior FSM
│   │   ├── conveyor_fsm.py          # Conveyor transport FSM
│   │   └── station_fsm.py           # Station orchestration FSM
│   ├── jaamsim_integration/          # 🏭 JAAMSIM Engine Integration
│   │   ├── __init__.py
│   │   ├── jaamsim_bridge.py        # JAAMSIM communication bridge
│   │   ├── turntable_fixtures/       # Existing 1-up & 3-up turntables
│   │   │   ├── turntable_1up.py     # Single DUT fixture simulation
│   │   │   └── turntable_3up.py     # Three DUT fixture simulation
│   │   ├── model_generator.py        # JAAMSIM model generation
│   │   └── results_processor.py     # Simulation results analysis
│   ├── simulation_engine/            # 🎯 Core Simulation Framework
│   │   ├── __init__.py
│   │   ├── discrete_event.py         # DiscreteEvent class definitions
│   │   ├── event_scheduler.py        # Event scheduling algorithms
│   │   ├── time_manager.py           # Simulation time management
│   │   ├── statistics_collector.py   # Performance metrics collection
│   │   └── digital_twin.py           # Digital twin synchronization
│   └── scenarios/                    # 📊 Simulation Scenarios
│       ├── __init__.py
│       ├── manufacturing_scenarios.py # Pre-defined test scenarios
│       ├── optimization_scenarios.py  # Optimization test cases
│       └── validation_scenarios.py    # Model validation scenarios
│
├── web_interfaces/                   # 🌐 Multi-Tier Web Architecture
│   ├── __init__.py                   # Web interface package
│   ├── super_admin/                  # 👑 Super Admin Interface
│   │   ├── dashboard.py              # System overview dashboard
│   │   ├── user_management.py        # User & role management
│   │   └── system_configuration.py   # Global system config
│   ├── line_manager/                 # 🏭 Line Manager Interface
│   │   ├── line_dashboard.py         # Line performance dashboard
│   │   ├── station_monitoring.py     # Station status monitoring
│   │   └── production_planning.py    # Production schedule management
│   ├── station_engineer/             # 🔧 Station Engineer Interface
│   │   ├── station_control.py        # Individual station control
│   │   ├── test_configuration.py     # Test setup & limits
│   │   └── diagnostics.py            # Station diagnostics
│   ├── component_vendor/             # 📦 Component Vendor Interface
│   │   ├── data_upload.py            # CAD/API/EE data upload
│   │   ├── component_status.py       # Component processing status
│   │   └── vendor_dashboard.py       # Vendor performance metrics
│   └── shared/                       # 🔄 Shared Web Components
│       ├── authentication.py         # Role-based access control
│       ├── api_gateway.py            # Unified API gateway
│       └── websocket_manager.py      # Real-time communication
│
├── database/                         # 🗄️ Data Persistence Layer
│   ├── __init__.py                   # Database package initialization
│   ├── pocketbase/                   # 📱 PocketBase Integration
│   │   ├── __init__.py
│   │   ├── client.py                 # PocketBase client wrapper
│   │   ├── schemas/                  # Database schema definitions
│   │   │   ├── users.py              # User & role schemas
│   │   │   ├── stations.py           # Station data schemas
│   │   │   ├── components.py         # Component data schemas
│   │   │   └── test_results.py       # Test result schemas
│   │   └── migrations/               # Database migrations
│   ├── models/                       # 📋 Data Models
│   │   ├── __init__.py
│   │   ├── user_models.py            # User & authentication models
│   │   ├── manufacturing_models.py   # Core manufacturing entities
│   │   ├── test_models.py            # Test execution & results
│   │   └── optimization_models.py    # AI optimization data models
│   └── repositories/                 # 🔍 Data Access Layer
│       ├── __init__.py
│       ├── user_repository.py        # User data operations
│       ├── station_repository.py     # Station data operations
│       └── test_repository.py        # Test data operations
│
├── line_controller/                  # 🎮 Line Control System
│   ├── __init__.py                   # Line controller package
│   ├── main_controller.py            # Master line controller
│   ├── station_controllers/          # 🏭 Individual Station Controllers
│   │   ├── __init__.py
│   │   ├── smt_controller.py         # SMT station controller
│   │   └── test_controller.py        # Test station controller
│   ├── coordination/                 # 🔄 Cross-Station Coordination
│   │   ├── __init__.py
│   │   ├── workflow_engine.py        # DUT workflow orchestration
│   │   ├── resource_manager.py       # Shared resource allocation
│   │   └── synchronization.py        # Station synchronization
│   └── monitoring/                   # 📊 Real-Time Monitoring
│       ├── __init__.py
│       ├── performance_monitor.py    # UPH & efficiency tracking
│       ├── alarm_manager.py          # Alarm & notification system
│       └── metrics_collector.py      # KPI collection & reporting
│
├── tests/                            # 🧪 Comprehensive Test Suite
│   ├── __init__.py                   # Test package initialization
│   ├── unit/                         # 🔍 Unit Tests
│   │   ├── __init__.py
│   │   ├── test_layers/              # Layer-specific unit tests
│   │   │   ├── test_component_layer.py
│   │   │   ├── test_station_layer.py
│   │   │   ├── test_line_layer.py
│   │   │   └── test_pm_layer.py
│   │   ├── test_common/              # Common component unit tests
│   │   │   ├── test_stations.py
│   │   │   ├── test_operators.py
│   │   │   ├── test_conveyors.py
│   │   │   ├── test_equipment.py
│   │   │   └── test_fixtures.py
│   │   ├── test_simulation/          # Simulation unit tests
│   │   │   ├── test_fsm.py
│   │   │   ├── test_jaamsim.py
│   │   │   └── test_discrete_events.py
│   │   └── test_interfaces/          # Interface unit tests
│   │       ├── test_sockets.py
│   │       └── test_data_transfer.py
│   ├── integration/                  # 🔗 Integration Tests
│   │   ├── __init__.py
│   │   ├── test_layer_integration.py # Cross-layer communication
│   │   ├── test_socket_pipeline.py   # End-to-end socket pipeline
│   │   ├── test_simulation_sync.py   # Simulation synchronization
│   │   └── test_web_api.py           # Web interface integration
│   ├── system/                       # 🏗️ System Tests
│   │   ├── __init__.py
│   │   ├── test_full_pipeline.py     # Complete manufacturing pipeline
│   │   ├── test_optimization.py      # AI optimization validation
│   │   ├── test_performance.py       # Performance benchmarking
│   │   └── test_scalability.py       # System scalability testing
│   ├── acceptance/                   # ✅ Acceptance Tests
│   │   ├── __init__.py
│   │   ├── test_user_stories.py      # User story validation
│   │   ├── test_business_rules.py    # Business logic validation
│   │   ├── test_compliance.py        # Regulatory compliance tests
│   │   └── test_production_readiness.py # Production deployment tests
│   ├── fixtures/                     # 🎯 Test Data & Fixtures
│   │   ├── __init__.py
│   │   ├── sample_data/              # Sample manufacturing data
│   │   │   ├── components.json       # Sample component data
│   │   │   ├── stations.json         # Sample station configurations
│   │   │   └── lines.json            # Sample line definitions
│   │   ├── mock_services/            # Mock external services
│   │   │   ├── mock_pocketbase.py    # Mock database service
│   │   │   └── mock_jaamsim.py       # Mock simulation service
│   │   └── test_scenarios/           # Predefined test scenarios
│   │       ├── optimization_test.py  # Optimization test scenarios
│   │       └── failure_scenarios.py  # Failure condition testing
│   ├── conftest.py                   # Pytest configuration
│   ├── pytest.ini                   # Pytest settings
│   └── coverage.ini                  # Coverage configuration
│
├── config/                           # ⚙️ Configuration Management
│   ├── __init__.py                   # Configuration package
│   ├── environments/                 # 🌍 Environment-Specific Configs
│   │   ├── development.yml           # Development environment
│   │   ├── staging.yml               # Staging environment
│   │   ├── production.yml            # Production environment
│   │   └── testing.yml               # Testing environment
│   ├── stations/                     # 🏭 Station Configurations
│   │   ├── station_limits.yml        # Test limits per station
│   │   ├── smt_config.yml            # SMT station configuration
│   │   └── test_config.yml           # Test station configuration
│   ├── simulation/                   # 🎮 Simulation Configurations
│   │   ├── fsm_config.yml            # FSM timing parameters
│   │   ├── jaamsim_config.yml        # JAAMSIM model parameters
│   │   └── optimization_config.yml   # AI optimization parameters
│   └── database/                     # 🗄️ Database Configurations
│       ├── pocketbase_config.yml     # PocketBase settings
│       └── schema_migrations.yml     # Migration tracking
│
├── deployment/                       # 🚀 Deployment & Infrastructure
│   ├── __init__.py                   # Deployment package
│   ├── docker/                       # 🐳 Docker Configuration
│   │   ├── Dockerfile                # Main application container
│   │   ├── docker-compose.prod.yml   # Production orchestration
│   │   ├── docker-compose.dev.yml    # Development orchestration
│   │   └── services/                 # Individual service containers
│   │       ├── pocketbase.dockerfile # Database container
│   │       └── web.dockerfile        # Web interface container
│   ├── kubernetes/                   # ☸️ Kubernetes Deployment
│   │   ├── namespace.yml             # Kubernetes namespace
│   │   ├── deployments/              # Application deployments
│   │   ├── services/                 # Service definitions
│   │   └── ingress/                  # Load balancer configuration
│   ├── cloud/                        # ☁️ Cloud Provider Configurations
│   │   ├── aws/                      # Amazon Web Services
│   │   ├── gcp/                      # Google Cloud Platform
│   │   └── azure/                    # Microsoft Azure
│   └── scripts/                      # 📜 Deployment Scripts
│       ├── deploy.sh                 # Main deployment script
│       ├── backup.sh                 # Database backup script
│       └── monitor.sh                # Health monitoring script
│
├── docs/                             # 📚 Project Documentation
│   ├── __init__.py                   # Documentation package
│   ├── architecture/                 # 🏗️ Architecture Documentation
│   │   ├── system_overview.md        # High-level system architecture
│   │   ├── layer_architecture.md     # Manufacturing layer details
│   │   ├── socket_architecture.md    # Standard data socket design
│   │   ├── simulation_architecture.md # Discrete event simulation design
│   │   └── deployment_architecture.md # Infrastructure & deployment
│   ├── api/                          # 📡 API Documentation
│   │   ├── layer_apis.md             # Manufacturing layer APIs
│   │   ├── socket_apis.md            # Data socket APIs
│   │   ├── web_apis.md               # Web interface APIs
│   │   └── simulation_apis.md        # Simulation engine APIs
│   ├── user_guides/                  # 👥 User Documentation
│   │   ├── super_admin_guide.md      # Super admin user guide
│   │   ├── line_manager_guide.md     # Line manager user guide
│   │   ├── station_engineer_guide.md # Station engineer guide
│   │   └── component_vendor_guide.md # Component vendor guide
│   ├── developer_guides/             # 👨‍💻 Developer Documentation
│   │   ├── getting_started.md        # Quick start for developers
│   │   ├── contributing.md           # Contribution guidelines
│   │   ├── testing.md                # Testing best practices
│   │   ├── deployment.md             # Deployment procedures
│   │   └── troubleshooting.md        # Common issues & solutions
│   ├── technical_specs/              # 🔧 Technical Specifications
│   │   ├── data_schemas.md           # Data structure definitions
│   │   ├── communication_protocols.md # Messaging specifications
│   │   ├── performance_requirements.md # Performance benchmarks
│   │   └── security_requirements.md  # Security specifications
│   └── examples/                     # 💡 Code Examples
│       ├── basic_usage.md            # Simple usage examples
│       ├── advanced_scenarios.md     # Complex implementation examples
│       └── integration_examples.md   # Third-party integration examples
│
└── tools/                            # 🔧 Development & Maintenance Tools
    ├── __init__.py                   # Tools package
    ├── data_migration/               # 📦 Data Migration Tools
    │   ├── __init__.py
    │   ├── legacy_importer.py        # Import from legacy systems
    │   ├── data_validator.py         # Validate migrated data
    │   └── schema_converter.py       # Convert between data schemas
    ├── monitoring/                   # 📊 System Monitoring Tools
    │   ├── __init__.py
    │   ├── health_checker.py         # System health monitoring
    │   ├── performance_profiler.py   # Performance analysis
    │   └── log_analyzer.py           # Log analysis & insights
    ├── code_generation/              # 🏭 Code Generation Tools
    │   ├── __init__.py
    │   ├── socket_generator.py       # Generate standard data sockets
    │   ├── fsm_generator.py          # Generate FSM templates
    │   └── api_generator.py          # Generate API endpoints
    └── maintenance/                  # 🛠️ Maintenance Scripts
        ├── __init__.py
        ├── database_cleanup.py       # Database maintenance
        ├── log_rotation.py           # Log file management
        └── backup_manager.py         # Automated backup management
```

## 🔄 **Migration Strategy**

### Phase 1: Core Architecture Migration
1. **Create new folder structure** with proper `__init__.py` files
2. **Migrate existing files** to appropriate locations
3. **Update import paths** throughout the codebase
4. **Establish layer interfaces** with standard data sockets

### Phase 2: Feature Integration
1. **Integrate discrete event FSM** as system backbone
2. **Connect JAAMSIM simulation** with existing turntable work
3. **Implement PM layer** with AI optimization
4. **Establish web interfaces** for multi-tier architecture

### Phase 3: Testing & Validation
1. **Create comprehensive test suite** (unit/integration/system/acceptance)
2. **Validate data pipeline** end-to-end
3. **Performance benchmarking** and optimization
4. **User acceptance testing** for all roles

### Phase 4: Deployment Preparation
1. **Containerization** with Docker
2. **Cloud deployment** configuration
3. **Monitoring & logging** infrastructure
4. **Documentation completion**

## 🎯 **Key Architecture Principles**

### 1. **Layer Separation with Standard Sockets**
- Each layer (Component → Station → Line → PM) operates independently
- Standard data sockets enable seamless communication
- MOS Algo-Engines process raw data into structured formats

### 2. **Discrete Event Simulation Backbone**
- All components inherit from BaseFiniteStateMachine
- Fixed-duration event methods with timing precision
- JAAMSIM integration for complex simulation scenarios

### 3. **Component-Based Organization**
- Dedicated folders for stations, operators, conveyors, equipment, fixtures
- Standardized interfaces for all component types
- Hierarchical import structure for clean code organization

### 4. **Multi-Tier Web Architecture**
- Role-based access control (Super Admin, Line Manager, Station Engineer, Component Vendor)
- Specialized interfaces optimized per user role
- Real-time communication via WebSocket

### 5. **AI-Enabled Optimization**
- PM layer with genetic algorithm optimization
- Yield vs MVA trade-off analysis
- Pareto optimal solution discovery

## 📊 **Benefits of New Structure**

### **Scalability**
- Users can purchase and operate at any single layer
- Independent development and deployment cycles
- Horizontal scaling per layer requirements

### **Maintainability**
- Clear separation of concerns
- Standardized interfaces reduce coupling
- Component-based architecture enables focused development

### **Extensibility**
- New component types easily added
- AI algorithms independently evolved
- Standard sockets support future layer additions

### **Testability**
- Comprehensive test coverage at all levels
- Component isolation enables focused testing
- Simulation integration validates system behavior

This reorganization transforms the manufacturing line system into a scalable, maintainable, and extensible architecture that reflects the comprehensive requirements while maintaining the discrete event simulation backbone as the core organizing principle.