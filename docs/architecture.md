# System Architecture

## Overview

The Factory Test Station system is designed with a modular architecture that separates concerns and enables maintainability, testing, and deployment.

## Module Structure

### 1. Stations Module (`stations/`)

**Purpose**: Station-specific implementations and main entry point

**Components**:
- `project_station_run.py` - Main entry point with multi-GUI support
- `station_config.py` - Configuration loading system
- `console_test_runner.py` - Console-based testing interface
- `simple_gui_fixed.py` - Cross-platform TKinter GUI
- `web_gui.py` - Flask web-based GUI
- `config/` - Station-specific configuration files
- `test_station/` - Station implementation

**Responsibilities**:
- Application entry points
- Station-specific logic
- Configuration management
- Cross-platform compatibility

### 2. GUI Module (`GUI/`)

**Purpose**: All user interface components

**Components**:
- `factory_test_gui_main.py` - Main WPF GUI orchestrator
- `wpf_gui_core.py` - Core WPF functionality
- `gui_test_runner.py` - Test execution logic for GUIs
- `gui_dialogs.py` - Dialog handling
- `operator_interface/` - Operator interface framework
- `UI_dep/` - UI dependencies (Windows DLLs)

**Responsibilities**:
- User interface presentation
- User interaction handling
- Cross-platform GUI support
- Test execution coordination

### 3. Log Module (`log/`)

**Purpose**: Logging infrastructure and output

**Components**:
- `test_log/` - Test logging framework
- `shop_floor_interface/` - Integration with shop floor systems
- `factory-test_logs/` - Generated log files
- `raw_logs/` - Raw log storage
- `io_utils.py` - I/O utilities for logging
- `profiling.py` - Performance profiling

**Responsibilities**:
- Test result logging
- Shop floor integration
- Performance monitoring
- Data persistence

### 4. Common Module (`common/`)

**Purpose**: Core infrastructure components

**Components**:
- `test_station/` - Core test station framework
  - `test_station.py` - Base test station class
  - `dut/` - Device Under Test implementations
  - `test_equipment/` - Test equipment interfaces
  - `test_fixture/` - Test fixture components
- `utils/` - Core utility functions
  - `os_utils.py` - OS-specific operations
  - `retries_util.py` - Retry logic
  - `serial_number.py` - Serial number handling
  - `thread_utils.py` - Threading utilities

**Responsibilities**:
- Core business logic
- Hardware abstraction
- Utility functions
- Cross-platform compatibility

### 5. Build Module (`build/`)

**Purpose**: Build scripts and deployment tools

**Components**:
- `build.sh` - Build script
- `setup.py` - Python package setup
- `new-station` - Station provisioning script
- `delete-station` - Station cleanup script

**Responsibilities**:
- Binary generation
- Deployment automation
- Environment setup
- Station management

### 6. Tests Module (`tests/`)

**Purpose**: Test suite for quality assurance

**Components**:
- Unit tests for each module
- Integration tests
- System tests

**Responsibilities**:
- Code quality assurance
- Regression testing
- Continuous integration support

### 7. Docs Module (`docs/`)

**Purpose**: Documentation and guides

**Components**:
- Architecture documentation
- API documentation  
- User guides
- Developer guides

**Responsibilities**:
- System documentation
- User guidance
- Developer onboarding

## Design Principles

### 1. Separation of Concerns
Each module has a clear, single responsibility:
- GUI handles only user interface
- Log handles only logging concerns
- Common provides only core infrastructure

### 2. Loose Coupling
Modules interact through well-defined interfaces, allowing independent development and testing.

### 3. High Cohesion
Components within each module work together closely to achieve the module's purpose.

### 4. Cross-Platform Support
The system supports Windows, macOS, and Linux through:
- Platform-specific implementations
- Graceful fallbacks
- Conditional imports

### 5. Extensibility
New station types, GUIs, or logging formats can be added without modifying existing code.

## Data Flow

```
User Input → GUI → Test Runner → Test Station → DUT/Equipment
                                      ↓
Log Output ← Log Module ← Test Results
```

## Dependencies

```
stations/ → GUI/, log/, common/
GUI/ → common/
log/ → common/
common/ → (minimal external dependencies)
```

This architecture ensures that the core infrastructure (common/) has minimal dependencies, making it stable and portable, while higher-level modules can use more specialized libraries.