# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Factory Test Station is a comprehensive, modular factory test station system for hardware testing with multiple GUI interfaces and cross-platform support. The system has been reorganized from the original `hardware_station_common` structure into a cleaner, more maintainable architecture.

## Development Commands

### Installation and Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Create required directories (if not exist)
mkdir -p log/raw_logs
mkdir -p log/factory-test_logs/project_summary
```

### Running the Test Station

#### Console Mode (Cross-platform)
```bash
cd stations
python simple_console_test.py                    # Interactive mode
python simple_console_test.py TEST123            # Single test
python simple_console_test.py --help             # Help
```

#### TKinter GUI (Cross-platform)
```bash
cd stations
python simple_gui_fixed.py                       # TKinter GUI mode
```

#### Web Browser GUI (Cross-platform)
```bash
cd stations  
python web_gui.py                                # Web interface at http://localhost:5000 (matches TK layout)
python simple_web_gui.py                         # Simple web interface (legacy)
```

#### Unified Entry Point (Fixed)
```bash
cd stations
python project_station_run.py --console          # Console mode (default)
python project_station_run.py --tk               # TKinter GUI mode
python project_station_run.py --web              # Web browser GUI mode
python project_station_run.py --help             # Help
```

### Testing Commands
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=.

# Run specific test module
python -m pytest tests/test_common.py
```

## Project Structure

```
factory-test-station/
├── stations/           # Station implementations & main entry points
│   ├── config/         # Station configurations and test limits
│   ├── DUT/           # Device Under Test implementations  
│   ├── equipment/     # Test equipment interfaces
│   ├── fixture/       # Test fixture components
│   └── test_station/  # Station orchestration classes
├── GUI/               # User interface components (WPF, TK, console)
├── log/               # Logging infrastructure & output
│   ├── test_log/      # Test logging framework
│   └── factory-test_logs/  # Generated log files
├── common/            # Core infrastructure components
│   ├── test_station/  # Core test station framework
│   └── utils/         # Utility functions
├── build/             # Build scripts & deployment tools
├── tests/             # Test suite
├── docs/              # Documentation
└── reference/         # Original source material (hardware_station_common)
```

## Architecture Overview

### Module Responsibilities

- **`stations/`**: Entry points, station-specific logic, configuration management
- **`GUI/`**: User interface components (console, TK, web, WPF) 
- **`log/`**: Test result logging, shop floor integration, performance monitoring
- **`common/`**: Core business logic, hardware abstraction, utility functions

### Cross-Platform Compatibility
- **Windows**: All GUI modes supported (console, TK, web, WPF)
- **macOS**: Console, TKinter, and Web modes
- **Linux**: Console, TKinter, and Web modes

## Configuration Management

### Station Configuration
```python
# stations/config/station_config_project_station.py
STATION_TYPE = 'project_station'
STATION_NUMBER = 0
IS_STATION_ACTIVE = True
SERIAL_NUMBER_VALIDATION = False  # Set to True for production
```

### Test Limits
```python
# stations/config/station_limits_project_station.py
STATION_LIMITS = [
    {'name': 'TEST ITEM 1', 'low_limit': 1, 'high_limit': 2, 'unique_id': 11},
    {'name': 'TEST ITEM 2', 'low_limit': None, 'high_limit': None, 'unique_id': 12},
]
```

## Key Development Patterns

### Import Path Fixes
The codebase was migrated from `hardware_station_common` imports to the new `common` structure. Key changes:
- `hardware_station_common.test_station.test_station` → `common.test_station.test_station`
- `hardware_station_common.utils` → `common.utils`
- Path management using `sys.path.insert()` for cross-module imports

### Test Execution Flow
1. **Station Initialization**: Load configuration, initialize hardware interfaces
2. **Test Sequence**: Execute `_do_test()` method with test items validation
3. **Results Processing**: Generate logs, update shop floor systems
4. **Cleanup**: Restore fixture to known state

### UI Mode Selection
The system automatically detects platform capabilities and selects appropriate UI:
- Console mode: Always available, cross-platform default
- TKinter: Cross-platform GUI when display available  
- Web: Browser-based interface with real-time updates
- WPF: Windows-only native interface

### Error Handling
- Custom exception classes for different failure modes
- Graceful degradation when modules unavailable
- Retry mechanisms for hardware operations
- Platform-specific fallbacks

## Data Flow

```
User Input → GUI → Test Runner → Test Station → DUT/Equipment/Fixture
                                        ↓
Log Output ← Log Module ← Test Results
```

## Dependencies

- **Core**: Python 3.7+, psutil, lxml, requests
- **GUI**: tkinter (built-in), flask, flask-socketio
- **Windows**: pywin32, pythonnet (for WPF)
- **Serial**: pyserial
- **Config**: pyyaml, configparser
- **Logging**: colorama

## File Organization

### Configuration Files
- Station configs: `stations/config/station_config_*.py`
- Station limits: `stations/config/station_limits_*.py`
- Main entry points: `stations/*_run.py`

### Naming Conventions
- Station implementations: `test_station_*.py`
- DUT implementations: `*Dut` classes
- Fixture implementations: `*Fixture` classes
- Configuration modules: `station_config_*.py`

## Logging and Output

Test results and logs are stored in:
- `log/factory-test_logs/`: Test results and summaries
- `log/raw_logs/`: Raw measurement data
- `log/test_log/`: Test logging framework

## Development Notes

### Working UI Modes (Verified)
- ✅ Console mode: `simple_console_test.py` - Fully functional
- ✅ TKinter GUI: `simple_gui_fixed.py` - Test items table from station limits config
- ✅ Web GUI: `web_gui.py` - Matches TKinter layout exactly, real-time test value updates, loads from station limits
- ✅ Web GUI (Simple): `simple_web_gui.py` - Basic Flask interface (legacy)
- ✅ Unified entry point: `project_station_run.py` - All modes working via reliable implementations

### Migration Status
- ✅ Core imports fixed from `hardware_station_common` to `common`
- ✅ Station configuration loading working
- ✅ Test execution flow functional
- ✅ Cross-platform compatibility maintained
- ✅ Unified entry point fixed to use reliable simple implementations

### Recommended Development Workflow
1. Use `simple_console_test.py` for basic functionality testing
2. Use `web_gui.py` for web interface development (matches TKinter layout exactly)
3. Use `simple_gui_fixed.py` for TKinter GUI work  
4. Use unified entry point `project_station_run.py --[console|tk|web]` for complete testing
5. For complex hardware integration, extend the simple implementations rather than fixing legacy imports

### Web GUI Features (TKinter Layout Match)
- **Identical Layout**: Header, serial input, test table, console log sections match TKinter exactly
- **Test Items Table**: Loaded from `station_limits_project_station.py` configuration 
- **Real-time Updates**: Test values update live during testing via WebSocket
- **Persistent Table**: Only "Test Value" column updates, structure remains constant
- **Status Indicators**: Visual feedback (pass ✓, fail ✗, testing ⏳)
- **Auto Scan**: Generate test serial numbers automatically
- **Console Log**: Real-time logging with timestamps and color coding

The system maintains backward compatibility while providing simplified, reliable entry points for each UI mode.