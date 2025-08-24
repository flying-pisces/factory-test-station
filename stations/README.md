# Factory Test Station - Reorganized Structure

This directory contains the reorganized minimum skeleton components for the project_station test system with improved separation of concerns.

## New Reorganized Structure

```
project/
├── stations/                       # Station-specific components
│   ├── project_station_run.py      # Main entry point with GUI selection
│   ├── station_config.py           # Configuration loader
│   ├── console_test_runner.py      # Console-based test interface
│   ├── simple_gui_fixed.py         # TKinter cross-platform GUI
│   ├── web_gui.py                  # Flask web-based GUI
│   ├── cleanup_utils.py            # Process cleanup utilities
│   ├── config/                     # Station configuration files
│   │   ├── station_config_project_station.py
│   │   └── station_limits_project_station.py
│   └── test_station/               # Test station implementation
│       ├── __init__.py
│       └── test_station_project_station.py
├── GUI/                            # All GUI-related components
│   ├── factory_test_gui_main.py    # Main WPF GUI (reorganized)
│   ├── wpf_gui_core.py            # Core WPF functionality
│   ├── gui_test_runner.py         # GUI test execution logic
│   ├── gui_dialogs.py             # Dialog components
│   ├── gui_utils.py               # GUI utilities
│   ├── operator_interface/         # Operator interface components
│   ├── UI_dep/                    # UI dependencies (Windows DLLs)
│   └── logo/                      # UI assets
├── log/                            # All logging-related components
│   ├── test_log/                   # Test logging framework
│   │   ├── test_log.py
│   │   └── shop_floor_interface/   # Shop floor integrations
│   ├── factory-test_logs/          # Generated log files
│   │   └── project_summary/
│   ├── raw_logs/                   # Raw log storage
│   ├── io_utils.py                 # I/O utilities for logging
│   └── profiling.py                # Profiling utilities
├── common/                         # Core infrastructure (minimal)
│   ├── test_station/               # Core test station framework
│   │   ├── test_station.py         # Base test station class
│   │   ├── dut/                    # Device Under Test
│   │   ├── test_equipment/         # Equipment interfaces
│   │   └── test_fixture/           # Test fixtures
│   └── utils/                      # Core utilities
│       ├── os_utils.py             # OS-specific utilities
│       ├── retries_util.py         # Retry logic
│       ├── serial_number.py        # Serial number handling
│       └── thread_utils.py         # Threading utilities
└── infrastructure/                 # Build and deployment scripts
```

## Key Improvements

### 1. Better Separation of Concerns
- **GUI**: All UI components moved to dedicated `GUI/` folder
- **Logging**: All log-related functionality in `log/` folder  
- **Common**: Only core infrastructure remains in `common/`

### 2. Code Organization
- Large GUI files split into focused components:
  - `wpf_gui_core.py`: Core WPF functionality
  - `gui_test_runner.py`: Test execution logic
  - `gui_dialogs.py`: Dialog handling
- Better module structure with proper `__init__.py` files

### 3. Improved Maintainability
- Focused, single-responsibility modules
- Clearer import paths
- Better documentation and organization

## Usage

The main entry point supports multiple GUI modes:

### Console Mode (Default)
```bash
cd stations
python project_station_run.py
python project_station_run.py --console
```

### TKinter GUI Mode (Cross-platform)
```bash
cd stations
python project_station_run.py --tk
```

### Web Browser GUI Mode
```bash
cd stations
python project_station_run.py --web
```

### Help
```bash
cd stations
python project_station_run.py --help
```

## Features

- **Console Mode**: Command-line interface for all platforms
- **TKinter GUI**: Cross-platform graphical interface
- **Web GUI**: Browser-based interface with real-time updates
- **Modular Logging**: Organized logging system in dedicated folder
- **Configuration**: Modular configuration system
- **Process Management**: Automatic cleanup utilities

## Dependencies

The system now uses a properly organized structure:
- **GUI/**: All user interface components (WPF, TK, web, operator interface)
- **log/**: Test logging framework and utilities
- **common/**: Core test station infrastructure
- **stations/**: Station-specific implementations

## Platform Support

- Windows (all GUI modes including WPF)
- macOS (console, TK, web modes)
- Linux (console, TK, web modes)