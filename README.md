# Factory Test Station

A comprehensive, modular factory test station system for hardware testing with multiple GUI interfaces and cross-platform support.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd factory-test-station

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Running the Test Station

```bash
# Console mode (default)
cd stations
python project_station_run.py

# TKinter GUI (cross-platform)
python project_station_run.py --tk

# Web browser GUI
python project_station_run.py --web

# Help
python project_station_run.py --help
```

## 📁 Project Structure

```
factory-test-station/
├── stations/           # Station implementations & main entry point
├── GUI/               # All user interface components
├── log/               # Logging infrastructure & output
├── common/            # Core infrastructure components  
├── build/             # Build scripts & deployment tools
├── tests/             # Test suite
├── docs/              # Documentation
└── reference/         # Original source material
```

## ✨ Key Features

### Multi-Interface Support
- **Console Mode**: Command-line interface for all platforms
- **TKinter GUI**: Cross-platform graphical interface  
- **Web GUI**: Browser-based interface with real-time updates
- **WPF GUI**: Native Windows interface (Windows only)

### Cross-Platform Compatibility
- **Windows**: All GUI modes supported
- **macOS**: Console, TKinter, and Web modes
- **Linux**: Console, TKinter, and Web modes

### Modular Architecture
- **Separation of Concerns**: Each module has a single, clear responsibility
- **Loose Coupling**: Modules interact through well-defined interfaces
- **High Cohesion**: Components within modules work together closely
- **Extensibility**: Easy to add new station types, GUIs, or logging formats

## 🏗️ Architecture Overview

The system follows a modular architecture with clear separation of concerns:

### Core Modules

- **`stations/`**: Station-specific implementations and configuration
- **`GUI/`**: User interface components (console, TK, web, WPF)
- **`log/`**: Logging framework and test result storage
- **`common/`**: Core infrastructure (test station framework, utilities)

### Design Principles

1. **Single Responsibility**: Each module handles one specific concern
2. **Clean Imports**: Minimal, well-defined dependencies between modules
3. **Cross-Platform**: Graceful fallbacks and platform-specific implementations
4. **Testability**: Modular design enables comprehensive testing

## 🔧 Configuration

Station configurations are stored in `stations/config/`:

```python
# stations/config/station_config_project_station.py
STATION_TYPE = 'project_station'
STATION_NUMBER = 0
IS_STATION_ACTIVE = True
SERIAL_NUMBER_VALIDATION = False  # Set to True for production
```

Test limits are defined in `stations/config/station_limits_project_station.py`:

```python
STATION_LIMITS = [
    {'name': 'TEST ITEM 1', 'low_limit': 1, 'high_limit': 2, 'unique_id': 11},
    {'name': 'TEST ITEM 2', 'low_limit': None, 'high_limit': None, 'unique_id': 12},
]
```

## 📊 Logging

Test results and logs are stored in the `log/` directory:

- `log/factory-test_logs/`: Test results and summaries
- `log/raw_logs/`: Raw measurement data
- `log/test_log/`: Test logging framework and shop floor integration

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=.

# Run specific test module
python -m pytest tests/test_common.py
```

## 🔨 Development

### Setting up Development Environment

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
python -m pytest

# Code formatting
black .

# Linting  
flake8 .
```

### Adding a New Station Type

1. Create configuration files in `stations/config/`
2. Implement station class in `stations/test_station/`
3. Add entry point in `stations/project_station_run.py`
4. Update documentation

### Adding a New GUI Interface

1. Create GUI module in `GUI/`
2. Implement interface following existing patterns
3. Add entry point in main runner
4. Update tests and documentation

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- [System Architecture](docs/architecture.md)
- [API Documentation](docs/api/)
- [User Guide](docs/user_guide/)
- [Developer Guide](docs/developer_guide/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/company/factory-test-station/issues)
- **Documentation**: [docs/](docs/)
- **Discussions**: [GitHub Discussions](https://github.com/company/factory-test-station/discussions)

## 🎯 Roadmap

- [ ] Enhanced web GUI with real-time charts
- [ ] Mobile app interface
- [ ] Advanced analytics dashboard
- [ ] Cloud integration for remote monitoring
- [ ] Additional station type templates