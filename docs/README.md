# Factory Test Station Documentation

This directory contains comprehensive documentation for the Factory Test Station system.

## Documentation Structure

```
docs/
├── README.md                    # This file
├── architecture.md              # System architecture overview
├── api/                        # API documentation
│   ├── common.md               # Common infrastructure API
│   ├── gui.md                  # GUI components API  
│   └── log.md                  # Logging framework API
├── user_guide/                 # End-user documentation
│   ├── installation.md        # Installation guide
│   ├── configuration.md       # Configuration guide
│   └── operation.md            # Operation manual
└── developer_guide/            # Developer documentation
    ├── contributing.md         # Contribution guidelines
    ├── testing.md              # Testing procedures
    └── deployment.md           # Deployment guide
```

## Quick Start

For users:
1. See [Installation Guide](user_guide/installation.md)
2. See [Configuration Guide](user_guide/configuration.md) 
3. See [Operation Manual](user_guide/operation.md)

For developers:
1. See [System Architecture](architecture.md)
2. See [Developer Guide](developer_guide/)
3. See [API Documentation](api/)

## System Overview

The Factory Test Station is organized into focused modules:

- **`stations/`** - Station-specific implementations and main entry point
- **`GUI/`** - All user interface components (console, TK, web, WPF)
- **`log/`** - Logging infrastructure and output
- **`common/`** - Core infrastructure components
- **`build/`** - Build scripts and deployment tools
- **`tests/`** - Test suite
- **`docs/`** - Documentation (this folder)