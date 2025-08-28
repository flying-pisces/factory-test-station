# Manufacturing Line Control System

Multi-tier manufacturing line orchestration system for factory automation.

## Architecture Overview

This system manages manufacturing lines with:
- 15 stations (SMT: ICT, Firmware, FCT | FATP: IQC, Camera, Display, RF, WiFi, OTA, Battery, Housing, Burn-in, OS-fusion, MMI, Packaging)
- Point-to-point conveyor systems
- Digital operators for manual interventions
- Cloud deployment with PocketBase backend

## Repository Structure

```
manufacturing-line/
├── line-controller/     # Line orchestration logic
├── stations/           # Submodule → factory-test-station repo
├── conveyors/          # Submodule → conveyor-system repo  
├── operators/          # Submodule → digital-operator repo
├── common/             # Shared libraries
├── web-portal/         # Web UI (React/Vue)
├── database/           # PocketBase schemas
├── config/             # JSON configurations
├── hooks/              # Integration endpoints
└── docs/               # Documentation
```

## Quick Start

```bash
# Clone with submodules
git clone --recursive https://github.com/[org]/manufacturing-line.git

# Install dependencies
pip install -r requirements.txt
npm install

# Configure PocketBase
cd database && ./pocketbase serve

# Run line controller
python line-controller/main.py

# Start web portal
cd web-portal && npm run dev
```

## Access Levels

| URL | Access Level | Description |
|-----|--------------|-------------|
| /line | Line Manager | Monitor entire production line |
| /line/station/* | Station Engineer | Configure and develop stations |
| /line/station/fixture/* | Component Vendor | Manage component specs |

## Documentation

- [PRD](./PRD_Manufacturing_Line_System.md) - Product Requirements
- [API Docs](./docs/api/) - REST API specifications
- [Integration Guide](./docs/integration/) - Hook system documentation

## License

Proprietary - All rights reserved