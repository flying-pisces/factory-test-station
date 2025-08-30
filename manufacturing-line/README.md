# Manufacturing Line Control System

Multi-tier manufacturing line orchestration system for factory automation.

![License](https://img.shields.io/badge/License-Proprietary-red.svg)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)

---

<div align="center">

# 💖 SPONSOR THIS PROJECT 💖

### 🚨 **Support Open Source Manufacturing Automation** 🚨

<table>
<tr>
<td align="center" width="25%">
<h3>🌟 Manufacturing Supporter</h3>
<h2>$4.99/month</h2>
<p>✅ Digital sponsor badge<br/>
✅ Monthly updates<br/>
✅ Discord access</p>
</td>
<td align="center" width="25%" style="background-color: #f0f8ff;">
<h3>🔧 Automation Enthusiast</h3>
<h2>$19.99/month</h2>
<p>✅ Everything above +<br/>
✅ <strong>Advanced manufacturing templates</strong><br/>
✅ Early access releases</p>
</td>
<td align="center" width="25%" style="background-color: #fff8dc;">
<h3>💼 Professional Developer</h3>
<h2>$99.99/month</h2>
<p>✅ Everything above +<br/>
✅ <strong>1-hour monthly consultation</strong><br/>
✅ Logo placement</p>
</td>
<td align="center" width="25%" style="background-color: #f0fff0;">
<h3>🚀 Enterprise Sponsor</h3>
<h2>$999.99/month</h2>
<p>✅ Everything above +<br/>
✅ <strong>Custom manufacturing solutions</strong><br/>
✅ Priority development</p>
</td>
</tr>
</table>

## 🎯 [**BECOME A SPONSOR NOW**](https://github.com/sponsors/flying-pisces) 🎯

[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-GitHub-red?style=for-the-badge&logo=github)](https://github.com/sponsors/flying-pisces)
[![PayPal](https://img.shields.io/badge/PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/yinye0)
[![Ko-Fi](https://img.shields.io/badge/Ko--fi-F16061?style=for-the-badge&logo=ko-fi&logoColor=white)](https://ko-fi.com/flyingpisces)

### 💡 **Why Sponsor?**
- 🔬 **Open Source Manufacturing** • **Growing Community** • **Professional Tool Development**
- 🎯 **Your funding directly develops new automation features**
- 🏆 **Join companies supporting open source manufacturing tools**

[📋 **VIEW ALL SPONSOR TIERS & BENEFITS**](SPONSORS.md)

</div>

---

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