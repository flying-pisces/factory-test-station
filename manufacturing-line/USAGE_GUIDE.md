# Manufacturing Line Control System - Usage Guide

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.8+ installed
- Git repository cloned to local machine
- Basic familiarity with command line operations

### Instant System Check
```bash
# Navigate to project directory
cd manufacturing-line

# Verify system health
python3 -c "
try:
    from layers.component_layer.component_layer_engine import ComponentLayerEngine
    from layers.station_layer.station_layer_engine import StationLayerEngine
    from layers.line_layer.line_layer_engine import LineLayerEngine
    print('✅ System Ready - All core engines imported successfully')
except Exception as e:
    print(f'❌ System Issue: {e}')
    print('👉 See SYSTEM_BUGS_AND_FIXES.md for troubleshooting')
"
```

**Expected Result**: `✅ System Ready - All core engines imported successfully`

---

## 📋 Complete Documentation Index

### 🎯 **Start Here** 
| Document | Purpose | Time Required |
|----------|---------|---------------|
| **`FIRST_TIME_USER_GUIDE.md`** | Step-by-step system verification | 15 minutes |
| **`USAGE_GUIDE.md`** | This document - how to use the system | 10 minutes |

### 🔧 **System Information**
| Document | Purpose | Use Case |
|----------|---------|----------|
| **`SYSTEM_SUMMARY_AND_STATUS.md`** | Executive overview and current status | Project managers, stakeholders |
| **`SYSTEM_BUGS_AND_FIXES.md`** | Known issues and resolution steps | Developers, system administrators |
| **`VISUALIZATION_CAPABILITIES.md`** | Animation and visualization features | UI/UX teams, end users |

### 📈 **Development Plans**
| Document | Purpose | Audience |
|----------|---------|----------|
| **`WEEK_16_DEVELOPMENT_PLAN.md`** | Production deployment strategy | DevOps, deployment teams |
| **`WEEK_15_DEVELOPMENT_PLAN.md`** | Integration testing framework | QA, testing teams |
| **`WEEK_13_DEVELOPMENT_PLAN.md`** | AI/ML optimization features | Data scientists, AI engineers |

---

## 🏭 How to Use the Manufacturing System

### 1. System Architecture Overview
The system is organized into layers representing different aspects of manufacturing control:

```
📁 /layers/
├── component_layer/     - Individual component processing (CAD, API, EE)
├── station_layer/       - Manufacturing station optimization
├── line_layer/          - Production line efficiency management  
├── pm_layer/            - Production management and coordination
├── ai_layer/            - AI-powered optimization and prediction
├── ui_layer/            - User interfaces and dashboards
└── production_deployment/ - Production infrastructure and monitoring
```

### 2. Core Manufacturing Operations

#### A. Component Processing
```python
# Process different component types
from layers.component_layer.component_layer_engine import ComponentLayerEngine

engine = ComponentLayerEngine()

# Process mechanical CAD components
cad_processor = engine.get_processor('mechanical_cad')
result = cad_processor.process_component(component_data)

# Process electrical API components  
api_processor = engine.get_processor('electrical_api')
result = api_processor.process_component(component_data)

# Process embedded electronics
ee_processor = engine.get_processor('ee_embedded')  
result = ee_processor.process_component(component_data)
```

#### B. Station Optimization
```python
# Optimize manufacturing station configuration
from layers.station_layer.station_layer_engine import StationLayerEngine

engine = StationLayerEngine()

station_config = {
    'station_id': 'STATION_001',
    'uptime_requirement': 0.95,
    'cost_constraint': 100000,
    'throughput_target': 120  # Units per hour
}

result = engine.optimize_station_configuration(station_config)
print(f"Optimized UPH: {result['projected_uph']}")
print(f"Total Cost: ${result['total_cost']}")
print(f"Optimization Score: {result['optimization_score']}")
```

#### C. Line Efficiency Analysis
```python
# Analyze production line efficiency and identify bottlenecks
from layers.line_layer.line_layer_engine import LineLayerEngine

engine = LineLayerEngine()

stations = [
    {'station_id': 'ST001', 'uph': 100, 'uptime': 0.95},
    {'station_id': 'ST002', 'uph': 120, 'uptime': 0.92}, 
    {'station_id': 'ST003', 'uph': 80, 'uptime': 0.98}
]

# Calculate overall line efficiency
efficiency = engine.calculate_line_efficiency(stations)
print(f"Line Efficiency: {efficiency:.2%}")

# Identify bottlenecks
bottlenecks = engine.identify_bottlenecks(stations)
print(f"Primary Bottleneck: {bottlenecks[0]['station_id']}")
```

### 3. Web Interface Usage

#### Starting the Web Dashboard
```bash
# Navigate to UI layer
cd layers/ui_layer/web_interfaces/line_manager

# Start the web dashboard
python line_manager_dashboard.py
```

**Then open browser to**: `http://localhost:5000`

#### Available Web Interfaces
- **Line Manager Dashboard**: Production oversight and control
- **Production Operator Interface**: Station-level operations  
- **Super Admin Panel**: System-wide administration
- **Station Engineer Tools**: Technical configuration and maintenance

### 4. Real-Time Monitoring

#### Starting Production Monitoring
```python
# Start comprehensive production monitoring
from layers.production_deployment.monitoring_system import ProductionMonitoringSystem

monitor = ProductionMonitoringSystem()

# Initialize monitoring with configuration
config = {
    'collection_interval': 5,  # seconds
    'alert_thresholds': {
        'cpu_usage': 80,
        'memory_usage': 85,
        'disk_usage': 90
    }
}

monitor.start_monitoring(config)
print("🔍 Production monitoring active")
```

#### Viewing Real-Time Metrics
- **Web Dashboard**: Real-time charts and graphs at `http://localhost:5000`
- **Console Output**: Live metrics printed to terminal
- **Alert Notifications**: Automated alerts via email/SMS/webhook

---

## 🎯 Common Use Cases

### Use Case 1: New Production Line Setup
```python
# 1. Configure components
component_engine = ComponentLayerEngine()
# Process all component types (CAD, API, EE)

# 2. Optimize each station
station_engine = StationLayerEngine() 
# Configure each station for optimal performance

# 3. Balance the entire line
line_engine = LineLayerEngine()
# Calculate efficiency and eliminate bottlenecks

# 4. Start monitoring
monitoring_system = ProductionMonitoringSystem()
# Track performance and alert on issues
```

### Use Case 2: Performance Troubleshooting
```python
# 1. Identify bottlenecks
line_engine = LineLayerEngine()
bottlenecks = line_engine.identify_bottlenecks(current_stations)

# 2. Analyze station performance
station_engine = StationLayerEngine()
performance = station_engine.analyze_performance(bottleneck_station)

# 3. Optimize configuration
optimized_config = station_engine.optimize_station_configuration(current_config)

# 4. Validate improvements
new_efficiency = line_engine.calculate_line_efficiency(updated_stations)
```

### Use Case 3: Predictive Maintenance
```python
# Use AI layer for predictive analytics
from layers.ai_layer.ai_layer_engine import AILayerEngine

ai_engine = AILayerEngine()

# Predict equipment failures
predictions = ai_engine.predict_equipment_failures(equipment_data)

# Schedule maintenance  
maintenance_schedule = ai_engine.optimize_maintenance_schedule(predictions)

# Monitor implementation
for item in maintenance_schedule:
    print(f"Equipment: {item['equipment_id']}")
    print(f"Predicted Failure: {item['failure_date']}")
    print(f"Recommended Maintenance: {item['maintenance_date']}")
```

---

## 🔧 Configuration and Customization

### System Configuration
The main configuration file is located at: `config/line_config.json`

```json
{
  "system": {
    "environment": "production",
    "debug_mode": false,
    "log_level": "INFO"
  },
  "database": {
    "type": "pocketbase", 
    "connection_string": "http://localhost:8090"
  },
  "monitoring": {
    "collection_interval": 5,
    "retention_days": 90,
    "alert_channels": ["email", "webhook"]
  }
}
```

### Customizing Dashboards
Dashboard templates are located in: `layers/ui_layer/templates/`
- Modify HTML templates for layout changes
- Update CSS for styling customizations  
- Edit JavaScript for interactive functionality

### Adding New Stations
```python
# Create new station configuration
new_station_config = {
    'station_id': 'NEW_STATION_001',
    'station_type': 'assembly',
    'capabilities': ['welding', 'inspection', 'packaging'],
    'uptime_target': 0.95,
    'throughput_target': 150
}

# Register with station engine
station_engine = StationLayerEngine()
station_engine.register_station(new_station_config)
```

---

## 🚨 Troubleshooting

### Common Issues and Solutions

**Issue**: Import errors when starting system  
**Solution**: Check `SYSTEM_BUGS_AND_FIXES.md` for missing dependencies

**Issue**: Web interface not accessible  
**Solution**: 
```bash
# Check if Flask is running
ps aux | grep python
# Install missing dependencies
pip install flask flask-socketio
```

**Issue**: Database connection failed  
**Solution**: Ensure PocketBase is running on port 8090

**Issue**: Performance optimization taking too long  
**Solution**: Reduce optimization complexity in station configuration

### Getting Help
1. **First**: Check `FIRST_TIME_USER_GUIDE.md` for verification steps
2. **Second**: Review `SYSTEM_BUGS_AND_FIXES.md` for known issues  
3. **Third**: Check system logs in `/logs/` directory
4. **Last**: Contact development team with error details

---

## 📊 System Performance

### Expected Performance Metrics
- **Startup Time**: < 30 seconds for full system
- **Web Response**: < 2 seconds for dashboard loads
- **Calculation Speed**: < 1 second for optimizations
- **Memory Usage**: < 2GB for normal operations  

### Monitoring System Health
```python
# Check system health programmatically
from layers.production_deployment.monitoring_system import ProductionMonitoringSystem

monitor = ProductionMonitoringSystem()
health_status = monitor.get_system_health()

print(f"System Status: {health_status['overall_status']}")
print(f"CPU Usage: {health_status['cpu_usage']:.1f}%")
print(f"Memory Usage: {health_status['memory_usage']:.1f}%")
print(f"Active Processes: {health_status['process_count']}")
```

---

## 🎓 Learning Path

### For New Users (First Week)
1. **Day 1**: Complete `FIRST_TIME_USER_GUIDE.md` (15 minutes)
2. **Day 2**: Explore web interfaces and dashboards (2 hours)
3. **Day 3**: Try basic component and station operations (1 hour)
4. **Day 4**: Practice line efficiency analysis (1 hour)  
5. **Day 5**: Set up monitoring and alerts (30 minutes)

### For Developers (First Month)
1. **Week 1**: Understand layer architecture and core engines
2. **Week 2**: Customize web interfaces and add new features
3. **Week 3**: Integrate with external systems and databases
4. **Week 4**: Optimize performance and add monitoring

### For System Administrators
1. **Production Deployment**: Follow `WEEK_16_DEVELOPMENT_PLAN.md`
2. **Monitoring Setup**: Configure alerts and dashboards
3. **Backup Procedures**: Set up data backup and recovery
4. **User Management**: Configure role-based access control

---

## ✅ Success Indicators

You'll know the system is working correctly when:

- ✅ All import tests pass without errors
- ✅ Web dashboards load and display real-time data
- ✅ Manufacturing calculations complete in < 1 second
- ✅ Station optimizations improve efficiency metrics
- ✅ Line analysis accurately identifies bottlenecks
- ✅ Monitoring system shows green status across all metrics

**🎉 Congratulations! Your Manufacturing Line Control System is operational and ready for production use.**

---

## 📞 Support and Resources

### Documentation
- **Technical Details**: See individual layer README files
- **API Reference**: `/docs/api/` directory (when implemented)
- **Architecture Diagrams**: `/docs/architecture/` directory

### Development
- **Source Code**: All code in `/layers/` directory with comprehensive comments
- **Test Suite**: Run `python -m pytest tests/` for validation
- **Contributing**: Follow existing code patterns and layer architecture

### Community
- **Issues**: Report bugs via repository issue tracker
- **Discussions**: Technical discussions in development team channels  
- **Updates**: Monitor repository for new releases and features

**System Version**: Week 16 - Production Deployment Complete  
**Last Updated**: Week 16 Implementation  
**Status**: 75% Production Ready - Bug fixes in progress