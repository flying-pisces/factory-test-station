"""
Dashboard Manager - Week 13: UI & Visualization Layer

Manages multiple dashboards for different user roles (operator, manager, technician)
with real-time data integration and customizable layouts.
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import threading
from enum import Enum


class DashboardType(Enum):
    """Dashboard types for different interfaces."""
    OPERATOR = "operator"
    MANAGER = "manager"
    TECHNICIAN = "technician"
    MOBILE = "mobile"
    EXECUTIVE = "executive"


class DashboardRole:
    """Dashboard roles and permissions."""
    OPERATOR = "operator"
    MANAGER = "manager"
    TECHNICIAN = "technician"
    ADMINISTRATOR = "administrator"


class DashboardManager:
    """
    Dashboard management system for manufacturing line interfaces.
    
    Manages operator, manager, and technician dashboards with role-based
    access, real-time updates, and customizable layouts.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Dashboard Manager.
        
        Args:
            config: Configuration dictionary for dashboard settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Performance targets
        self.dashboard_load_target_ms = 2000    # Dashboard load under 2 seconds
        self.widget_update_target_ms = 100      # Widget updates under 100ms
        self.layout_change_target_ms = 500      # Layout changes under 500ms
        
        # Dashboard storage
        self.dashboards = {}
        self.active_sessions = {}
        self.widget_registry = {}
        
        # User role configurations
        self.role_permissions = {}
        self.role_defaults = {}
        
        # Performance metrics
        self.dashboard_metrics = {
            'dashboards_created': 0,
            'active_dashboards': 0,
            'widget_updates': 0,
            'avg_load_time_ms': 0.0,
            'error_count': 0,
            'user_sessions': 0
        }
        
        # Thread safety
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dashboard")
        
        # Initialize dashboard system
        self._initialize_dashboard_system()
        
        self.logger.info("DashboardManager initialized successfully")
    
    def _initialize_dashboard_system(self):
        """Initialize dashboard management system."""
        try:
            # Initialize role permissions
            self.role_permissions = {
                DashboardRole.OPERATOR: {
                    'widgets': [
                        'production_status', 'quality_metrics', 'equipment_status',
                        'ai_alerts', 'emergency_controls', 'shift_summary'
                    ],
                    'actions': ['acknowledge_alert', 'emergency_stop', 'log_issue'],
                    'data_access': ['real_time', 'current_shift']
                },
                DashboardRole.MANAGER: {
                    'widgets': [
                        'kpi_overview', 'production_trends', 'efficiency_metrics',
                        'cost_analysis', 'resource_utilization', 'ai_insights',
                        'performance_reports'
                    ],
                    'actions': ['schedule_maintenance', 'adjust_targets', 'generate_reports'],
                    'data_access': ['real_time', 'historical', 'analytics']
                },
                DashboardRole.TECHNICIAN: {
                    'widgets': [
                        'equipment_diagnostics', 'maintenance_schedule', 'sensor_data',
                        'predictive_maintenance', 'troubleshooting_guides', 'parts_inventory'
                    ],
                    'actions': ['update_maintenance', 'run_diagnostics', 'order_parts'],
                    'data_access': ['real_time', 'historical', 'diagnostic']
                },
                DashboardRole.ADMINISTRATOR: {
                    'widgets': '*',  # All widgets
                    'actions': '*',   # All actions
                    'data_access': '*'  # All data
                }
            }
            
            # Initialize default dashboard layouts
            self._initialize_default_dashboards()
            
            # Initialize widget registry
            self._initialize_widget_registry()
            
            self.logger.info("Dashboard system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize dashboard system: {e}")
            raise
    
    def _initialize_default_dashboards(self):
        """Initialize default dashboard configurations for each role."""
        # Operator Dashboard
        operator_dashboard = {
            'dashboard_id': 'operator_default',
            'role': DashboardRole.OPERATOR,
            'title': 'Production Control Dashboard',
            'layout': {
                'type': 'grid',
                'columns': 3,
                'rows': 4,
                'responsive': True
            },
            'widgets': [
                {
                    'widget_id': 'production_status',
                    'type': 'status_card',
                    'position': {'x': 0, 'y': 0, 'w': 1, 'h': 1},
                    'config': {
                        'title': 'Production Status',
                        'data_source': 'real_time_production',
                        'refresh_interval': 1000
                    }
                },
                {
                    'widget_id': 'quality_metrics',
                    'type': 'gauge_chart',
                    'position': {'x': 1, 'y': 0, 'w': 1, 'h': 1},
                    'config': {
                        'title': 'Quality Score',
                        'data_source': 'quality_analytics',
                        'min': 0, 'max': 100,
                        'target': 95
                    }
                },
                {
                    'widget_id': 'equipment_status',
                    'type': '3d_equipment_view',
                    'position': {'x': 2, 'y': 0, 'w': 1, 'h': 2},
                    'config': {
                        'title': 'Equipment Status',
                        'data_source': 'equipment_monitoring',
                        'view_mode': '3d_factory'
                    }
                },
                {
                    'widget_id': 'ai_alerts',
                    'type': 'alert_list',
                    'position': {'x': 0, 'y': 1, 'w': 2, 'h': 1},
                    'config': {
                        'title': 'AI Insights & Alerts',
                        'data_source': 'ai_predictions',
                        'max_alerts': 5
                    }
                },
                {
                    'widget_id': 'throughput_chart',
                    'type': 'line_chart',
                    'position': {'x': 0, 'y': 2, 'w': 3, 'h': 1},
                    'config': {
                        'title': 'Production Throughput',
                        'data_source': 'throughput_metrics',
                        'time_range': '8h'
                    }
                },
                {
                    'widget_id': 'emergency_controls',
                    'type': 'control_panel',
                    'position': {'x': 0, 'y': 3, 'w': 3, 'h': 1},
                    'config': {
                        'title': 'Emergency Controls',
                        'controls': ['emergency_stop', 'alarm_silence', 'supervisor_call']
                    }
                }
            ],
            'theme': {
                'primary_color': '#1976d2',
                'alert_color': '#f44336',
                'success_color': '#4caf50',
                'warning_color': '#ff9800'
            },
            'created_at': datetime.now().isoformat()
        }
        
        # Manager Dashboard
        manager_dashboard = {
            'dashboard_id': 'manager_default',
            'role': DashboardRole.MANAGER,
            'title': 'Manufacturing Intelligence Dashboard',
            'layout': {
                'type': 'grid',
                'columns': 4,
                'rows': 3,
                'responsive': True
            },
            'widgets': [
                {
                    'widget_id': 'kpi_overview',
                    'type': 'kpi_grid',
                    'position': {'x': 0, 'y': 0, 'w': 2, 'h': 1},
                    'config': {
                        'title': 'Key Performance Indicators',
                        'kpis': ['oee', 'throughput', 'quality', 'efficiency'],
                        'show_trends': True
                    }
                },
                {
                    'widget_id': 'production_trends',
                    'type': 'multi_line_chart',
                    'position': {'x': 2, 'y': 0, 'w': 2, 'h': 1},
                    'config': {
                        'title': 'Production Trends',
                        'metrics': ['actual', 'target', 'forecast'],
                        'time_range': '7d'
                    }
                },
                {
                    'widget_id': 'ai_insights',
                    'type': 'ai_insights_panel',
                    'position': {'x': 0, 'y': 1, 'w': 2, 'h': 1},
                    'config': {
                        'title': 'AI Recommendations',
                        'insight_types': ['optimization', 'maintenance', 'quality']
                    }
                },
                {
                    'widget_id': 'cost_analysis',
                    'type': 'cost_breakdown_chart',
                    'position': {'x': 2, 'y': 1, 'w': 2, 'h': 1},
                    'config': {
                        'title': 'Cost Analysis',
                        'categories': ['labor', 'materials', 'energy', 'maintenance']
                    }
                },
                {
                    'widget_id': 'performance_heatmap',
                    'type': 'heatmap_chart',
                    'position': {'x': 0, 'y': 2, 'w': 4, 'h': 1},
                    'config': {
                        'title': 'Equipment Performance Heatmap',
                        'data_source': 'equipment_performance',
                        'time_granularity': 'hour'
                    }
                }
            ],
            'theme': {
                'primary_color': '#673ab7',
                'accent_color': '#9c27b0',
                'background_color': '#f5f5f5'
            },
            'created_at': datetime.now().isoformat()
        }
        
        # Technician Dashboard
        technician_dashboard = {
            'dashboard_id': 'technician_default',
            'role': DashboardRole.TECHNICIAN,
            'title': 'Maintenance & Diagnostics Dashboard',
            'layout': {
                'type': 'grid',
                'columns': 3,
                'rows': 3,
                'responsive': True
            },
            'widgets': [
                {
                    'widget_id': 'equipment_health',
                    'type': 'equipment_health_grid',
                    'position': {'x': 0, 'y': 0, 'w': 2, 'h': 1},
                    'config': {
                        'title': 'Equipment Health Status',
                        'health_indicators': ['temperature', 'vibration', 'pressure']
                    }
                },
                {
                    'widget_id': 'maintenance_schedule',
                    'type': 'schedule_calendar',
                    'position': {'x': 2, 'y': 0, 'w': 1, 'h': 2},
                    'config': {
                        'title': 'Maintenance Schedule',
                        'view_mode': 'week',
                        'show_predictions': True
                    }
                },
                {
                    'widget_id': 'predictive_alerts',
                    'type': 'predictive_maintenance_panel',
                    'position': {'x': 0, 'y': 1, 'w': 2, 'h': 1},
                    'config': {
                        'title': 'Predictive Maintenance Alerts',
                        'priority_filter': 'high',
                        'time_horizon': '30d'
                    }
                },
                {
                    'widget_id': 'sensor_dashboard',
                    'type': 'sensor_monitoring_grid',
                    'position': {'x': 0, 'y': 2, 'w': 3, 'h': 1},
                    'config': {
                        'title': 'Real-time Sensor Data',
                        'sensor_categories': ['temperature', 'vibration', 'pressure', 'current']
                    }
                }
            ],
            'theme': {
                'primary_color': '#ff5722',
                'secondary_color': '#795548',
                'accent_color': '#607d8b'
            },
            'created_at': datetime.now().isoformat()
        }
        
        # Store default dashboards
        self.dashboards[operator_dashboard['dashboard_id']] = operator_dashboard
        self.dashboards[manager_dashboard['dashboard_id']] = manager_dashboard
        self.dashboards[technician_dashboard['dashboard_id']] = technician_dashboard
        
        self.role_defaults = {
            DashboardRole.OPERATOR: operator_dashboard['dashboard_id'],
            DashboardRole.MANAGER: manager_dashboard['dashboard_id'],
            DashboardRole.TECHNICIAN: technician_dashboard['dashboard_id']
        }
    
    def _initialize_widget_registry(self):
        """Initialize registry of available widgets."""
        self.widget_registry = {
            'status_card': {
                'type': 'status_card',
                'category': 'information',
                'data_requirements': ['status', 'value', 'timestamp'],
                'real_time': True
            },
            'gauge_chart': {
                'type': 'gauge_chart',
                'category': 'visualization',
                'data_requirements': ['current_value', 'target_value', 'min', 'max'],
                'real_time': True
            },
            '3d_equipment_view': {
                'type': '3d_equipment_view',
                'category': 'visualization',
                'data_requirements': ['equipment_status', 'positions', 'metrics'],
                'real_time': True
            },
            'alert_list': {
                'type': 'alert_list',
                'category': 'information',
                'data_requirements': ['alerts', 'priority', 'timestamp'],
                'real_time': True
            },
            'line_chart': {
                'type': 'line_chart',
                'category': 'visualization',
                'data_requirements': ['time_series_data'],
                'real_time': True
            },
            'control_panel': {
                'type': 'control_panel',
                'category': 'interaction',
                'data_requirements': ['control_states', 'permissions'],
                'real_time': True
            },
            'kpi_grid': {
                'type': 'kpi_grid',
                'category': 'information',
                'data_requirements': ['kpi_values', 'targets', 'trends'],
                'real_time': True
            },
            'ai_insights_panel': {
                'type': 'ai_insights_panel',
                'category': 'ai',
                'data_requirements': ['ai_recommendations', 'confidence', 'impact'],
                'real_time': True
            }
        }
    
    async def create_dashboard(self, dashboard_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new dashboard with specified configuration.
        
        Args:
            dashboard_config: Dashboard configuration dictionary
        
        Returns:
            Created dashboard information
        """
        start_time = time.time()
        
        try:
            dashboard_id = dashboard_config.get('dashboard_id', f"dashboard_{int(time.time() * 1000)}")
            role = dashboard_config.get('role', DashboardRole.OPERATOR)
            
            # Validate role permissions
            if not self._validate_role_permissions(role, dashboard_config):
                raise ValueError(f"Invalid permissions for role: {role}")
            
            # Create dashboard structure
            dashboard = {
                'dashboard_id': dashboard_id,
                'role': role,
                'title': dashboard_config.get('title', f'Dashboard {dashboard_id}'),
                'layout': dashboard_config.get('layout', {'type': 'grid', 'columns': 3, 'rows': 3}),
                'widgets': dashboard_config.get('widgets', []),
                'theme': dashboard_config.get('theme', self._get_default_theme(role)),
                'permissions': self.role_permissions.get(role, {}),
                'created_at': datetime.now().isoformat(),
                'last_modified': datetime.now().isoformat(),
                'status': 'active'
            }
            
            # Initialize widgets
            widget_init_results = []
            for widget_config in dashboard['widgets']:
                widget_result = await self._initialize_widget(widget_config, dashboard_id)
                widget_init_results.append(widget_result)
            
            # Store dashboard
            self.dashboards[dashboard_id] = dashboard
            
            creation_time = (time.time() - start_time) * 1000
            
            # Update metrics
            with self.lock:
                self.dashboard_metrics['dashboards_created'] += 1
                self.dashboard_metrics['active_dashboards'] = len(self.dashboards)
                self._update_dashboard_metrics(creation_time)
            
            # Performance validation
            if creation_time > self.dashboard_load_target_ms:
                self.logger.warning(f"Dashboard creation exceeded target: {creation_time:.1f}ms > {self.dashboard_load_target_ms}ms")
            
            self.logger.info(f"Created dashboard: {dashboard_id} for role {role} in {creation_time:.1f}ms")
            
            return {
                'success': True,
                'dashboard_id': dashboard_id,
                'dashboard': dashboard,
                'creation_time_ms': creation_time,
                'widget_results': widget_init_results
            }
            
        except Exception as e:
            creation_time = (time.time() - start_time) * 1000
            with self.lock:
                self.dashboard_metrics['error_count'] += 1
            self.logger.error(f"Failed to create dashboard: {e}")
            raise
    
    async def get_dashboard_for_role(self, role: str, user_preferences: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Get appropriate dashboard for user role.
        
        Args:
            role: User role (operator, manager, technician, administrator)
            user_preferences: Optional user customizations
        
        Returns:
            Dashboard configuration for the role
        """
        try:
            # Get default dashboard for role
            default_dashboard_id = self.role_defaults.get(role)
            
            if not default_dashboard_id or default_dashboard_id not in self.dashboards:
                raise ValueError(f"No default dashboard found for role: {role}")
            
            dashboard = self.dashboards[default_dashboard_id].copy()
            
            # Apply user preferences if provided
            if user_preferences:
                dashboard = await self._apply_user_preferences(dashboard, user_preferences)
            
            return {
                'success': True,
                'dashboard': dashboard,
                'role': role,
                'customized': user_preferences is not None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get dashboard for role {role}: {e}")
            raise
    
    async def update_dashboard_widget(self, dashboard_id: str, widget_id: str, 
                                    update_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update specific widget with new data.
        
        Args:
            dashboard_id: Dashboard identifier
            widget_id: Widget identifier
            update_data: New data for widget
        
        Returns:
            Update result with performance metrics
        """
        start_time = time.time()
        
        try:
            if dashboard_id not in self.dashboards:
                raise ValueError(f"Dashboard not found: {dashboard_id}")
            
            dashboard = self.dashboards[dashboard_id]
            
            # Find widget in dashboard
            widget_found = False
            for widget in dashboard['widgets']:
                if widget['widget_id'] == widget_id:
                    # Update widget data
                    if 'data' not in widget:
                        widget['data'] = {}
                    
                    widget['data'].update(update_data)
                    widget['last_update'] = datetime.now().isoformat()
                    widget_found = True
                    break
            
            if not widget_found:
                raise ValueError(f"Widget not found: {widget_id}")
            
            update_time = (time.time() - start_time) * 1000
            
            # Update metrics
            with self.lock:
                self.dashboard_metrics['widget_updates'] += 1
            
            # Performance validation
            if update_time > self.widget_update_target_ms:
                self.logger.warning(f"Widget update exceeded target: {update_time:.1f}ms > {self.widget_update_target_ms}ms")
            
            return {
                'success': True,
                'dashboard_id': dashboard_id,
                'widget_id': widget_id,
                'update_time_ms': update_time,
                'data_size': len(str(update_data))
            }
            
        except Exception as e:
            update_time = (time.time() - start_time) * 1000
            with self.lock:
                self.dashboard_metrics['error_count'] += 1
            self.logger.error(f"Failed to update widget: {e}")
            raise
    
    def _validate_role_permissions(self, role: str, dashboard_config: Dict) -> bool:
        """Validate that dashboard configuration matches role permissions."""
        try:
            if role not in self.role_permissions:
                return False
            
            permissions = self.role_permissions[role]
            
            # Check widget permissions
            allowed_widgets = permissions.get('widgets', [])
            if allowed_widgets != '*':
                for widget in dashboard_config.get('widgets', []):
                    widget_type = widget.get('type', '')
                    if widget_type not in allowed_widgets:
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Permission validation failed: {e}")
            return False
    
    async def _initialize_widget(self, widget_config: Dict, dashboard_id: str) -> Dict[str, Any]:
        """Initialize individual widget."""
        try:
            widget_id = widget_config['widget_id']
            widget_type = widget_config['type']
            
            # Validate widget type
            if widget_type not in self.widget_registry:
                raise ValueError(f"Unknown widget type: {widget_type}")
            
            widget_info = self.widget_registry[widget_type]
            
            # Initialize widget data structure
            initialized_widget = {
                'widget_id': widget_id,
                'type': widget_type,
                'category': widget_info['category'],
                'status': 'initialized',
                'data': {},
                'last_update': None,
                'error_count': 0
            }
            
            return {
                'success': True,
                'widget_id': widget_id,
                'widget': initialized_widget
            }
            
        except Exception as e:
            return {
                'success': False,
                'widget_id': widget_config.get('widget_id', 'unknown'),
                'error': str(e)
            }
    
    async def _apply_user_preferences(self, dashboard: Dict, preferences: Dict) -> Dict:
        """Apply user customizations to dashboard."""
        try:
            customized_dashboard = dashboard.copy()
            
            # Apply theme preferences
            if 'theme' in preferences:
                customized_dashboard['theme'].update(preferences['theme'])
            
            # Apply layout preferences
            if 'layout' in preferences:
                customized_dashboard['layout'].update(preferences['layout'])
            
            # Apply widget preferences
            if 'widget_preferences' in preferences:
                for widget in customized_dashboard['widgets']:
                    widget_id = widget['widget_id']
                    if widget_id in preferences['widget_preferences']:
                        widget['config'].update(preferences['widget_preferences'][widget_id])
            
            return customized_dashboard
            
        except Exception as e:
            self.logger.error(f"Failed to apply user preferences: {e}")
            return dashboard
    
    def _get_default_theme(self, role: str) -> Dict[str, str]:
        """Get default theme colors for role."""
        themes = {
            DashboardRole.OPERATOR: {
                'primary_color': '#1976d2',
                'secondary_color': '#42a5f5',
                'alert_color': '#f44336',
                'success_color': '#4caf50',
                'warning_color': '#ff9800',
                'background_color': '#fafafa'
            },
            DashboardRole.MANAGER: {
                'primary_color': '#673ab7',
                'secondary_color': '#9c27b0',
                'accent_color': '#e1bee7',
                'background_color': '#f3e5f5',
                'text_color': '#4a148c'
            },
            DashboardRole.TECHNICIAN: {
                'primary_color': '#ff5722',
                'secondary_color': '#795548',
                'accent_color': '#607d8b',
                'background_color': '#fafafa',
                'border_color': '#bdbdbd'
            }
        }
        
        return themes.get(role, themes[DashboardRole.OPERATOR])
    
    def _update_dashboard_metrics(self, creation_time: float):
        """Update dashboard performance metrics."""
        with self.lock:
            current_avg = self.dashboard_metrics['avg_load_time_ms']
            count = self.dashboard_metrics['dashboards_created']
            self.dashboard_metrics['avg_load_time_ms'] = (
                (current_avg * (count - 1) + creation_time) / count
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get dashboard manager performance metrics."""
        with self.lock:
            return {
                'dashboard_manager_metrics': self.dashboard_metrics.copy(),
                'performance_targets': {
                    'dashboard_load_target_ms': self.dashboard_load_target_ms,
                    'widget_update_target_ms': self.widget_update_target_ms,
                    'layout_change_target_ms': self.layout_change_target_ms
                },
                'active_dashboards': len(self.dashboards),
                'registered_widgets': len(self.widget_registry),
                'timestamp': datetime.now().isoformat()
            }
    
    async def validate_dashboard_manager(self) -> Dict[str, Any]:
        """Validate dashboard manager functionality and performance."""
        validation_results = {
            'engine_name': 'DashboardManager',
            'validation_timestamp': datetime.now().isoformat(),
            'tests': {},
            'performance_metrics': {},
            'overall_status': 'unknown'
        }
        
        try:
            # Test 1: Dashboard Creation
            test_dashboard_config = {
                'dashboard_id': 'validation_dashboard',
                'role': DashboardRole.OPERATOR,
                'title': 'Validation Dashboard',
                'widgets': [
                    {
                        'widget_id': 'test_widget',
                        'type': 'production_status',
                        'position': {'x': 0, 'y': 0, 'w': 1, 'h': 1}
                    }
                ]
            }
            
            dashboard_result = await self.create_dashboard(test_dashboard_config)
            
            validation_results['tests']['dashboard_creation'] = {
                'status': 'pass' if dashboard_result['success'] else 'fail',
                'creation_time_ms': dashboard_result.get('creation_time_ms', 0),
                'target_ms': self.dashboard_load_target_ms,
                'details': f"Created dashboard: {dashboard_result.get('dashboard_id', 'unknown')}"
            }
            
            # Test 2: Role-based Dashboard Access
            role_result = await self.get_dashboard_for_role(DashboardRole.MANAGER)
            
            validation_results['tests']['role_based_access'] = {
                'status': 'pass' if role_result['success'] else 'fail',
                'role': role_result.get('role', 'unknown'),
                'details': f"Retrieved dashboard for role: {role_result.get('role', 'unknown')}"
            }
            
            # Test 3: Widget Update Performance
            if dashboard_result['success']:
                update_result = await self.update_dashboard_widget(
                    'validation_dashboard', 
                    'test_widget',
                    {'status': 'active', 'value': 42}
                )
                
                validation_results['tests']['widget_update'] = {
                    'status': 'pass' if update_result['success'] else 'fail',
                    'update_time_ms': update_result.get('update_time_ms', 0),
                    'target_ms': self.widget_update_target_ms,
                    'details': f"Updated widget in {update_result.get('update_time_ms', 0):.1f}ms"
                }
            
            # Performance metrics
            validation_results['performance_metrics'] = self.get_performance_metrics()
            
            # Overall status
            passed_tests = sum(1 for test in validation_results['tests'].values() 
                             if test['status'] == 'pass')
            total_tests = len(validation_results['tests'])
            
            validation_results['overall_status'] = 'pass' if passed_tests == total_tests else 'fail'
            validation_results['test_summary'] = f"{passed_tests}/{total_tests} tests passed"
            
            self.logger.info(f"Dashboard manager validation completed: {validation_results['test_summary']}")
            
        except Exception as e:
            validation_results['overall_status'] = 'error'
            validation_results['error'] = str(e)
            self.logger.error(f"Dashboard manager validation failed: {e}")
        
        return validation_results
    
    def shutdown(self):
        """Shutdown dashboard manager and cleanup resources."""
        try:
            # Clear all dashboards and sessions
            self.dashboards.clear()
            self.active_sessions.clear()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("Dashboard manager shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during dashboard manager shutdown: {e}")


# Integration functions for dashboard management
async def integrate_with_visualization_engine(dashboard_manager: DashboardManager, 
                                            visualization_engine) -> Dict[str, Any]:
    """Integrate dashboard manager with visualization engine."""
    try:
        return {
            'integration_type': 'dashboard_visualization',
            'status': 'connected',
            'capabilities': [
                'real_time_chart_integration',
                'widget_visualization_updates',
                'dashboard_chart_management'
            ]
        }
    except Exception as e:
        return {
            'integration_type': 'dashboard_visualization',
            'status': 'error',
            'error': str(e)
        }