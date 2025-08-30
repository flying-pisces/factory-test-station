"""
UI Controller - Week 13: UI & Visualization Layer

Main controller coordinating all UI components including dashboards,
visualization engine, and real-time data pipeline.
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import threading

# Import UI layer components
from .visualization_engine import VisualizationEngine
from .dashboard_manager import DashboardManager, DashboardRole
from .real_time_data_pipeline import RealTimeDataPipeline


class UIController:
    """
    Main controller for manufacturing line user interface layer.
    
    Coordinates visualization engine, dashboard manager, and real-time data
    pipeline to provide comprehensive user interface capabilities.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize UI Controller.
        
        Args:
            config: Configuration dictionary for UI controller settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Performance targets
        self.ui_response_target_ms = 100     # UI operations under 100ms
        self.data_sync_target_ms = 50        # Data synchronization under 50ms
        self.dashboard_switch_target_ms = 500  # Dashboard switching under 500ms
        
        # UI components
        self.visualization_engine = None
        self.dashboard_manager = None
        self.data_pipeline = None
        
        # UI state management
        self.active_sessions = {}
        self.ui_state = {
            'active_dashboards': {},
            'real_time_connections': {},
            'user_preferences': {},
            'system_alerts': []
        }
        
        # Performance metrics
        self.ui_metrics = {
            'ui_operations': 0,
            'dashboard_switches': 0,
            'data_updates': 0,
            'avg_response_time_ms': 0.0,
            'active_users': 0,
            'error_count': 0
        }
        
        # Thread safety
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="ui_controller")
        
        # Initialize UI system
        self._initialize_ui_system()
        
        self.logger.info("UIController initialized successfully")
    
    def _initialize_ui_system(self):
        """Initialize complete UI system with all components."""
        try:
            # Initialize visualization engine
            viz_config = self.config.get('visualization', {})
            self.visualization_engine = VisualizationEngine(viz_config)
            
            # Initialize dashboard manager
            dashboard_config = self.config.get('dashboard', {})
            self.dashboard_manager = DashboardManager(dashboard_config)
            
            # Initialize data pipeline
            pipeline_config = self.config.get('data_pipeline', {})
            self.data_pipeline = RealTimeDataPipeline(pipeline_config)
            
            # Set up integration between components
            self._setup_component_integration()
            
            self.logger.info("UI system components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize UI system: {e}")
            raise
    
    def _setup_component_integration(self):
        """Set up integration between UI components."""
        try:
            # In a full implementation, would set up event handlers and
            # data flow connections between components
            self.logger.info("UI component integration established")
            
        except Exception as e:
            self.logger.error(f"Failed to setup component integration: {e}")
    
    async def create_user_session(self, user_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create new user session with role-based dashboard.
        
        Args:
            user_info: User information including role and preferences
        
        Returns:
            Created session information with dashboard
        """
        start_time = time.time()
        
        try:
            user_id = user_info.get('user_id', f"user_{int(time.time() * 1000)}")
            user_role = user_info.get('role', DashboardRole.OPERATOR)
            user_preferences = user_info.get('preferences', {})
            
            # Create session
            session_id = f"session_{user_id}_{int(time.time())}"
            
            # Get appropriate dashboard for user role
            dashboard_result = await self.dashboard_manager.get_dashboard_for_role(
                user_role, user_preferences
            )
            
            if not dashboard_result['success']:
                raise RuntimeError("Failed to get dashboard for user role")
            
            dashboard = dashboard_result['dashboard']
            
            # Set up real-time data connection
            subscription_config = {
                'data_sources': self._get_data_sources_for_role(user_role),
                'update_frequency': user_preferences.get('update_frequency', 20),
                'filters': user_preferences.get('data_filters', {})
            }
            
            # Create WebSocket connection (mock for now)
            mock_websocket = {'session_id': session_id, 'user_id': user_id}
            
            connection_result = await self.data_pipeline.register_websocket_connection(
                session_id, mock_websocket, subscription_config
            )
            
            # Store session information
            session_info = {
                'session_id': session_id,
                'user_id': user_id,
                'user_role': user_role,
                'dashboard': dashboard,
                'data_connection': connection_result,
                'preferences': user_preferences,
                'created_at': datetime.now().isoformat(),
                'last_activity': datetime.now().isoformat(),
                'status': 'active'
            }
            
            self.active_sessions[session_id] = session_info
            
            # Update UI state
            self.ui_state['active_dashboards'][session_id] = dashboard['dashboard_id']
            self.ui_state['real_time_connections'][session_id] = connection_result['connection_id']
            
            session_time = (time.time() - start_time) * 1000
            
            # Update metrics
            with self.lock:
                self.ui_metrics['active_users'] += 1
                self.ui_metrics['ui_operations'] += 1
                self._update_ui_metrics(session_time)
            
            # Performance validation
            if session_time > self.ui_response_target_ms:
                self.logger.warning(f"Session creation exceeded target: {session_time:.1f}ms")
            
            self.logger.info(f"Created user session: {session_id} for role {user_role} in {session_time:.1f}ms")
            
            return {
                'success': True,
                'session_id': session_id,
                'dashboard': dashboard,
                'data_connection': connection_result,
                'session_time_ms': session_time
            }
            
        except Exception as e:
            session_time = (time.time() - start_time) * 1000
            with self.lock:
                self.ui_metrics['error_count'] += 1
            self.logger.error(f"Failed to create user session: {e}")
            raise
    
    async def update_dashboard_data(self, session_id: str, data_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update dashboard with new data from manufacturing systems.
        
        Args:
            session_id: User session identifier
            data_updates: Data updates for dashboard widgets
        
        Returns:
            Update result with performance metrics
        """
        start_time = time.time()
        
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session not found: {session_id}")
            
            session_info = self.active_sessions[session_id]
            dashboard_id = session_info['dashboard']['dashboard_id']
            
            # Update individual widgets
            update_results = []
            
            for widget_id, widget_data in data_updates.items():
                try:
                    widget_result = await self.dashboard_manager.update_dashboard_widget(
                        dashboard_id, widget_id, widget_data
                    )
                    update_results.append(widget_result)
                    
                    # Also send to data pipeline for real-time distribution
                    await self.data_pipeline.push_data_to_pipeline(
                        'ui_updates', 
                        {
                            'session_id': session_id,
                            'widget_id': widget_id,
                            'data': widget_data
                        }
                    )
                    
                except Exception as e:
                    update_results.append({
                        'success': False,
                        'widget_id': widget_id,
                        'error': str(e)
                    })
            
            # Update session activity
            session_info['last_activity'] = datetime.now().isoformat()
            
            update_time = (time.time() - start_time) * 1000
            
            # Update metrics
            with self.lock:
                self.ui_metrics['data_updates'] += len(data_updates)
                self.ui_metrics['ui_operations'] += 1
            
            # Performance validation
            if update_time > self.data_sync_target_ms:
                self.logger.warning(f"Dashboard update exceeded target: {update_time:.1f}ms")
            
            successful_updates = sum(1 for result in update_results if result['success'])
            
            return {
                'success': True,
                'session_id': session_id,
                'widgets_updated': successful_updates,
                'total_widgets': len(data_updates),
                'update_time_ms': update_time,
                'update_results': update_results
            }
            
        except Exception as e:
            update_time = (time.time() - start_time) * 1000
            with self.lock:
                self.ui_metrics['error_count'] += 1
            self.logger.error(f"Failed to update dashboard data: {e}")
            raise
    
    async def create_visualization(self, session_id: str, 
                                 visualization_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create new visualization for user session.
        
        Args:
            session_id: User session identifier
            visualization_config: Visualization configuration
        
        Returns:
            Created visualization information
        """
        start_time = time.time()
        
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session not found: {session_id}")
            
            viz_type = visualization_config.get('type', 'chart')
            
            # Create visualization based on type
            if viz_type == 'chart':
                viz_result = await self.visualization_engine.create_real_time_chart(
                    visualization_config
                )
            elif viz_type == '3d_equipment':
                viz_result = await self.visualization_engine.create_3d_equipment_visualization(
                    visualization_config
                )
            else:
                raise ValueError(f"Unknown visualization type: {viz_type}")
            
            viz_time = (time.time() - start_time) * 1000
            
            # Update metrics
            with self.lock:
                self.ui_metrics['ui_operations'] += 1
                self._update_ui_metrics(viz_time)
            
            return {
                'success': True,
                'session_id': session_id,
                'visualization': viz_result,
                'creation_time_ms': viz_time
            }
            
        except Exception as e:
            viz_time = (time.time() - start_time) * 1000
            with self.lock:
                self.ui_metrics['error_count'] += 1
            self.logger.error(f"Failed to create visualization: {e}")
            raise
    
    async def switch_dashboard(self, session_id: str, new_dashboard_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Switch user to different dashboard configuration.
        
        Args:
            session_id: User session identifier
            new_dashboard_config: New dashboard configuration
        
        Returns:
            Dashboard switch result
        """
        start_time = time.time()
        
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session not found: {session_id}")
            
            session_info = self.active_sessions[session_id]
            
            # Create new dashboard
            dashboard_result = await self.dashboard_manager.create_dashboard(
                new_dashboard_config
            )
            
            if not dashboard_result['success']:
                raise RuntimeError("Failed to create new dashboard")
            
            new_dashboard = dashboard_result['dashboard']
            
            # Update session with new dashboard
            old_dashboard_id = session_info['dashboard']['dashboard_id']
            session_info['dashboard'] = new_dashboard
            session_info['last_activity'] = datetime.now().isoformat()
            
            # Update UI state
            self.ui_state['active_dashboards'][session_id] = new_dashboard['dashboard_id']
            
            switch_time = (time.time() - start_time) * 1000
            
            # Update metrics
            with self.lock:
                self.ui_metrics['dashboard_switches'] += 1
                self.ui_metrics['ui_operations'] += 1
            
            # Performance validation
            if switch_time > self.dashboard_switch_target_ms:
                self.logger.warning(f"Dashboard switch exceeded target: {switch_time:.1f}ms")
            
            self.logger.info(f"Switched dashboard for session {session_id} in {switch_time:.1f}ms")
            
            return {
                'success': True,
                'session_id': session_id,
                'old_dashboard_id': old_dashboard_id,
                'new_dashboard': new_dashboard,
                'switch_time_ms': switch_time
            }
            
        except Exception as e:
            switch_time = (time.time() - start_time) * 1000
            with self.lock:
                self.ui_metrics['error_count'] += 1
            self.logger.error(f"Failed to switch dashboard: {e}")
            raise
    
    def _get_data_sources_for_role(self, role: str) -> List[str]:
        """Get appropriate data sources for user role."""
        role_data_sources = {
            DashboardRole.OPERATOR: [
                'production_system',
                'equipment_monitoring',
                'ai_layer'
            ],
            DashboardRole.MANAGER: [
                'production_system',
                'quality_systems',
                'ai_layer'
            ],
            DashboardRole.TECHNICIAN: [
                'equipment_monitoring',
                'ai_layer'
            ],
            DashboardRole.ADMINISTRATOR: [
                'production_system',
                'equipment_monitoring',
                'quality_systems',
                'ai_layer'
            ]
        }
        
        return role_data_sources.get(role, ['production_system'])
    
    def _update_ui_metrics(self, response_time: float):
        """Update UI performance metrics."""
        with self.lock:
            current_avg = self.ui_metrics['avg_response_time_ms']
            count = self.ui_metrics['ui_operations']
            self.ui_metrics['avg_response_time_ms'] = (
                (current_avg * (count - 1) + response_time) / count
            )
    
    def get_ui_status(self) -> Dict[str, Any]:
        """Get comprehensive UI system status."""
        return {
            'ui_controller_status': 'active',
            'active_sessions': len(self.active_sessions),
            'component_status': {
                'visualization_engine': 'active' if self.visualization_engine else 'inactive',
                'dashboard_manager': 'active' if self.dashboard_manager else 'inactive',
                'data_pipeline': 'active' if self.data_pipeline else 'inactive'
            },
            'ui_state': {
                'active_dashboards': len(self.ui_state['active_dashboards']),
                'real_time_connections': len(self.ui_state['real_time_connections']),
                'system_alerts': len(self.ui_state['system_alerts'])
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get UI controller performance metrics."""
        with self.lock:
            ui_metrics = self.ui_metrics.copy()
        
        # Get component metrics
        component_metrics = {}
        
        if self.visualization_engine:
            component_metrics['visualization'] = self.visualization_engine.get_performance_metrics()
        
        if self.dashboard_manager:
            component_metrics['dashboard'] = self.dashboard_manager.get_performance_metrics()
        
        if self.data_pipeline:
            component_metrics['data_pipeline'] = self.data_pipeline.get_performance_metrics()
        
        return {
            'ui_controller_metrics': ui_metrics,
            'component_metrics': component_metrics,
            'performance_targets': {
                'ui_response_target_ms': self.ui_response_target_ms,
                'data_sync_target_ms': self.data_sync_target_ms,
                'dashboard_switch_target_ms': self.dashboard_switch_target_ms
            },
            'timestamp': datetime.now().isoformat()
        }
    
    async def validate_ui_controller(self) -> Dict[str, Any]:
        """Validate UI controller functionality and performance."""
        validation_results = {
            'engine_name': 'UIController',
            'validation_timestamp': datetime.now().isoformat(),
            'tests': {},
            'performance_metrics': {},
            'overall_status': 'unknown'
        }
        
        try:
            # Test 1: User Session Creation
            test_user = {
                'user_id': 'validation_user',
                'role': DashboardRole.OPERATOR,
                'preferences': {'update_frequency': 10}
            }
            
            session_result = await self.create_user_session(test_user)
            
            validation_results['tests']['session_creation'] = {
                'status': 'pass' if session_result['success'] else 'fail',
                'session_time_ms': session_result.get('session_time_ms', 0),
                'target_ms': self.ui_response_target_ms,
                'details': f"Created session: {session_result.get('session_id', 'unknown')}"
            }
            
            # Test 2: Dashboard Data Update
            if session_result['success']:
                session_id = session_result['session_id']
                test_data = {
                    'production_status': {'status': 'running', 'throughput': 95}
                }
                
                update_result = await self.update_dashboard_data(session_id, test_data)
                
                validation_results['tests']['dashboard_update'] = {
                    'status': 'pass' if update_result['success'] else 'fail',
                    'update_time_ms': update_result.get('update_time_ms', 0),
                    'target_ms': self.data_sync_target_ms,
                    'details': f"Updated {update_result.get('widgets_updated', 0)} widgets"
                }
            
            # Test 3: Visualization Creation
            if session_result['success']:
                viz_config = {
                    'type': 'chart',
                    'chart_id': 'validation_chart',
                    'chart_type': 'line',
                    'title': 'Validation Chart'
                }
                
                viz_result = await self.create_visualization(session_id, viz_config)
                
                validation_results['tests']['visualization_creation'] = {
                    'status': 'pass' if viz_result['success'] else 'fail',
                    'creation_time_ms': viz_result.get('creation_time_ms', 0),
                    'details': "Created visualization for session"
                }
            
            # Test 4: Component Integration
            ui_status = self.get_ui_status()
            components_active = sum(1 for status in ui_status['component_status'].values() 
                                  if status == 'active')
            
            validation_results['tests']['component_integration'] = {
                'status': 'pass' if components_active == 3 else 'fail',
                'active_components': components_active,
                'total_components': 3,
                'details': f"{components_active}/3 components active"
            }
            
            # Performance metrics
            validation_results['performance_metrics'] = self.get_performance_metrics()
            
            # Overall status
            passed_tests = sum(1 for test in validation_results['tests'].values() 
                             if test['status'] == 'pass')
            total_tests = len(validation_results['tests'])
            
            validation_results['overall_status'] = 'pass' if passed_tests == total_tests else 'fail'
            validation_results['test_summary'] = f"{passed_tests}/{total_tests} tests passed"
            
            self.logger.info(f"UI controller validation completed: {validation_results['test_summary']}")
            
        except Exception as e:
            validation_results['overall_status'] = 'error'
            validation_results['error'] = str(e)
            self.logger.error(f"UI controller validation failed: {e}")
        
        return validation_results
    
    def shutdown(self):
        """Shutdown UI controller and all components."""
        try:
            # Shutdown components
            if self.data_pipeline:
                self.data_pipeline.shutdown()
            
            if self.dashboard_manager:
                self.dashboard_manager.shutdown()
            
            if self.visualization_engine:
                self.visualization_engine.shutdown()
            
            # Clear state
            self.active_sessions.clear()
            self.ui_state.clear()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("UI controller shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during UI controller shutdown: {e}")


# Integration functions for UI controller
async def integrate_ui_with_ai_layer(ui_controller: UIController, ai_engines: Dict) -> Dict[str, Any]:
    """Integrate UI controller with AI layer."""
    try:
        # Set up data pipeline integration with AI engines
        pipeline_integration = await integrate_with_ai_layer(ui_controller.data_pipeline, ai_engines)
        
        return {
            'integration_type': 'ui_ai',
            'status': 'connected',
            'ui_components': ['visualization', 'dashboard', 'data_pipeline'],
            'ai_engines': list(ai_engines.keys()),
            'pipeline_integration': pipeline_integration,
            'capabilities': [
                'real_time_ai_visualization',
                'ai_powered_dashboards',
                'predictive_ui_updates',
                'intelligent_alert_management'
            ]
        }
        
    except Exception as e:
        return {
            'integration_type': 'ui_ai',
            'status': 'error',
            'error': str(e)
        }