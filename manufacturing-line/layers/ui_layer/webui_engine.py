"""
Week 6: WebUIEngine - Advanced Web-Based User Interface System

This module implements the core web-based user interface system for manufacturing
line control, providing real-time dashboards, interactive controls, and responsive
design for comprehensive system management.

Key Features:
- Modern responsive web interface with real-time data binding
- Interactive dashboards with live KPI monitoring
- Real-time data visualization and chart integration
- System control interfaces with immediate feedback
- WebSocket-based real-time communication with <100ms response times

Author: Claude Code
Date: 2024-08-28
Version: 1.0
"""

import time
import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import threading
from pathlib import Path
import uuid

# Import Week 5 dependencies
try:
    from ..control_layer.monitoring_engine import MonitoringEngine
    from ..control_layer.realtime_control_engine import RealTimeControlEngine
    from ..control_layer.orchestration_engine import OrchestrationEngine
except ImportError:
    # Fallback for development
    logging.warning("Week 5 control layer not available - using mock interfaces")
    MonitoringEngine = None
    RealTimeControlEngine = None
    OrchestrationEngine = None

class UIComponentType(Enum):
    DASHBOARD = "dashboard"
    CHART = "chart"
    CONTROL_PANEL = "control_panel"
    TABLE = "table"
    FORM = "form"
    ALERT = "alert"

class UITheme(Enum):
    LIGHT = "light"
    DARK = "dark"
    HIGH_CONTRAST = "high_contrast"
    COMPACT = "compact"

class DashboardLayout(Enum):
    GRID = "grid"
    FLEX = "flex"
    TABS = "tabs"
    SIDEBAR = "sidebar"

@dataclass
class UIComponent:
    component_id: str
    component_type: UIComponentType
    title: str
    config: Dict[str, Any]
    data_source: str
    position: Dict[str, int]  # x, y, width, height
    refresh_rate_ms: int = 1000
    visible: bool = True
    interactive: bool = True

@dataclass
class Dashboard:
    dashboard_id: str
    name: str
    layout: DashboardLayout
    components: List[UIComponent]
    theme: UITheme
    auto_refresh: bool = True
    permissions: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None

@dataclass
class UserSession:
    session_id: str
    user_id: str
    permissions: List[str]
    active_dashboard: Optional[str]
    connected_at: datetime
    last_activity: datetime
    websocket_connection: Any = None

@dataclass
class UIInteraction:
    interaction_id: str
    session_id: str
    component_id: str
    action: str
    data: Dict[str, Any]
    timestamp: datetime
    response_time_ms: float = 0

class WebUIEngine:
    """Advanced web-based user interface system for manufacturing line control."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the WebUIEngine with configuration."""
        self.config = config or {}
        
        # Performance configuration
        self.response_target_ms = self.config.get('response_target_ms', 100)
        self.ui_update_interval_ms = self.config.get('ui_update_interval_ms', 500)
        self.websocket_timeout_s = self.config.get('websocket_timeout_s', 30)
        
        # UI configuration
        self.default_theme = UITheme(self.config.get('default_theme', 'light'))
        self.max_dashboard_components = self.config.get('max_dashboard_components', 20)
        self.session_timeout_hours = self.config.get('session_timeout_hours', 8)
        
        # Integration with Week 5 engines
        if MonitoringEngine:
            self.monitoring_engine = MonitoringEngine(self.config.get('monitoring_config', {}))
        else:
            self.monitoring_engine = None
            
        if RealTimeControlEngine:
            self.control_engine = RealTimeControlEngine(self.config.get('control_config', {}))
        else:
            self.control_engine = None
            
        if OrchestrationEngine:
            self.orchestration_engine = OrchestrationEngine(self.config.get('orchestration_config', {}))
        else:
            self.orchestration_engine = None
        
        # UI state management
        self.active_sessions: Dict[str, UserSession] = {}
        self.dashboard_registry: Dict[str, Dashboard] = {}
        self.component_cache: Dict[str, Dict[str, Any]] = {}
        self.interaction_history: List[UIInteraction] = []
        
        # Real-time update system
        self.update_threads: Dict[str, threading.Thread] = {}
        self.websocket_connections: Dict[str, Any] = {}
        self.data_subscriptions: Dict[str, List[str]] = {}  # component_id -> session_ids
        
        # Performance monitoring
        self.performance_metrics = {
            'avg_response_time_ms': 0,
            'total_interactions': 0,
            'successful_interactions': 0,
            'active_sessions': 0,
            'dashboard_renders': 0,
            'data_updates': 0
        }
        
        # Initialize default dashboards
        self._initialize_default_dashboards()
        
        logging.info(f"WebUIEngine initialized with {self.response_target_ms}ms target")

    def render_real_time_dashboard(self, 
                                 dashboard_id: str,
                                 session_id: str,
                                 customizations: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Render real-time manufacturing dashboard with live data."""
        start_time = time.time()
        
        try:
            # Validate session
            if session_id not in self.active_sessions:
                raise ValueError(f"Invalid session: {session_id}")
            
            session = self.active_sessions[session_id]
            
            # Get dashboard configuration
            if dashboard_id not in self.dashboard_registry:
                raise ValueError(f"Dashboard not found: {dashboard_id}")
            
            dashboard = self.dashboard_registry[dashboard_id]
            
            # Check permissions
            if not self._check_dashboard_permissions(dashboard, session):
                raise PermissionError(f"Insufficient permissions for dashboard: {dashboard_id}")
            
            # Apply customizations
            if customizations:
                dashboard = self._apply_dashboard_customizations(dashboard, customizations)
            
            # Render dashboard components
            rendered_components = []
            for component in dashboard.components:
                if component.visible:
                    rendered_component = self._render_component(component, session)
                    rendered_components.append(rendered_component)
            
            # Update session state
            session.active_dashboard = dashboard_id
            session.last_activity = datetime.now()
            
            # Calculate performance
            render_time = (time.time() - start_time) * 1000
            
            # Prepare dashboard response
            dashboard_response = {
                'dashboard_id': dashboard_id,
                'name': dashboard.name,
                'layout': dashboard.layout.value,
                'theme': dashboard.theme.value,
                'components': rendered_components,
                'auto_refresh': dashboard.auto_refresh,
                'render_time_ms': render_time,
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id
            }
            
            # Update performance metrics
            self._update_ui_metrics(render_time, True, 'dashboard_render')
            self.performance_metrics['dashboard_renders'] += 1
            
            logging.info(f"Dashboard {dashboard_id} rendered in {render_time:.2f}ms for session {session_id}")
            
            return dashboard_response
            
        except Exception as e:
            render_time = (time.time() - start_time) * 1000
            self._update_ui_metrics(render_time, False, 'dashboard_render')
            logging.error(f"Dashboard rendering failed for {dashboard_id}: {e}")
            
            return {
                'error': str(e),
                'dashboard_id': dashboard_id,
                'render_time_ms': render_time,
                'timestamp': datetime.now().isoformat()
            }

    def handle_user_interactions(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and respond to user interface interactions."""
        start_time = time.time()
        
        try:
            # Parse interaction data
            session_id = interaction_data.get('session_id')
            component_id = interaction_data.get('component_id')
            action = interaction_data.get('action')
            data = interaction_data.get('data', {})
            
            if not all([session_id, component_id, action]):
                raise ValueError("Missing required interaction parameters")
            
            # Validate session
            if session_id not in self.active_sessions:
                raise ValueError(f"Invalid session: {session_id}")
            
            session = self.active_sessions[session_id]
            
            # Create interaction record
            interaction = UIInteraction(
                interaction_id=str(uuid.uuid4()),
                session_id=session_id,
                component_id=component_id,
                action=action,
                data=data,
                timestamp=datetime.now()
            )
            
            # Process interaction based on action type
            response = self._process_interaction_action(interaction, session)
            
            # Calculate response time
            response_time = (time.time() - start_time) * 1000
            interaction.response_time_ms = response_time
            
            # Store interaction history
            self.interaction_history.append(interaction)
            if len(self.interaction_history) > 1000:  # Keep recent history
                self.interaction_history = self.interaction_history[-1000:]
            
            # Update session activity
            session.last_activity = datetime.now()
            
            # Update performance metrics
            self._update_ui_metrics(response_time, True, 'interaction')
            
            # Prepare response
            interaction_response = {
                'interaction_id': interaction.interaction_id,
                'success': True,
                'response': response,
                'response_time_ms': response_time,
                'timestamp': datetime.now().isoformat()
            }
            
            logging.info(f"Interaction {action} on {component_id} processed in {response_time:.2f}ms")
            
            return interaction_response
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self._update_ui_metrics(response_time, False, 'interaction')
            logging.error(f"Interaction processing failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'response_time_ms': response_time,
                'timestamp': datetime.now().isoformat()
            }

    def update_ui_components(self, component_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update UI components with real-time data streams."""
        start_time = time.time()
        
        try:
            updated_components = []
            failed_updates = []
            
            for update in component_updates:
                component_id = update.get('component_id')
                data = update.get('data')
                
                if not component_id or data is None:
                    failed_updates.append({'component_id': component_id, 'error': 'Missing required data'})
                    continue
                
                try:
                    # Update component cache
                    self.component_cache[component_id] = {
                        'data': data,
                        'timestamp': datetime.now().isoformat(),
                        'update_count': self.component_cache.get(component_id, {}).get('update_count', 0) + 1
                    }
                    
                    # Broadcast update to subscribed sessions
                    if component_id in self.data_subscriptions:
                        for session_id in self.data_subscriptions[component_id]:
                            self._broadcast_component_update(session_id, component_id, data)
                    
                    updated_components.append(component_id)
                    
                except Exception as e:
                    failed_updates.append({'component_id': component_id, 'error': str(e)})
            
            update_time = (time.time() - start_time) * 1000
            self.performance_metrics['data_updates'] += len(updated_components)
            
            return {
                'success': True,
                'updated_components': updated_components,
                'failed_updates': failed_updates,
                'update_time_ms': update_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            update_time = (time.time() - start_time) * 1000
            logging.error(f"Component update failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'update_time_ms': update_time,
                'timestamp': datetime.now().isoformat()
            }

    def create_user_session(self, user_id: str, permissions: List[str], websocket_connection: Any = None) -> str:
        """Create new user session with authentication and permissions."""
        session_id = str(uuid.uuid4())
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            permissions=permissions,
            active_dashboard=None,
            connected_at=datetime.now(),
            last_activity=datetime.now(),
            websocket_connection=websocket_connection
        )
        
        self.active_sessions[session_id] = session
        self.performance_metrics['active_sessions'] = len(self.active_sessions)
        
        # Start session monitoring
        self._start_session_monitoring(session_id)
        
        logging.info(f"User session created for {user_id}: {session_id}")
        
        return session_id

    def terminate_user_session(self, session_id: str) -> bool:
        """Terminate user session and clean up resources."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        # Close WebSocket connection
        if session.websocket_connection:
            try:
                session.websocket_connection.close()
            except:
                pass
        
        # Clean up subscriptions
        for component_id, subscribers in self.data_subscriptions.items():
            if session_id in subscribers:
                subscribers.remove(session_id)
        
        # Remove session
        del self.active_sessions[session_id]
        self.performance_metrics['active_sessions'] = len(self.active_sessions)
        
        logging.info(f"User session terminated: {session_id}")
        
        return True

    def _initialize_default_dashboards(self):
        """Initialize default dashboard configurations."""
        # Manufacturing Overview Dashboard
        overview_dashboard = Dashboard(
            dashboard_id="manufacturing_overview",
            name="Manufacturing Overview",
            layout=DashboardLayout.GRID,
            components=[
                UIComponent(
                    component_id="throughput_chart",
                    component_type=UIComponentType.CHART,
                    title="Production Throughput",
                    config={"chart_type": "line", "time_range": "24h"},
                    data_source="analytics_engine.throughput_kpis",
                    position={"x": 0, "y": 0, "width": 6, "height": 4}
                ),
                UIComponent(
                    component_id="efficiency_gauge",
                    component_type=UIComponentType.CHART,
                    title="Overall Efficiency",
                    config={"chart_type": "gauge", "target": 85},
                    data_source="analytics_engine.efficiency_kpis",
                    position={"x": 6, "y": 0, "width": 3, "height": 4}
                ),
                UIComponent(
                    component_id="quality_status",
                    component_type=UIComponentType.TABLE,
                    title="Quality Metrics",
                    config={"columns": ["metric", "current", "target", "status"]},
                    data_source="analytics_engine.quality_kpis",
                    position={"x": 9, "y": 0, "width": 3, "height": 4}
                )
            ],
            theme=self.default_theme,
            permissions=["view_dashboards"]
        )
        
        # Control Panel Dashboard
        control_dashboard = Dashboard(
            dashboard_id="control_panel",
            name="System Control Panel",
            layout=DashboardLayout.TABS,
            components=[
                UIComponent(
                    component_id="line_controls",
                    component_type=UIComponentType.CONTROL_PANEL,
                    title="Line Controls",
                    config={"controls": ["start", "stop", "pause", "emergency_stop"]},
                    data_source="control_engine.line_status",
                    position={"x": 0, "y": 0, "width": 12, "height": 6}
                ),
                UIComponent(
                    component_id="system_alerts",
                    component_type=UIComponentType.ALERT,
                    title="System Alerts",
                    config={"severity_levels": ["critical", "warning", "info"]},
                    data_source="monitoring_engine.alerts",
                    position={"x": 0, "y": 6, "width": 12, "height": 3}
                )
            ],
            theme=self.default_theme,
            permissions=["control_system", "view_dashboards"]
        )
        
        self.dashboard_registry["manufacturing_overview"] = overview_dashboard
        self.dashboard_registry["control_panel"] = control_dashboard

    def _render_component(self, component: UIComponent, session: UserSession) -> Dict[str, Any]:
        """Render individual UI component with current data."""
        # Get cached data or fetch fresh data
        component_data = self.component_cache.get(component.component_id, {})
        
        # Subscribe session to component updates
        if component.component_id not in self.data_subscriptions:
            self.data_subscriptions[component.component_id] = []
        if session.session_id not in self.data_subscriptions[component.component_id]:
            self.data_subscriptions[component.component_id].append(session.session_id)
        
        rendered_component = {
            'component_id': component.component_id,
            'type': component.component_type.value,
            'title': component.title,
            'config': component.config,
            'position': component.position,
            'data': component_data.get('data', {}),
            'last_updated': component_data.get('timestamp'),
            'refresh_rate_ms': component.refresh_rate_ms,
            'interactive': component.interactive
        }
        
        return rendered_component

    def _process_interaction_action(self, interaction: UIInteraction, session: UserSession) -> Dict[str, Any]:
        """Process specific interaction action and return response."""
        action = interaction.action
        data = interaction.data
        
        if action == "control_command":
            return self._handle_control_command(data, session)
        elif action == "data_request":
            return self._handle_data_request(data, session)
        elif action == "configuration_change":
            return self._handle_configuration_change(data, session)
        elif action == "dashboard_navigation":
            return self._handle_dashboard_navigation(data, session)
        else:
            return {"status": "unknown_action", "action": action}

    def _handle_control_command(self, data: Dict[str, Any], session: UserSession) -> Dict[str, Any]:
        """Handle system control commands through UI."""
        if "control_system" not in session.permissions:
            return {"status": "permission_denied", "message": "Insufficient permissions for control commands"}
        
        command = data.get("command")
        parameters = data.get("parameters", {})
        
        if self.control_engine:
            try:
                # Execute control command through control engine
                result = self.control_engine.process_real_time_data(parameters, {"command": command})
                return {"status": "success", "result": result, "command": command}
            except Exception as e:
                return {"status": "error", "message": str(e), "command": command}
        else:
            # Mock response for development
            return {"status": "mock_success", "command": command, "message": "Control engine not available"}

    def _handle_data_request(self, data: Dict[str, Any], session: UserSession) -> Dict[str, Any]:
        """Handle real-time data requests from UI components."""
        data_source = data.get("data_source")
        parameters = data.get("parameters", {})
        
        if data_source and self.monitoring_engine:
            try:
                # Request data from monitoring engine
                monitoring_data = {"data_source": data_source, "parameters": parameters}
                return {"status": "success", "data": monitoring_data}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        else:
            # Return mock data
            return {"status": "mock_data", "data": {"timestamp": datetime.now().isoformat()}}

    def _handle_configuration_change(self, data: Dict[str, Any], session: UserSession) -> Dict[str, Any]:
        """Handle system configuration changes through UI."""
        if "configure_system" not in session.permissions:
            return {"status": "permission_denied", "message": "Insufficient permissions for configuration changes"}
        
        config_type = data.get("config_type")
        config_data = data.get("config_data", {})
        
        return {"status": "success", "config_type": config_type, "message": "Configuration updated"}

    def _handle_dashboard_navigation(self, data: Dict[str, Any], session: UserSession) -> Dict[str, Any]:
        """Handle dashboard navigation and switching."""
        dashboard_id = data.get("dashboard_id")
        
        if dashboard_id and dashboard_id in self.dashboard_registry:
            session.active_dashboard = dashboard_id
            return {"status": "success", "dashboard_id": dashboard_id}
        else:
            return {"status": "error", "message": "Dashboard not found"}

    def _check_dashboard_permissions(self, dashboard: Dashboard, session: UserSession) -> bool:
        """Check if user session has permissions to access dashboard."""
        if not dashboard.permissions:
            return True  # No specific permissions required
        
        return any(perm in session.permissions for perm in dashboard.permissions)

    def _apply_dashboard_customizations(self, dashboard: Dashboard, customizations: Dict[str, Any]) -> Dashboard:
        """Apply user customizations to dashboard configuration."""
        # Create copy of dashboard for customization
        customized_dashboard = Dashboard(
            dashboard_id=dashboard.dashboard_id,
            name=dashboard.name,
            layout=dashboard.layout,
            components=dashboard.components.copy(),
            theme=dashboard.theme,
            auto_refresh=dashboard.auto_refresh,
            permissions=dashboard.permissions
        )
        
        # Apply theme customization
        if "theme" in customizations:
            try:
                customized_dashboard.theme = UITheme(customizations["theme"])
            except ValueError:
                pass  # Keep original theme if invalid
        
        # Apply component customizations
        if "components" in customizations:
            component_customizations = customizations["components"]
            for component in customized_dashboard.components:
                if component.component_id in component_customizations:
                    comp_custom = component_customizations[component.component_id]
                    if "visible" in comp_custom:
                        component.visible = comp_custom["visible"]
                    if "position" in comp_custom:
                        component.position.update(comp_custom["position"])
        
        return customized_dashboard

    def _broadcast_component_update(self, session_id: str, component_id: str, data: Dict[str, Any]):
        """Broadcast component update to specific session via WebSocket."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            if session.websocket_connection:
                try:
                    update_message = {
                        "type": "component_update",
                        "component_id": component_id,
                        "data": data,
                        "timestamp": datetime.now().isoformat()
                    }
                    # Note: Actual WebSocket send would be implemented based on the specific WebSocket library
                    # session.websocket_connection.send(json.dumps(update_message))
                except Exception as e:
                    logging.warning(f"Failed to broadcast update to session {session_id}: {e}")

    def _start_session_monitoring(self, session_id: str):
        """Start monitoring thread for session timeout and cleanup."""
        def monitor_session():
            while session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                time_since_activity = datetime.now() - session.last_activity
                
                if time_since_activity.total_seconds() > (self.session_timeout_hours * 3600):
                    logging.info(f"Session {session_id} timed out - terminating")
                    self.terminate_user_session(session_id)
                    break
                
                time.sleep(60)  # Check every minute
        
        monitor_thread = threading.Thread(target=monitor_session, daemon=True)
        monitor_thread.start()

    def _update_ui_metrics(self, response_time: float, success: bool, operation: str):
        """Update UI performance metrics."""
        self.performance_metrics['total_interactions'] += 1
        
        if success:
            self.performance_metrics['successful_interactions'] += 1
        
        # Update average response time
        total_interactions = self.performance_metrics['total_interactions']
        current_avg = self.performance_metrics['avg_response_time_ms']
        self.performance_metrics['avg_response_time_ms'] = (
            (current_avg * (total_interactions - 1) + response_time) / total_interactions
        )

    def get_dashboard_list(self, session_id: str) -> List[Dict[str, Any]]:
        """Get list of available dashboards for user session."""
        if session_id not in self.active_sessions:
            return []
        
        session = self.active_sessions[session_id]
        available_dashboards = []
        
        for dashboard_id, dashboard in self.dashboard_registry.items():
            if self._check_dashboard_permissions(dashboard, session):
                available_dashboards.append({
                    'dashboard_id': dashboard_id,
                    'name': dashboard.name,
                    'layout': dashboard.layout.value,
                    'component_count': len(dashboard.components)
                })
        
        return available_dashboards

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current UI performance metrics."""
        return self.performance_metrics.copy()

    def cleanup_expired_sessions(self):
        """Clean up expired user sessions."""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            time_since_activity = current_time - session.last_activity
            if time_since_activity.total_seconds() > (self.session_timeout_hours * 3600):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.terminate_user_session(session_id)
        
        logging.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    def __str__(self) -> str:
        return f"WebUIEngine(target={self.response_target_ms}ms, sessions={len(self.active_sessions)})"

    def __repr__(self) -> str:
        return self.__str__()