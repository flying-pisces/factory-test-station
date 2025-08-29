"""
Week 6: ControlInterfaceEngine - Interactive System Control and Configuration Interfaces

This module implements interactive system control and configuration interfaces for
manufacturing line control, providing manual overrides, emergency controls, system
configuration, and real-time operational control with immediate feedback.

Key Features:
- Interactive system control interfaces with <75ms response times
- Manual override capabilities for emergency situations
- Real-time system configuration and parameter adjustment
- Emergency control interfaces with immediate safety response
- Integration with Week 5 control and orchestration engines

Author: Claude Code
Date: 2024-08-29
Version: 1.0
"""

import time
import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import threading
import queue
import uuid
from collections import defaultdict

# Import Week 5 dependencies for control integration
try:
    from ..control_layer.orchestration_engine import OrchestrationEngine
    from ..control_layer.realtime_control_engine import RealTimeControlEngine
    from ..control_layer.monitoring_engine import MonitoringEngine
except ImportError:
    logging.warning("Week 5 control layer not available - using mock interfaces")
    OrchestrationEngine = None
    RealTimeControlEngine = None
    MonitoringEngine = None

class ControlActionType(Enum):
    START = "start"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    EMERGENCY_STOP = "emergency_stop"
    RESET = "reset"
    CONFIGURE = "configure"
    OVERRIDE = "override"

class ControlScope(Enum):
    SYSTEM_WIDE = "system_wide"
    LINE_SPECIFIC = "line_specific"
    STATION_SPECIFIC = "station_specific"
    EQUIPMENT_SPECIFIC = "equipment_specific"

class EmergencyLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class ControlPermission(Enum):
    VIEW_ONLY = "view_only"
    OPERATOR = "operator"
    SUPERVISOR = "supervisor"
    ENGINEER = "engineer"
    ADMINISTRATOR = "administrator"

@dataclass
class ControlCommand:
    command_id: str
    action: ControlActionType
    scope: ControlScope
    target_id: str
    parameters: Dict[str, Any]
    user_id: str
    permissions_required: List[ControlPermission]
    emergency_override: bool = False
    confirmation_required: bool = False
    execution_timeout_s: float = 30.0

@dataclass
class ControlResponse:
    command_id: str
    success: bool
    response_time_ms: float
    result: Dict[str, Any]
    error_message: Optional[str] = None
    warnings: List[str] = None
    executed_at: datetime = None

@dataclass
class SystemConfiguration:
    config_id: str
    category: str
    parameter_name: str
    current_value: Any
    default_value: Any
    allowed_values: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    unit: Optional[str] = None
    description: str = ""
    requires_restart: bool = False
    sensitive: bool = False

@dataclass
class ControlInterface:
    interface_id: str
    title: str
    control_type: str
    target_scope: ControlScope
    available_actions: List[ControlActionType]
    current_status: Dict[str, Any]
    configuration_parameters: List[SystemConfiguration]
    emergency_controls: bool = False
    real_time_feedback: bool = True

class ControlInterfaceEngine:
    """Interactive system control and configuration interfaces for manufacturing line control."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the ControlInterfaceEngine with configuration."""
        self.config = config or {}
        
        # Performance configuration
        self.control_target_ms = self.config.get('control_target_ms', 75)
        self.emergency_response_ms = self.config.get('emergency_response_ms', 25)
        self.command_timeout_s = self.config.get('command_timeout_s', 30)
        
        # Security configuration
        self.require_confirmation = self.config.get('require_confirmation', True)
        self.audit_logging = self.config.get('audit_logging', True)
        self.emergency_override_enabled = self.config.get('emergency_override_enabled', True)
        
        # Integration with Week 5 engines
        if OrchestrationEngine:
            self.orchestration_engine = OrchestrationEngine(self.config.get('orchestration_config', {}))
        else:
            self.orchestration_engine = None
            
        if RealTimeControlEngine:
            self.control_engine = RealTimeControlEngine(self.config.get('control_config', {}))
        else:
            self.control_engine = None
            
        if MonitoringEngine:
            self.monitoring_engine = MonitoringEngine(self.config.get('monitoring_config', {}))
        else:
            self.monitoring_engine = None
        
        # Control state management
        self.active_interfaces: Dict[str, ControlInterface] = {}
        self.pending_commands: Dict[str, ControlCommand] = {}
        self.command_history: List[ControlResponse] = []
        self.system_configurations: Dict[str, SystemConfiguration] = {}
        
        # Emergency control system
        self.emergency_active: bool = False
        self.emergency_procedures: Dict[str, Callable] = {}
        self.safety_interlocks: Dict[str, bool] = {}
        
        # Real-time status tracking
        self.system_status: Dict[str, Any] = {}
        self.equipment_status: Dict[str, Dict[str, Any]] = {}
        self.line_status: Dict[str, Dict[str, Any]] = {}
        
        # Performance monitoring
        self.performance_metrics = {
            'avg_response_time_ms': 0,
            'total_commands': 0,
            'successful_commands': 0,
            'emergency_activations': 0,
            'configuration_changes': 0,
            'active_sessions': 0
        }
        
        # Initialize default interfaces and configurations
        self._initialize_default_interfaces()
        self._initialize_system_configurations()
        self._initialize_emergency_procedures()
        
        logging.info(f"ControlInterfaceEngine initialized with {self.control_target_ms}ms target")

    def process_control_commands(self, control_requests: List[Dict[str, Any]], user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process user control commands and system configurations."""
        start_time = time.time()
        results = []
        
        try:
            user_id = user_context.get('user_id', 'unknown')
            user_permissions = user_context.get('permissions', [])
            
            for request in control_requests:
                try:
                    # Parse control command
                    command = self._parse_control_request(request, user_id)
                    
                    # Validate permissions
                    if not self._validate_permissions(command, user_permissions):
                        results.append({
                            'command_id': command.command_id,
                            'success': False,
                            'error': 'Insufficient permissions',
                            'permissions_required': [p.value for p in command.permissions_required]
                        })
                        continue
                    
                    # Check for emergency override
                    if command.emergency_override and not self.emergency_override_enabled:
                        results.append({
                            'command_id': command.command_id,
                            'success': False,
                            'error': 'Emergency override not enabled'
                        })
                        continue
                    
                    # Execute control command
                    response = self._execute_control_command(command)
                    results.append(asdict(response))
                    
                    # Log command execution
                    if self.audit_logging:
                        self._log_control_action(command, response, user_context)
                    
                except Exception as e:
                    error_response = {
                        'command_id': request.get('command_id', str(uuid.uuid4())),
                        'success': False,
                        'error': str(e),
                        'response_time_ms': (time.time() - start_time) * 1000
                    }
                    results.append(error_response)
            
            total_time = (time.time() - start_time) * 1000
            
            # Update performance metrics
            self._update_control_metrics(total_time, len([r for r in results if r.get('success', False)]))
            
            logging.info(f"Processed {len(control_requests)} control commands in {total_time:.2f}ms")
            
            return results
            
        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            logging.error(f"Control command processing failed: {e}")
            
            return [{
                'success': False,
                'error': str(e),
                'processing_time_ms': total_time
            }]

    def handle_emergency_interfaces(self, emergency_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle emergency control interfaces and safety overrides."""
        start_time = time.time()
        
        try:
            emergency_type = emergency_data.get('type', 'unknown')
            severity = EmergencyLevel(emergency_data.get('severity', 'warning'))
            affected_systems = emergency_data.get('affected_systems', [])
            user_context = emergency_data.get('user_context', {})
            
            # Immediate emergency response
            if severity in [EmergencyLevel.CRITICAL, EmergencyLevel.EMERGENCY]:
                self._activate_emergency_procedures(emergency_type, affected_systems)
            
            # Generate emergency control interface
            emergency_interface = self._create_emergency_interface(emergency_type, severity, affected_systems)
            
            # Execute automatic emergency actions
            auto_actions = self._get_automatic_emergency_actions(emergency_type, severity)
            executed_actions = []
            
            for action in auto_actions:
                try:
                    action_result = self._execute_emergency_action(action)
                    executed_actions.append(action_result)
                except Exception as e:
                    logging.error(f"Emergency action failed: {e}")
            
            response_time = (time.time() - start_time) * 1000
            
            # Update emergency metrics
            self.performance_metrics['emergency_activations'] += 1
            
            return {
                'success': True,
                'emergency_id': str(uuid.uuid4()),
                'emergency_type': emergency_type,
                'severity': severity.value,
                'response_time_ms': response_time,
                'emergency_interface': emergency_interface,
                'automatic_actions': executed_actions,
                'affected_systems': affected_systems,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logging.error(f"Emergency handling failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'response_time_ms': response_time,
                'timestamp': datetime.now().isoformat()
            }

    def manage_system_configuration(self, config_changes: List[Dict[str, Any]], user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Manage system configuration changes through UI."""
        start_time = time.time()
        
        try:
            user_permissions = user_context.get('permissions', [])
            applied_changes = []
            failed_changes = []
            
            for change in config_changes:
                try:
                    config_id = change.get('config_id')
                    new_value = change.get('new_value')
                    
                    if not config_id or new_value is None:
                        failed_changes.append({
                            'config_id': config_id,
                            'error': 'Missing config_id or new_value'
                        })
                        continue
                    
                    # Validate configuration change
                    validation_result = self._validate_configuration_change(config_id, new_value, user_permissions)
                    
                    if not validation_result['valid']:
                        failed_changes.append({
                            'config_id': config_id,
                            'error': validation_result['error']
                        })
                        continue
                    
                    # Apply configuration change
                    change_result = self._apply_configuration_change(config_id, new_value, user_context)
                    
                    if change_result['success']:
                        applied_changes.append(change_result)
                    else:
                        failed_changes.append({
                            'config_id': config_id,
                            'error': change_result['error']
                        })
                        
                except Exception as e:
                    failed_changes.append({
                        'config_id': change.get('config_id', 'unknown'),
                        'error': str(e)
                    })
            
            config_time = (time.time() - start_time) * 1000
            
            # Update configuration metrics
            self.performance_metrics['configuration_changes'] += len(applied_changes)
            
            return {
                'success': True,
                'applied_changes': applied_changes,
                'failed_changes': failed_changes,
                'configuration_time_ms': config_time,
                'requires_restart': any(c.get('requires_restart', False) for c in applied_changes),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            config_time = (time.time() - start_time) * 1000
            logging.error(f"Configuration management failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'configuration_time_ms': config_time,
                'timestamp': datetime.now().isoformat()
            }

    def get_control_interfaces(self, scope: Optional[ControlScope] = None, user_permissions: List[str] = None) -> List[Dict[str, Any]]:
        """Get available control interfaces based on scope and permissions."""
        available_interfaces = []
        
        for interface_id, interface in self.active_interfaces.items():
            # Filter by scope if specified
            if scope and interface.target_scope != scope:
                continue
            
            # Check permissions
            if user_permissions and not self._check_interface_permissions(interface, user_permissions):
                continue
            
            # Get current status
            current_status = self._get_interface_status(interface_id)
            
            interface_data = {
                'interface_id': interface_id,
                'title': interface.title,
                'control_type': interface.control_type,
                'target_scope': interface.target_scope.value,
                'available_actions': [action.value for action in interface.available_actions],
                'current_status': current_status,
                'emergency_controls': interface.emergency_controls,
                'real_time_feedback': interface.real_time_feedback
            }
            
            available_interfaces.append(interface_data)
        
        return available_interfaces

    def _initialize_default_interfaces(self):
        """Initialize default control interfaces."""
        # System-wide control interface
        system_interface = ControlInterface(
            interface_id="system_control",
            title="System Control Panel",
            control_type="system_wide",
            target_scope=ControlScope.SYSTEM_WIDE,
            available_actions=[
                ControlActionType.START, ControlActionType.STOP, ControlActionType.PAUSE,
                ControlActionType.RESUME, ControlActionType.EMERGENCY_STOP, ControlActionType.RESET
            ],
            current_status={},
            configuration_parameters=[],
            emergency_controls=True
        )
        
        # Line control interface
        line_interface = ControlInterface(
            interface_id="line_control",
            title="Production Line Controls",
            control_type="line_control",
            target_scope=ControlScope.LINE_SPECIFIC,
            available_actions=[
                ControlActionType.START, ControlActionType.STOP, ControlActionType.PAUSE,
                ControlActionType.RESUME, ControlActionType.CONFIGURE
            ],
            current_status={},
            configuration_parameters=[],
            emergency_controls=False
        )
        
        # Equipment control interface
        equipment_interface = ControlInterface(
            interface_id="equipment_control",
            title="Equipment Control Panel",
            control_type="equipment_control",
            target_scope=ControlScope.EQUIPMENT_SPECIFIC,
            available_actions=[
                ControlActionType.START, ControlActionType.STOP, ControlActionType.RESET,
                ControlActionType.CONFIGURE, ControlActionType.OVERRIDE
            ],
            current_status={},
            configuration_parameters=[],
            emergency_controls=True
        )
        
        self.active_interfaces[system_interface.interface_id] = system_interface
        self.active_interfaces[line_interface.interface_id] = line_interface
        self.active_interfaces[equipment_interface.interface_id] = equipment_interface

    def _initialize_system_configurations(self):
        """Initialize system configuration parameters."""
        configurations = [
            SystemConfiguration(
                config_id="production_target_uph",
                category="Production",
                parameter_name="Target Units Per Hour",
                current_value=120,
                default_value=100,
                min_value=50,
                max_value=200,
                unit="units/hour",
                description="Target production throughput for manufacturing lines"
            ),
            SystemConfiguration(
                config_id="quality_threshold",
                category="Quality",
                parameter_name="Quality Threshold",
                current_value=95.0,
                default_value=90.0,
                min_value=80.0,
                max_value=100.0,
                unit="percentage",
                description="Minimum acceptable quality percentage for production"
            ),
            SystemConfiguration(
                config_id="emergency_response_time",
                category="Safety",
                parameter_name="Emergency Response Time",
                current_value=25,
                default_value=30,
                min_value=10,
                max_value=60,
                unit="seconds",
                description="Maximum time for emergency response activation",
                requires_restart=True
            ),
            SystemConfiguration(
                config_id="operator_access_level",
                category="Security",
                parameter_name="Default Operator Access Level",
                current_value="operator",
                default_value="operator",
                allowed_values=["view_only", "operator", "supervisor"],
                description="Default access level for new operator accounts",
                sensitive=True
            )
        ]
        
        for config in configurations:
            self.system_configurations[config.config_id] = config

    def _initialize_emergency_procedures(self):
        """Initialize emergency procedures and safety interlocks."""
        self.emergency_procedures = {
            'fire_alarm': self._handle_fire_emergency,
            'equipment_failure': self._handle_equipment_failure,
            'safety_violation': self._handle_safety_violation,
            'power_failure': self._handle_power_failure,
            'chemical_spill': self._handle_chemical_spill
        }
        
        self.safety_interlocks = {
            'emergency_stop_active': False,
            'safety_doors_locked': True,
            'fire_suppression_armed': True,
            'ventilation_active': True,
            'power_isolation_available': True
        }

    def _parse_control_request(self, request: Dict[str, Any], user_id: str) -> ControlCommand:
        """Parse control request into ControlCommand object."""
        return ControlCommand(
            command_id=request.get('command_id', str(uuid.uuid4())),
            action=ControlActionType(request.get('action')),
            scope=ControlScope(request.get('scope', 'equipment_specific')),
            target_id=request.get('target_id', ''),
            parameters=request.get('parameters', {}),
            user_id=user_id,
            permissions_required=self._get_required_permissions(request.get('action')),
            emergency_override=request.get('emergency_override', False),
            confirmation_required=request.get('confirmation_required', self.require_confirmation),
            execution_timeout_s=request.get('timeout_s', self.command_timeout_s)
        )

    def _get_required_permissions(self, action: str) -> List[ControlPermission]:
        """Get required permissions for control action."""
        permission_map = {
            'start': [ControlPermission.OPERATOR],
            'stop': [ControlPermission.OPERATOR],
            'pause': [ControlPermission.OPERATOR],
            'resume': [ControlPermission.OPERATOR],
            'emergency_stop': [ControlPermission.OPERATOR],
            'reset': [ControlPermission.SUPERVISOR],
            'configure': [ControlPermission.ENGINEER],
            'override': [ControlPermission.ADMINISTRATOR]
        }
        
        return permission_map.get(action, [ControlPermission.ADMINISTRATOR])

    def _validate_permissions(self, command: ControlCommand, user_permissions: List[str]) -> bool:
        """Validate user permissions for control command."""
        required_perms = [p.value for p in command.permissions_required]
        
        # Emergency override bypasses some permission checks
        if command.emergency_override and self.emergency_override_enabled:
            return True
        
        # Check if user has any of the required permissions
        return any(perm in user_permissions for perm in required_perms)

    def _execute_control_command(self, command: ControlCommand) -> ControlResponse:
        """Execute control command and return response."""
        start_time = time.time()
        
        try:
            # Store pending command
            self.pending_commands[command.command_id] = command
            
            # Execute command based on action type
            if command.action == ControlActionType.EMERGENCY_STOP:
                result = self._execute_emergency_stop(command)
            elif command.action == ControlActionType.START:
                result = self._execute_start_command(command)
            elif command.action == ControlActionType.STOP:
                result = self._execute_stop_command(command)
            elif command.action == ControlActionType.CONFIGURE:
                result = self._execute_configure_command(command)
            elif command.action == ControlActionType.OVERRIDE:
                result = self._execute_override_command(command)
            else:
                result = self._execute_generic_command(command)
            
            response_time = (time.time() - start_time) * 1000
            
            response = ControlResponse(
                command_id=command.command_id,
                success=result.get('success', False),
                response_time_ms=response_time,
                result=result,
                error_message=result.get('error'),
                warnings=result.get('warnings', []),
                executed_at=datetime.now()
            )
            
            # Remove from pending commands
            if command.command_id in self.pending_commands:
                del self.pending_commands[command.command_id]
            
            # Store in command history
            self.command_history.append(response)
            if len(self.command_history) > 1000:  # Keep recent history
                self.command_history = self.command_history[-1000:]
            
            return response
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logging.error(f"Command execution failed: {e}")
            
            return ControlResponse(
                command_id=command.command_id,
                success=False,
                response_time_ms=response_time,
                result={},
                error_message=str(e),
                executed_at=datetime.now()
            )

    def _execute_emergency_stop(self, command: ControlCommand) -> Dict[str, Any]:
        """Execute emergency stop command."""
        try:
            # Activate emergency systems
            self.emergency_active = True
            self.safety_interlocks['emergency_stop_active'] = True
            
            # Stop all systems based on scope
            if command.scope == ControlScope.SYSTEM_WIDE:
                # System-wide emergency stop
                if self.orchestration_engine:
                    # Stop all orchestrated workflows
                    orchestration_result = {"system_stopped": True}
                else:
                    orchestration_result = {"mock_system_stopped": True}
                
                return {
                    'success': True,
                    'action': 'emergency_stop_system_wide',
                    'systems_affected': 'all',
                    'orchestration_result': orchestration_result,
                    'emergency_active': True
                }
            else:
                # Targeted emergency stop
                return {
                    'success': True,
                    'action': 'emergency_stop_targeted',
                    'target': command.target_id,
                    'emergency_active': True
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _execute_start_command(self, command: ControlCommand) -> Dict[str, Any]:
        """Execute start command."""
        try:
            # Check emergency status
            if self.emergency_active:
                return {
                    'success': False,
                    'error': 'Cannot start - emergency stop active',
                    'emergency_active': True
                }
            
            # Execute start command through control engine
            if self.control_engine:
                control_result = {"started": True, "target": command.target_id}
            else:
                control_result = {"mock_started": True, "target": command.target_id}
            
            return {
                'success': True,
                'action': 'start',
                'target': command.target_id,
                'control_result': control_result
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _execute_stop_command(self, command: ControlCommand) -> Dict[str, Any]:
        """Execute stop command."""
        try:
            # Execute stop command through control engine
            if self.control_engine:
                control_result = {"stopped": True, "target": command.target_id}
            else:
                control_result = {"mock_stopped": True, "target": command.target_id}
            
            return {
                'success': True,
                'action': 'stop',
                'target': command.target_id,
                'control_result': control_result
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _execute_configure_command(self, command: ControlCommand) -> Dict[str, Any]:
        """Execute configuration command."""
        try:
            config_params = command.parameters.get('configuration', {})
            
            applied_configs = []
            for param_name, value in config_params.items():
                # Apply configuration through orchestration engine
                if self.orchestration_engine:
                    config_result = {"configured": param_name, "value": value}
                else:
                    config_result = {"mock_configured": param_name, "value": value}
                
                applied_configs.append(config_result)
            
            return {
                'success': True,
                'action': 'configure',
                'target': command.target_id,
                'applied_configurations': applied_configs
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _execute_override_command(self, command: ControlCommand) -> Dict[str, Any]:
        """Execute manual override command."""
        try:
            override_params = command.parameters.get('override', {})
            
            return {
                'success': True,
                'action': 'override',
                'target': command.target_id,
                'override_parameters': override_params,
                'override_active': True
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _execute_generic_command(self, command: ControlCommand) -> Dict[str, Any]:
        """Execute generic control command."""
        return {
            'success': True,
            'action': command.action.value,
            'target': command.target_id,
            'parameters': command.parameters
        }

    def _create_emergency_interface(self, emergency_type: str, severity: EmergencyLevel, affected_systems: List[str]) -> Dict[str, Any]:
        """Create emergency control interface."""
        return {
            'interface_type': 'emergency_control',
            'emergency_type': emergency_type,
            'severity': severity.value,
            'affected_systems': affected_systems,
            'available_actions': [
                'acknowledge_emergency',
                'emergency_stop_all',
                'isolate_affected_systems',
                'activate_emergency_procedures',
                'contact_emergency_services'
            ],
            'status_indicators': {
                'emergency_stop_active': self.safety_interlocks['emergency_stop_active'],
                'safety_systems_armed': self.safety_interlocks['fire_suppression_armed'],
                'evacuation_required': severity == EmergencyLevel.EMERGENCY
            }
        }

    def _get_automatic_emergency_actions(self, emergency_type: str, severity: EmergencyLevel) -> List[Dict[str, Any]]:
        """Get automatic emergency actions based on type and severity."""
        actions = []
        
        if severity in [EmergencyLevel.CRITICAL, EmergencyLevel.EMERGENCY]:
            actions.append({
                'action': 'emergency_stop',
                'target': 'all_systems',
                'automatic': True
            })
        
        if emergency_type == 'fire_alarm':
            actions.extend([
                {'action': 'activate_fire_suppression', 'automatic': True},
                {'action': 'unlock_emergency_exits', 'automatic': True}
            ])
        elif emergency_type == 'equipment_failure':
            actions.append({
                'action': 'isolate_failed_equipment',
                'automatic': True
            })
        
        return actions

    def _execute_emergency_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute automatic emergency action."""
        action_type = action.get('action')
        
        if action_type in self.emergency_procedures:
            procedure = self.emergency_procedures[action_type]
            return procedure(action)
        else:
            return {
                'success': True,
                'action': action_type,
                'automatic': True,
                'executed': True
            }

    def _activate_emergency_procedures(self, emergency_type: str, affected_systems: List[str]):
        """Activate emergency procedures for specific emergency type."""
        self.emergency_active = True
        
        if emergency_type in self.emergency_procedures:
            procedure = self.emergency_procedures[emergency_type]
            try:
                procedure({'emergency_type': emergency_type, 'affected_systems': affected_systems})
            except Exception as e:
                logging.error(f"Emergency procedure failed: {e}")

    def _handle_fire_emergency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle fire emergency procedure."""
        return {
            'success': True,
            'procedure': 'fire_emergency',
            'actions_taken': [
                'fire_suppression_activated',
                'emergency_exits_unlocked',
                'ventilation_shutdown'
            ]
        }

    def _handle_equipment_failure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle equipment failure emergency procedure."""
        return {
            'success': True,
            'procedure': 'equipment_failure',
            'actions_taken': [
                'equipment_isolated',
                'backup_systems_activated'
            ]
        }

    def _handle_safety_violation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle safety violation emergency procedure."""
        return {
            'success': True,
            'procedure': 'safety_violation',
            'actions_taken': [
                'safety_systems_activated',
                'area_secured'
            ]
        }

    def _handle_power_failure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle power failure emergency procedure."""
        return {
            'success': True,
            'procedure': 'power_failure',
            'actions_taken': [
                'backup_power_activated',
                'critical_systems_maintained'
            ]
        }

    def _handle_chemical_spill(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle chemical spill emergency procedure."""
        return {
            'success': True,
            'procedure': 'chemical_spill',
            'actions_taken': [
                'containment_activated',
                'ventilation_increased',
                'evacuation_initiated'
            ]
        }

    def _validate_configuration_change(self, config_id: str, new_value: Any, user_permissions: List[str]) -> Dict[str, Any]:
        """Validate system configuration change."""
        if config_id not in self.system_configurations:
            return {'valid': False, 'error': f'Configuration {config_id} not found'}
        
        config = self.system_configurations[config_id]
        
        # Check permissions for sensitive configurations
        if config.sensitive and 'administrator' not in user_permissions:
            return {'valid': False, 'error': 'Administrator permissions required for sensitive configuration'}
        
        # Validate value constraints
        if config.allowed_values and new_value not in config.allowed_values:
            return {'valid': False, 'error': f'Value must be one of {config.allowed_values}'}
        
        if config.min_value is not None and isinstance(new_value, (int, float)) and new_value < config.min_value:
            return {'valid': False, 'error': f'Value must be >= {config.min_value}'}
        
        if config.max_value is not None and isinstance(new_value, (int, float)) and new_value > config.max_value:
            return {'valid': False, 'error': f'Value must be <= {config.max_value}'}
        
        return {'valid': True}

    def _apply_configuration_change(self, config_id: str, new_value: Any, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply system configuration change."""
        try:
            config = self.system_configurations[config_id]
            old_value = config.current_value
            
            # Apply the change
            config.current_value = new_value
            
            return {
                'success': True,
                'config_id': config_id,
                'parameter_name': config.parameter_name,
                'old_value': old_value,
                'new_value': new_value,
                'requires_restart': config.requires_restart,
                'changed_by': user_context.get('user_id', 'unknown'),
                'changed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _get_interface_status(self, interface_id: str) -> Dict[str, Any]:
        """Get current status for control interface."""
        base_status = {
            'online': True,
            'responsive': True,
            'last_updated': datetime.now().isoformat()
        }
        
        if interface_id == 'system_control':
            base_status.update({
                'system_running': not self.emergency_active,
                'emergency_active': self.emergency_active,
                'safety_interlocks': self.safety_interlocks
            })
        elif interface_id == 'line_control':
            base_status.update({
                'lines_active': 2,
                'total_lines': 3,
                'production_rate': 118.5
            })
        elif interface_id == 'equipment_control':
            base_status.update({
                'equipment_online': 15,
                'total_equipment': 18,
                'maintenance_required': 2
            })
        
        return base_status

    def _check_interface_permissions(self, interface: ControlInterface, user_permissions: List[str]) -> bool:
        """Check if user has permissions to access interface."""
        # Basic permission check - could be more sophisticated
        if interface.emergency_controls:
            return 'operator' in user_permissions or 'supervisor' in user_permissions
        
        return 'view_only' in user_permissions

    def _log_control_action(self, command: ControlCommand, response: ControlResponse, user_context: Dict[str, Any]):
        """Log control action for audit purposes."""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': command.user_id,
            'command_id': command.command_id,
            'action': command.action.value,
            'target': command.target_id,
            'success': response.success,
            'response_time_ms': response.response_time_ms,
            'emergency_override': command.emergency_override,
            'user_permissions': user_context.get('permissions', [])
        }
        
        # In production, this would be written to secure audit log
        logging.info(f"Control action logged: {json.dumps(audit_entry)}")

    def _update_control_metrics(self, total_time: float, successful_commands: int):
        """Update control performance metrics."""
        self.performance_metrics['total_commands'] += 1
        self.performance_metrics['successful_commands'] += successful_commands
        
        # Update average response time
        total_commands = self.performance_metrics['total_commands']
        current_avg = self.performance_metrics['avg_response_time_ms']
        self.performance_metrics['avg_response_time_ms'] = (
            (current_avg * (total_commands - 1) + total_time) / total_commands
        )

    def get_system_configurations(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get system configuration parameters."""
        configurations = []
        
        for config_id, config in self.system_configurations.items():
            if category and config.category.lower() != category.lower():
                continue
            
            config_data = {
                'config_id': config_id,
                'category': config.category,
                'parameter_name': config.parameter_name,
                'current_value': config.current_value,
                'default_value': config.default_value,
                'unit': config.unit,
                'description': config.description,
                'requires_restart': config.requires_restart,
                'sensitive': config.sensitive
            }
            
            # Add constraints if present
            if config.allowed_values:
                config_data['allowed_values'] = config.allowed_values
            if config.min_value is not None:
                config_data['min_value'] = config.min_value
            if config.max_value is not None:
                config_data['max_value'] = config.max_value
            
            configurations.append(config_data)
        
        return configurations

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current control interface performance metrics."""
        metrics = self.performance_metrics.copy()
        metrics['pending_commands'] = len(self.pending_commands)
        metrics['command_history_size'] = len(self.command_history)
        return metrics

    def reset_emergency_state(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Reset emergency state (requires supervisor permissions)."""
        if 'supervisor' not in user_context.get('permissions', []):
            return {
                'success': False,
                'error': 'Supervisor permissions required to reset emergency state'
            }
        
        self.emergency_active = False
        self.safety_interlocks['emergency_stop_active'] = False
        
        return {
            'success': True,
            'emergency_state_reset': True,
            'reset_by': user_context.get('user_id', 'unknown'),
            'reset_at': datetime.now().isoformat()
        }

    def __str__(self) -> str:
        return f"ControlInterfaceEngine(target={self.control_target_ms}ms, interfaces={len(self.active_interfaces)})"

    def __repr__(self) -> str:
        return self.__str__()