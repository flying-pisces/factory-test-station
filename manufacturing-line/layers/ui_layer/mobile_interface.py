"""
Mobile Interface - Week 13: UI & Visualization Layer

Mobile-optimized interface for field operators and technicians providing
quick access to critical manufacturing data, equipment status, and controls
on smartphones and tablets.
"""

from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
import uuid
import random

from .dashboard_manager import DashboardManager, DashboardType
from .real_time_data_pipeline import RealTimeDataPipeline
from .visualization_engine import VisualizationEngine


class MobileInterface:
    """
    Mobile Interface for manufacturing line monitoring and control.
    
    Provides mobile-optimized interface for field operators and technicians
    with touch-friendly controls, offline capability, and critical data access.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Mobile Interface.
        
        Args:
            config: Configuration dictionary for mobile interface settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Mobile interface configuration
        self.port = self.config.get('port', 5003)
        self.host = self.config.get('host', '0.0.0.0')
        self.debug = self.config.get('debug', False)
        self.update_interval_ms = self.config.get('update_interval_ms', 3000)  # 3 seconds for mobile
        self.offline_mode = self.config.get('offline_mode', False)
        
        # Initialize Flask app
        self.app = Flask(__name__, template_folder='templates', static_folder='static')
        self.app.config['SECRET_KEY'] = self.config.get('secret_key', 'mobile_interface_secret')
        
        # Initialize SocketIO
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", 
                               async_mode='threading', logger=self.debug)
        
        # Initialize components
        self.dashboard_manager = DashboardManager()
        self.data_pipeline = RealTimeDataPipeline()
        self.visualization_engine = VisualizationEngine()
        
        # Mobile-optimized data structure
        self.mobile_data = {
            'critical_alerts': [],
            'equipment_overview': {},
            'quick_metrics': {
                'line_status': 'running',
                'current_output': 85,
                'efficiency': 87.3,
                'quality_score': 98.2,
                'alert_count': 2
            },
            'maintenance_tasks': [
                {
                    'id': 'MT-001',
                    'equipment': 'Line 1 Conveyor',
                    'task': 'Lubrication Check',
                    'priority': 'medium',
                    'due_date': (datetime.now() + timedelta(hours=4)).isoformat(),
                    'estimated_time': '15 min',
                    'status': 'pending'
                },
                {
                    'id': 'MT-002',
                    'equipment': 'Quality Station 2',
                    'task': 'Calibration Verification',
                    'priority': 'high',
                    'due_date': (datetime.now() + timedelta(hours=2)).isoformat(),
                    'estimated_time': '30 min',
                    'status': 'pending'
                }
            ],
            'recent_activities': []
        }
        
        # Mobile session management
        self.mobile_sessions = {}
        self.mobile_metrics = {
            'active_mobile_users': 0,
            'touch_interactions': 0,
            'offline_sessions': 0,
            'average_session_time': 0.0,
            'feature_usage': {
                'equipment_control': 0,
                'alert_acknowledgment': 0,
                'maintenance_updates': 0,
                'status_checks': 0
            }
        }
        
        # Thread management
        self.lock = threading.Lock()
        self.update_thread = None
        self.is_running = False
        self.start_time = datetime.now()
        
        # Set up routes and handlers
        self._setup_routes()
        self._setup_socketio_handlers()
        
        # Initialize mobile data
        self._initialize_mobile_data()
        
        self.logger.info("MobileInterface initialized successfully")
    
    def _setup_routes(self):
        """Set up Flask routes for mobile interface."""
        
        @self.app.route('/')
        def mobile_home():
            """Main mobile interface page."""
            start_time = time.time()
            
            try:
                # Create mobile session
                if 'session_id' not in session:
                    session['session_id'] = str(uuid.uuid4())
                    session['user_type'] = 'mobile_operator'
                    session['device_type'] = self._detect_device_type(request)
                    session['login_time'] = datetime.now().isoformat()
                
                device_type = session.get('device_type', 'mobile')
                
                response_time = (time.time() - start_time) * 1000
                
                return render_template('mobile_interface.html',
                                     session_id=session['session_id'],
                                     device_type=device_type,
                                     update_interval=self.update_interval_ms,
                                     mobile_data=self.mobile_data,
                                     offline_mode=self.offline_mode)
                
            except Exception as e:
                self.logger.error(f"Mobile home route error: {e}")
                return f"Mobile Interface Error: {e}", 500
        
        @self.app.route('/api/mobile-data')
        def get_mobile_data():
            """API endpoint for mobile-optimized data."""
            try:
                return jsonify(self.mobile_data)
            except Exception as e:
                self.logger.error(f"Mobile data API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/equipment/<equipment_id>/status')
        def get_equipment_status(equipment_id):
            """API endpoint for specific equipment status (mobile-optimized)."""
            try:
                if equipment_id in self.mobile_data['equipment_overview']:
                    equipment = self.mobile_data['equipment_overview'][equipment_id]
                    return jsonify({
                        'equipment_id': equipment_id,
                        'status': equipment['status'],
                        'health_score': equipment['health_score'],
                        'temperature': equipment['temperature'],
                        'last_update': equipment['last_update']
                    })
                else:
                    return jsonify({'error': 'Equipment not found'}), 404
                    
            except Exception as e:
                self.logger.error(f"Equipment status API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/maintenance/<task_id>/update', methods=['POST'])
        def update_maintenance_task(task_id):
            """API endpoint for updating maintenance tasks."""
            try:
                new_status = request.json.get('status')
                notes = request.json.get('notes', '')
                operator_id = session.get('session_id', 'unknown')
                
                # Find and update maintenance task
                for task in self.mobile_data['maintenance_tasks']:
                    if task['id'] == task_id:
                        task['status'] = new_status
                        task['completed_by'] = operator_id
                        task['completed_at'] = datetime.now().isoformat()
                        if notes:
                            task['notes'] = notes
                        break
                
                # Track feature usage
                with self.lock:
                    self.mobile_metrics['feature_usage']['maintenance_updates'] += 1
                
                return jsonify({'success': True, 'task_id': task_id})
                
            except Exception as e:
                self.logger.error(f"Maintenance task update error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/quick-action/<action_type>', methods=['POST'])
        def quick_action(action_type):
            """API endpoint for quick mobile actions."""
            try:
                action_data = request.json
                result = self._execute_quick_action(action_type, action_data)
                
                # Track touch interactions
                with self.lock:
                    self.mobile_metrics['touch_interactions'] += 1
                
                return jsonify(result)
                
            except Exception as e:
                self.logger.error(f"Quick action error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/offline-sync', methods=['POST'])
        def sync_offline_data():
            """API endpoint for syncing offline data."""
            try:
                offline_data = request.json
                sync_result = self._process_offline_sync(offline_data)
                
                with self.lock:
                    self.mobile_metrics['offline_sessions'] += 1
                
                return jsonify(sync_result)
                
            except Exception as e:
                self.logger.error(f"Offline sync error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/mobile-metrics')
        def get_mobile_metrics():
            """API endpoint for mobile interface metrics."""
            try:
                with self.lock:
                    metrics = self.mobile_metrics.copy()
                    metrics['uptime_seconds'] = (datetime.now() - self.start_time).total_seconds()
                
                return jsonify(metrics)
                
            except Exception as e:
                self.logger.error(f"Mobile metrics error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _setup_socketio_handlers(self):
        """Set up SocketIO event handlers for mobile communication."""
        
        @self.socketio.on('connect')
        def handle_mobile_connect():
            """Handle mobile device connection."""
            try:
                session_id = request.sid
                user_agent = request.headers.get('User-Agent', '')
                
                device_info = self._parse_device_info(user_agent)
                
                self.mobile_sessions[session_id] = {
                    'session_id': session_id,
                    'connected_at': datetime.now().isoformat(),
                    'user_type': 'mobile_operator',
                    'device_info': device_info,
                    'last_activity': datetime.now().isoformat()
                }
                
                join_room('mobile_users')
                
                with self.lock:
                    self.mobile_metrics['active_mobile_users'] += 1
                
                self.logger.info(f"Mobile device connected: {session_id} ({device_info['type']})")
                
                # Send initial mobile-optimized data
                emit('mobile_data', self.mobile_data)
                
            except Exception as e:
                self.logger.error(f"Mobile SocketIO connect error: {e}")
        
        @self.socketio.on('disconnect')
        def handle_mobile_disconnect():
            """Handle mobile device disconnection."""
            try:
                session_id = request.sid
                
                if session_id in self.mobile_sessions:
                    # Calculate session duration
                    connect_time = datetime.fromisoformat(
                        self.mobile_sessions[session_id]['connected_at']
                    )
                    duration = (datetime.now() - connect_time).total_seconds()
                    
                    # Update average session time
                    with self.lock:
                        current_avg = self.mobile_metrics['average_session_time']
                        user_count = self.mobile_metrics['active_mobile_users']
                        if user_count > 0:
                            self.mobile_metrics['average_session_time'] = (
                                (current_avg * (user_count - 1) + duration) / user_count
                            )
                        self.mobile_metrics['active_mobile_users'] -= 1
                    
                    del self.mobile_sessions[session_id]
                
                leave_room('mobile_users')
                self.logger.info(f"Mobile device disconnected: {session_id}")
                
            except Exception as e:
                self.logger.error(f"Mobile SocketIO disconnect error: {e}")
        
        @self.socketio.on('touch_interaction')
        def handle_touch_interaction(data):
            """Handle touch interaction tracking."""
            try:
                interaction_type = data.get('type')
                element = data.get('element')
                
                with self.lock:
                    self.mobile_metrics['touch_interactions'] += 1
                    if interaction_type in self.mobile_metrics['feature_usage']:
                        self.mobile_metrics['feature_usage'][interaction_type] += 1
                
                # Update last activity
                session_id = request.sid
                if session_id in self.mobile_sessions:
                    self.mobile_sessions[session_id]['last_activity'] = datetime.now().isoformat()
                
            except Exception as e:
                self.logger.error(f"Touch interaction tracking error: {e}")
        
        @self.socketio.on('request_equipment_details')
        def handle_equipment_details_request(data):
            """Handle request for detailed equipment information."""
            try:
                equipment_id = data.get('equipment_id')
                
                if equipment_id in self.mobile_data['equipment_overview']:
                    equipment = self.mobile_data['equipment_overview'][equipment_id]
                    
                    # Create mobile-optimized equipment details
                    detailed_info = {
                        'basic_info': {
                            'id': equipment_id,
                            'name': equipment['name'],
                            'status': equipment['status'],
                            'health_score': equipment['health_score']
                        },
                        'current_metrics': {
                            'temperature': equipment['temperature'],
                            'vibration': equipment.get('vibration', 0.2),
                            'efficiency': equipment.get('efficiency', 85),
                            'runtime_hours': equipment.get('runtime_hours', 1240)
                        },
                        'maintenance_info': {
                            'last_maintenance': equipment.get('last_maintenance'),
                            'next_maintenance': equipment.get('next_maintenance'),
                            'maintenance_score': equipment.get('maintenance_score', 8.5)
                        },
                        'quick_actions': [
                            'start', 'stop', 'pause', 'reset', 'maintenance_mode'
                        ]
                    }
                    
                    emit('equipment_details', detailed_info)
                else:
                    emit('equipment_details_error', {'error': 'Equipment not found'})
                
            except Exception as e:
                self.logger.error(f"Equipment details request error: {e}")
                emit('equipment_details_error', {'error': str(e)})
    
    def _initialize_mobile_data(self):
        """Initialize mobile-optimized data."""
        try:
            # Initialize equipment overview for mobile
            equipment_list = [
                'line_1_conveyor', 'line_1_robot', 'line_2_conveyor', 
                'line_2_robot', 'quality_station', 'packaging_unit'
            ]
            
            for equipment_id in equipment_list:
                self.mobile_data['equipment_overview'][equipment_id] = {
                    'id': equipment_id,
                    'name': equipment_id.replace('_', ' ').title(),
                    'status': 'running' if hash(equipment_id) % 3 != 0 else 'stopped',
                    'health_score': 92.0 + (hash(equipment_id) % 8),
                    'temperature': 45.0 + (hash(equipment_id) % 15),
                    'last_update': datetime.now().isoformat(),
                    'priority': 'high' if hash(equipment_id) % 4 == 0 else 'medium'
                }
            
            # Initialize critical alerts for mobile
            self.mobile_data['critical_alerts'] = [
                {
                    'id': 'CA-001',
                    'type': 'warning',
                    'title': 'High Temperature',
                    'equipment': 'Line 1 Conveyor',
                    'message': 'Temperature 65°C exceeds threshold',
                    'timestamp': datetime.now().isoformat(),
                    'priority': 'high',
                    'acknowledged': False
                },
                {
                    'id': 'CA-002',
                    'type': 'info',
                    'title': 'Maintenance Due',
                    'equipment': 'Quality Station',
                    'message': 'Scheduled maintenance in 2 hours',
                    'timestamp': (datetime.now() - timedelta(minutes=30)).isoformat(),
                    'priority': 'medium',
                    'acknowledged': False
                }
            ]
            
            # Initialize recent activities
            self.mobile_data['recent_activities'] = [
                {
                    'id': 'RA-001',
                    'action': 'Equipment Started',
                    'equipment': 'Line 2 Robot',
                    'operator': 'Mobile User',
                    'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat()
                },
                {
                    'id': 'RA-002',
                    'action': 'Alert Acknowledged',
                    'equipment': 'Packaging Unit',
                    'operator': 'Mobile User',
                    'timestamp': (datetime.now() - timedelta(minutes=15)).isoformat()
                }
            ]
            
            self.logger.info("Mobile interface data initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing mobile data: {e}")
    
    def _detect_device_type(self, request_obj) -> str:
        """Detect device type from request."""
        user_agent = request_obj.headers.get('User-Agent', '').lower()
        
        if 'tablet' in user_agent or 'ipad' in user_agent:
            return 'tablet'
        elif 'mobile' in user_agent or 'android' in user_agent or 'iphone' in user_agent:
            return 'mobile'
        else:
            return 'desktop'
    
    def _parse_device_info(self, user_agent: str) -> Dict[str, str]:
        """Parse device information from user agent."""
        user_agent_lower = user_agent.lower()
        
        device_info = {
            'type': 'unknown',
            'os': 'unknown',
            'browser': 'unknown'
        }
        
        # Detect device type
        if 'ipad' in user_agent_lower:
            device_info['type'] = 'tablet'
            device_info['os'] = 'iOS'
        elif 'tablet' in user_agent_lower or 'android' in user_agent_lower and 'mobile' not in user_agent_lower:
            device_info['type'] = 'tablet'
            device_info['os'] = 'Android'
        elif 'mobile' in user_agent_lower or 'iphone' in user_agent_lower:
            device_info['type'] = 'mobile'
            device_info['os'] = 'iOS' if 'iphone' in user_agent_lower else 'Android'
        else:
            device_info['type'] = 'desktop'
        
        # Detect browser
        if 'chrome' in user_agent_lower:
            device_info['browser'] = 'Chrome'
        elif 'safari' in user_agent_lower:
            device_info['browser'] = 'Safari'
        elif 'firefox' in user_agent_lower:
            device_info['browser'] = 'Firefox'
        
        return device_info
    
    def _execute_quick_action(self, action_type: str, action_data: Dict) -> Dict[str, Any]:
        """Execute quick mobile actions."""
        try:
            if action_type == 'emergency_stop':
                equipment_id = action_data.get('equipment_id', 'all')
                
                # Simulate emergency stop
                if equipment_id == 'all':
                    for eq_id in self.mobile_data['equipment_overview']:
                        self.mobile_data['equipment_overview'][eq_id]['status'] = 'emergency_stop'
                    message = 'All equipment emergency stopped'
                else:
                    if equipment_id in self.mobile_data['equipment_overview']:
                        self.mobile_data['equipment_overview'][equipment_id]['status'] = 'emergency_stop'
                        message = f'Equipment {equipment_id} emergency stopped'
                    else:
                        return {'success': False, 'error': 'Equipment not found'}
                
                # Add to recent activities
                self.mobile_data['recent_activities'].insert(0, {
                    'id': f'RA-{int(time.time())}',
                    'action': 'Emergency Stop',
                    'equipment': equipment_id,
                    'operator': 'Mobile User',
                    'timestamp': datetime.now().isoformat()
                })
                
                return {'success': True, 'message': message}
            
            elif action_type == 'acknowledge_alert':
                alert_id = action_data.get('alert_id')
                
                # Find and acknowledge alert
                for alert in self.mobile_data['critical_alerts']:
                    if alert['id'] == alert_id:
                        alert['acknowledged'] = True
                        alert['acknowledged_at'] = datetime.now().isoformat()
                        break
                
                with self.lock:
                    self.mobile_metrics['feature_usage']['alert_acknowledgment'] += 1
                
                return {'success': True, 'message': 'Alert acknowledged'}
            
            elif action_type == 'quick_status':
                # Return quick status for mobile dashboard
                return {
                    'success': True,
                    'status': {
                        'line_status': self.mobile_data['quick_metrics']['line_status'],
                        'active_alerts': len([a for a in self.mobile_data['critical_alerts'] if not a['acknowledged']]),
                        'equipment_running': len([eq for eq in self.mobile_data['equipment_overview'].values() if eq['status'] == 'running']),
                        'maintenance_pending': len([t for t in self.mobile_data['maintenance_tasks'] if t['status'] == 'pending'])
                    }
                }
            
            else:
                return {'success': False, 'error': f'Unknown action type: {action_type}'}
                
        except Exception as e:
            self.logger.error(f"Quick action execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _process_offline_sync(self, offline_data: Dict) -> Dict[str, Any]:
        """Process offline data synchronization."""
        try:
            sync_result = {
                'sync_timestamp': datetime.now().isoformat(),
                'processed_items': 0,
                'errors': [],
                'success': True
            }
            
            # Process offline actions
            actions = offline_data.get('actions', [])
            for action in actions:
                try:
                    action_type = action.get('type')
                    action_data = action.get('data', {})
                    
                    # Process action
                    result = self._execute_quick_action(action_type, action_data)
                    
                    if result.get('success'):
                        sync_result['processed_items'] += 1
                    else:
                        sync_result['errors'].append(f"Action {action_type}: {result.get('error')}")
                
                except Exception as e:
                    sync_result['errors'].append(f"Action processing error: {str(e)}")
            
            # Update sync metrics
            if sync_result['errors']:
                sync_result['success'] = False
            
            return sync_result
            
        except Exception as e:
            self.logger.error(f"Offline sync processing error: {e}")
            return {'success': False, 'error': str(e)}
    
    def start_mobile_updates(self):
        """Start mobile data update thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.update_thread = threading.Thread(target=self._mobile_update_worker, daemon=True)
        self.update_thread.start()
        
        self.logger.info("Mobile update thread started")
    
    def _mobile_update_worker(self):
        """Background worker for updating mobile data."""
        while self.is_running:
            try:
                # Update quick metrics
                metrics = self.mobile_data['quick_metrics']
                
                # Simulate realistic mobile-relevant updates
                metrics['current_output'] = max(0, min(100, metrics['current_output'] + random.uniform(-3, 5)))
                metrics['efficiency'] = max(75, min(95, metrics['efficiency'] + random.uniform(-0.5, 1.0)))
                
                # Update equipment temperatures
                for equipment in self.mobile_data['equipment_overview'].values():
                    temp_change = random.uniform(-2, 3)
                    equipment['temperature'] = max(20, min(75, equipment['temperature'] + temp_change))
                    equipment['last_update'] = datetime.now().isoformat()
                
                # Generate new alerts occasionally
                if random.random() < 0.05:  # 5% chance per update
                    self._generate_mobile_alert()
                
                # Update maintenance task progress
                for task in self.mobile_data['maintenance_tasks']:
                    if task['status'] == 'pending':
                        due_time = datetime.fromisoformat(task['due_date'])
                        if datetime.now() > due_time:
                            task['status'] = 'overdue'
                
                # Broadcast updates to mobile users
                if self.mobile_sessions:
                    self.socketio.emit('mobile_data_update', {
                        'quick_metrics': self.mobile_data['quick_metrics'],
                        'equipment_overview': self.mobile_data['equipment_overview'],
                        'maintenance_tasks': self.mobile_data['maintenance_tasks'],
                        'timestamp': datetime.now().isoformat()
                    }, room='mobile_users')
                
                # Sleep until next update
                time.sleep(self.update_interval_ms / 1000.0)
                
            except Exception as e:
                self.logger.error(f"Mobile update worker error: {e}")
                time.sleep(2.0)  # Wait before retrying
    
    def _generate_mobile_alert(self):
        """Generate new mobile alert."""
        alert_templates = [
            ('warning', 'Temperature Alert', 'Temperature exceeds safe limits'),
            ('info', 'Maintenance Reminder', 'Scheduled maintenance approaching'),
            ('error', 'Equipment Fault', 'Equipment sensor malfunction detected'),
            ('warning', 'Quality Issue', 'Quality parameters out of specification')
        ]
        
        alert_type, title_template, message_template = random.choice(alert_templates)
        equipment_id = random.choice(list(self.mobile_data['equipment_overview'].keys()))
        equipment_name = self.mobile_data['equipment_overview'][equipment_id]['name']
        
        new_alert = {
            'id': f'CA-{int(time.time())}-{random.randint(100, 999)}',
            'type': alert_type,
            'title': title_template,
            'equipment': equipment_name,
            'message': f'{equipment_name}: {message_template}',
            'timestamp': datetime.now().isoformat(),
            'priority': 'high' if alert_type == 'error' else 'medium',
            'acknowledged': False
        }
        
        # Add to beginning of alerts list
        self.mobile_data['critical_alerts'].insert(0, new_alert)
        
        # Keep only last 10 alerts for mobile
        if len(self.mobile_data['critical_alerts']) > 10:
            self.mobile_data['critical_alerts'] = self.mobile_data['critical_alerts'][:10]
        
        # Update alert count
        self.mobile_data['quick_metrics']['alert_count'] = len([
            a for a in self.mobile_data['critical_alerts'] if not a['acknowledged']
        ])
        
        # Broadcast new alert to mobile users
        self.socketio.emit('new_mobile_alert', new_alert, room='mobile_users')
        
        self.logger.info(f"Generated mobile alert: {title_template} for {equipment_name}")
    
    def run(self):
        """Run the mobile interface server."""
        try:
            # Start mobile updates
            self.start_mobile_updates()
            
            self.logger.info(f"Starting Mobile Interface on {self.host}:{self.port}")
            
            # Run the Flask app with SocketIO
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=self.debug,
                use_reloader=False
            )
            
        except Exception as e:
            self.logger.error(f"Error running mobile interface: {e}")
            raise
    
    def stop(self):
        """Stop the mobile interface."""
        self.is_running = False
        self.logger.info("Mobile Interface stopped")
    
    async def validate_mobile_interface(self) -> Dict[str, Any]:
        """Validate mobile interface functionality."""
        validation_results = {
            'component': 'MobileInterface',
            'validation_timestamp': datetime.now().isoformat(),
            'tests': {},
            'performance_metrics': {},
            'overall_status': 'unknown'
        }
        
        try:
            # Test 1: Mobile Data Structure
            required_sections = ['critical_alerts', 'equipment_overview', 'quick_metrics', 
                               'maintenance_tasks', 'recent_activities']
            missing_sections = [section for section in required_sections 
                              if section not in self.mobile_data]
            
            validation_results['tests']['mobile_data_structure'] = {
                'status': 'pass' if not missing_sections else 'fail',
                'details': f"All mobile sections present" if not missing_sections 
                          else f"Missing sections: {missing_sections}"
            }
            
            # Test 2: Quick Action Execution
            test_action = self._execute_quick_action('quick_status', {})
            
            validation_results['tests']['quick_actions'] = {
                'status': 'pass' if test_action['success'] else 'fail',
                'details': f"Quick action test: {test_action}"
            }
            
            # Test 3: Mobile Alert Generation
            initial_alert_count = len(self.mobile_data['critical_alerts'])
            self._generate_mobile_alert()
            new_alert_count = len(self.mobile_data['critical_alerts'])
            
            validation_results['tests']['mobile_alerts'] = {
                'status': 'pass' if new_alert_count > initial_alert_count else 'fail',
                'details': f"Mobile alert generated: {new_alert_count - initial_alert_count} new alerts"
            }
            
            # Test 4: Device Detection
            mock_request = type('Request', (), {
                'headers': {'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)'}
            })
            device_type = self._detect_device_type(mock_request)
            
            validation_results['tests']['device_detection'] = {
                'status': 'pass' if device_type == 'mobile' else 'fail',
                'details': f"Device detection: {device_type}"
            }
            
            # Test 5: Offline Sync Processing
            offline_data = {
                'actions': [
                    {'type': 'acknowledge_alert', 'data': {'alert_id': 'CA-001'}}
                ]
            }
            sync_result = self._process_offline_sync(offline_data)
            
            validation_results['tests']['offline_sync'] = {
                'status': 'pass' if sync_result['success'] else 'fail',
                'details': f"Offline sync: processed {sync_result['processed_items']} items"
            }
            
            # Performance metrics
            with self.lock:
                validation_results['performance_metrics'] = {
                    'mobile_metrics': self.mobile_metrics.copy(),
                    'active_sessions': len(self.mobile_sessions),
                    'alerts_count': len(self.mobile_data['critical_alerts']),
                    'equipment_count': len(self.mobile_data['equipment_overview']),
                    'maintenance_tasks_count': len(self.mobile_data['maintenance_tasks']),
                    'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
                }
            
            # Overall status
            passed_tests = sum(1 for test in validation_results['tests'].values() 
                             if test['status'] == 'pass')
            total_tests = len(validation_results['tests'])
            
            validation_results['overall_status'] = 'pass' if passed_tests == total_tests else 'fail'
            validation_results['test_summary'] = f"{passed_tests}/{total_tests} tests passed"
            
            self.logger.info(f"Mobile interface validation completed: {validation_results['test_summary']}")
            
        except Exception as e:
            validation_results['overall_status'] = 'error'
            validation_results['error'] = str(e)
            self.logger.error(f"Mobile interface validation failed: {e}")
        
        return validation_results


# Utility function to create mobile interface
def create_mobile_interface(config: Optional[Dict] = None) -> MobileInterface:
    """Create and configure a mobile interface instance."""
    return MobileInterface(config)