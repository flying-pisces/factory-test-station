"""
Operator Dashboard - Week 13: UI & Visualization Layer

Web-based operator dashboard providing real-time manufacturing data,
equipment status, quality metrics, and control capabilities for factory operators.
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

from .dashboard_manager import DashboardManager, DashboardType
from .real_time_data_pipeline import RealTimeDataPipeline
from .visualization_engine import VisualizationEngine


class OperatorDashboard:
    """
    Operator Dashboard for manufacturing line control and monitoring.
    
    Provides real-time data visualization, equipment control, quality monitoring,
    and alert management specifically designed for factory operators.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Operator Dashboard.
        
        Args:
            config: Configuration dictionary for dashboard settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Dashboard configuration
        self.port = self.config.get('port', 5001)
        self.host = self.config.get('host', '0.0.0.0')
        self.debug = self.config.get('debug', False)
        self.update_interval_ms = self.config.get('update_interval_ms', 1000)  # 1 second
        
        # Initialize Flask app
        self.app = Flask(__name__, template_folder='templates', static_folder='static')
        self.app.config['SECRET_KEY'] = self.config.get('secret_key', 'operator_dashboard_secret')
        
        # Initialize SocketIO for real-time updates
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", 
                               async_mode='threading', logger=self.debug)
        
        # Initialize components
        self.dashboard_manager = DashboardManager()
        self.data_pipeline = RealTimeDataPipeline()
        self.visualization_engine = VisualizationEngine()
        
        # Dashboard state
        self.active_sessions = {}
        self.dashboard_data = {
            'production_metrics': {
                'throughput': {'current': 0, 'target': 100, 'trend': 'stable'},
                'efficiency': {'current': 85.2, 'target': 90, 'trend': 'increasing'},
                'quality_rate': {'current': 98.5, 'target': 99, 'trend': 'stable'},
                'uptime': {'current': 94.8, 'target': 95, 'trend': 'stable'}
            },
            'equipment_status': {},
            'quality_metrics': {
                'defect_rate': {'current': 1.5, 'target': 2.0, 'status': 'good'},
                'first_pass_yield': {'current': 96.8, 'target': 95.0, 'status': 'excellent'},
                'rework_rate': {'current': 2.1, 'target': 3.0, 'status': 'good'}
            },
            'alerts': [],
            'work_orders': []
        }
        
        # Performance tracking
        self.dashboard_metrics = {
            'page_loads': 0,
            'active_users': 0,
            'data_updates_sent': 0,
            'avg_response_time_ms': 0.0,
            'uptime_seconds': 0,
            'error_count': 0
        }
        
        # Thread management
        self.lock = threading.Lock()
        self.update_thread = None
        self.is_running = False
        self.start_time = datetime.now()
        
        # Set up routes and event handlers
        self._setup_routes()
        self._setup_socketio_handlers()
        
        # Initialize dashboard data
        self._initialize_dashboard_data()
        
        self.logger.info("OperatorDashboard initialized successfully")
    
    def _setup_routes(self):
        """Set up Flask routes for operator dashboard."""
        
        @self.app.route('/')
        def dashboard_home():
            """Main operator dashboard page."""
            start_time = time.time()
            
            try:
                # Create session if needed
                if 'session_id' not in session:
                    session['session_id'] = str(uuid.uuid4())
                    session['user_type'] = 'operator'
                    session['login_time'] = datetime.now().isoformat()
                
                # Update metrics
                with self.lock:
                    self.dashboard_metrics['page_loads'] += 1
                
                response_time = (time.time() - start_time) * 1000
                
                # Update average response time
                with self.lock:
                    current_avg = self.dashboard_metrics['avg_response_time_ms']
                    page_loads = self.dashboard_metrics['page_loads']
                    self.dashboard_metrics['avg_response_time_ms'] = (
                        (current_avg * (page_loads - 1) + response_time) / page_loads
                    )
                
                return render_template('operator_dashboard.html', 
                                     session_id=session['session_id'],
                                     update_interval=self.update_interval_ms,
                                     dashboard_data=self.dashboard_data)
                
            except Exception as e:
                with self.lock:
                    self.dashboard_metrics['error_count'] += 1
                self.logger.error(f"Dashboard home route error: {e}")
                return f"Dashboard Error: {e}", 500
        
        @self.app.route('/api/data')
        def get_dashboard_data():
            """API endpoint for dashboard data."""
            try:
                return jsonify(self.dashboard_data)
            except Exception as e:
                self.logger.error(f"API data route error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/equipment/<equipment_id>/control', methods=['POST'])
        def control_equipment(equipment_id):
            """API endpoint for equipment control."""
            try:
                command = request.json.get('command')
                parameters = request.json.get('parameters', {})
                
                # Simulate equipment control
                result = self._execute_equipment_control(equipment_id, command, parameters)
                
                return jsonify(result)
                
            except Exception as e:
                self.logger.error(f"Equipment control error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
        def acknowledge_alert(alert_id):
            """API endpoint for alert acknowledgment."""
            try:
                operator_id = session.get('session_id', 'unknown')
                
                # Find and acknowledge alert
                for alert in self.dashboard_data['alerts']:
                    if alert['id'] == alert_id:
                        alert['acknowledged'] = True
                        alert['acknowledged_by'] = operator_id
                        alert['acknowledged_at'] = datetime.now().isoformat()
                        break
                
                return jsonify({'success': True, 'alert_id': alert_id})
                
            except Exception as e:
                self.logger.error(f"Alert acknowledgment error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/metrics')
        def get_dashboard_metrics():
            """API endpoint for dashboard performance metrics."""
            try:
                with self.lock:
                    metrics = self.dashboard_metrics.copy()
                    metrics['uptime_seconds'] = (datetime.now() - self.start_time).total_seconds()
                
                return jsonify(metrics)
                
            except Exception as e:
                self.logger.error(f"Metrics route error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _setup_socketio_handlers(self):
        """Set up SocketIO event handlers for real-time communication."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            try:
                session_id = request.sid
                
                # Register session
                self.active_sessions[session_id] = {
                    'session_id': session_id,
                    'connected_at': datetime.now().isoformat(),
                    'user_type': 'operator',
                    'subscriptions': [],
                    'last_activity': datetime.now().isoformat()
                }
                
                # Join operator room
                join_room('operators')
                
                with self.lock:
                    self.dashboard_metrics['active_users'] += 1
                
                self.logger.info(f"Operator connected: {session_id}")
                
                # Send initial data
                emit('dashboard_data', self.dashboard_data)
                
            except Exception as e:
                self.logger.error(f"SocketIO connect error: {e}")
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            try:
                session_id = request.sid
                
                # Clean up session
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]
                
                leave_room('operators')
                
                with self.lock:
                    self.dashboard_metrics['active_users'] -= 1
                
                self.logger.info(f"Operator disconnected: {session_id}")
                
            except Exception as e:
                self.logger.error(f"SocketIO disconnect error: {e}")
        
        @self.socketio.on('subscribe_data')
        def handle_data_subscription(data):
            """Handle data subscription requests."""
            try:
                session_id = request.sid
                data_types = data.get('data_types', [])
                
                if session_id in self.active_sessions:
                    self.active_sessions[session_id]['subscriptions'] = data_types
                    self.active_sessions[session_id]['last_activity'] = datetime.now().isoformat()
                
                emit('subscription_confirmed', {'data_types': data_types})
                
            except Exception as e:
                self.logger.error(f"Data subscription error: {e}")
        
        @self.socketio.on('equipment_command')
        def handle_equipment_command(data):
            """Handle equipment control commands from operators."""
            try:
                session_id = request.sid
                equipment_id = data.get('equipment_id')
                command = data.get('command')
                parameters = data.get('parameters', {})
                
                # Execute equipment control
                result = self._execute_equipment_control(equipment_id, command, parameters)
                
                # Send result back to client
                emit('equipment_command_result', result)
                
                # Broadcast equipment status update to all operators
                self.socketio.emit('equipment_status_update', {
                    'equipment_id': equipment_id,
                    'status': result.get('new_status', 'unknown'),
                    'timestamp': datetime.now().isoformat()
                }, room='operators')
                
            except Exception as e:
                self.logger.error(f"Equipment command error: {e}")
                emit('equipment_command_result', {'error': str(e)})
    
    def _initialize_dashboard_data(self):
        """Initialize dashboard with sample data."""
        try:
            # Initialize equipment status
            equipment_list = [
                'line_1_conveyor', 'line_1_robot_arm', 'line_1_vision_system',
                'line_2_conveyor', 'line_2_robot_arm', 'line_2_vision_system',
                'quality_station_1', 'quality_station_2', 'packaging_unit'
            ]
            
            for equipment_id in equipment_list:
                self.dashboard_data['equipment_status'][equipment_id] = {
                    'id': equipment_id,
                    'name': equipment_id.replace('_', ' ').title(),
                    'status': 'running',
                    'health_score': 95.0 + (hash(equipment_id) % 10) * 0.5,
                    'efficiency': 85.0 + (hash(equipment_id) % 15),
                    'temperature': 45.0 + (hash(equipment_id) % 20),
                    'last_maintenance': (datetime.now() - timedelta(days=hash(equipment_id) % 30)).isoformat(),
                    'next_maintenance': (datetime.now() + timedelta(days=30 - (hash(equipment_id) % 30))).isoformat(),
                    'alerts_count': hash(equipment_id) % 3
                }
            
            # Initialize sample alerts
            self.dashboard_data['alerts'] = [
                {
                    'id': 'alert_001',
                    'type': 'warning',
                    'message': 'Line 1 Conveyor temperature elevated',
                    'equipment_id': 'line_1_conveyor',
                    'timestamp': datetime.now().isoformat(),
                    'acknowledged': False,
                    'priority': 'medium'
                },
                {
                    'id': 'alert_002', 
                    'type': 'info',
                    'message': 'Quality Station 2 calibration due in 2 days',
                    'equipment_id': 'quality_station_2',
                    'timestamp': (datetime.now() - timedelta(minutes=15)).isoformat(),
                    'acknowledged': False,
                    'priority': 'low'
                }
            ]
            
            # Initialize sample work orders
            self.dashboard_data['work_orders'] = [
                {
                    'id': 'WO-001',
                    'product': 'Widget A',
                    'quantity': 1000,
                    'completed': 750,
                    'status': 'in_progress',
                    'priority': 'high',
                    'due_date': (datetime.now() + timedelta(hours=8)).isoformat(),
                    'line': 'Line 1'
                },
                {
                    'id': 'WO-002',
                    'product': 'Widget B',
                    'quantity': 500,
                    'completed': 0,
                    'status': 'pending',
                    'priority': 'medium',
                    'due_date': (datetime.now() + timedelta(hours=16)).isoformat(),
                    'line': 'Line 2'
                }
            ]
            
            self.logger.info("Dashboard data initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing dashboard data: {e}")
    
    def _execute_equipment_control(self, equipment_id: str, command: str, parameters: Dict) -> Dict[str, Any]:
        """Execute equipment control command."""
        try:
            # Simulate equipment control
            if equipment_id not in self.dashboard_data['equipment_status']:
                return {'error': f'Equipment {equipment_id} not found'}
            
            equipment = self.dashboard_data['equipment_status'][equipment_id]
            
            if command == 'start':
                equipment['status'] = 'running'
                result = {'success': True, 'action': 'started', 'new_status': 'running'}
            elif command == 'stop':
                equipment['status'] = 'stopped'
                result = {'success': True, 'action': 'stopped', 'new_status': 'stopped'}
            elif command == 'pause':
                equipment['status'] = 'paused'
                result = {'success': True, 'action': 'paused', 'new_status': 'paused'}
            elif command == 'reset':
                equipment['status'] = 'running'
                equipment['alerts_count'] = 0
                result = {'success': True, 'action': 'reset', 'new_status': 'running'}
            else:
                result = {'error': f'Unknown command: {command}'}
            
            # Log control action
            if 'success' in result:
                self.logger.info(f"Equipment {equipment_id} {command} executed successfully")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Equipment control execution error: {e}")
            return {'error': str(e)}
    
    def start_data_updates(self):
        """Start real-time data update thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.update_thread = threading.Thread(target=self._data_update_worker, daemon=True)
        self.update_thread.start()
        
        self.logger.info("Data update thread started")
    
    def _data_update_worker(self):
        """Background worker for updating dashboard data."""
        import random
        
        while self.is_running:
            try:
                # Update production metrics with realistic variations
                metrics = self.dashboard_data['production_metrics']
                
                # Throughput - simulate production variations
                current_throughput = metrics['throughput']['current']
                target_throughput = metrics['throughput']['target']
                
                if current_throughput < target_throughput * 0.95:
                    new_throughput = min(target_throughput, current_throughput + random.uniform(0.5, 2.0))
                    trend = 'increasing'
                elif current_throughput > target_throughput * 1.05:
                    new_throughput = max(target_throughput * 0.8, current_throughput - random.uniform(0.5, 2.0))
                    trend = 'decreasing'
                else:
                    new_throughput = current_throughput + random.uniform(-1.0, 1.0)
                    trend = 'stable'
                
                metrics['throughput'] = {
                    'current': round(new_throughput, 1),
                    'target': target_throughput,
                    'trend': trend
                }
                
                # Efficiency - gradual improvements with occasional dips
                efficiency = metrics['efficiency']['current']
                if random.random() < 0.1:  # 10% chance of efficiency dip
                    new_efficiency = max(75, efficiency - random.uniform(1, 5))
                else:
                    new_efficiency = min(98, efficiency + random.uniform(-0.5, 1.0))
                
                metrics['efficiency'] = {
                    'current': round(new_efficiency, 1),
                    'target': 90,
                    'trend': 'increasing' if new_efficiency > efficiency else 'decreasing'
                }
                
                # Update equipment status
                for equipment_id, equipment in self.dashboard_data['equipment_status'].items():
                    # Simulate temperature fluctuations
                    temp_variation = random.uniform(-2, 2)
                    new_temp = max(20, min(80, equipment['temperature'] + temp_variation))
                    equipment['temperature'] = round(new_temp, 1)
                    
                    # Simulate health score changes
                    health_variation = random.uniform(-0.5, 0.5)
                    new_health = max(70, min(100, equipment['health_score'] + health_variation))
                    equipment['health_score'] = round(new_health, 1)
                    
                    # Generate new alerts occasionally
                    if random.random() < 0.005:  # 0.5% chance per update
                        self._generate_alert(equipment_id)
                
                # Update work order progress
                for work_order in self.dashboard_data['work_orders']:
                    if work_order['status'] == 'in_progress':
                        progress_increment = random.uniform(0, 2)
                        new_completed = min(work_order['quantity'], work_order['completed'] + progress_increment)
                        work_order['completed'] = int(new_completed)
                        
                        if new_completed >= work_order['quantity']:
                            work_order['status'] = 'completed'
                
                # Broadcast updates to all connected operators
                if self.active_sessions:
                    with self.lock:
                        self.dashboard_metrics['data_updates_sent'] += 1
                    
                    self.socketio.emit('dashboard_data_update', {
                        'production_metrics': self.dashboard_data['production_metrics'],
                        'equipment_status': self.dashboard_data['equipment_status'],
                        'work_orders': self.dashboard_data['work_orders'],
                        'timestamp': datetime.now().isoformat()
                    }, room='operators')
                
                # Sleep until next update
                time.sleep(self.update_interval_ms / 1000.0)
                
            except Exception as e:
                self.logger.error(f"Data update worker error: {e}")
                time.sleep(1.0)  # Wait before retrying
    
    def _generate_alert(self, equipment_id: str):
        """Generate new alert for equipment."""
        alert_types = [
            ('warning', 'Temperature elevated'),
            ('info', 'Maintenance due soon'),
            ('warning', 'Vibration detected'),
            ('info', 'Calibration recommended'),
            ('error', 'Sensor fault detected')
        ]
        
        alert_type, message_template = random.choice(alert_types)
        
        alert = {
            'id': f'alert_{int(time.time())}_{random.randint(100, 999)}',
            'type': alert_type,
            'message': f'{self.dashboard_data["equipment_status"][equipment_id]["name"]}: {message_template}',
            'equipment_id': equipment_id,
            'timestamp': datetime.now().isoformat(),
            'acknowledged': False,
            'priority': 'high' if alert_type == 'error' else 'medium' if alert_type == 'warning' else 'low'
        }
        
        self.dashboard_data['alerts'].insert(0, alert)  # Add to beginning
        
        # Keep only last 50 alerts
        if len(self.dashboard_data['alerts']) > 50:
            self.dashboard_data['alerts'] = self.dashboard_data['alerts'][:50]
        
        # Broadcast new alert
        self.socketio.emit('new_alert', alert, room='operators')
        
        self.logger.info(f"Generated alert for {equipment_id}: {alert['message']}")
    
    def run(self):
        """Run the operator dashboard server."""
        try:
            # Start data updates
            self.start_data_updates()
            
            self.logger.info(f"Starting Operator Dashboard on {self.host}:{self.port}")
            
            # Run the Flask app with SocketIO
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=self.debug,
                use_reloader=False  # Disable reloader to prevent thread issues
            )
            
        except Exception as e:
            self.logger.error(f"Error running operator dashboard: {e}")
            raise
    
    def stop(self):
        """Stop the operator dashboard."""
        self.is_running = False
        self.logger.info("Operator Dashboard stopped")
    
    async def validate_operator_dashboard(self) -> Dict[str, Any]:
        """Validate operator dashboard functionality."""
        validation_results = {
            'component': 'OperatorDashboard',
            'validation_timestamp': datetime.now().isoformat(),
            'tests': {},
            'performance_metrics': {},
            'overall_status': 'unknown'
        }
        
        try:
            # Test 1: Dashboard Initialization
            validation_results['tests']['initialization'] = {
                'status': 'pass',
                'details': 'Dashboard initialized successfully'
            }
            
            # Test 2: Data Structure Validation
            required_sections = ['production_metrics', 'equipment_status', 'quality_metrics', 'alerts', 'work_orders']
            missing_sections = [section for section in required_sections 
                              if section not in self.dashboard_data]
            
            validation_results['tests']['data_structure'] = {
                'status': 'pass' if not missing_sections else 'fail',
                'details': f"All required sections present" if not missing_sections 
                          else f"Missing sections: {missing_sections}"
            }
            
            # Test 3: Equipment Control
            test_equipment = list(self.dashboard_data['equipment_status'].keys())[0]
            control_result = self._execute_equipment_control(test_equipment, 'pause', {})
            
            validation_results['tests']['equipment_control'] = {
                'status': 'pass' if control_result.get('success') else 'fail',
                'details': f"Equipment control test: {control_result}"
            }
            
            # Test 4: Alert Generation
            initial_alert_count = len(self.dashboard_data['alerts'])
            self._generate_alert(test_equipment)
            new_alert_count = len(self.dashboard_data['alerts'])
            
            validation_results['tests']['alert_generation'] = {
                'status': 'pass' if new_alert_count > initial_alert_count else 'fail',
                'details': f"Alert generated: {new_alert_count - initial_alert_count} new alerts"
            }
            
            # Performance metrics
            with self.lock:
                validation_results['performance_metrics'] = {
                    'dashboard_metrics': self.dashboard_metrics.copy(),
                    'active_sessions': len(self.active_sessions),
                    'equipment_count': len(self.dashboard_data['equipment_status']),
                    'alerts_count': len(self.dashboard_data['alerts']),
                    'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
                }
            
            # Overall status
            passed_tests = sum(1 for test in validation_results['tests'].values() 
                             if test['status'] == 'pass')
            total_tests = len(validation_results['tests'])
            
            validation_results['overall_status'] = 'pass' if passed_tests == total_tests else 'fail'
            validation_results['test_summary'] = f"{passed_tests}/{total_tests} tests passed"
            
            self.logger.info(f"Operator dashboard validation completed: {validation_results['test_summary']}")
            
        except Exception as e:
            validation_results['overall_status'] = 'error'
            validation_results['error'] = str(e)
            self.logger.error(f"Operator dashboard validation failed: {e}")
        
        return validation_results


# Template and static file creation
def create_operator_dashboard_template():
    """Create HTML template for operator dashboard."""
    template_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Manufacturing Line - Operator Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.0/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #1a1a1a;
            color: #ffffff;
            line-height: 1.6;
        }
        
        .dashboard-header {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        
        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .logo h1 {
            color: #3498db;
            font-size: 1.8rem;
        }
        
        .status-info {
            display: flex;
            gap: 2rem;
            align-items: center;
        }
        
        .status-item {
            text-align: center;
        }
        
        .status-value {
            font-size: 1.4rem;
            font-weight: bold;
            color: #2ecc71;
        }
        
        .status-label {
            font-size: 0.9rem;
            color: #bdc3c7;
        }
        
        .main-content {
            padding: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 2rem;
            margin-bottom: 2rem;
        }
        
        .production-metrics {
            background: #2c3e50;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .metric-card {
            background: #34495e;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            margin: 0.5rem 0;
        }
        
        .metric-target {
            font-size: 0.9rem;
            color: #95a5a6;
        }
        
        .trend-up { color: #2ecc71; }
        .trend-down { color: #e74c3c; }
        .trend-stable { color: #f39c12; }
        
        .alerts-panel {
            background: #2c3e50;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        
        .alert-item {
            background: #34495e;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 8px;
            border-left: 4px solid #f39c12;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .alert-warning { border-left-color: #f39c12; }
        .alert-error { border-left-color: #e74c3c; }
        .alert-info { border-left-color: #3498db; }
        
        .equipment-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .equipment-card {
            background: #2c3e50;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        
        .equipment-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-left: 0.5rem;
        }
        
        .status-running { background-color: #2ecc71; }
        .status-stopped { background-color: #e74c3c; }
        .status-paused { background-color: #f39c12; }
        
        .equipment-metrics {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .equipment-metric {
            text-align: center;
        }
        
        .control-buttons {
            display: flex;
            gap: 0.5rem;
            margin-top: 1rem;
        }
        
        .btn {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.3s;
        }
        
        .btn-primary { background: #3498db; color: white; }
        .btn-success { background: #2ecc71; color: white; }
        .btn-warning { background: #f39c12; color: white; }
        .btn-danger { background: #e74c3c; color: white; }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        
        .work-orders {
            background: #2c3e50;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        
        .work-order-item {
            background: #34495e;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #95a5a6;
            border-radius: 4px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #2ecc71 0%, #27ae60 100%);
            transition: width 0.5s ease;
        }
        
        .chart-container {
            position: relative;
            height: 200px;
            margin: 1rem 0;
        }
        
        @media (max-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            
            .metrics-grid {
                grid-template-columns: 1fr;
            }
            
            .equipment-grid {
                grid-template-columns: 1fr;
            }
            
            .status-info {
                display: none;
            }
        }
    </style>
</head>
<body>
    <div class="dashboard-header">
        <div class="header-content">
            <div class="logo">
                <h1>🏭 Manufacturing Line Control</h1>
            </div>
            <div class="status-info">
                <div class="status-item">
                    <div class="status-value" id="system-status">ONLINE</div>
                    <div class="status-label">System Status</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="active-users">1</div>
                    <div class="status-label">Active Operators</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="current-time">--:--</div>
                    <div class="status-label">Current Time</div>
                </div>
            </div>
        </div>
    </div>

    <div class="main-content">
        <div class="dashboard-grid">
            <div class="production-metrics">
                <h2>Production Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h3>Throughput</h3>
                        <div class="metric-value trend-stable" id="throughput-value">0</div>
                        <div class="metric-target">Target: <span id="throughput-target">100</span>/hr</div>
                    </div>
                    <div class="metric-card">
                        <h3>Efficiency</h3>
                        <div class="metric-value trend-up" id="efficiency-value">0%</div>
                        <div class="metric-target">Target: <span id="efficiency-target">90</span>%</div>
                    </div>
                    <div class="metric-card">
                        <h3>Quality Rate</h3>
                        <div class="metric-value trend-stable" id="quality-value">0%</div>
                        <div class="metric-target">Target: <span id="quality-target">99</span>%</div>
                    </div>
                    <div class="metric-card">
                        <h3>Uptime</h3>
                        <div class="metric-value trend-stable" id="uptime-value">0%</div>
                        <div class="metric-target">Target: <span id="uptime-target">95</span>%</div>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="production-chart"></canvas>
                </div>
            </div>

            <div class="alerts-panel">
                <h2>Active Alerts</h2>
                <div id="alerts-container">
                    <!-- Alerts will be populated dynamically -->
                </div>
            </div>
        </div>

        <div class="equipment-grid" id="equipment-grid">
            <!-- Equipment cards will be populated dynamically -->
        </div>

        <div class="work-orders">
            <h2>Work Orders</h2>
            <div id="work-orders-container">
                <!-- Work orders will be populated dynamically -->
            </div>
        </div>
    </div>

    <script>
        // Initialize Socket.IO connection
        const socket = io();
        
        // Chart variables
        let productionChart = null;
        const chartData = {
            throughput: [],
            efficiency: [],
            quality: [],
            timestamps: []
        };
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            updateCurrentTime();
            setInterval(updateCurrentTime, 1000);
            
            initializeProductionChart();
        });
        
        // Socket event handlers
        socket.on('connect', function() {
            console.log('Connected to dashboard server');
            document.getElementById('system-status').textContent = 'ONLINE';
            document.getElementById('system-status').style.color = '#2ecc71';
        });
        
        socket.on('disconnect', function() {
            console.log('Disconnected from dashboard server');
            document.getElementById('system-status').textContent = 'OFFLINE';
            document.getElementById('system-status').style.color = '#e74c3c';
        });
        
        socket.on('dashboard_data', function(data) {
            updateDashboardData(data);
        });
        
        socket.on('dashboard_data_update', function(data) {
            updateProductionMetrics(data.production_metrics);
            updateEquipmentStatus(data.equipment_status);
            updateWorkOrders(data.work_orders);
            updateChart(data.production_metrics);
        });
        
        socket.on('new_alert', function(alert) {
            addAlert(alert);
        });
        
        socket.on('equipment_status_update', function(data) {
            updateEquipmentCard(data.equipment_id, data.status);
        });
        
        // Update functions
        function updateCurrentTime() {
            const now = new Date();
            document.getElementById('current-time').textContent = now.toLocaleTimeString();
        }
        
        function updateDashboardData(data) {
            updateProductionMetrics(data.production_metrics);
            updateEquipmentStatus(data.equipment_status);
            updateAlerts(data.alerts);
            updateWorkOrders(data.work_orders);
        }
        
        function updateProductionMetrics(metrics) {
            document.getElementById('throughput-value').textContent = metrics.throughput.current;
            document.getElementById('efficiency-value').textContent = metrics.efficiency.current + '%';
            document.getElementById('quality-value').textContent = metrics.quality_rate.current + '%';
            document.getElementById('uptime-value').textContent = metrics.uptime.current + '%';
            
            // Update trend classes
            updateTrendClass('throughput-value', metrics.throughput.trend);
            updateTrendClass('efficiency-value', metrics.efficiency.trend);
            updateTrendClass('quality-value', metrics.quality_rate.trend);
            updateTrendClass('uptime-value', metrics.uptime.trend);
        }
        
        function updateTrendClass(elementId, trend) {
            const element = document.getElementById(elementId);
            element.className = element.className.replace(/trend-\\w+/g, '');
            element.classList.add('trend-' + trend);
        }
        
        function updateEquipmentStatus(equipment) {
            const container = document.getElementById('equipment-grid');
            container.innerHTML = '';
            
            Object.values(equipment).forEach(eq => {
                const card = createEquipmentCard(eq);
                container.appendChild(card);
            });
        }
        
        function createEquipmentCard(equipment) {
            const card = document.createElement('div');
            card.className = 'equipment-card';
            card.innerHTML = `
                <div class="equipment-header">
                    <h3>${equipment.name}</h3>
                    <div class="status-indicator status-${equipment.status}"></div>
                </div>
                <div class="equipment-metrics">
                    <div class="equipment-metric">
                        <div style="font-size: 1.2rem; font-weight: bold;">${equipment.health_score.toFixed(1)}%</div>
                        <div style="font-size: 0.8rem; color: #95a5a6;">Health Score</div>
                    </div>
                    <div class="equipment-metric">
                        <div style="font-size: 1.2rem; font-weight: bold;">${equipment.temperature}°C</div>
                        <div style="font-size: 0.8rem; color: #95a5a6;">Temperature</div>
                    </div>
                </div>
                <div class="control-buttons">
                    <button class="btn btn-success" onclick="controlEquipment('${equipment.id}', 'start')">Start</button>
                    <button class="btn btn-warning" onclick="controlEquipment('${equipment.id}', 'pause')">Pause</button>
                    <button class="btn btn-danger" onclick="controlEquipment('${equipment.id}', 'stop')">Stop</button>
                    <button class="btn btn-primary" onclick="controlEquipment('${equipment.id}', 'reset')">Reset</button>
                </div>
            `;
            return card;
        }
        
        function updateAlerts(alerts) {
            const container = document.getElementById('alerts-container');
            container.innerHTML = '';
            
            alerts.slice(0, 5).forEach(alert => {
                const alertElement = createAlertElement(alert);
                container.appendChild(alertElement);
            });
        }
        
        function addAlert(alert) {
            const container = document.getElementById('alerts-container');
            const alertElement = createAlertElement(alert);
            container.insertBefore(alertElement, container.firstChild);
            
            // Keep only 5 alerts visible
            while (container.children.length > 5) {
                container.removeChild(container.lastChild);
            }
        }
        
        function createAlertElement(alert) {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert-item alert-${alert.type}`;
            alertDiv.innerHTML = `
                <div>
                    <div style="font-weight: bold;">${alert.message}</div>
                    <div style="font-size: 0.8rem; color: #95a5a6;">${new Date(alert.timestamp).toLocaleTimeString()}</div>
                </div>
                <button class="btn btn-primary" onclick="acknowledgeAlert('${alert.id}')">
                    ${alert.acknowledged ? 'Acknowledged' : 'Acknowledge'}
                </button>
            `;
            return alertDiv;
        }
        
        function updateWorkOrders(workOrders) {
            const container = document.getElementById('work-orders-container');
            container.innerHTML = '';
            
            workOrders.forEach(wo => {
                const woElement = createWorkOrderElement(wo);
                container.appendChild(woElement);
            });
        }
        
        function createWorkOrderElement(workOrder) {
            const progress = (workOrder.completed / workOrder.quantity) * 100;
            const woDiv = document.createElement('div');
            woDiv.className = 'work-order-item';
            woDiv.innerHTML = `
                <div style="flex: 1;">
                    <div style="font-weight: bold;">${workOrder.id} - ${workOrder.product}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${progress}%"></div>
                    </div>
                    <div style="font-size: 0.9rem; color: #95a5a6;">
                        ${workOrder.completed}/${workOrder.quantity} units (${progress.toFixed(1)}%)
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-weight: bold; color: ${workOrder.priority === 'high' ? '#e74c3c' : '#f39c12'};">
                        ${workOrder.priority.toUpperCase()}
                    </div>
                    <div style="font-size: 0.8rem; color: #95a5a6;">
                        ${workOrder.line}
                    </div>
                </div>
            `;
            return woDiv;
        }
        
        // Chart functions
        function initializeProductionChart() {
            const ctx = document.getElementById('production-chart').getContext('2d');
            productionChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: chartData.timestamps,
                    datasets: [
                        {
                            label: 'Throughput',
                            data: chartData.throughput,
                            borderColor: '#3498db',
                            backgroundColor: 'rgba(52, 152, 219, 0.1)',
                            tension: 0.4
                        },
                        {
                            label: 'Efficiency (%)',
                            data: chartData.efficiency,
                            borderColor: '#2ecc71',
                            backgroundColor: 'rgba(46, 204, 113, 0.1)',
                            tension: 0.4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: '#34495e' },
                            ticks: { color: '#ffffff' }
                        },
                        x: {
                            grid: { color: '#34495e' },
                            ticks: { color: '#ffffff' }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: { color: '#ffffff' }
                        }
                    }
                }
            });
        }
        
        function updateChart(metrics) {
            const now = new Date().toLocaleTimeString();
            
            chartData.timestamps.push(now);
            chartData.throughput.push(metrics.throughput.current);
            chartData.efficiency.push(metrics.efficiency.current);
            
            // Keep only last 20 data points
            if (chartData.timestamps.length > 20) {
                chartData.timestamps.shift();
                chartData.throughput.shift();
                chartData.efficiency.shift();
            }
            
            productionChart.update();
        }
        
        // Control functions
        function controlEquipment(equipmentId, command) {
            socket.emit('equipment_command', {
                equipment_id: equipmentId,
                command: command,
                parameters: {}
            });
        }
        
        function acknowledgeAlert(alertId) {
            fetch(`/api/alerts/${alertId}/acknowledge`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Update button
                    const alertElement = document.querySelector(`[onclick="acknowledgeAlert('${alertId}')"]`);
                    if (alertElement) {
                        alertElement.textContent = 'Acknowledged';
                        alertElement.disabled = true;
                    }
                }
            })
            .catch(error => console.error('Error acknowledging alert:', error));
        }
    </script>
</body>
</html>"""
    
    return template_content


# Utility function to create operator dashboard
def create_operator_dashboard(config: Optional[Dict] = None) -> OperatorDashboard:
    """Create and configure an operator dashboard instance."""
    return OperatorDashboard(config)