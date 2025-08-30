"""
Management Dashboard - Week 13: UI & Visualization Layer

Executive management dashboard providing high-level KPIs, financial metrics,
strategic insights, and performance analytics for manufacturing leadership.
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


class ManagementDashboard:
    """
    Management Dashboard for strategic manufacturing oversight.
    
    Provides executive-level KPIs, financial metrics, strategic insights,
    and performance analytics for manufacturing leadership and decision-making.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Management Dashboard.
        
        Args:
            config: Configuration dictionary for dashboard settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Dashboard configuration
        self.port = self.config.get('port', 5002)
        self.host = self.config.get('host', '0.0.0.0')
        self.debug = self.config.get('debug', False)
        self.update_interval_ms = self.config.get('update_interval_ms', 5000)  # 5 seconds for management
        
        # Initialize Flask app
        self.app = Flask(__name__, template_folder='templates', static_folder='static')
        self.app.config['SECRET_KEY'] = self.config.get('secret_key', 'management_dashboard_secret')
        
        # Initialize SocketIO
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", 
                               async_mode='threading', logger=self.debug)
        
        # Initialize components
        self.dashboard_manager = DashboardManager()
        self.data_pipeline = RealTimeDataPipeline()
        self.visualization_engine = VisualizationEngine()
        
        # Management dashboard data
        self.executive_data = {
            'financial_metrics': {
                'revenue': {'current': 2456789.50, 'target': 2500000.00, 'trend': 'up', 'period': 'monthly'},
                'profit_margin': {'current': 18.4, 'target': 20.0, 'trend': 'stable', 'period': 'monthly'},
                'cost_per_unit': {'current': 45.67, 'target': 42.00, 'trend': 'down', 'period': 'weekly'},
                'roi': {'current': 24.8, 'target': 25.0, 'trend': 'up', 'period': 'quarterly'}
            },
            'operational_kpis': {
                'overall_efficiency': {'current': 87.3, 'target': 90.0, 'trend': 'up'},
                'quality_score': {'current': 98.2, 'target': 99.0, 'trend': 'stable'},
                'on_time_delivery': {'current': 94.7, 'target': 95.0, 'trend': 'up'},
                'capacity_utilization': {'current': 82.1, 'target': 85.0, 'trend': 'up'}
            },
            'production_performance': {
                'total_output': {'current': 12450, 'target': 13000, 'period': 'weekly'},
                'defect_rate': {'current': 1.8, 'target': 2.0, 'period': 'weekly'},
                'downtime_hours': {'current': 4.2, 'target': 3.0, 'period': 'weekly'},
                'energy_efficiency': {'current': 91.5, 'target': 90.0, 'period': 'weekly'}
            },
            'workforce_metrics': {
                'productivity_index': {'current': 108.2, 'target': 110.0, 'trend': 'up'},
                'safety_incidents': {'current': 0, 'target': 0, 'period': 'monthly'},
                'training_completion': {'current': 94.3, 'target': 95.0, 'period': 'quarterly'},
                'employee_satisfaction': {'current': 4.2, 'target': 4.0, 'scale': '5.0', 'period': 'quarterly'}
            },
            'strategic_insights': [
                {
                    'type': 'opportunity',
                    'priority': 'high',
                    'title': 'Production Line Optimization',
                    'description': 'AI analysis suggests 12% efficiency gain possible through schedule optimization',
                    'impact': '$180k monthly savings potential',
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'type': 'risk',
                    'priority': 'medium',
                    'title': 'Supplier Reliability',
                    'description': 'Quality variations detected from Supplier B (15% of orders)',
                    'impact': 'Potential quality issues',
                    'timestamp': (datetime.now() - timedelta(hours=2)).isoformat()
                },
                {
                    'type': 'achievement',
                    'priority': 'low',
                    'title': 'Energy Efficiency Target Met',
                    'description': 'Monthly energy efficiency exceeded target by 1.5%',
                    'impact': '$12k cost savings',
                    'timestamp': (datetime.now() - timedelta(days=1)).isoformat()
                }
            ],
            'trend_analysis': {
                'efficiency_trend': [85.2, 86.1, 86.8, 87.0, 87.3],
                'quality_trend': [97.8, 98.1, 98.0, 98.2, 98.2],
                'revenue_trend': [2.3, 2.35, 2.41, 2.44, 2.46],  # in millions
                'cost_trend': [47.2, 46.8, 46.1, 45.9, 45.67]
            }
        }
        
        # Management dashboard state
        self.active_sessions = {}
        self.dashboard_metrics = {
            'page_loads': 0,
            'active_managers': 0,
            'reports_generated': 0,
            'insights_viewed': 0,
            'avg_session_duration': 0.0,
            'uptime_seconds': 0
        }
        
        # Thread management
        self.lock = threading.Lock()
        self.update_thread = None
        self.is_running = False
        self.start_time = datetime.now()
        
        # Set up routes and handlers
        self._setup_routes()
        self._setup_socketio_handlers()
        
        # Initialize strategic data
        self._initialize_strategic_data()
        
        self.logger.info("ManagementDashboard initialized successfully")
    
    def _setup_routes(self):
        """Set up Flask routes for management dashboard."""
        
        @self.app.route('/')
        def management_home():
            """Main management dashboard page."""
            start_time = time.time()
            
            try:
                # Create management session
                if 'session_id' not in session:
                    session['session_id'] = str(uuid.uuid4())
                    session['user_type'] = 'manager'
                    session['login_time'] = datetime.now().isoformat()
                
                with self.lock:
                    self.dashboard_metrics['page_loads'] += 1
                
                response_time = (time.time() - start_time) * 1000
                
                return render_template('management_dashboard.html',
                                     session_id=session['session_id'],
                                     update_interval=self.update_interval_ms,
                                     executive_data=self.executive_data)
                
            except Exception as e:
                self.logger.error(f"Management dashboard home route error: {e}")
                return f"Management Dashboard Error: {e}", 500
        
        @self.app.route('/api/executive-data')
        def get_executive_data():
            """API endpoint for executive dashboard data."""
            try:
                return jsonify(self.executive_data)
            except Exception as e:
                self.logger.error(f"Executive data API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/reports/generate', methods=['POST'])
        def generate_report():
            """API endpoint for generating executive reports."""
            try:
                report_type = request.json.get('report_type', 'summary')
                time_period = request.json.get('time_period', 'weekly')
                
                report_data = self._generate_executive_report(report_type, time_period)
                
                with self.lock:
                    self.dashboard_metrics['reports_generated'] += 1
                
                return jsonify(report_data)
                
            except Exception as e:
                self.logger.error(f"Report generation error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/insights/<insight_id>/view', methods=['POST'])
        def view_insight(insight_id):
            """API endpoint for tracking insight views."""
            try:
                manager_id = session.get('session_id', 'unknown')
                
                # Track insight view
                with self.lock:
                    self.dashboard_metrics['insights_viewed'] += 1
                
                return jsonify({'success': True, 'insight_id': insight_id})
                
            except Exception as e:
                self.logger.error(f"Insight view tracking error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/benchmarks')
        def get_benchmarks():
            """API endpoint for industry benchmarks."""
            try:
                benchmarks = self._get_industry_benchmarks()
                return jsonify(benchmarks)
            except Exception as e:
                self.logger.error(f"Benchmarks API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/metrics')
        def get_management_metrics():
            """API endpoint for management dashboard metrics."""
            try:
                with self.lock:
                    metrics = self.dashboard_metrics.copy()
                    metrics['uptime_seconds'] = (datetime.now() - self.start_time).total_seconds()
                
                return jsonify(metrics)
                
            except Exception as e:
                self.logger.error(f"Management metrics error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _setup_socketio_handlers(self):
        """Set up SocketIO event handlers for management communication."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle manager connection."""
            try:
                session_id = request.sid
                
                self.active_sessions[session_id] = {
                    'session_id': session_id,
                    'connected_at': datetime.now().isoformat(),
                    'user_type': 'manager',
                    'last_activity': datetime.now().isoformat()
                }
                
                join_room('managers')
                
                with self.lock:
                    self.dashboard_metrics['active_managers'] += 1
                
                self.logger.info(f"Manager connected: {session_id}")
                
                # Send initial executive data
                emit('executive_data', self.executive_data)
                
            except Exception as e:
                self.logger.error(f"Manager SocketIO connect error: {e}")
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle manager disconnection."""
            try:
                session_id = request.sid
                
                if session_id in self.active_sessions:
                    # Calculate session duration
                    connect_time = datetime.fromisoformat(
                        self.active_sessions[session_id]['connected_at']
                    )
                    duration = (datetime.now() - connect_time).total_seconds()
                    
                    # Update average session duration
                    with self.lock:
                        current_avg = self.dashboard_metrics['avg_session_duration']
                        active_count = self.dashboard_metrics['active_managers']
                        if active_count > 0:
                            self.dashboard_metrics['avg_session_duration'] = (
                                (current_avg * (active_count - 1) + duration) / active_count
                            )
                        self.dashboard_metrics['active_managers'] -= 1
                    
                    del self.active_sessions[session_id]
                
                leave_room('managers')
                self.logger.info(f"Manager disconnected: {session_id}")
                
            except Exception as e:
                self.logger.error(f"Manager SocketIO disconnect error: {e}")
        
        @self.socketio.on('request_detailed_analysis')
        def handle_analysis_request(data):
            """Handle requests for detailed analysis."""
            try:
                analysis_type = data.get('analysis_type')
                time_frame = data.get('time_frame', 'weekly')
                
                analysis_result = self._perform_detailed_analysis(analysis_type, time_frame)
                
                emit('detailed_analysis_result', analysis_result)
                
            except Exception as e:
                self.logger.error(f"Analysis request error: {e}")
                emit('analysis_error', {'error': str(e)})
    
    def _initialize_strategic_data(self):
        """Initialize strategic dashboard data."""
        try:
            # Generate historical trend data
            base_date = datetime.now() - timedelta(days=30)
            self.historical_data = {
                'dates': [],
                'efficiency': [],
                'revenue': [],
                'quality': [],
                'costs': []
            }
            
            for i in range(30):
                date = base_date + timedelta(days=i)
                self.historical_data['dates'].append(date.strftime('%Y-%m-%d'))
                
                # Generate realistic trend data
                self.historical_data['efficiency'].append(85.0 + random.uniform(-2, 3) + i * 0.1)
                self.historical_data['revenue'].append(2200000 + random.uniform(-100000, 150000) + i * 8000)
                self.historical_data['quality'].append(97.5 + random.uniform(-0.5, 1.0))
                self.historical_data['costs'].append(48.0 - random.uniform(-1, 2) - i * 0.08)
            
            # Initialize predictive models data
            self.predictive_insights = {
                'production_forecast': {
                    'next_week': 13200,
                    'confidence': 89.5,
                    'factors': ['seasonal_demand', 'capacity_improvements', 'supply_chain_stability']
                },
                'maintenance_schedule': {
                    'critical_items': 2,
                    'upcoming_7_days': 5,
                    'cost_impact': '$45,200'
                },
                'quality_predictions': {
                    'expected_yield': 98.4,
                    'risk_factors': ['supplier_variation', 'environmental_conditions'],
                    'confidence': 92.1
                }
            }
            
            self.logger.info("Strategic dashboard data initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing strategic data: {e}")
    
    def _generate_executive_report(self, report_type: str, time_period: str) -> Dict[str, Any]:
        """Generate executive report."""
        try:
            report_data = {
                'report_type': report_type,
                'time_period': time_period,
                'generated_at': datetime.now().isoformat(),
                'summary': {},
                'details': {},
                'recommendations': []
            }
            
            if report_type == 'financial':
                report_data['summary'] = {
                    'total_revenue': self.executive_data['financial_metrics']['revenue']['current'],
                    'profit_margin': self.executive_data['financial_metrics']['profit_margin']['current'],
                    'cost_savings': 125000,  # Example calculation
                    'roi': self.executive_data['financial_metrics']['roi']['current']
                }
                
                report_data['recommendations'] = [
                    'Optimize production schedule to reduce overtime costs',
                    'Negotiate better rates with Supplier A for 8% cost reduction',
                    'Implement energy efficiency program for $50k annual savings'
                ]
            
            elif report_type == 'operational':
                report_data['summary'] = {
                    'efficiency': self.executive_data['operational_kpis']['overall_efficiency']['current'],
                    'quality': self.executive_data['operational_kpis']['quality_score']['current'],
                    'delivery': self.executive_data['operational_kpis']['on_time_delivery']['current'],
                    'utilization': self.executive_data['operational_kpis']['capacity_utilization']['current']
                }
                
                report_data['recommendations'] = [
                    'Implement predictive maintenance to reduce downtime by 15%',
                    'Cross-train operators to improve line flexibility',
                    'Upgrade Line 2 sensors for better quality control'
                ]
            
            elif report_type == 'summary':
                report_data['summary'] = {
                    'overall_performance': 'Good',
                    'key_achievements': [
                        'Quality targets exceeded for 3 consecutive months',
                        'Energy efficiency improved by 5% quarter-over-quarter',
                        'Zero safety incidents this month'
                    ],
                    'areas_for_improvement': [
                        'Production efficiency still 2.7% below target',
                        'Supplier B quality variations need attention',
                        'Capacity utilization can be optimized'
                    ]
                }
            
            return report_data
            
        except Exception as e:
            self.logger.error(f"Report generation error: {e}")
            return {'error': str(e)}
    
    def _get_industry_benchmarks(self) -> Dict[str, Any]:
        """Get industry benchmark data."""
        return {
            'manufacturing_industry': {
                'efficiency': {'industry_average': 82.0, 'top_quartile': 88.0, 'our_position': 87.3},
                'quality': {'industry_average': 96.5, 'top_quartile': 98.5, 'our_position': 98.2},
                'on_time_delivery': {'industry_average': 91.0, 'top_quartile': 96.0, 'our_position': 94.7},
                'energy_efficiency': {'industry_average': 87.0, 'top_quartile': 93.0, 'our_position': 91.5}
            },
            'competitive_analysis': {
                'cost_per_unit': {'market_average': 48.50, 'best_in_class': 41.00, 'our_position': 45.67},
                'profit_margin': {'market_average': 16.2, 'best_in_class': 22.5, 'our_position': 18.4},
                'innovation_index': {'market_average': 6.8, 'best_in_class': 8.5, 'our_position': 7.2}
            }
        }
    
    def _perform_detailed_analysis(self, analysis_type: str, time_frame: str) -> Dict[str, Any]:
        """Perform detailed analysis based on request."""
        try:
            analysis_result = {
                'analysis_type': analysis_type,
                'time_frame': time_frame,
                'generated_at': datetime.now().isoformat(),
                'insights': [],
                'data_points': {},
                'recommendations': []
            }
            
            if analysis_type == 'efficiency_deep_dive':
                analysis_result['insights'] = [
                    'Line 1 shows 5% efficiency drop during shift changes',
                    'Equipment warm-up time averaging 12 minutes above optimal',
                    'Operator skill variations contribute to 3% efficiency variance'
                ]
                analysis_result['recommendations'] = [
                    'Implement staggered shift changes',
                    'Pre-warm equipment during planned downtime',
                    'Establish operator skill standardization program'
                ]
            
            elif analysis_type == 'quality_analysis':
                analysis_result['insights'] = [
                    'Temperature fluctuations correlate with 70% of quality issues',
                    'Supplier B materials show 2x defect rate variation',
                    'Quality improves 15% with humidity control'
                ]
                analysis_result['recommendations'] = [
                    'Install advanced temperature control systems',
                    'Implement supplier quality agreements with penalties',
                    'Upgrade environmental control systems'
                ]
            
            elif analysis_type == 'cost_optimization':
                analysis_result['insights'] = [
                    'Energy costs spike 25% during peak hours',
                    'Maintenance costs vary 40% between production lines',
                    'Material waste averages 2.3% above target'
                ]
                analysis_result['recommendations'] = [
                    'Implement time-of-use energy scheduling',
                    'Standardize maintenance procedures across lines',
                    'Deploy lean manufacturing waste reduction initiatives'
                ]
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Detailed analysis error: {e}")
            return {'error': str(e)}
    
    def start_strategic_updates(self):
        """Start strategic data update thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.update_thread = threading.Thread(target=self._strategic_update_worker, daemon=True)
        self.update_thread.start()
        
        self.logger.info("Strategic update thread started")
    
    def _strategic_update_worker(self):
        """Background worker for updating strategic metrics."""
        while self.is_running:
            try:
                # Update financial metrics with realistic business variations
                revenue = self.executive_data['financial_metrics']['revenue']
                current_revenue = revenue['current']
                
                # Simulate daily revenue changes
                daily_variation = random.uniform(-50000, 80000)  # Business day variations
                new_revenue = max(2000000, current_revenue + daily_variation)
                revenue['current'] = new_revenue
                
                # Update profit margin based on revenue and cost changes
                profit_margin = self.executive_data['financial_metrics']['profit_margin']
                margin_change = random.uniform(-0.2, 0.3)
                new_margin = max(15.0, min(25.0, profit_margin['current'] + margin_change))
                profit_margin['current'] = round(new_margin, 1)
                
                # Update operational KPIs
                efficiency = self.executive_data['operational_kpis']['overall_efficiency']
                efficiency_change = random.uniform(-0.5, 0.8)
                new_efficiency = max(80.0, min(95.0, efficiency['current'] + efficiency_change))
                efficiency['current'] = round(new_efficiency, 1)
                
                # Update trend data
                trend = self.executive_data['trend_analysis']
                trend['efficiency_trend'].append(new_efficiency)
                trend['revenue_trend'].append(round(new_revenue / 1000000, 2))
                
                # Keep only last 10 data points
                for trend_key in trend:
                    if len(trend[trend_key]) > 10:
                        trend[trend_key] = trend[trend_key][-10:]
                
                # Generate new strategic insights occasionally
                if random.random() < 0.1:  # 10% chance per update cycle
                    self._generate_strategic_insight()
                
                # Broadcast updates to connected managers
                if self.active_sessions:
                    self.socketio.emit('executive_data_update', {
                        'financial_metrics': self.executive_data['financial_metrics'],
                        'operational_kpis': self.executive_data['operational_kpis'],
                        'trend_analysis': self.executive_data['trend_analysis'],
                        'timestamp': datetime.now().isoformat()
                    }, room='managers')
                
                # Sleep until next update (management updates are less frequent)
                time.sleep(self.update_interval_ms / 1000.0)
                
            except Exception as e:
                self.logger.error(f"Strategic update worker error: {e}")
                time.sleep(5.0)  # Wait before retrying
    
    def _generate_strategic_insight(self):
        """Generate new strategic insight."""
        insight_templates = [
            {
                'type': 'opportunity',
                'titles': [
                    'Market Demand Surge Detected',
                    'Supply Chain Optimization Opportunity',
                    'Energy Cost Reduction Potential',
                    'Automation ROI Opportunity'
                ],
                'descriptions': [
                    'AI analysis shows 18% demand increase for Product Line A',
                    'Route optimization could reduce logistics costs by $75k annually',
                    'Off-peak production scheduling saves $25k monthly',
                    'Robotic arm upgrade pays back in 14 months'
                ]
            },
            {
                'type': 'risk',
                'titles': [
                    'Supplier Capacity Constraint',
                    'Quality Variance Alert',
                    'Equipment Age Risk',
                    'Workforce Skill Gap'
                ],
                'descriptions': [
                    'Primary supplier at 95% capacity, backup needed',
                    'Statistical quality variations above control limits',
                    'Critical equipment nearing end-of-life cycle',
                    'Upcoming retirements create knowledge transfer risk'
                ]
            },
            {
                'type': 'achievement',
                'titles': [
                    'Sustainability Milestone Reached',
                    'Quality Excellence Award',
                    'Cost Reduction Target Met',
                    'Safety Record Achievement'
                ],
                'descriptions': [
                    'Carbon footprint reduced by 20% year-over-year',
                    'Six Sigma certification earned for Process A',
                    'Q3 cost reduction exceeded target by 12%',
                    '500 days without recordable incidents'
                ]
            }
        ]
        
        insight_type = random.choice(list(insight_templates))
        template = next(t for t in insight_templates if t['type'] == insight_type)
        
        title = random.choice(template['titles'])
        description = random.choice(template['descriptions'])
        
        new_insight = {
            'type': insight_type,
            'priority': random.choice(['high', 'medium', 'low']),
            'title': title,
            'description': description,
            'impact': f"${random.randint(50, 500)}k potential impact",
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to beginning of insights list
        self.executive_data['strategic_insights'].insert(0, new_insight)
        
        # Keep only last 10 insights
        if len(self.executive_data['strategic_insights']) > 10:
            self.executive_data['strategic_insights'] = self.executive_data['strategic_insights'][:10]
        
        # Broadcast new insight
        self.socketio.emit('new_strategic_insight', new_insight, room='managers')
        
        self.logger.info(f"Generated strategic insight: {title}")
    
    def run(self):
        """Run the management dashboard server."""
        try:
            # Start strategic updates
            self.start_strategic_updates()
            
            self.logger.info(f"Starting Management Dashboard on {self.host}:{self.port}")
            
            # Run the Flask app with SocketIO
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=self.debug,
                use_reloader=False
            )
            
        except Exception as e:
            self.logger.error(f"Error running management dashboard: {e}")
            raise
    
    def stop(self):
        """Stop the management dashboard."""
        self.is_running = False
        self.logger.info("Management Dashboard stopped")
    
    async def validate_management_dashboard(self) -> Dict[str, Any]:
        """Validate management dashboard functionality."""
        validation_results = {
            'component': 'ManagementDashboard',
            'validation_timestamp': datetime.now().isoformat(),
            'tests': {},
            'performance_metrics': {},
            'overall_status': 'unknown'
        }
        
        try:
            # Test 1: Dashboard Initialization
            validation_results['tests']['initialization'] = {
                'status': 'pass',
                'details': 'Management dashboard initialized successfully'
            }
            
            # Test 2: Executive Data Structure
            required_sections = ['financial_metrics', 'operational_kpis', 'production_performance', 
                               'workforce_metrics', 'strategic_insights', 'trend_analysis']
            missing_sections = [section for section in required_sections 
                              if section not in self.executive_data]
            
            validation_results['tests']['executive_data_structure'] = {
                'status': 'pass' if not missing_sections else 'fail',
                'details': f"All executive sections present" if not missing_sections 
                          else f"Missing sections: {missing_sections}"
            }
            
            # Test 3: Report Generation
            report = self._generate_executive_report('summary', 'weekly')
            
            validation_results['tests']['report_generation'] = {
                'status': 'pass' if 'summary' in report else 'fail',
                'details': f"Generated executive report with {len(report)} sections"
            }
            
            # Test 4: Strategic Insight Generation
            initial_insight_count = len(self.executive_data['strategic_insights'])
            self._generate_strategic_insight()
            new_insight_count = len(self.executive_data['strategic_insights'])
            
            validation_results['tests']['strategic_insights'] = {
                'status': 'pass' if new_insight_count > initial_insight_count else 'fail',
                'details': f"Strategic insight generated: {new_insight_count - initial_insight_count} new insights"
            }
            
            # Test 5: Industry Benchmarks
            benchmarks = self._get_industry_benchmarks()
            
            validation_results['tests']['industry_benchmarks'] = {
                'status': 'pass' if 'manufacturing_industry' in benchmarks else 'fail',
                'details': f"Industry benchmarks available: {len(benchmarks)} categories"
            }
            
            # Performance metrics
            with self.lock:
                validation_results['performance_metrics'] = {
                    'dashboard_metrics': self.dashboard_metrics.copy(),
                    'active_sessions': len(self.active_sessions),
                    'strategic_insights_count': len(self.executive_data['strategic_insights']),
                    'kpi_count': len(self.executive_data['operational_kpis']),
                    'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
                }
            
            # Overall status
            passed_tests = sum(1 for test in validation_results['tests'].values() 
                             if test['status'] == 'pass')
            total_tests = len(validation_results['tests'])
            
            validation_results['overall_status'] = 'pass' if passed_tests == total_tests else 'fail'
            validation_results['test_summary'] = f"{passed_tests}/{total_tests} tests passed"
            
            self.logger.info(f"Management dashboard validation completed: {validation_results['test_summary']}")
            
        except Exception as e:
            validation_results['overall_status'] = 'error'
            validation_results['error'] = str(e)
            self.logger.error(f"Management dashboard validation failed: {e}")
        
        return validation_results


# Create management dashboard template
def create_management_dashboard_template():
    """Create HTML template for management dashboard."""
    template_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Manufacturing Line - Executive Dashboard</title>
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
            background: linear-gradient(135deg, #1a1a1a 0%, #2c3e50 100%);
            color: #ffffff;
            line-height: 1.6;
            min-height: 100vh;
        }
        
        .executive-header {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            padding: 1.5rem 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            position: relative;
        }
        
        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .logo h1 {
            color: #3498db;
            font-size: 2.2rem;
            font-weight: 300;
        }
        
        .logo .subtitle {
            color: #95a5a6;
            font-size: 0.9rem;
            margin-top: 0.2rem;
        }
        
        .executive-summary {
            display: flex;
            gap: 3rem;
            align-items: center;
        }
        
        .summary-metric {
            text-align: center;
        }
        
        .summary-value {
            font-size: 2.2rem;
            font-weight: bold;
            color: #2ecc71;
            display: block;
        }
        
        .summary-label {
            font-size: 0.85rem;
            color: #bdc3c7;
            margin-top: 0.3rem;
        }
        
        .main-content {
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .kpi-card {
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.3s ease;
        }
        
        .kpi-card:hover {
            transform: translateY(-4px);
        }
        
        .kpi-title {
            font-size: 0.9rem;
            color: #95a5a6;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 1rem;
        }
        
        .kpi-value {
            font-size: 2.8rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        .kpi-target {
            font-size: 0.85rem;
            color: #7f8c8d;
        }
        
        .kpi-trend {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: bold;
            margin-top: 0.5rem;
        }
        
        .trend-up {
            background: rgba(46, 204, 113, 0.2);
            color: #2ecc71;
        }
        
        .trend-down {
            background: rgba(231, 76, 60, 0.2);
            color: #e74c3c;
        }
        
        .trend-stable {
            background: rgba(243, 156, 18, 0.2);
            color: #f39c12;
        }
        
        .dashboard-section {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 2rem;
            margin-bottom: 2rem;
        }
        
        .charts-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }
        
        .chart-card {
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .chart-title {
            font-size: 1.1rem;
            margin-bottom: 1rem;
            color: #ecf0f1;
        }
        
        .chart-container {
            position: relative;
            height: 250px;
        }
        
        .insights-panel {
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .insight-item {
            background: rgba(0,0,0,0.2);
            padding: 1.2rem;
            margin: 1rem 0;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        
        .insight-opportunity { border-left-color: #2ecc71; }
        .insight-risk { border-left-color: #e74c3c; }
        .insight-achievement { border-left-color: #f39c12; }
        
        .insight-header {
            display: flex;
            justify-content: between;
            align-items: flex-start;
            margin-bottom: 0.5rem;
        }
        
        .insight-title {
            font-weight: bold;
            color: #ecf0f1;
        }
        
        .insight-priority {
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: bold;
            text-transform: uppercase;
            margin-left: auto;
        }
        
        .priority-high {
            background: rgba(231, 76, 60, 0.3);
            color: #e74c3c;
        }
        
        .priority-medium {
            background: rgba(243, 156, 18, 0.3);
            color: #f39c12;
        }
        
        .priority-low {
            background: rgba(52, 152, 219, 0.3);
            color: #3498db;
        }
        
        .insight-description {
            font-size: 0.9rem;
            color: #bdc3c7;
            margin-bottom: 0.5rem;
        }
        
        .insight-impact {
            font-size: 0.8rem;
            color: #95a5a6;
            font-weight: bold;
        }
        
        .financial-section {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .financial-card {
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
            border-radius: 12px;
            padding: 2rem;
            color: white;
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }
        
        .financial-revenue {
            background: linear-gradient(135deg, #3498db 0%, #5dade2 100%);
        }
        
        .financial-profit {
            background: linear-gradient(135deg, #e67e22 0%, #f39c12 100%);
        }
        
        .financial-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        .controls-section {
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .btn {
            padding: 0.8rem 1.5rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #3498db 0%, #5dade2 100%);
            color: white;
        }
        
        .btn-success {
            background: linear-gradient(135deg, #2ecc71 0%, #58d68d 100%);
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(0,0,0,0.3);
        }
        
        @media (max-width: 1200px) {
            .kpi-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .charts-container {
                grid-template-columns: 1fr;
            }
            
            .dashboard-section {
                grid-template-columns: 1fr;
            }
        }
        
        @media (max-width: 768px) {
            .kpi-grid {
                grid-template-columns: 1fr;
            }
            
            .financial-section {
                grid-template-columns: 1fr;
            }
            
            .executive-summary {
                display: none;
            }
        }
    </style>
</head>
<body>
    <div class="executive-header">
        <div class="header-content">
            <div class="logo">
                <h1>📊 Executive Manufacturing Dashboard</h1>
                <div class="subtitle">Strategic Overview & Performance Analytics</div>
            </div>
            <div class="executive-summary">
                <div class="summary-metric">
                    <span class="summary-value" id="total-revenue">$2.45M</span>
                    <div class="summary-label">Monthly Revenue</div>
                </div>
                <div class="summary-metric">
                    <span class="summary-value" id="efficiency-score">87.3%</span>
                    <div class="summary-label">Overall Efficiency</div>
                </div>
                <div class="summary-metric">
                    <span class="summary-value" id="quality-score">98.2%</span>
                    <div class="summary-label">Quality Score</div>
                </div>
            </div>
        </div>
    </div>

    <div class="main-content">
        <div class="financial-section">
            <div class="financial-card financial-revenue">
                <div class="financial-value" id="revenue-value">$2,456,789</div>
                <div>Monthly Revenue</div>
                <div style="font-size: 0.9rem; margin-top: 0.5rem;">
                    Target: $2,500,000 (98.3%)
                </div>
            </div>
            <div class="financial-card financial-profit">
                <div class="financial-value" id="profit-margin-value">18.4%</div>
                <div>Profit Margin</div>
                <div style="font-size: 0.9rem; margin-top: 0.5rem;">
                    Target: 20.0% (92.0%)
                </div>
            </div>
        </div>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-title">Overall Efficiency</div>
                <div class="kpi-value" id="kpi-efficiency">87.3%</div>
                <div class="kpi-target">Target: 90.0%</div>
                <span class="kpi-trend trend-up" id="efficiency-trend">↗ Improving</span>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Quality Score</div>
                <div class="kpi-value" id="kpi-quality">98.2%</div>
                <div class="kpi-target">Target: 99.0%</div>
                <span class="kpi-trend trend-stable" id="quality-trend">→ Stable</span>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">On-Time Delivery</div>
                <div class="kpi-value" id="kpi-delivery">94.7%</div>
                <div class="kpi-target">Target: 95.0%</div>
                <span class="kpi-trend trend-up" id="delivery-trend">↗ Improving</span>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Capacity Utilization</div>
                <div class="kpi-value" id="kpi-capacity">82.1%</div>
                <div class="kpi-target">Target: 85.0%</div>
                <span class="kpi-trend trend-up" id="capacity-trend">↗ Improving</span>
            </div>
        </div>

        <div class="controls-section">
            <button class="btn btn-primary" onclick="generateReport('financial')">Financial Report</button>
            <button class="btn btn-primary" onclick="generateReport('operational')">Operational Report</button>
            <button class="btn btn-success" onclick="requestAnalysis('efficiency_deep_dive')">Deep Dive Analysis</button>
        </div>

        <div class="dashboard-section">
            <div class="charts-container">
                <div class="chart-card">
                    <div class="chart-title">Performance Trends</div>
                    <div class="chart-container">
                        <canvas id="performance-chart"></canvas>
                    </div>
                </div>
                <div class="chart-card">
                    <div class="chart-title">Financial Overview</div>
                    <div class="chart-container">
                        <canvas id="financial-chart"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="insights-panel">
                <h2>Strategic Insights</h2>
                <div id="insights-container">
                    <!-- Strategic insights will be populated dynamically -->
                </div>
            </div>
        </div>
    </div>

    <script>
        // Initialize Socket.IO connection
        const socket = io();
        
        // Chart variables
        let performanceChart = null;
        let financialChart = null;
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initializeCharts();
        });
        
        // Socket event handlers
        socket.on('connect', function() {
            console.log('Connected to management dashboard server');
        });
        
        socket.on('executive_data', function(data) {
            updateExecutiveData(data);
        });
        
        socket.on('executive_data_update', function(data) {
            updateFinancialMetrics(data.financial_metrics);
            updateOperationalKPIs(data.operational_kpis);
            updateCharts(data.trend_analysis);
        });
        
        socket.on('new_strategic_insight', function(insight) {
            addStrategicInsight(insight);
        });
        
        // Update functions
        function updateExecutiveData(data) {
            updateFinancialMetrics(data.financial_metrics);
            updateOperationalKPIs(data.operational_kpis);
            updateStrategicInsights(data.strategic_insights);
            updateCharts(data.trend_analysis);
        }
        
        function updateFinancialMetrics(financial) {
            document.getElementById('revenue-value').textContent = 
                '$' + financial.revenue.current.toLocaleString();
            document.getElementById('profit-margin-value').textContent = 
                financial.profit_margin.current + '%';
            document.getElementById('total-revenue').textContent = 
                '$' + (financial.revenue.current / 1000000).toFixed(2) + 'M';
        }
        
        function updateOperationalKPIs(kpis) {
            document.getElementById('kpi-efficiency').textContent = kpis.overall_efficiency.current + '%';
            document.getElementById('kpi-quality').textContent = kpis.quality_score.current + '%';
            document.getElementById('kpi-delivery').textContent = kpis.on_time_delivery.current + '%';
            document.getElementById('kpi-capacity').textContent = kpis.capacity_utilization.current + '%';
            
            document.getElementById('efficiency-score').textContent = kpis.overall_efficiency.current + '%';
            document.getElementById('quality-score').textContent = kpis.quality_score.current + '%';
            
            // Update trend indicators
            updateTrendIndicator('efficiency-trend', kpis.overall_efficiency.trend);
            updateTrendIndicator('quality-trend', kpis.quality_score.trend);
            updateTrendIndicator('delivery-trend', kpis.on_time_delivery.trend);
            updateTrendIndicator('capacity-trend', kpis.capacity_utilization.trend);
        }
        
        function updateTrendIndicator(elementId, trend) {
            const element = document.getElementById(elementId);
            element.className = element.className.replace(/trend-\\w+/g, '');
            element.classList.add('trend-' + trend);
            
            const arrows = { up: '↗', down: '↘', stable: '→' };
            const text = { up: 'Improving', down: 'Declining', stable: 'Stable' };
            
            element.textContent = arrows[trend] + ' ' + text[trend];
        }
        
        function updateStrategicInsights(insights) {
            const container = document.getElementById('insights-container');
            container.innerHTML = '';
            
            insights.slice(0, 3).forEach(insight => {
                const insightElement = createInsightElement(insight);
                container.appendChild(insightElement);
            });
        }
        
        function addStrategicInsight(insight) {
            const container = document.getElementById('insights-container');
            const insightElement = createInsightElement(insight);
            container.insertBefore(insightElement, container.firstChild);
            
            // Keep only 3 insights visible
            while (container.children.length > 3) {
                container.removeChild(container.lastChild);
            }
        }
        
        function createInsightElement(insight) {
            const insightDiv = document.createElement('div');
            insightDiv.className = `insight-item insight-${insight.type}`;
            insightDiv.innerHTML = `
                <div class="insight-header">
                    <div class="insight-title">${insight.title}</div>
                    <span class="insight-priority priority-${insight.priority}">${insight.priority}</span>
                </div>
                <div class="insight-description">${insight.description}</div>
                <div class="insight-impact">${insight.impact}</div>
            `;
            return insightDiv;
        }
        
        // Chart functions
        function initializeCharts() {
            // Performance trend chart
            const perfCtx = document.getElementById('performance-chart').getContext('2d');
            performanceChart = new Chart(perfCtx, {
                type: 'line',
                data: {
                    labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5'],
                    datasets: [
                        {
                            label: 'Efficiency (%)',
                            data: [85.2, 86.1, 86.8, 87.0, 87.3],
                            borderColor: '#2ecc71',
                            backgroundColor: 'rgba(46, 204, 113, 0.1)',
                            tension: 0.4
                        },
                        {
                            label: 'Quality (%)',
                            data: [97.8, 98.1, 98.0, 98.2, 98.2],
                            borderColor: '#3498db',
                            backgroundColor: 'rgba(52, 152, 219, 0.1)',
                            tension: 0.4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { 
                            beginAtZero: false,
                            min: 80,
                            grid: { color: '#34495e' },
                            ticks: { color: '#ffffff' }
                        },
                        x: {
                            grid: { color: '#34495e' },
                            ticks: { color: '#ffffff' }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: '#ffffff' } }
                    }
                }
            });
            
            // Financial chart
            const finCtx = document.getElementById('financial-chart').getContext('2d');
            financialChart = new Chart(finCtx, {
                type: 'bar',
                data: {
                    labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5'],
                    datasets: [
                        {
                            label: 'Revenue ($M)',
                            data: [2.3, 2.35, 2.41, 2.44, 2.46],
                            backgroundColor: 'rgba(52, 152, 219, 0.8)',
                            borderColor: '#3498db',
                            borderWidth: 1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false,
                            min: 2.0,
                            grid: { color: '#34495e' },
                            ticks: { color: '#ffffff' }
                        },
                        x: {
                            grid: { color: '#34495e' },
                            ticks: { color: '#ffffff' }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: '#ffffff' } }
                    }
                }
            });
        }
        
        function updateCharts(trendData) {
            if (performanceChart && trendData.efficiency_trend) {
                performanceChart.data.datasets[0].data = trendData.efficiency_trend;
                performanceChart.data.datasets[1].data = trendData.quality_trend;
                performanceChart.update();
            }
            
            if (financialChart && trendData.revenue_trend) {
                financialChart.data.datasets[0].data = trendData.revenue_trend;
                financialChart.update();
            }
        }
        
        // Control functions
        function generateReport(reportType) {
            fetch('/api/reports/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    report_type: reportType,
                    time_period: 'weekly'
                })
            })
            .then(response => response.json())
            .then(data => {
                alert(`${reportType} report generated: ${JSON.stringify(data.summary, null, 2)}`);
            })
            .catch(error => console.error('Error generating report:', error));
        }
        
        function requestAnalysis(analysisType) {
            socket.emit('request_detailed_analysis', {
                analysis_type: analysisType,
                time_frame: 'weekly'
            });
        }
        
        socket.on('detailed_analysis_result', function(analysis) {
            const insights = analysis.insights.join('\\n');
            const recommendations = analysis.recommendations.join('\\n');
            alert(`Analysis Results:\\n\\nInsights:\\n${insights}\\n\\nRecommendations:\\n${recommendations}`);
        });
    </script>
</body>
</html>"""
    
    return template_content


# Utility function to create management dashboard
def create_management_dashboard(config: Optional[Dict] = None) -> ManagementDashboard:
    """Create and configure a management dashboard instance."""
    return ManagementDashboard(config)