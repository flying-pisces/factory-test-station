#!/usr/bin/env python3
"""
Week 12 Interactive Demo Case
Manufacturing Line Control System - Complete AI Integration Demo

This is a fully interactive demonstration showcasing all AI engines
working together in realistic manufacturing scenarios.
"""

import sys
import time
import asyncio
import json
import random
from datetime import datetime, timedelta

# Add project root to path
sys.path.append('.')

class ManufacturingLineDemo:
    """Interactive demo of complete manufacturing line with AI integration"""
    
    def __init__(self):
        self.ai_engines = {}
        self.production_data = {}
        self.demo_running = True
        self.scenario_count = 0
        
    def initialize_system(self):
        """Initialize complete manufacturing system with AI"""
        print("🏭 MANUFACTURING LINE CONTROL SYSTEM")
        print("🤖 Week 12: Complete AI Integration Demo")
        print("=" * 70)
        print("🚀 INITIALIZING INTELLIGENT MANUFACTURING SYSTEM...")
        
        # Initialize all AI engines
        try:
            from layers.ai_layer.ai_engine import AIEngine
            from layers.ai_layer.predictive_maintenance_engine import PredictiveMaintenanceEngine
            from layers.ai_layer.vision_engine import VisionEngine
            from layers.ai_layer.nlp_engine import NLPEngine
            from layers.ai_layer.optimization_ai_engine import OptimizationAIEngine
            
            self.ai_engines = {
                'ai': AIEngine(),
                'maintenance': PredictiveMaintenanceEngine(),
                'vision': VisionEngine(),
                'nlp': NLPEngine(),
                'optimization': OptimizationAIEngine()
            }
            
            print("✅ All AI engines initialized successfully")
            
            # Initialize production data
            self.production_data = {
                'line_status': 'RUNNING',
                'current_throughput': 98,
                'quality_score': 0.94,
                'energy_consumption': 875,
                'shift': 'Day Shift',
                'products_completed': 1247,
                'defects_found': 12,
                'last_maintenance': '2024-08-20',
                'operator_notes': []
            }
            
            print("📊 Production line data initialized")
            print("=" * 70)
            
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            return False
            
        return True
    
    def simulate_production_scenario(self, scenario_type):
        """Simulate different production scenarios"""
        self.scenario_count += 1
        
        scenarios = {
            'normal': {
                'name': 'Normal Production',
                'description': 'Steady state production with optimal performance',
                'equipment_temp': random.uniform(65, 72),
                'vibration': random.uniform(0.1, 0.3),
                'throughput': random.uniform(95, 105),
                'quality': random.uniform(0.92, 0.98),
                'operator_note': "Production running smoothly, all stations operational."
            },
            'quality_issue': {
                'name': 'Quality Issue Detection',
                'description': 'AI detects quality problems and recommends actions',
                'equipment_temp': random.uniform(70, 75),
                'vibration': random.uniform(0.2, 0.4),
                'throughput': random.uniform(85, 95),
                'quality': random.uniform(0.85, 0.91),
                'operator_note': "Noticed some defects in Station 3, quality seems inconsistent."
            },
            'maintenance_alert': {
                'name': 'Predictive Maintenance Alert',
                'description': 'AI predicts equipment failure and schedules maintenance',
                'equipment_temp': random.uniform(78, 85),  # High temperature
                'vibration': random.uniform(0.6, 0.9),      # High vibration
                'throughput': random.uniform(80, 90),
                'quality': random.uniform(0.88, 0.94),
                'operator_note': "Station 2 conveyor making unusual noise, vibration increased."
            },
            'optimization': {
                'name': 'Production Optimization',
                'description': 'AI optimizes line parameters for maximum efficiency',
                'equipment_temp': random.uniform(68, 74),
                'vibration': random.uniform(0.15, 0.35),
                'throughput': random.uniform(90, 100),
                'quality': random.uniform(0.90, 0.95),
                'operator_note': "Looking to increase throughput while maintaining quality standards."
            }
        }
        
        return scenarios.get(scenario_type, scenarios['normal'])
    
    def run_ai_analysis(self, scenario_data):
        """Run complete AI analysis on scenario data"""
        print(f"\n🤖 AI ANALYSIS - Scenario #{self.scenario_count}")
        print(f"📋 Scenario: {scenario_data['name']}")
        print(f"💬 Description: {scenario_data['description']}")
        print("-" * 50)
        
        results = {}
        
        # 1. Predictive Maintenance Analysis
        print("🔧 PREDICTIVE MAINTENANCE ANALYSIS")
        equipment_data = {
            'equipment_id': 'MAIN_CONVEYOR_LINE',
            'sensor_data': {
                'temperature': scenario_data['equipment_temp'],
                'vibration': scenario_data['vibration'],
                'pressure': random.uniform(40, 50),
                'current': random.uniform(10, 15)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            maintenance_result = self.ai_engines['maintenance'].detect_anomalies(equipment_data)
            failure_risk = self.ai_engines['maintenance'].predict_equipment_failure(equipment_data)
            rul_result = self.ai_engines['maintenance'].estimate_remaining_useful_life(equipment_data)
            
            print(f"   🌡️ Temperature: {equipment_data['sensor_data']['temperature']:.1f}°C")
            print(f"   📳 Vibration: {equipment_data['sensor_data']['vibration']:.2f}G")
            print(f"   🚨 Anomaly Detected: {maintenance_result.get('anomaly_detected', False)}")
            print(f"   📊 Anomaly Score: {maintenance_result.get('anomaly_score', 0):.3f}")
            print(f"   ⚠️ Failure Risk: {failure_risk.get('failure_probability', 0):.1%}")
            print(f"   ⏰ Est. RUL: {rul_result.get('estimated_rul_hours', 0):.1f} hours")
            
            results['maintenance'] = {
                'anomaly_detected': maintenance_result.get('anomaly_detected', False),
                'failure_risk': failure_risk.get('failure_probability', 0),
                'rul_hours': rul_result.get('estimated_rul_hours', 0)
            }
            
        except Exception as e:
            print(f"   ❌ Maintenance analysis error: {e}")
            results['maintenance'] = {'error': str(e)}
        
        # 2. Vision System Analysis
        print("\n👁️ COMPUTER VISION ANALYSIS")
        image_data = {
            'image_id': f'QUALITY_SCAN_{self.scenario_count:03d}',
            'width': 1920,
            'height': 1080,
            'station_id': 'INSPECTION_STATION',
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            defect_result = self.ai_engines['vision'].detect_defects(image_data)
            component_result = self.ai_engines['vision'].classify_components({'images': [image_data]})
            # Note: Using available methods from vision engine
            
            print(f"   📷 Image: {image_data['image_id']}")
            print(f"   🔍 Defects Found: {defect_result.get('defects_found', 0)}")
            print(f"   📦 Components Classified: {component_result.get('total_components', 0)}")
            print(f"   ✨ Classification Accuracy: {component_result.get('classification_accuracy', 0):.1%}")
            print(f"   ⚡ Processing Time: {defect_result.get('processing_time_ms', 0):.1f}ms")
            
            results['vision'] = {
                'defects_found': defect_result.get('defects_found', 0),
                'components_classified': component_result.get('total_components', 0),
                'classification_accuracy': component_result.get('classification_accuracy', 0)
            }
            
        except Exception as e:
            print(f"   ❌ Vision analysis error: {e}")
            results['vision'] = {'error': str(e)}
        
        # 3. Natural Language Processing
        print("\n📝 NATURAL LANGUAGE PROCESSING")
        operator_report = scenario_data['operator_note']
        
        try:
            nlp_result = asyncio.run(self.ai_engines['nlp'].analyze_text(operator_report, 'full'))
            
            print(f"   📄 Operator Note: \"{operator_report}\"")
            print(f"   🔤 Tokens: {len(nlp_result.get('tokens', []))}")
            
            sentiment = nlp_result.get('sentiment', {})
            print(f"   😊 Sentiment: {sentiment.get('sentiment', 'Unknown')} ({sentiment.get('confidence', 0):.1%})")
            
            entities = nlp_result.get('entities', {})
            if entities:
                print(f"   🏷️ Entities:")
                for entity_type, entity_list in entities.items():
                    print(f"      {entity_type}: {', '.join(entity_list)}")
            
            print(f"   ⚡ Processing Time: {nlp_result.get('processing_time_ms', 0):.1f}ms")
            
            results['nlp'] = {
                'sentiment': sentiment.get('sentiment', 'Unknown'),
                'entities': entities,
                'processing_time': nlp_result.get('processing_time_ms', 0)
            }
            
        except Exception as e:
            print(f"   ❌ NLP analysis error: {e}")
            results['nlp'] = {'error': str(e)}
        
        # 4. Optimization Engine
        print("\n🎯 OPTIMIZATION AI ANALYSIS")
        current_state = {
            'throughput': scenario_data['throughput'],
            'quality': scenario_data['quality'],
            'energy_consumption': random.uniform(800, 900),
            'defect_rate': random.uniform(0.02, 0.08)
        }
        
        target_metrics = {
            'target_throughput': 110,
            'target_quality': 0.96,
            'max_energy': 850,
            'max_defect_rate': 0.03
        }
        
        try:
            opt_result = asyncio.run(self.ai_engines['optimization'].real_time_optimization(
                current_state, target_metrics
            ))
            
            print(f"   📊 Current Performance:")
            print(f"      Throughput: {current_state['throughput']:.1f} units/hr")
            print(f"      Quality: {current_state['quality']:.1%}")
            print(f"      Energy: {current_state['energy_consumption']:.0f}kW")
            
            print(f"   🎯 Optimization Results:")
            print(f"      Best Fitness: {opt_result.get('best_fitness', 0):.3f}")
            print(f"      Processing Time: {opt_result.get('processing_time_ms', 0):.1f}ms")
            
            if opt_result.get('best_solution'):
                adjustments = opt_result['best_solution']
                print(f"      Recommended Adjustments:")
                if len(adjustments) >= 3:
                    print(f"        Speed: {adjustments[0]:.2f}x")
                    print(f"        Pressure: {adjustments[1]:.2f}x") 
                    print(f"        Temperature: {adjustments[2]:.2f}x")
            
            results['optimization'] = {
                'fitness_score': opt_result.get('best_fitness', 0),
                'processing_time': opt_result.get('processing_time_ms', 0),
                'solution': opt_result.get('best_solution', [])
            }
            
        except Exception as e:
            print(f"   ❌ Optimization error: {e}")
            results['optimization'] = {'error': str(e)}
        
        return results
    
    def generate_ai_recommendations(self, scenario_data, analysis_results):
        """Generate intelligent recommendations based on AI analysis"""
        print("\n💡 AI SYSTEM RECOMMENDATIONS")
        print("-" * 50)
        
        recommendations = []
        priority_actions = []
        
        # Maintenance recommendations
        if 'maintenance' in analysis_results and 'error' not in analysis_results['maintenance']:
            maintenance = analysis_results['maintenance']
            
            if maintenance.get('anomaly_detected', False):
                priority_actions.append("🚨 IMMEDIATE: Investigate equipment anomaly")
                recommendations.append("Schedule immediate inspection of main conveyor system")
            
            if maintenance.get('failure_risk', 0) > 0.3:
                priority_actions.append("⚠️ HIGH PRIORITY: Preventive maintenance required")
                recommendations.append(f"Schedule maintenance within {maintenance.get('rul_hours', 48):.0f} hours")
            
            elif maintenance.get('failure_risk', 0) > 0.1:
                recommendations.append("Plan preventive maintenance for next scheduled window")
        
        # Vision system recommendations
        if 'vision' in analysis_results and 'error' not in analysis_results['vision']:
            vision = analysis_results['vision']
            
            if vision.get('defects_found', 0) > 2:
                priority_actions.append("🔍 QUALITY ALERT: Multiple defects detected")
                recommendations.append("Inspect upstream processes for quality issues")
            
            if vision.get('quality_score', 1.0) < 0.9:
                recommendations.append("Review quality control parameters")
                recommendations.append("Consider slowing line speed to improve quality")
        
        # NLP-based recommendations
        if 'nlp' in analysis_results and 'error' not in analysis_results['nlp']:
            nlp = analysis_results['nlp']
            
            if nlp.get('sentiment') == 'negative':
                recommendations.append("Address operator concerns mentioned in reports")
            
            entities = nlp.get('entities', {})
            if 'EQUIPMENT' in entities:
                recommendations.append(f"Focus attention on equipment: {', '.join(entities['EQUIPMENT'])}")
        
        # Optimization recommendations
        if 'optimization' in analysis_results and 'error' not in analysis_results['optimization']:
            opt = analysis_results['optimization']
            
            if opt.get('fitness_score', 0) > 0.5:
                recommendations.append("Apply optimized parameters to improve performance")
                solution = opt.get('solution', [])
                if len(solution) >= 3:
                    if solution[0] > 1.05:
                        recommendations.append("Increase conveyor speed by 5-10%")
                    elif solution[0] < 0.95:
                        recommendations.append("Reduce conveyor speed for better quality")
        
        # Display recommendations
        if priority_actions:
            print("🚨 PRIORITY ACTIONS:")
            for action in priority_actions:
                print(f"   {action}")
        
        if recommendations:
            print(f"\n📋 SYSTEM RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        else:
            print("✅ All systems operating within normal parameters")
        
        # Overall system status
        print(f"\n🎯 OVERALL SYSTEM STATUS:")
        
        # Calculate system health score
        health_score = 100
        if any('anomaly_detected' in r and r.get('anomaly_detected') for r in analysis_results.values()):
            health_score -= 20
        
        if any('failure_risk' in r and r.get('failure_risk', 0) > 0.3 for r in analysis_results.values()):
            health_score -= 30
        
        if any('defects_found' in r and r.get('defects_found', 0) > 2 for r in analysis_results.values()):
            health_score -= 15
        
        if any('quality_score' in r and r.get('quality_score', 1) < 0.9 for r in analysis_results.values()):
            health_score -= 10
        
        health_score = max(0, health_score)
        
        if health_score >= 90:
            status = "🟢 EXCELLENT"
            color = "green"
        elif health_score >= 75:
            status = "🟡 GOOD"
            color = "yellow"
        elif health_score >= 60:
            status = "🟠 FAIR"
            color = "orange"
        else:
            status = "🔴 NEEDS ATTENTION"
            color = "red"
        
        print(f"   System Health: {health_score}% - {status}")
        
        return {
            'recommendations': recommendations,
            'priority_actions': priority_actions,
            'health_score': health_score,
            'status': status
        }
    
    def run_interactive_demo(self):
        """Run interactive demonstration"""
        if not self.initialize_system():
            return
        
        scenarios = ['normal', 'quality_issue', 'maintenance_alert', 'optimization']
        
        print("\n🎮 INTERACTIVE DEMO MODE")
        print("Choose scenarios to demonstrate AI capabilities:")
        print("1. Normal Production")
        print("2. Quality Issue Detection") 
        print("3. Predictive Maintenance Alert")
        print("4. Production Optimization")
        print("5. Run All Scenarios")
        print("6. Exit Demo")
        
        while self.demo_running:
            try:
                print(f"\n{'='*70}")
                choice = input("Enter your choice (1-6): ").strip()
                
                if choice == '6':
                    print("👋 Demo completed. Thank you!")
                    break
                elif choice == '5':
                    # Run all scenarios
                    for scenario_type in scenarios:
                        self.run_single_scenario(scenario_type)
                        if self.scenario_count < len(scenarios):
                            input("\nPress Enter to continue to next scenario...")
                elif choice in ['1', '2', '3', '4']:
                    scenario_map = {
                        '1': 'normal',
                        '2': 'quality_issue', 
                        '3': 'maintenance_alert',
                        '4': 'optimization'
                    }
                    self.run_single_scenario(scenario_map[choice])
                else:
                    print("❌ Invalid choice. Please select 1-6.")
                
            except KeyboardInterrupt:
                print("\n👋 Demo interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Demo error: {e}")
    
    def run_single_scenario(self, scenario_type):
        """Run a single demonstration scenario"""
        scenario_data = self.simulate_production_scenario(scenario_type)
        analysis_results = self.run_ai_analysis(scenario_data)
        recommendations = self.generate_ai_recommendations(scenario_data, analysis_results)
        
        print(f"\n📈 SCENARIO SUMMARY:")
        print(f"   Scenario: {scenario_data['name']}")
        print(f"   Health Score: {recommendations['health_score']}%")
        print(f"   Status: {recommendations['status']}")
        print(f"   Recommendations: {len(recommendations['recommendations'])}")
        print(f"   Priority Actions: {len(recommendations['priority_actions'])}")
    
    def run_automated_demo(self):
        """Run automated demonstration of all scenarios"""
        if not self.initialize_system():
            return
        
        print("\n🤖 AUTOMATED DEMO - Running All Scenarios")
        print("=" * 70)
        
        scenarios = ['normal', 'quality_issue', 'maintenance_alert', 'optimization']
        
        for scenario_type in scenarios:
            self.run_single_scenario(scenario_type)
            print(f"\n{'⏳ Processing next scenario...' if scenario_type != scenarios[-1] else ''}")
            time.sleep(2)  # Brief pause between scenarios
        
        print(f"\n🎊 AUTOMATED DEMO COMPLETE")
        print(f"📊 Total Scenarios Run: {self.scenario_count}")
        print("✅ All AI engines demonstrated successfully")

def main():
    """Main demo entry point"""
    demo = ManufacturingLineDemo()
    
    print("🏭 WEEK 12 MANUFACTURING LINE AI DEMO")
    print("Choose demo mode:")
    print("1. Interactive Demo (choose scenarios)")
    print("2. Automated Demo (run all scenarios)")
    
    try:
        choice = input("Select mode (1 or 2): ").strip()
        
        if choice == '1':
            demo.run_interactive_demo()
        elif choice == '2':
            demo.run_automated_demo()
        else:
            print("❌ Invalid choice. Running automated demo...")
            demo.run_automated_demo()
            
    except KeyboardInterrupt:
        print("\n👋 Demo cancelled. Goodbye!")
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    main()