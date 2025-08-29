#!/usr/bin/env python3
"""
Week 12 Pre-Commit Validation
Quick validation script to run before git commit
"""

import sys
import time
import traceback

def validate_ai_engines():
    """Validate all AI engines can initialize"""
    print("🧠 Validating AI Engines...")
    
    engines = [
        ('AIEngine', 'layers.ai_layer.ai_engine'),
        ('PredictiveMaintenanceEngine', 'layers.ai_layer.predictive_maintenance_engine'),
        ('VisionEngine', 'layers.ai_layer.vision_engine'),
        ('NLPEngine', 'layers.ai_layer.nlp_engine'),
        ('OptimizationAIEngine', 'layers.ai_layer.optimization_ai_engine')
    ]
    
    success_count = 0
    for engine_name, module_path in engines:
        try:
            module = __import__(module_path, fromlist=[engine_name])
            engine_class = getattr(module, engine_name)
            engine_instance = engine_class()
            print(f"   ✅ {engine_name}")
            success_count += 1
        except Exception as e:
            print(f"   ❌ {engine_name}: {e}")
    
    return success_count == 5

def validate_demo_functionality():
    """Validate core demo functionality"""
    print("\n🎮 Validating Demo Functionality...")
    
    try:
        # Test predictive maintenance
        from layers.ai_layer.predictive_maintenance_engine import PredictiveMaintenanceEngine
        pm_engine = PredictiveMaintenanceEngine()
        
        test_data = {
            'equipment_id': 'VALIDATION_TEST',
            'sensor_data': {'temperature': 70, 'vibration': 0.3}
        }
        result = pm_engine.detect_anomalies(test_data)
        
        if 'anomaly_score' in result:
            print(f"   ✅ Predictive Maintenance: Anomaly score {result['anomaly_score']}")
        else:
            print(f"   ❌ Predictive Maintenance: Missing anomaly_score")
            return False
        
        # Test vision engine
        from layers.ai_layer.vision_engine import VisionEngine
        vision_engine = VisionEngine()
        
        image_data = {'image_id': 'VALIDATION_TEST', 'width': 640, 'height': 480}
        vision_result = vision_engine.detect_defects(image_data)
        
        if 'processing_time_ms' in vision_result:
            print(f"   ✅ Vision Engine: {vision_result['processing_time_ms']:.1f}ms processing")
        else:
            print(f"   ❌ Vision Engine: Missing processing time")
            return False
        
        # Test NLP engine
        from layers.ai_layer.nlp_engine import NLPEngine
        import asyncio
        
        nlp_engine = NLPEngine()
        nlp_result = asyncio.run(nlp_engine.analyze_text("Test validation text", 'sentiment'))
        
        if 'sentiment' in nlp_result:
            sentiment = nlp_result['sentiment']['sentiment']
            print(f"   ✅ NLP Engine: Sentiment '{sentiment}' detected")
        else:
            print(f"   ❌ NLP Engine: Missing sentiment analysis")
            return False
        
        print("   ✅ Core demo functionality validated")
        return True
        
    except Exception as e:
        print(f"   ❌ Demo validation failed: {e}")
        return False

def validate_demo_files():
    """Validate demo files exist and are importable"""
    print("\n📁 Validating Demo Files...")
    
    demo_files = [
        'week12_quick_demo.py',
        'week12_milestone_demo.py', 
        'week12_interactive_demo.py'
    ]
    
    success_count = 0
    for demo_file in demo_files:
        try:
            with open(demo_file, 'r') as f:
                content = f.read()
                if len(content) > 1000:  # Basic content check
                    print(f"   ✅ {demo_file}")
                    success_count += 1
                else:
                    print(f"   ❌ {demo_file}: File too small")
        except FileNotFoundError:
            print(f"   ❌ {demo_file}: Not found")
        except Exception as e:
            print(f"   ❌ {demo_file}: {e}")
    
    return success_count >= 2  # At least 2 demo files should be available

def main():
    """Main validation function"""
    print("🏭 WEEK 12 PRE-COMMIT VALIDATION")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run validations
    validations = [
        ("AI Engines", validate_ai_engines),
        ("Demo Functionality", validate_demo_functionality),
        ("Demo Files", validate_demo_files)
    ]
    
    passed_validations = 0
    total_validations = len(validations)
    
    for validation_name, validation_func in validations:
        try:
            if validation_func():
                passed_validations += 1
        except Exception as e:
            print(f"\n❌ {validation_name} validation failed with exception: {e}")
            print(f"Traceback: {traceback.format_exc()}")
    
    # Summary
    execution_time = (time.time() - start_time) * 1000
    print(f"\n{'='*50}")
    print(f"📊 VALIDATION SUMMARY")
    print(f"{'='*50}")
    print(f"✅ Passed: {passed_validations}/{total_validations} validations")
    print(f"⚡ Time: {execution_time:.1f}ms")
    
    if passed_validations == total_validations:
        print(f"🎉 ALL VALIDATIONS PASSED")
        print(f"✅ Week 12 ready for git commit!")
        return True
    else:
        print(f"❌ {total_validations - passed_validations} validation(s) failed")
        print(f"🚫 Fix issues before git commit")
        return False

if __name__ == '__main__':
    sys.path.append('.')
    
    success = main()
    sys.exit(0 if success else 1)