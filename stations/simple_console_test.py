#!/usr/bin/env python3
"""
Simple console test runner to demonstrate basic functionality.
"""
import os
import sys
import time
import platform
import station_config

def test_basic_functionality():
    """Test basic station functionality without complex imports."""
    print("🚀 Starting simple console test...")
    
    # Load station configuration
    print("📋 Loading station configuration...")
    station_config.load_station('project_station')
    print(f"✅ Station loaded: {station_config.STATION_TYPE}")
    print(f"   Station number: {station_config.STATION_NUMBER}")
    print(f"   Active: {station_config.IS_STATION_ACTIVE}")
    
    # Simulate a simple test sequence
    print("\n🧪 Running simple test sequence...")
    
    # Test item 1: Basic calculation
    test_value_1 = 1.1
    print(f"   Test Item 1: Measured value = {test_value_1}")
    test_1_pass = 1.0 <= test_value_1 <= 2.0
    print(f"   Test Item 1: {'✅ PASS' if test_1_pass else '❌ FAIL'}")
    
    # Test item 2: Another calculation
    test_value_2 = 1.4  
    print(f"   Test Item 2: Measured value = {test_value_2}")
    test_2_pass = True  # No limits set, always pass
    print(f"   Test Item 2: {'✅ PASS' if test_2_pass else '❌ FAIL'}")
    
    # Test item 3: Directory check (non-parametric)
    test_3_pass = os.path.exists(station_config.RAW_DIR)
    print(f"   Test Item 3: Directory check = {test_3_pass}")
    print(f"   Test Item 3: {'✅ PASS' if test_3_pass else '❌ FAIL'}")
    
    overall_result = test_1_pass and test_2_pass and test_3_pass
    
    print(f"\n📊 Overall Result: {'✅ PASS' if overall_result else '❌ FAIL'}")
    
    return overall_result

def interactive_mode():
    """Simple interactive testing mode."""
    print("\n" + "="*50)
    print("   SIMPLE CONSOLE TEST MODE")
    print("="*50)
    print("Enter serial numbers to test, 'quit' to exit")
    
    while True:
        try:
            serial_number = input("\nSerial Number: ").strip()
            
            if serial_number.lower() in ['quit', 'exit', 'q']:
                break
                
            if not serial_number:
                continue
            
            print(f"\n🔬 Testing unit: {serial_number}")
            start_time = time.time()
            
            result = test_basic_functionality()
            
            elapsed = time.time() - start_time
            print(f"⏱️  Test completed in {elapsed:.2f} seconds")
            
            if result:
                print(f"🎉 Unit {serial_number} PASSED")
            else:
                print(f"💥 Unit {serial_number} FAILED")
                
        except KeyboardInterrupt:
            print("\n👋 Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main entry point."""
    print(f"🌍 Platform: {platform.system()}")
    print(f"🐍 Python: {sys.version}")
    
    try:
        if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
            print("""
Simple Console Test Runner

Usage:
    python simple_console_test.py [serial_number]

Options:
    --help, -h    Show this help message
    
Examples:
    python simple_console_test.py              # Interactive mode
    python simple_console_test.py TEST123      # Test specific serial number
""")
            return
        
        if len(sys.argv) > 1:
            # Single test mode
            serial_number = sys.argv[1]
            print(f"\n🔬 Testing unit: {serial_number}")
            result = test_basic_functionality()
            print(f"🎯 Final result: {'PASS' if result else 'FAIL'}")
            sys.exit(0 if result else 1)
        else:
            # Interactive mode
            interactive_mode()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()