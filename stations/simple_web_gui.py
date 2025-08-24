#!/usr/bin/env python3
"""
Simple Flask web-based GUI for factory test stations.
Provides a browser interface for testing using the simple console test.
"""

from flask import Flask, render_template, request, jsonify
import threading
import time
import sys
import os
import subprocess
import json
import platform
import webbrowser
from datetime import datetime
import signal

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import station components
import station_config

app = Flask(__name__)
app.config['SECRET_KEY'] = 'factory_test_secret_key'

# Global variables for test state
test_results = []
test_running = False

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/status')
def status():
    """Get current status."""
    try:
        station_config.load_station('project_station')
        return jsonify({
            'status': 'ready',
            'station_type': station_config.STATION_TYPE,
            'station_number': station_config.STATION_NUMBER,
            'is_active': station_config.IS_STATION_ACTIVE,
            'platform': platform.system(),
            'python_version': sys.version
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        })

@app.route('/run_test', methods=['POST'])
def run_test():
    """Run a test for the given serial number."""
    global test_running
    
    if test_running:
        return jsonify({'error': 'Test already running'}), 400
    
    data = request.json
    serial_number = data.get('serial_number', '').strip()
    
    if not serial_number:
        return jsonify({'error': 'Serial number required'}), 400
    
    test_running = True
    
    try:
        # Run the simple console test as a subprocess
        result = subprocess.run([
            sys.executable, 'simple_console_test.py', serial_number
        ], capture_output=True, text=True, timeout=30)
        
        test_result = {
            'serial_number': serial_number,
            'timestamp': datetime.now().isoformat(),
            'passed': result.returncode == 0,
            'output': result.stdout,
            'error': result.stderr if result.stderr else None,
            'duration': 0  # We don't track duration in subprocess mode
        }
        
        test_results.append(test_result)
        test_running = False
        
        return jsonify(test_result)
        
    except subprocess.TimeoutExpired:
        test_running = False
        return jsonify({'error': 'Test timeout'}), 500
    except Exception as e:
        test_running = False
        return jsonify({'error': str(e)}), 500

@app.route('/test_results')
def get_test_results():
    """Get all test results."""
    return jsonify(test_results)

@app.route('/clear_results', methods=['POST'])
def clear_results():
    """Clear all test results."""
    global test_results
    test_results = []
    return jsonify({'message': 'Results cleared'})

def create_html_template():
    """Create the HTML template if it doesn't exist."""
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    template_path = os.path.join(template_dir, 'index.html')
    
    if not os.path.exists(template_path):
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Factory Test Station - Web Interface</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        .status-panel { background: #e8f4fd; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .test-panel { background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .results-panel { background: #f8f9fa; padding: 15px; border-radius: 5px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input[type="text"] { width: 300px; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin-right: 10px; }
        button:hover { background: #0056b3; }
        button:disabled { background: #6c757d; cursor: not-allowed; }
        .btn-danger { background: #dc3545; }
        .btn-danger:hover { background: #c82333; }
        .test-result { padding: 10px; margin-bottom: 10px; border-radius: 4px; border-left: 4px solid #ddd; }
        .test-result.pass { border-left-color: #28a745; background: #d4edda; }
        .test-result.fail { border-left-color: #dc3545; background: #f8d7da; }
        .test-output { font-family: monospace; font-size: 12px; white-space: pre-wrap; background: #f8f9fa; padding: 10px; border-radius: 4px; margin-top: 5px; }
        #loading { display: none; color: #007bff; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏭 Factory Test Station - Web Interface</h1>
        
        <div class="status-panel">
            <h3>📊 Station Status</h3>
            <div id="status">Loading...</div>
        </div>
        
        <div class="test-panel">
            <h3>🔬 Run Test</h3>
            <div class="form-group">
                <label for="serial_number">Serial Number:</label>
                <input type="text" id="serial_number" placeholder="Enter serial number" onkeypress="if(event.key==='Enter') runTest()">
            </div>
            <button onclick="runTest()" id="runBtn">Run Test</button>
            <button onclick="clearResults()" class="btn-danger">Clear Results</button>
            <div id="loading">🔄 Running test...</div>
        </div>
        
        <div class="results-panel">
            <h3>📋 Test Results</h3>
            <div id="results">No tests run yet.</div>
        </div>
    </div>

    <script>
        let testResults = [];

        function loadStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'ready') {
                        document.getElementById('status').innerHTML = `
                            ✅ <strong>Station Ready</strong><br>
                            Station: ${data.station_type} #${data.station_number}<br>
                            Active: ${data.is_active ? 'Yes' : 'No'}<br>
                            Platform: ${data.platform}<br>
                            Python: ${data.python_version.split(' ')[0]}
                        `;
                    } else {
                        document.getElementById('status').innerHTML = `
                            ❌ <strong>Station Error:</strong> ${data.error}
                        `;
                    }
                })
                .catch(error => {
                    document.getElementById('status').innerHTML = `
                        ❌ <strong>Connection Error:</strong> ${error.message}
                    `;
                });
        }

        function runTest() {
            const serialNumber = document.getElementById('serial_number').value.trim();
            if (!serialNumber) {
                alert('Please enter a serial number');
                return;
            }

            document.getElementById('runBtn').disabled = true;
            document.getElementById('loading').style.display = 'block';

            fetch('/run_test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ serial_number: serialNumber })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert('Error: ' + data.error);
                } else {
                    testResults.unshift(data);
                    updateResults();
                    document.getElementById('serial_number').value = '';
                }
            })
            .catch(error => {
                alert('Error: ' + error.message);
            })
            .finally(() => {
                document.getElementById('runBtn').disabled = false;
                document.getElementById('loading').style.display = 'none';
            });
        }

        function clearResults() {
            fetch('/clear_results', { method: 'POST' })
                .then(() => {
                    testResults = [];
                    updateResults();
                });
        }

        function updateResults() {
            const resultsDiv = document.getElementById('results');
            
            if (testResults.length === 0) {
                resultsDiv.innerHTML = 'No tests run yet.';
                return;
            }

            resultsDiv.innerHTML = testResults.map(result => `
                <div class="test-result ${result.passed ? 'pass' : 'fail'}">
                    <strong>${result.serial_number}</strong> - 
                    ${result.passed ? '✅ PASS' : '❌ FAIL'} - 
                    ${new Date(result.timestamp).toLocaleString()}
                    <div class="test-output">${result.output}</div>
                </div>
            `).join('');
        }

        // Load status on page load
        loadStatus();
        
        // Refresh status every 30 seconds
        setInterval(loadStatus, 30000);
        
        // Focus serial number input
        document.getElementById('serial_number').focus();
    </script>
</body>
</html>'''
        
        with open(template_path, 'w') as f:
            f.write(html_content)
        
        print(f"✅ Created HTML template at: {template_path}")

def open_browser():
    """Open browser after a short delay."""
    time.sleep(1.5)
    try:
        webbrowser.open('http://localhost:5000')
        print("🌐 Browser opened to http://localhost:5000")
    except Exception as e:
        print(f"⚠️  Could not open browser: {e}")
        print("   Please manually open: http://localhost:5000")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n👋 Shutting down web server...")
    sys.exit(0)

def main():
    """Main entry point."""
    print("🚀 Starting Factory Test Station Web Interface...")
    print(f"🌍 Platform: {platform.system()}")
    
    # Create HTML template
    create_html_template()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start browser in background thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    try:
        print("🌐 Starting web server on http://localhost:5000")
        print("📝 Press Ctrl+C to stop")
        
        # Run Flask app
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Web server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()