"""Comprehensive Test Suite for Manufacturing Line System.

This package contains unit, integration, system, and acceptance tests
for all components of the manufacturing line control system.

Test Structure:
- unit/: Component-level unit tests
- integration/: Cross-component integration tests  
- system/: End-to-end system tests
- acceptance/: User acceptance tests
- fixtures/: Test data and mock services
"""

import os
import sys

# Add project root to path for test imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)