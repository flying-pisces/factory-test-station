# Manufacturing Line Testing Framework

## Overview
Centralized testing framework for the Manufacturing Line Control System with automated test execution, logging, and git commit tracking.

## Directory Structure
```
testing/
├── README.md                          # This file
├── scripts/                          # Test execution scripts
│   ├── run_all_tests.py              # Master test runner
│   ├── run_week1_tests.py             # Week 1 specific tests
│   ├── run_week2_tests.py             # Week 2 specific tests
│   └── run_performance_tests.py       # Performance benchmarking
├── logs/                             # Test execution logs
│   ├── test_runs/                    # Individual test run logs
│   ├── performance/                  # Performance benchmark logs
│   └── git_tracking/                 # Git commit tracking logs
├── reports/                          # Test reports and summaries
│   ├── week1/                        # Week 1 test reports
│   ├── week2/                        # Week 2 test reports
│   └── integration/                  # Integration test reports
├── fixtures/                         # Test data and fixtures
│   ├── component_data/               # Sample component data
│   ├── station_configs/              # Sample station configurations
│   └── expected_results/             # Expected test results
└── validators/                       # Test result validators
    ├── performance_validator.py      # Performance validation
    ├── result_validator.py           # Result validation
    └── coverage_validator.py         # Coverage validation
```

## Usage

### Run All Tests
```bash
cd testing/scripts
python run_all_tests.py
```

### Run Week-Specific Tests
```bash
python run_week1_tests.py
python run_week2_tests.py
```

### Run Performance Tests
```bash
python run_performance_tests.py --target-ms 100
```

## Test Tracking
- All test runs are logged with timestamps
- Git commit hash is recorded for each test run
- Performance metrics are tracked over time
- Test results are validated against expected outcomes

## Test Categories
1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Cross-component interaction testing
3. **System Tests**: End-to-end functionality testing
4. **Performance Tests**: Timing and resource usage validation
5. **Acceptance Tests**: Business requirement validation