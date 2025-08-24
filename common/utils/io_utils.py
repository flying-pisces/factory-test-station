#!/usr/bin/env python3
"""
I/O utilities for factory test station.
"""

import os
import time
from datetime import datetime

def timestamp(dt=None):
    """Generate timestamp string from datetime object."""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y%m%d-%H%M%S")

def read_line_from_file(filename):
    """Read first line from file."""
    try:
        with open(filename, 'r') as f:
            return f.readline().strip()
    except:
        return ""

def append_results_log_to_csv(csv_file, log_content):
    """Append log content to CSV file."""
    try:
        with open(csv_file, 'a') as f:
            f.write(log_content + '\n')
    except Exception as e:
        print(f"Error writing to CSV file {csv_file}: {e}")

def mkdir_p(path):
    """Create directory and all parent directories."""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {path}: {e}")