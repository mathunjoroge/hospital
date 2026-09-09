#!/usr/bin/env python3
"""Dump ESI triage module structure to plan integration."""
import os

files = [
    "departments/nursing/triage.py",
    "departments/models/nursing.py"
]

for f in files:
    if os.path.exists(f):
        print(f"\n{'='*60}")
        print(f"FILE: {f}")
        print('='*60)
        with open(f, "r", encoding="utf-8") as fh:
            content = fh.read()
        
        # Print imports, classes, functions, and columns
        for line in content.split('\n'):
            stripped = line.strip()
            if (stripped.startswith('import ') or stripped.startswith('from ') or 
                stripped.startswith('class ') or stripped.startswith('def ') or 
                '    def ' in line or 'Column' in line or 'relationship' in line):
                print(line)