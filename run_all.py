#!/usr/bin/env python3
"""Run both research matchers under the same protocol."""
import subprocess,sys
for matcher in ['minutiae','orb']:
    subprocess.run([sys.executable,'run.py','--matcher',matcher,'--output',f'results/{matcher}',*sys.argv[1:]],check=True)
