"""
Test runner for DirectKV correctness tests.

Runs tests/test_directkv_correctness.py via unittest.
The ae_python virtualenv must be active and install.sh must have been run first.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))
loader = unittest.TestLoader()
suite = loader.loadTestsFromName('test_directkv_correctness')
runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
result = runner.run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
