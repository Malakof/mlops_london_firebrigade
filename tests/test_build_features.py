# tests/test_build_features.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import unittest
# from src.features import build_features

class TestBuildFeatures(unittest.TestCase):

    def test_load_data(self):
        # This test would check if data is loaded correctly, possibly by checking the output shape
        pass

    def test_clean_data(self):
        # Test if cleaning operations such as dropping and renaming are done correctly
        pass

if __name__ == '__main__':
    unittest.main()
