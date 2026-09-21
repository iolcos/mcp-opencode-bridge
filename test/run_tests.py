"""
Runs the whole test suite: one module per server implementation, plus the
shared trunk. Add a future integration's test module to TEST_MODULES when it
joins serveur/orca.py and serveur/generic.py.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

TEST_MODULES = ["test_package", "test_orca", "test_generic"]
# Add here the future test_<nouvelle_intégration>.py the day a new
# integration (another "front" than Orca) joins serveur/orca.py and
# serveur/generic.py.


def load_tests() -> unittest.TestSuite:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in TEST_MODULES:
        suite.addTests(loader.loadTestsFromName(name))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(load_tests())
    raise SystemExit(0 if result.wasSuccessful() else 1)
