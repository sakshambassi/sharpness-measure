""" Tests for sharpness

To run all tests inside directory Tests/, 
run `python -m unittest discover tests` from the repo directory

"""
import logging
import unittest

class TestSharpness(unittest.TestCase):
    def test_import(self):
        """ Tests import sharpness
        """
        try:
            from strikegen.sharpness import Sharpness
            strike_sharpness = Sharpness()
        except Exception as e:
            logging.info(f"Exception thrown while testing import: {e}")
            raise e

if __name__ == "__main__":
    unittest.main()
