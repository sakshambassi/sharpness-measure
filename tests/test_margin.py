""" Tests for sharpness

To run all tests inside directory Tests/, 
run `python -m unittest discover tests` from the repo directory

"""
import logging
import torch
import unittest

class TestMargin(unittest.TestCase):
    def test_import(self):
        """ Tests import sharpness
        """
        try:
            from strikegen.margin import Margin
            strike_margin = Margin()
        except Exception as e:
            logging.info(f"Exception thrown while testing import: {e}")
            raise e
    
    def test_mean_margin_success(self):
        """ Test to check if mean margin runs fine
        """
        from strikegen.margin import Margin
        strike_margin = Margin()
        x = torch.rand((5,3))
        sx = torch.nn.functional.softmax(x, dim=-1)
        labels = torch.randint(3, (5, 1))
        # margin for the given sample
        margin = strike_margin.calculate(probabilities=sx,
                        true_label_indexes=labels,
                        margin_type='mean'
                    )
        self.assertTrue(type(margin) is float)
        logging.info(f"Got successful margin = {margin}")


if __name__ == "__main__":
    unittest.main()
