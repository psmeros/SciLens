import unittest
import inspect

from src.feature_extraction.feature_factory import FeatureFactory
from src.feature_extraction.feature import Feature


class TestFetcherFactory(unittest.TestCase):

    def test_feature_factory_on_various_features(self):
        """
        It tests that all class methods of feature factory return a `Feature` type of object.
        """
        class_methods = inspect.getmembers(FeatureFactory, predicate=inspect.ismethod)

        for class_method in class_methods:
            self.assertIsInstance(class_method[1]('', ''), Feature)


if __name__ == '__main__':
    unittest.main()