import unittest

from src.etl.fetcher_factory import get_fetcher_from_source
from src.etl.fetcher import Fetcher
from utils import get_project_root


class TestFetcherFactory(unittest.TestCase):

    def test_get_fetcher_from_source_on_various_sources(self):
        """
        It tests that the fetcher factory returns a fetcher object according to the requested fetcher.
        """
        requested_fetchers = [('Twitter', None),
                              ('CWUR', None),
                              ('CSV', {'path': get_project_root()})]

        for fetcher in requested_fetchers:
            self.assertIsInstance(get_fetcher_from_source(fetcher[0], params=fetcher[1]), Fetcher)

    def test_get_fetcher_from_source_on_unknown_source(self):
        """
        It tests that the fetcher factory raises the proper exception id unknown fetcher is requested.
        """
        requested_fetcher = 'unknown_fetcher'

        with self.assertRaises(NameError):
            get_fetcher_from_source(requested_fetcher, params=None)


if __name__ == '__main__':
    unittest.main()
