from src.etl.twitter_fetcher import *
from src.etl.csv_fetcher import *


def get_fetcher_from_source(source, params=None):
    """
    Factory method that returns the data fetcher class according to requested source.
    Currently supported sources are the following:
        - Twitter: 'twitter', 'Twitter', 'tweet', 'tweets'
        - CSV: 'csv'

    :param source: str. the source name
    :param params: dict
    :return: Fetcher object
    """

    fetcher_cl = eval('{}Fetcher'.format(source))
    fetcher = fetcher_cl(params)
    return fetcher
