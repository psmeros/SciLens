from src.etl.twitter_fetcher import *
from src.etl.csv_fetcher import *
from src.etl.cwur_fetcher import *
from src.etl.fetcher import Fetcher


def get_fetcher_from_source(source, params=None) -> Fetcher:
    """
    Factory method that returns the data fetcher class according to requested source.
    Currently supported sources are the following:
        - Twitter: 'Twitter'
        - CSV: 'CSV'
        - World University Rankings: 'CWUR'

    :param source: str. the source name
    :param params: dict
    :return: Fetcher object
    """

    fetcher_cl = eval('{}Fetcher'.format(source))
    fetcher = fetcher_cl(params=params)
    return fetcher
