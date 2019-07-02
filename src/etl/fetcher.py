import pandas as pd
from utils import get_logger


class Fetcher:
    """ Base Fetcher class. All other fetchers inherit. """

    def __init__(self):
        self.logger = get_logger('SciLens', 'fetcher_log.log')

    def get_data(self, **kwargs):
        """
        This method serves as a wrapper to the custom logic of each fetcher sub-class.

        :return: pd Dataframe
        """
        data = self._fetch(**kwargs)

        if isinstance(data, pd.DataFrame):
            self.logger.info("Dataframe fetched")
            return data
        else:
            self.logger.error("Returned non-Dataframe type")
            raise Exception("Returned non-Dataframe type")

    def _fetch(self, **kwargs):
        """
        Abstract private method that all fetcher children classes should implement.
        According to each fetcher mechanism, this method should return the provided data in Pandas format.

        :return: pd Dataframe
        """
        raise NotImplementedError("Please implement _load method")



