import pandas as pd
import os

from src.etl.fetcher import Fetcher
from utils import get_project_root


class CSVFetcher(Fetcher):
    """
    CSV fetcher. Loads data from CSV files and returns a pandas dataframe.
    """
    def __init__(self):
        Fetcher.__init__(self)
        self.path = os.path.join(get_project_root(), 'datasets/raw/scilens_3M_replies.tsv')

    def _fetch(self, **kwargs):
        """
        Implements the main logic of the class. It reads csv file and return in to pd Dataframe.

        :param kwargs: -separator: str. the delimiter of the file
        :return: pd.Dataframe
        """
        return self._read_csv(kwargs.get('separator', None))

    def _read_csv(self, separator):
        """
        Reads a csv file and returns the pd Dataframe with its content

        :param separator: str, the delimiter of the file.
        :return: pd.Dataframe
        """
        df = pd.read_csv(self.path, sep=separator)

        return df
