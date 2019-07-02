from src.etl.fetcher import Fetcher


class TwitterFetcher(Fetcher):
    """
    CSV fetcher. Loads data from CSV files and returns a pd.Dataframe.
    """
    def __init__(self):
        Fetcher.__init__(self)

    def _fetch(self, **kwargs):
        raise NotImplementedError
