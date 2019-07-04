from utils import get_logger


class Feature:
    def __init__(self, columns, new_column):
        """
        Feature class. All other features inherit.

        :param columns: list[str], the list of the names of features we want to apply our pre-processing upon.
        :param new_column: str., the name of the new feature
        """
        self.logger = get_logger('SciLens', 'feature_extraction.log')
        self.columns = columns
        self.new_column = new_column

    def apply(self, data):
        """
        A wrapper of the `_process()` method.

        :param data: pd.Dataframe
        :return: pd.Dataframe
        """
        # Get the sub-space of features of interest that needs to be processed
        data_subset = data[self.columns]

        # Add a new column with the features of interest.
        self.logger.info('Performing feature extraction...')
        data[self.new_column] = self._process(data=data_subset)

        return data

    def _process(self, data):
        """
        Abstract private method that all feature children classes should implement.
        According to each feature logic, this method should return the processed data in Pandas format.

        :param data: pd.Dataframe
        :return: pd.Dataframe
        """
        raise NotImplementedError
