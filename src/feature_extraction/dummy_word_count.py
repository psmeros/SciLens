from src.feature_extraction.feature import Feature


class WordCount(Feature):
    def __init__(self, columns, new_column):
        """
        WordCount feature. Counts the tokens in a sentence.

        :param columns: list[str], the list of the names of features we want to apply our pre-processing upon.
        :param new_column: str., the name of the new feature
        """
        Feature.__init__(self, columns, new_column)

    def _process(self, data):
        """
        This method performs the main logic of the class. by counting the tokens in a sentence.

        :param data: pd.Dataframe
        :return: pd.Dataframe
        """
        for column in self.columns:
            data = data[column].apply(lambda x: len(x.split()))
        return data
