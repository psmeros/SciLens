from src.feature_extraction.dummy_word_count import WordCount


class FeatureFactory:
    """
    Factory class that returns a feature class according to the called from tha caller class method.

    Currently supported features are the following:
        - word_count: count of tokens in a sentence
    """

    @classmethod
    def word_count(cls, columns, new_column):
        return WordCount(columns, new_column)
