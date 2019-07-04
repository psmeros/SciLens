from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from src.feature_extraction.feature import Feature


class Polarity(Feature):
    def __init__(self, column, new_column):
        """
        Polarity feature.

        :param column: str, the name of the feature we want to apply our pre-processing upon.
        :param new_column: str., the name of the new feature
        """
        Feature.__init__(self, column, new_column)

    def _process(self, data):
        return data.apply(lambda x: TextBlob(x).sentiment.polarity)


class Subjectivity(Feature):
    def __init__(self, column, new_column):
        """
        Subjectivity feature.

        :param column: str, the name of the feature we want to apply our pre-processing upon.
        :param new_column: str., the name of the new feature
        """
        Feature.__init__(self, column, new_column)

    def _process(self, data):
        return data.apply(lambda x: TextBlob(x).sentiment.subjectivity)


class Sentiment(Feature):
    def __init__(self, column, new_column):
        """
        Sentiment feature.

        :param column: str, the name of the feature we want to apply our pre-processing upon.
        :param new_column: str., the name of the new feature
        """
        Feature.__init__(self, column, new_column)

    def _process(self, data):
        sid = SentimentIntensityAnalyzer()
        return data.apply(lambda x: sid.polarity_scores(x))
