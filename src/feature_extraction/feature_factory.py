from src.feature_extraction.text_features import *
from src.feature_extraction.twitter_features import *


class FeatureFactory:
    """
    Factory class that returns a feature class according to the called from tha caller class method.

    Currently supported features are the following:
        - word_count: count of tokens in a sentence
    """

    @classmethod
    def word_count(cls, column, new_column):
        return WordCount(column, new_column)

    @classmethod
    def negation(cls, column, new_column):
        return Negation(column, new_column)

    @classmethod
    def word_sentiment(cls, column, new_column, has_positive=True):
        return WordSentiment(column, new_column, has_positive)

    @classmethod
    def sentence_length(cls, column, new_column):
        return SentenceLength(column, new_column)

    @classmethod
    def has_url(cls, column, new_column):
        return HasUrl(column, new_column)

    @classmethod
    def count_question_marks(cls, column, new_column):
        return CountQuestionMark(column, new_column)

    @classmethod
    def count_exclamation_marks(cls, column, new_column):
        return CountExclamationMark(column, new_column)

    @classmethod
    def polarity(cls, column, new_column):
        return Polarity(column, new_column)

    @classmethod
    def subjectivity(cls, column, new_column):
        return Subjectivity(column, new_column)

    @classmethod
    def sentiment(cls, column, new_column):
        return Sentiment(column, new_column)

    @classmethod
    def mean_polarity(cls, column, new_column):
        return MeanPolarity(column, new_column)

    @classmethod
    def mean_subjectivity(cls, column, new_column):
        return MeanSubjectivity(column, new_column)
