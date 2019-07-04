import re
import os

from src.feature_extraction.feature import Feature
from utils import get_project_root


class WordCount(Feature):
    def __init__(self, column, new_column):
        """
        WordCount feature. Counts the tokens in a sentence.

        :param column: str, the name of the feature we want to apply our pre-processing upon.
        :param new_column: str., the name of the new feature
        """
        Feature.__init__(self, column, new_column)

    def _process(self, data):
        return data.apply(lambda x: len((re.sub(' +', ' ', re.sub(r'[^a-zA-Z0-9 ]', '', x))).strip().split(' ')))


class Negation(Feature):
    def __init__(self, column, new_column):
        """
        Negation feature.

        :param column: str, the name of the feature we want to apply our pre-processing upon.
        :param new_column: str., the name of the new feature
        """
        Feature.__init__(self, column, new_column)

    def _process(self, data):
        return data.apply(lambda x: any(n in x for n in [' no ', ' not ', 'n\'t ']))


class WordSentiment(Feature):
    def __init__(self, column, new_column, has_positive):
        """
        Word sentiment feature.

        :param column: str, the name of the feature we want to apply our pre-processing upon.
        :param new_column: str., the name of the new feature
        :param has_positive: boolean.,  is true the sentiment is positive, if false the sentiment is negative.
        """
        Feature.__init__(self, column, new_column)
        self.has_positive = has_positive

        if self.has_positive:
            f = os.path.join(get_project_root(), 'datasets/vocabularies_and_collections/positive-words.txt')
            self.positive_words = open(f, encoding='utf-8', errors='ignore').read().splitlines()
        else:
            f = os.path.join(get_project_root(), 'datasets/vocabularies_and_collections/negative-words.txt')
            self.negative_words = open(f, encoding='utf-8', errors='ignore').read().splitlines()

    def _process(self, data):
        if self.has_positive:
            data = data.apply(lambda x: sum(n in x for n in self.positive_words))
        else:
            data = data.apply(lambda x: sum(n in x for n in self.negative_words))

        return data


class SentenceLength(Feature):
    def __init__(self, column, new_column):
        """
        Sentence length feature. Count the length of the characters of a sentence.

        :param column: str, the name of the feature we want to apply our pre-processing upon.
        :param new_column: str., the name of the new feature
        """
        Feature.__init__(self, column, new_column)

    def _process(self, data):
        return data.apply(len)


class HasUrl(Feature):
    def __init__(self, column, new_column):
        """
        Has url feature.

        :param column: str, the name of the feature we want to apply our pre-processing upon.
        :param new_column: str., the name of the new feature
        """
        Feature.__init__(self, column, new_column)

    def _process(self, data):
        return data.apply(lambda x: bool(re.search('http(s)?://', x)))


class CountQuestionMark(Feature):
    def __init__(self, column, new_column):
        """
        Count the number of question marks feature.

        :param column: str, the name of the feature we want to apply our pre-processing upon.
        :param new_column: str., the name of the new feature
        """
        Feature.__init__(self, column, new_column)

    def _process(self, data):
        return data.apply(lambda x: x.count('?'))


class CountExclamationMark(Feature):
    def __init__(self, column, new_column):
        """
        Count the number of exclamation marks feature.

        :param column: str, the name of the feature we want to apply our pre-processing upon.
        :param new_column: str., the name of the new feature
        """
        Feature.__init__(self, column, new_column)

    def _process(self, data):
        return data.apply(lambda x: x.count('!'))
