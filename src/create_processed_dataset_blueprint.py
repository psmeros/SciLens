"""
This scripts serves as a blueprint for creating a processed dataset given a source
and a list of features to be created upon raw data.

Steps:
- Fetch data from a source
- Select features and feature transformations
- Perform feature extraction
- Store processed data

In this toy example, we load a csv file with dummy tweets and replies,
and we perform a word count pre-processing.
"""
from utils import get_project_root
import os

from src.etl.fetcher_factory import get_fetcher_from_source
from src.feature_extraction.feature_factory import FeatureFactory

if __name__ == '__main__':

    # fetch data
    data_fetcher = get_fetcher_from_source('CSV')
    data = data_fetcher.get_data(separator=',')

    # set features configuration
    feature_set = [FeatureFactory.word_count('tweet', 'word_count'),
                   FeatureFactory.negation('tweet', 'negation'),
                   FeatureFactory.word_sentiment('tweet', 'positive', has_positive=True),
                   FeatureFactory.word_sentiment('tweet', 'negative', has_positive=False),
                   FeatureFactory.sentence_length('tweet', 'length'),
                   FeatureFactory.has_url('tweet', 'has_url'),
                   FeatureFactory.count_question_marks('tweet', 'quest_mark'),
                   FeatureFactory.count_exclamation_marks('tweet', 'excl_mark'),
                   FeatureFactory.polarity('tweet', 'polarity'),
                   FeatureFactory.subjectivity('tweet', 'subjectivity'),
                   FeatureFactory.sentiment('tweet', 'sentiment')]

    # perform feature extraction
    for feature in feature_set:
        data = feature.apply(data)

    # store df
    data.to_csv(os.path.join(get_project_root(), 'datasets/processed/new_data.csv'),
                sep=',',
                encoding='utf-8')
