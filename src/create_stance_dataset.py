"""
- Fetch data from a source
- Select features and feature transformations
- Perform feature extraction
- Store processed data
"""
from utils import get_project_root
import os

from src.etl.fetcher_factory import get_fetcher_from_source
from src.feature_extraction.feature_factory import FeatureFactory

if __name__ == '__main__':

    # fetch data
    data_fetcher = get_fetcher_from_source('CSV', params={'path': 'datasets/raw/scilens_3M_replies.tsv'})
    data = data_fetcher.get_data(separator='\t')

    data = data.rename(index=str, columns={"full_text": "tweet"})

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
    data.to_csv(os.path.join(get_project_root(), 'datasets/processed/tweet_replies.csv'),
                sep=',',
                encoding='utf-8')
