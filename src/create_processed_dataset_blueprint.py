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
    data_fetcher = get_fetcher_from_source('CSV', params={'path': 'datasets/raw/sample_data.csv'})
    data = data_fetcher.get_data(separator=',')

    # set features configuration
    feature_set = [FeatureFactory.word_count('tweet', 'word_count')]

    # perform feature extraction
    for feature in feature_set:
        data = feature.apply(data)

    # store df
    data.to_csv(os.path.join(get_project_root(), 'datasets/processed/new_sample_data.csv'),
                sep=',',
                encoding='utf-8')
