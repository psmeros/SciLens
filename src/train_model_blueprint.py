"""
This script serves as a blueprint for training a machine learning model and test it in unseen data.

Steps:
- Fetch processed data
- Create x and y arrays
- Train a model
- Validate a model
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from src.etl.fetcher_factory import get_fetcher_from_source
from src.modeling.utils import split_train_test
from src.modeling.ml_model import MLModel


if __name__ == '__main__':

    # fetch data
    data_fetcher = get_fetcher_from_source('CSV', params={'path': 'datasets/processed/processed_sample_data.csv'})
    processed_data = data_fetcher.get_data(separator=',')

    data = {'x': processed_data[['word_count', 'word_count']].values,
            'y': processed_data['target'].values}

    # split to train and test data
    data = split_train_test(data, test_ratio=0.5)

    # set classifier and cross-validation
    classifier = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1)
    kf = KFold(n_splits=2, shuffle=True)

    # instantiate trainer
    trainer = MLModel(ml_model=classifier, cross_validation=kf, project_name='example_model')

    # train the model
    history = trainer.train(data['train'])
    print(history)

    # evaluate the model
    score = trainer.validate(data['test'])
    print(score)
