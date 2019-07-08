import pickle
import os

from utils import get_project_root


class MLModelTrainer:
    def __init__(self, ml_model, cross_validation=None, project_name=None):
        """
        This class is responsible for training, validate and predict a given sklearn ml model.

        :param cross_validation: sklearn.model_selection object., supported validations are: "k-fold" and "cross-validation"
        :param ml_model: sklearn model object
        """
        self.ml_model = ml_model
        self.cross_validation = cross_validation
        self.model_name = '{}_{}_{}.{}'.format(project_name,
                                               ml_model.__class__.__name__,
                                               cross_validation.__class__.__name__,
                                               'pkl')

    def train(self, data):
        """
        This method performs a model fit according to the cross validation strategy that is set by the caller.
        In case of no provided validation strategy, the model fitting is performed on all the provided data
        and a single validation score on train data is returned.
        After model fitting, the trained model is stored in the `model` dir.

        :param data: dict. of np.arrays with the features and the targets.
        :return: list of float, the performance scores of each iteration of the training process
        """
        x = data['x']
        y = data['y']

        history = []

        if self.cross_validation:

            # run cross validation while fitting the model
            for train_index, test_index in self.cross_validation.split(x):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]

                self.ml_model.fit(x_train, y_train)
                history.append(self.ml_model.score(x_test, y_test))  # store performance of each iteration
        else:
            self.ml_model.fit(x, y)
            history.append(self.ml_model.score(x, y))

        pickle.dump(self.ml_model, open(os.path.join(get_project_root(), 'model', self.model_name), 'wb'))
        return history

    def validate(self, data):
        """
        This method evaluates a given dataset to an existing pre-trained model. The score it computes and returned
        is aligned with the type of the problem and the respective sklearn model used.

        :param data: dict. of np.arrays with the features and the targets.
        :return: float, the performance score
        """
        x = data['x']
        y = data['y']

        return self.ml_model.score(x, y)

    def predict(self, data):
        """
        This method perform predictions upon unseen data and returns the predicted y for each input sample.

        :param data: np.array with the given inputs
        :return: np.array with the predicted outputs
        """
        return self.ml_model.predict(data)
