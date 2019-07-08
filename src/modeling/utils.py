import math


def split_train_test(data, test_ratio=0.2):
    """
    This method splits the provided data to train and test sets according to the given `test_ratio` param.

    :param data: dict with np.array
    :param test_ratio: float, the percentage of test data
    :return: dict of dict with np.array
    """
    data_size = data['x'].shape[0]
    splitting_point = data_size - math.ceil(data_size * test_ratio)

    return {
        'train': {'x': data['x'][:splitting_point, :],
                  'y': data['y'][:splitting_point]},
        'test': {'x': data['x'][splitting_point:, :],
                 'y': data['y'][splitting_point:]}
    }
