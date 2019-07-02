
class FeatureMappingFactory:
    def __init__(self, feature_set):
        self.feature_set = self._feature_set_factory(feature_set)

    @staticmethod
    def _feature_set_factory(feature_set):
        """
        Factory method to instantiate Feature object and return them.

        :param feature_set: list of dict.
        :return: list of dict.
        """
        return [Feature(feature['name'], feature['params']) for feature in feature_set]


class Feature:
    def __init__(self, feature_name, params):
        self.feature = eval(feature_name)
        self.params = params
