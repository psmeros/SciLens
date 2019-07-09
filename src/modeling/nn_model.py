
class NNModel:
    def __init__(self):
        self.model = None

    def build_model(self):
        raise NotImplementedError

    def train(self, data):
        return self.model.fit(data)

    def validate(self):
        raise NotImplementedError("Please implement validate method")

    def predict(self, data):
        return self.model.predict(data)


