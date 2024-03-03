from sklearn.preprocessing import MinMaxScaler


class Scaler:
    def __init__(self):
        self.data_Scaler = MinMaxScaler()
        self.counter = 0

    def fit(self, data):
        assert self.counter == 0, "The data is already fitted once, cannot do it again!"
        fit_data = self.data_Scaler.fit_transform(data)
        self.counter += 1
        return fit_data

    def transform(self, data):
        transformed_data = self.data_Scaler.transform(data)
        return transformed_data

    def inverse_transform(self, data):
        transformed_data = self.data_Scaler.inverse_transform(data)
        return transformed_data
