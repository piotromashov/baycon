from sklearn.preprocessing import LabelEncoder


class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode
        self.encoders = [None] * len(columns)

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X):
        """
        Transforms columns of X specified in self.columns using
        LabelEncoder().
        """
        output = X
        for idx, encode in enumerate(self.columns):
            if encode:
                self.encoders[idx] = LabelEncoder().fit(output[:, idx])
                output[:, idx] = self.encoders[idx].transform(output[:, idx])
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        # output = np.array([]*len(X.shape[1]))
        for idx, decode in enumerate(self.columns):
            if decode:
                X[:, idx] = self.encoders[idx].inverse_transform(X[:, idx].astype(int))
