from sklearn.preprocessing import LabelEncoder


class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode
        self.encoders = None

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X):
        """
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        """
        output = X
        if self.columns is not None and len(self.columns):
            self.encoders = dict.fromkeys(self.columns)
            for col in self.columns:
                self.encoders[col] = LabelEncoder().fit(output[col])
                output[col] = self.encoders[col].transform(output[col])
        else:
            self.encoders = dict.fromkeys(output.columns)
            for colname, col in output.iteritems():
                self.encoders[colname] = LabelEncoder().fit(col)
                output[colname] = self.encoders[colname].transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        output = X.copy()
        for colname, encoder in self.encoders.items():
            output[colname] = encoder.inverse_transform(output[colname])
        return output
