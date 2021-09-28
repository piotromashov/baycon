from DataConstraints import DataConstraints
import unittest
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import KBinsDiscretizer

class Test_DataConstraints(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_features_possibilities(self):
        dataset = fetch_openml(name='diabetes', version=1)
        discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy='uniform')
        discrete_dataset = discretizer.fit_transform(dataset.data)
        data_constraints = DataConstraints(discrete_dataset)

        features_possibilities = int(data_constraints.features_possibilities_all())

        self.assertEqual(features_possibilities, dataset.data.shape[1]*discretizer.n_bins)
