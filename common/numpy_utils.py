import numpy as np
import numpy.random as rnd
from scipy.stats import truncnorm


def not_repeated(known_instances, new_instances):
    last_idx = len(known_instances) - 1
    instances = np.concatenate((known_instances, new_instances))
    _, idx_arr = np.unique(instances, axis=0, return_index=True)
    idx_arr = idx_arr[idx_arr > last_idx]
    instances = instances[idx_arr]
    return instances


def unique_concatenate(known_instances, new_instances):
    instances = np.concatenate((known_instances, new_instances))
    return np.unique(instances, axis=0)


class RandomDistMocker:
    def __init__(self, value):
        self._value = value

    def rvs(self, sample_size):
        return np.repeat(self._value, sample_size)


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    if sd < 1 or low == upp:
        return RandomDistMocker(mean)
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def normal_dist_sample(means, sds, bottoms, tops, sample_size):
    normal_distributions = [get_truncated_normal(means[k], sds[k], bottoms[k], tops[k]) for k in range(len(means))]
    return [np.round(nd.rvs(sample_size)) for nd in normal_distributions]


def uniform_dist_sample(bottoms, tops, sample_size):
    features = [np.floor(rnd.uniform(bottoms[k], tops[k] + 1, sample_size)) for k in range(len(bottoms))]
    return np.array(features)


def random_pick(column_labels, sample_size):
    column_values = [np.random.choice(column_labels[idx], sample_size) for idx in range(len(column_labels))]
    return np.reshape(column_values, (len(column_values), sample_size))
