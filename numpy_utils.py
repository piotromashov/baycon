import numpy as np
import numpy.random as rnd
from scipy.stats import truncnorm

RANDOM_SAMPLE_SIZE = 1000
NEIGHBOURS_SAMPLE_SIZE = 100


# distance metrics between two samples
def distance(alternative, template):
    distance = np.abs(alternative - template)
    return sum(distance)


# same as distance() applied over an array of alternatives
def distance_arr(alternatives, template):
    distance = np.abs(alternatives - template)
    return np.sum(distance, axis=1)


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


def normal_dist_sample(means, sds, bottoms, tops):
    normal_distributions = [get_truncated_normal(means[k], sds[k], bottoms[k], tops[k]) for k in range(len(means))]
    features = [np.round(nd.rvs(NEIGHBOURS_SAMPLE_SIZE)) for nd in normal_distributions]
    samples = np.array(features).transpose()
    return samples


def uniform_dist_sample(bottoms, tops):
    features = [np.floor(rnd.uniform(bottoms[k], tops[k] + 1, RANDOM_SAMPLE_SIZE)) for k in range(len(bottoms))]
    samples = np.array(features).transpose()
    return samples
