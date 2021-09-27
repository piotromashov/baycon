import unittest
import numpy_utils as npu
import numpy as np

class Test_Distance(unittest.TestCase):
    def test_distance(self):
        anInstance = np.array([1,2,3])
        anotherInstance = np.array([3,2,1])
        self.assertEqual(npu.distance(anInstance, anotherInstance), 4)

    def test_distance_negative_instances(self):
        aNegativeFeatureInstance = np.array([-2,-4,-1])
        anotherInstance = np.array([1,2,3])
        self.assertEqual(npu.distance(aNegativeFeatureInstance, anotherInstance), 13)


class Test_DistanceArr(unittest.TestCase):
    def test_distanceArray(self):
        pass

class Test_RemoveDuplicates(unittest.TestCase):
    def test_removesAllDuplicates(self):
        instancesA = [np.array([1,2,3,4]),np.array([1,1,1,1])]
        instance2s = np.array([2,2,2,2])
        instancesB = [instance2s,np.array([1,2,3,4])]

        uniqueInstances = npu.not_repeated(instancesA,instancesB)

        self.assertTrue(instance2s in uniqueInstances)

if __name__ == '__main__':
    unittest.main()