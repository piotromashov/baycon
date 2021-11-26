class Target:
    TARGET_TYPES = ['classification', 'regression']
    TYPE_CLASSIFICATION = TARGET_TYPES[0]
    TYPE_REGRESSION = TARGET_TYPES[1]
    REGRESSION_VALUES = [float('-inf'), float('inf')]
    REGRESSION_DECREASE = REGRESSION_VALUES[0]
    REGRESSION_INCREASE = REGRESSION_VALUES[1]

    def __init__(self, target_type, target_feature, target_value):
        assert target_type in self.TARGET_TYPES
        if target_type == self.TYPE_CLASSIFICATION:
            assert isinstance(target_value, int) or isinstance(target_value, str)
        elif target_type == self.TYPE_REGRESSION:
            self._target_value = target_value
            if self.is_range():
                assert target_value[0] < target_value[1]
            else:
                target_value = float(target_value)
                assert target_value in self.REGRESSION_VALUES

        self._target_value = target_value
        self._target_type = target_type
        self._target_feature = target_feature

    def is_range(self):
        return isinstance(self._target_value, tuple)

    def target_type(self):
        return self._target_type

    def target_value(self):
        return self._target_value

    def target_feature(self):
        return self._target_feature

    def __str__(self):
        return self._target_type + " " + str(self._target_value)
