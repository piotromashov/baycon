class Target:
    TARGET_TYPES = ['classification', 'range', 'regression']
    TYPE_CLASSIFICATION = TARGET_TYPES[0]
    TYPE_RANGE = TARGET_TYPES[1]
    TYPE_REGRESSION = TARGET_TYPES[2]
    REGRESSION_VALUES = ['increase', 'decrease']
    REGRESSION_INCREASE = REGRESSION_VALUES[0]
    REGRESSION_DECREASE = REGRESSION_VALUES[1]

    def __init__(self, target_type, target_feature, target_value):
        assert target_type in self.TARGET_TYPES
        if target_type == self.TYPE_CLASSIFICATION:
            assert isinstance(target_value, int) or isinstance(target_value, str)
        elif target_type == self.TYPE_RANGE:
            assert isinstance(target_value, tuple)
            assert target_value[0] < target_value[1]
        elif target_type == self.TYPE_REGRESSION:
            assert target_value in self.REGRESSION_VALUES
        self._target_value = target_value
        self._target_type = target_type
        self._target_feature = target_feature

    def target_type(self):
        return self._target_type

    def target_value(self):
        return self._target_value

    def target_feature(self):
        return self._target_feature

    def __str__(self):
        return self._target_type + " " + str(self._target_value)
