class Target:
    TARGET_TYPES = ['classification', 'regression']
    TYPE_CLASSIFICATION = TARGET_TYPES[0]
    TYPE_REGRESSION = TARGET_TYPES[1]
    REGRESSION_VALUES = [float('-inf'), float('inf')]
    REGRESSION_DECREASE = REGRESSION_VALUES[0]
    REGRESSION_INCREASE = REGRESSION_VALUES[1]

    def __init__(self, target_type, target_feature, target_value):
        self._target_type = target_type
        self._target_value = target_value
        self._target_feature = target_feature

        assert target_type in self.TARGET_TYPES
        if target_type == self.TYPE_CLASSIFICATION:
            assert isinstance(target_value, int) or isinstance(target_value, str)
        elif target_type == self.TYPE_REGRESSION:
            if self.is_range():
                self._target_value = [float(self._target_value[0]), float(self._target_value[1])]
                assert self._target_value[0] < self._target_value[1]
            else:
                self._target_value = float(self._target_value)
                assert self._target_value in self.REGRESSION_VALUES

    def is_range(self):
        if isinstance(self._target_value, str) and "," in self._target_value:
            self._target_value = self._target_value.split(",")
        return (isinstance(self._target_value, tuple) or isinstance(self._target_value, list)) and len(
            self._target_value) == 2

    def target_type(self):
        return self._target_type

    def target_value(self):
        return self._target_value

    def target_feature(self):
        return self._target_feature

    def __str__(self):
        return self._target_type + " " + self._target_feature + " " + str(self._target_value)

    def target_value_as_string(self):
        return "({},{})".format(str(self._target_value[0]), str(self._target_value[1])) if self.is_range() \
            else self._target_value
