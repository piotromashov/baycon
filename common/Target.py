TARGET_TYPES = ['classification', 'range', 'numerical']


class Target:
    def __init__(self, target_type, target_value):
        assert target_type in TARGET_TYPES
        if target_type == TARGET_TYPES[0]:
            assert isinstance(target_value, int) or isinstance(target_value, str)
        elif target_type == TARGET_TYPES[1]:
            assert isinstance(target_value, tuple)
            assert target_value[0] < target_value[1]
        else:
            assert isinstance(target_value, bool)
        self._target_value = target_value
        self._target_type = target_type

    def target_type(self):
        return self._target_type

    def target_value(self):
        return self._target_value

    def __str__(self):
        return self._target_type + " " + str(self._target_value)
