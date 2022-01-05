import glob
import json
import os


def check_numerical(potential_float):
    try:
        float(potential_float)
        return True
    except ValueError:
        return False


def _decode(o):
    if isinstance(o, str) and check_numerical(o):
        try:
            return int(o)
        except ValueError:
            return float(o)
    elif isinstance(o, dict):
        return {k: _decode(v) for k, v in o.items()}
    elif isinstance(o, list):
        return [_decode(v) for v in o]
    else:
        return o


class Parser:
    def __init__(self, experiment_file):
        print("--- Running parser on {} ---".format(experiment_file))
        with open(experiment_file) as f:
            data = json.load(f, object_hook=_decode)
            name_without_extension = experiment_file.split(".json")[0]
            with open(name_without_extension + ".json", 'w') as w:
                json.dump(data, w)


for experiment_file in glob.iglob('moc/*.json'):
    Parser(experiment_file)
    # os.remove(experiment_file)
