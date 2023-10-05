from ruamel import yaml
"""pip install ruamel.yaml"""


def save_yaml(opt, yaml_path):
    f = open(yaml_path, 'x', encoding='utf-8')
    yaml.dump(opt, f, Dumper=yaml.RoundTripDumper, indent=2)
    f.close()


def read_yaml(yaml_path):
    f = open(yaml_path, 'r', encoding='utf-8')
    data = yaml.load(f.read(), Loader=yaml.Loader)
    f.close()
    return data
