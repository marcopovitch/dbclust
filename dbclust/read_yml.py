import yaml


def ymljoin(loader, node):
    seq = loader.construct_sequence(node)
    return "".join([str(i) for i in seq])


def yml_read_config(filename: str):
    with open(filename, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


def read_config(conf_file: str):
    yaml.add_constructor("!join", ymljoin)
    cfg = yml_read_config(conf_file)
    return cfg
