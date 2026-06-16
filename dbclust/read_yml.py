import os

import yaml


def ymljoin(loader, node):
    seq = loader.construct_sequence(node)
    return "".join([str(i) for i in seq])


class _ExtendMarker(list):
    """Sentinel: a list returned by !extend that must be flattened into its parent."""
    pass


def make_loader(base_dir: str):
    """Build a FullLoader that supports !join, !include, and !extend.

    !include path  — inserts the file content as-is (scalar, dict, or list).
    !extend path   — when used as a list item, merges the file's list items
                     directly into the parent list (like Python's list.extend).

        before:
          - rule1
          - rule2
          - !extend extra_rules.yml   # items merged inline, no extra nesting
    """

    class DBClustLoader(yaml.FullLoader):
        pass

    def ymlinclude(loader, node):
        path = loader.construct_scalar(node)
        if not os.path.isabs(path):
            path = os.path.join(base_dir, path)
        with open(path, "r", encoding="utf-8") as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def ymlextend(loader, node):
        content = ymlinclude(loader, node)
        if not isinstance(content, list):
            raise ValueError(f"!extend requires a YAML file containing a list, got {type(content)}")
        return _ExtendMarker(content)

    def construct_sequence_flat(loader, node, deep=False):
        raw = yaml.FullLoader.construct_sequence(loader, node, deep=True)
        result = []
        for item in raw:
            if isinstance(item, _ExtendMarker):
                result.extend(item)
            else:
                result.append(item)
        return result

    DBClustLoader.add_constructor("!join", ymljoin)
    DBClustLoader.add_constructor("!include", ymlinclude)
    DBClustLoader.add_constructor("!extend", ymlextend)
    DBClustLoader.add_constructor("tag:yaml.org,2002:seq", construct_sequence_flat)

    return DBClustLoader


def read_config(conf_file: str):
    loader = make_loader(os.path.dirname(os.path.abspath(conf_file)))
    with open(conf_file, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=loader)
