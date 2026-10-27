"""Where things live, resolved once.

Every path a script needs comes from here, so that moving the tree or pointing at another corpus is one environment variable. Defaults are
relative to the repository root, which is one level above this file.

    PINN_UNET_ROOT         repository root
    PINN_UNET_DATA         the folder that holds the material folders `copper/` and `zener-8/` (default <root>/data)
    PINN_UNET_CHECKPOINTS  trained weights (default <root>/checkpoints)

The YAML configs of `uv run train` and `uv run infer` are read by load_config; a value may hold {ROOT}, {DATA} or {CHECKPOINTS}, which expand to the
paths above.
"""

import os
from pathlib import Path

TREE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PINN_UNET_ROOT", TREE))
DATA = Path(os.environ.get("PINN_UNET_DATA", ROOT / "data"))
CHECKPOINTS = Path(os.environ.get("PINN_UNET_CHECKPOINTS", ROOT / "checkpoints"))

PLACEHOLDERS = ("ROOT", "DATA", "CHECKPOINTS")


def expand(value):
    """A string (or the strings of a list) with {ROOT}, {DATA} and {CHECKPOINTS} replaced; other braces stay."""
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, str):
        for name in PLACEHOLDERS:
            value = value.replace("{" + name + "}", str(globals()[name]))
    return value


def load_config(path, overrides=()):
    """The YAML config at `path` as a dict in file order, after the overrides.

    An override is "key=value". The value true or false becomes a boolean, null (or nothing after the =) removes the key,
    any other value stays the text that was written; a key that the file lacks is added at the end. Strings are expanded by `expand`.
    """
    import yaml

    try:
        cfg = yaml.safe_load(Path(path).read_text()) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"{path}: {e}") from None
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: a config is a mapping of keys to values, not {type(cfg).__name__}")
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"{item!r} is not key=value")
        if value in ("", "null"):
            cfg.pop(key, None)
        else:
            cfg[key] = {"true": True, "false": False}.get(value.lower(), value)
    return {k: expand(v) for k, v in cfg.items()}


def check_keys(model, options, allowed, required=()):
    """SystemExit, naming the keys that `model` takes, unless the config has no key outside `allowed` and a value for every key of `required`."""
    unknown = sorted(set(options) - set(allowed))
    missing = [k for k in required if options.get(k) is None]
    if unknown or missing:
        raise SystemExit(f"{model} takes the keys {', '.join(allowed)} ({', '.join(required)} required); the config has {sorted(options)}")
