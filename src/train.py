"""Train a model from a YAML config:

    uv run train --config configs/train/<name>.yaml [--set key=value ...] [--dry-run]

The key `model:` of the config chooses the trainer (TRAINERS below). Every other key is an option of that trainer, with the underscores of
the key for the dashes of the option (max_rewinds: 20 gives --max-rewinds 20): a true value gives the bare flag, false and null give nothing, a list
gives its items; a mapping, an empty list or a key that is not a text stops the run. The one flag that is on by default has a false value, which gives its
negation: midpoint_guard: false gives --no-midpoint-guard. The trainer keeps its own argparse and its own defaults, so a key
that the config does not set has the default of the trainer. A string with {ROOT}, {DATA} or {CHECKPOINTS} in it expands to the path of
src/settings.py. A number that must stay a text (1.25e-4) is written in quotes.

--set key=value changes or adds a key after the file is read (true and false are booleans, null removes the key).
--dry-run prints the command line of the trainer and stops. A value of the form {NAME} that is still open stops a real run: give it with --set.
"""

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

from src import settings

TRAINERS = {"single_pass_unet": "src.training.single_pass"}
OPEN = re.compile(r"\{[A-Z][A-Z0-9_]*\}")
BARE = re.compile(r"^[A-Za-z0-9_@%+=:,./{}-]+$")
OFF_FLAGS = ("midpoint_guard",)


def options(cfg):
    """The command-line items for the keys of a config, in file order, without `model`."""
    items = []
    for key, value in cfg.items():
        if key == "model" or value is None:
            continue
        if not isinstance(key, str) or isinstance(value, dict) or value == []:
            raise SystemExit(f"config key {key!r}: a key is a text and a value is a text, a number, true or a list that is not empty")
        flag = "--" + key.replace("_", "-")
        if value is False:
            if key in OFF_FLAGS:
                items.append("--no-" + flag[2:])
        elif value is True:
            items.append(flag)
        elif isinstance(value, list):
            items += [flag, *map(str, value)]
        else:
            items += [flag, str(value)]
    return items


def command(cfg):
    """[python, -m, module, ...] for a config; SystemExit when `model` is missing or unknown."""
    model = cfg.get("model")
    if model not in TRAINERS:
        raise SystemExit(f"model {model!r} is not one of {sorted(TRAINERS)}")
    return [sys.executable, "-m", TRAINERS[model], *options(cfg)]


def shown(cmd):
    """The command as one line: the interpreter relative to the repository when it lies in it, a token quoted only when it needs it."""
    exe = Path(cmd[0])
    try:
        head = str(exe.relative_to(settings.ROOT))
    except ValueError:
        head = str(exe)
    return " ".join(t if BARE.match(t) else shlex.quote(t) for t in [head, *cmd[1:]])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/train/<name>.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="change or add one key of the config (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="print the command line of the trainer and stop")
    a = ap.parse_args(argv)
    try:
        cfg = settings.load_config(a.config, a.set)
    except (OSError, ValueError) as e:
        ap.error(str(e))
    cmd = command(cfg)
    open_ = [t for t in cmd if OPEN.search(t)]
    if a.dry_run:
        print(shown(cmd))
        if open_:
            print(f"open values: {' '.join(open_)}; give each with --set key=value before a real run", file=sys.stderr)
        return 0
    if open_:
        ap.error(f"the config holds open values {' '.join(open_)}; give each with --set key=value")
    return subprocess.call(cmd, cwd=settings.TREE)


if __name__ == "__main__":
    sys.exit(main())
