"""The names of the release (docs/NAMES.md): the six training configurations, the two materials and the three volume sets.

The module imports the standard library only, so that every module and every test can load it.
"""

CONFIGURATIONS = ("labels-only", "energy-only", "labels-energy-fixed", "labels-energy-adaptive", "labels-energy-uncertainty", "labels-energy-lagrangian")
LABELLED = tuple(c for c in CONFIGURATIONS if c != "energy-only")
MATERIALS = ("copper", "zener-8")
VOLUME_SETS = ("train", "val", "test")
