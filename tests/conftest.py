import pytest

from tests import synth

SEEDS = list(range(1000, 1006)) + [1140, 1141]
VAL_SEEDS = list(range(1140, 1160))


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    """A data folder with two material folders of synthetic volumes: `copper` has the seeds 1000 to 1005 (four for training, two for validation in the trainer) and the seeds
    1140 and 1141 (the first two validation seeds, which the L-BFGS phase reads), and `zener-8` the 20 validation seeds 1140 to 1159."""
    root = tmp_path_factory.mktemp("data")
    synth.write_volumes(root / "copper", SEEDS)
    synth.write_volumes(root / "zener-8", VAL_SEEDS)
    return root
