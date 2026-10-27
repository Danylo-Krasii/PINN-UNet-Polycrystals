# Installation

## Requirements

| what | needed for | version of the paper |
|---|---|---|
| Linux, Python 3.10 to 3.12 | everything (torch 2.3.1 has no wheel for 3.13) | Python 3.12 |
| [uv](https://docs.astral.sh/uv/) | the environment | |
| NVIDIA GPU with CUDA 11.8 drivers | training (inference, scoring and the tests run on the CPU) | NVIDIA RTX A5000 |

## Steps

1. Clone the repository and enter it.

       git clone https://github.com/Danylo-Krasii/PINN-UNet-Polycrystals.git
       cd PINN-UNet-Polycrystals

2. Create the environment. This makes `.venv/` with torch 2.3.1+cu118, the test tools and the project in editable mode.

       uv sync --all-extras

   A bare `uv sync` removes the extras again, so always add `--all-extras`. For another CUDA build, change the `pytorch-cu118` index in `pyproject.toml`.

3. Get the data and the trained weights. [Placeholder: where the data and the weights are obtained.] Put the volumes in `data/` and the weights in
   `checkpoints/` ([DATA.md](DATA.md), [TRAINING.md](TRAINING.md)). Another place for either one is set with the environment variables
   `PINN_UNET_DATA` and `PINN_UNET_CHECKPOINTS`.

4. Check the installation. The tests use synthetic volumes and need neither a GPU nor the data.

       uv run pytest

## Why the versions are pinned

`pyproject.toml` fixes torch 2.3.1 and numpy 1.26.4. The training runs of the paper used torch 2.3.1+cu118. Without the pins, a fresh
environment can resolve a newer numpy than the one the code was tested with.
