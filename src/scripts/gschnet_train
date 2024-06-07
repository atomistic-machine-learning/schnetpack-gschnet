#!/usr/bin/env python3
import schnetpack.cli as cli
import math
from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    "compute_refresh_rate",
    lambda num_train, batch_size, updates: max(
        1, int(math.ceil(num_train / batch_size) / updates)
    ),
)

if __name__ == "__main__":
    cli.train()
