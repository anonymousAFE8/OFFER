# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# pyre-unsafe

"""
Main entry point for model training. Please refer to README.md for usage instructions.
"""
import numpy as np
import logging
import os

import random
from typing import List, Optional



os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide excessive tensorflow debug messages
import sys

#import fbgemm_gpu  # noqa: F401, E402
import gin

import torch
import torch.multiprocessing as mp

from absl import app, flags
from generative_recommenders.research.trainer.train import train_fn

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# 正确设置新的日志输出
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

def delete_flags(FLAGS, keys_to_delete: List[str]) -> None:  # pyre-ignore [2]
    keys = [key for key in FLAGS._flags()]
    for key in keys:
        if key in keys_to_delete:
            delattr(FLAGS, key)


delete_flags(flags.FLAGS, ["gin_config_file", "master_port"])
flags.DEFINE_string("gin_config_file", None, "Path to the config file.")
flags.DEFINE_integer("master_port", 12355, "Master port.")
FLAGS = flags.FLAGS  # pyre-ignore [5]


def mp_train_fn(
    rank: int,
    world_size: int,
    master_port: int,
    gin_config_file: Optional[str],
) -> None:
    set_seed(42 + rank)

    if gin_config_file is not None:
        # Hack as absl doesn't support flag parsing inside multiprocessing.
        logging.info(f"Rank {rank}: loading gin config from {gin_config_file}")
        gin.parse_config_file(gin_config_file)

    train_fn(rank, world_size, master_port)


def _main(argv) -> None:  # pyre-ignore [2]
    world_size = torch.cuda.device_count()

    mp.set_start_method("forkserver")
    mp.spawn(
        mp_train_fn,
        args=(world_size, FLAGS.master_port, FLAGS.gin_config_file),
        nprocs=world_size,
        join=True,
    )


def main() -> None:
    app.run(_main)


if __name__ == "__main__":
    main()
