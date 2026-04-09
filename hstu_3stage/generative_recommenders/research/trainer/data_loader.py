# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

import os
from typing import Optional, Tuple

import gin
import torch


@gin.configurable
def create_data_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    world_size: int,
    rank: int,
    shuffle: bool,
    prefetch_factor: int = 4,
    num_workers: Optional[int] = min(os.cpu_count() or 0, 8),
    pin_memory: bool = True,
    persistent_workers: bool = True,
    drop_last: bool = False,
) -> Tuple[
    Optional[torch.utils.data.distributed.DistributedSampler[torch.utils.data.Dataset]],
    torch.utils.data.DataLoader,
]:
    if shuffle:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=0,
            drop_last=drop_last,
        )
    else:
        sampler = None
    worker_count = num_workers or 0
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        # shuffle=True, cannot use with sampler
        num_workers=worker_count,
        sampler=sampler,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if worker_count > 0 else None,
        #collate_fn = collate_fn,
        # ????????poch??????num_works
        persistent_workers=persistent_workers if worker_count > 0 else False,
    )
    return sampler, data_loader


