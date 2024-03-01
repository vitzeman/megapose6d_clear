"""
Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""



# Third Party
import numpy as np
import torch
from torch.utils.data import Sampler

# MegaPose
from megapose.utils.random import temp_numpy_seed


class PartialSampler(Sampler):
    def __init__(self, ds, epoch_size):
        self.n_items = len(ds)
        self.epoch_size = min(epoch_size, len(ds))
        super().__init__(None)

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        return (i.item() for i in torch.randperm(self.n_items)[: len(self)])


class DistributedSceneSampler(Sampler):
    def __init__(self, scene_ds, num_replicas, rank, shuffle=True):
        indices = np.arange(len(scene_ds))
        if shuffle:
            with temp_numpy_seed(0):
                indices = np.random.permutation(indices)
        all_indices = np.array_split(indices, num_replicas)
        local_indices = all_indices[rank].tolist()
        self.local_indices = local_indices

    def __len__(self):
        return len(self.local_indices)

    def __iter__(self):
        return iter(self.local_indices)

class ListSampler(Sampler):
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        return iter(self.ids)


class CustomDistributedSampler(Sampler):
    def __init__(self, ds, num_replicas, rank, epoch_size, seed=0, shuffle=True):
        # NOTE: Epoch size is local.
        total_epoch_size = epoch_size * num_replicas
        n_repeats = 1 + total_epoch_size // len(ds)
        self.all_indices = np.concatenate([np.arange(len(ds)) for _ in range(n_repeats)])
        assert len(self.all_indices) >= total_epoch_size
        self.total_epoch_size = total_epoch_size
        self.seed = seed
        self.epoch = 0
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        self.epoch += 1
        with temp_numpy_seed(self.epoch + self.seed):
            indices_shuffled = np.random.permutation(self.all_indices)[: self.total_epoch_size]
            local_indices = np.array_split(indices_shuffled, self.num_replicas)[self.rank]
        return iter(local_indices)

class CustomiterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(CustomiterableDataset, self).__init__()
        self.start = start
        self.end = end

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = self.start
            iter_end = self.end
        else:
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        for i in range(iter_start, iter_end):
            yield i

