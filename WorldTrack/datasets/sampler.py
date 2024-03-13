from typing import Iterator, Sized
import torch
from torch.utils.data import Sampler


class TemporalSampler(Sampler[int]):
    def __init__(self, data_source: Sized, batch_size: int = 2, accumulate_grad_batches: int = 8) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.accumulate_grad_batches = accumulate_grad_batches

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        m = n - (n % (self.batch_size * self.accumulate_grad_batches))

        idx = torch.arange(m, dtype=torch.long).view(self.batch_size, self.accumulate_grad_batches, -1)
        idx = idx.transpose(0, 1).permute(*torch.arange(idx.ndim - 1, -1, -1)).flatten().tolist()
        idx = idx + list(range(m, n))

        yield from idx
