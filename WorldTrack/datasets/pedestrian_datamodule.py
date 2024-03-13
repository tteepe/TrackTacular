import os
from typing import Optional

import lightning as pl
from torch.utils.data import DataLoader

from datasets.multiviewx_dataset import MultiviewX
from datasets.wildtrack_dataset import Wildtrack
from datasets.pedestrian_dataset import PedestrianDataset
from datasets.sampler import TemporalSampler


class PedestrianDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "../data/MultiviewX",
            batch_size: int = 2,
            num_workers: int = 4,
            resolution=None,
            bounds=None,
            accumulate_grad_batches=8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.bounds = bounds
        self.accumulate_grad_batches = accumulate_grad_batches
        self.dataset = os.path.basename(self.data_dir)

        self.data_predict = None
        self.data_test = None
        self.data_val = None
        self.data_train = None

    def setup(self, stage: Optional[str] = None):
        if 'wildtrack' in self.dataset.lower():
            base = Wildtrack(self.data_dir)
        elif 'multiviewx' in self.dataset.lower():
            base = MultiviewX(self.data_dir)
        else:
            raise ValueError(f'Unknown dataset name {self.dataset}')

        if stage == 'fit':
            self.data_train = PedestrianDataset(
                base,
                is_train=True,
                resolution=self.resolution,
                bounds=self.bounds,
            )
        if stage == 'fit' or stage == 'validate':
            self.data_val = PedestrianDataset(
                base,
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds,
            )
        if stage == 'test':
            self.data_test = PedestrianDataset(
                base,
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds
            )
        if stage == 'predict':
            self.data_predict = PedestrianDataset(
                base,
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=TemporalSampler(self.data_train, batch_size=self.batch_size,
                                    accumulate_grad_batches=self.accumulate_grad_batches),
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=TemporalSampler(self.data_val, batch_size=self.batch_size,
                                    accumulate_grad_batches=self.accumulate_grad_batches),
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=1,
            num_workers=1,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
