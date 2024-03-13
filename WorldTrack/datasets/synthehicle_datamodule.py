from typing import Optional

import lightning as pl
from torch.utils.data import DataLoader

from datasets.sampler import TemporalSampler
from datasets.synthehicle_dataset import SynthehicleDataset


class SynthehicleDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "../data/synthehicle",
            batch_size: int = 6,
            num_workers: int = 8,
            resolution=None,
            bounds=None,
            test_split='test',
            accumulate_grad_batches=8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_split = test_split

        self.resolution = resolution
        self.bounds = bounds
        self.accumulate_grad_batches = accumulate_grad_batches

        self.data_predict = None
        self.data_test = None
        self.data_val = None
        self.data_train = None

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            self.data_train = SynthehicleDataset(
                self.data_dir,
                split='train',
                resolution=self.resolution,
                bounds=self.bounds,
            )
        if stage == 'fit' or stage == 'validate':
            self.data_val = SynthehicleDataset(
                self.data_dir,
                is_train=True,
                split='val',
                resolution=self.resolution,
                bounds=self.bounds,
            )
        if stage == 'test':
            self.data_test = SynthehicleDataset(
                self.data_dir,
                split=self.test_split,
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
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=TemporalSampler(self.data_val, batch_size=self.batch_size,
                                    accumulate_grad_batches=self.accumulate_grad_batches),
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=1,
            num_workers=self.num_workers
        )
