import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class PackedBatchesDataset(Dataset):
    """
    Dataset поверх готовых packed batches.
    batches: (N, seq_len) — токены
    masks:   (N, seq_len) — seq_ids (0 для PAD, 1, 2, ... для объектов)
    """

    def __init__(self, batches: np.ndarray, masks: np.ndarray):
        self.batches = torch.from_numpy(batches).long()
        self.masks = torch.from_numpy(masks).long()

    def __len__(self):
        return self.batches.size(0)

    def __getitem__(self, idx):
        return self.batches[idx], self.masks[idx]


class WikiTextDataModule(pl.LightningDataModule):
    def __init__(self, npz_path: str, batch_size: int = 8, val_split: float = 0.05):
        super().__init__()
        self.npz_path = npz_path
        self.batch_size = batch_size
        self.val_split = val_split

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return

        data = np.load(self.npz_path)
        batches = data["batches"]    # (N, seq_len)
        masks = data["masks"]        # (N, seq_len)

        print(f"Загружено батчей: {batches.shape[0]}, длина: {batches.shape[1]}")

        n = batches.shape[0]
        n_val = int(n * self.val_split)
        n_train = n - n_val

        rng = np.random.default_rng(seed=42)
        idx = rng.permutation(n)

        train_idx = idx[:n_train]
        val_idx = idx[n_train:]

        self.train_dataset = PackedBatchesDataset(batches[train_idx], masks[train_idx])
        self.val_dataset = PackedBatchesDataset(batches[val_idx], masks[val_idx])

        print(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
