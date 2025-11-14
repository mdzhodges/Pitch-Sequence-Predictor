import random
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model.custom_types.trainer_type import TrainerComponents
from utils.logger import Logger

import torch.nn.functional as F

from utils.constants import Constants

from evaluation.eval import PitchSequenceEvaluator


class PitchSequenceTrainer:

    def __init__(self, model_params: TrainerComponents):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )

        self.num_epochs = model_params.num_epochs
        self.batch_size = model_params.batch_size

        self.dataset = model_params.dataset
        self.encoder = model_params.pitch_seq_encoder.to(self.device)

        # splits
        self.train_loader, self.val_loader, self.test_loader = self.get_loaders(
            val_split=0.1, test_split=0.1
        )

        self.logger = Logger(self.__class__.__name__)

        # Adam with weight decay
        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.encoder.learning_rate,
            weight_decay=1e-4
        )


    def train(self):
        for epoch in range(self.num_epochs):
            self.encoder.train()
            total_loss = 0.0
            num_batches = 0

            for batch in self.train_loader:
                labels = batch["label"].to(self.device)
                numeric = batch["numeric"].to(self.device)
                pitcher_id = batch["pitcher_id"].to(self.device)
                categorical = {k: v.to(self.device)
                            for k, v in batch["categorical"].items()}

                self.optimizer.zero_grad()

                out = self.encoder(
                    numeric=numeric,
                    categorical=categorical,
                    pitcher_id=pitcher_id,
                )

                logits = out["logits"]
                loss = F.cross_entropy(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 5.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches


            self.logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} — "
                f"Epoch Loss (avg): {avg_loss:.4f} | "
                f"Total Loss Sum: {total_loss:.4f}"
            )
        PitchSequenceEvaluator(self.encoder, test_dataset=self.test_loader).run()


    def get_indices_for_split(self, val_split: float = .1, test_split: float = .1):
        """Return train, val, test index lists based on your split ratios."""
        total_length = len(self.dataset)
        indices = list(range(total_length))

        # shuffle
        random.shuffle(indices)

        # compute split points
        test_size = int(total_length * test_split)
        val_size = int(total_length * val_split)

        train_end = total_length - test_size - val_size
        val_end = total_length - test_size

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        return train_indices, val_indices, test_indices

    def get_loaders(self, val_split: float = .1, test_split: float = .1):
        # Get indices for the split
        train_idx, val_idx, test_idx = self.get_indices_for_split(
            val_split=val_split, test_split=test_split)

        # take subset of the data
        train_subset = Subset(self.dataset, train_idx)
        val_subset = Subset(self.dataset, val_idx)
        test_subset = Subset(self.dataset, test_idx)

        # data loaders for test/val/train
        train_loader = DataLoader(
            dataset=train_subset,
            batch_size=self.batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            dataset=val_subset,
            batch_size=self.batch_size,
            shuffle=False
        )

        test_loader = DataLoader(
            dataset=test_subset,
            batch_size=self.batch_size,
            shuffle=False
        )

        return train_loader, val_loader, test_loader
