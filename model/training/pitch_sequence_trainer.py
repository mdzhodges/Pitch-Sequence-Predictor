from model.custom_types.trainer_type import TrainerComponents
from torch.utils.data import Dataset, DataLoader, Subset
from utils.logger import Logger
from tqdm import tqdm
import torch
import random


class PitchSequenceTrainer:

    def __init__(self, model_params: TrainerComponents):

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Various training needs
        self.num_epochs = model_params.num_epochs
        self.batch_size = model_params.batch_size

        # Initiate dataset and encoder
        self.dataset = model_params.dataset
        self.encoder = model_params.pitch_seq_encoder
        
        # send to device
        self.encoder = self.encoder.to(
            self.device)

        # Get the three loaders for train/val/test
        self.train_loader, self.val_loader, self.test_loader = self.get_loaders(
            val_split=.1, test_split=.1)
        
        # logger
        self.logger = Logger(self.__class__.__name__)

    def train(self):
        for _ in tqdm(range(self.num_epochs)):
            self.encoder.train()
            for batch in self.train_loader:
                self.encoder(**batch)
                
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
