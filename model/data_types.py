from dataclasses import dataclass
from preprocessing.hitter_dataset import HitterDataset
from preprocessing.pitcher_dataset import PitcherDataset
from preprocessing.context_dataset import ContextDataset
from preprocessing.pitch_sequence_dataset import PitchSequenceDataset
from typing import Generic, TypeVar

dataset_types = TypeVar("dataset_types", HitterDataset, PitcherDataset, ContextDataset, PitchSequenceDataset)


@dataclass
class ModelComponents(Generic[dataset_types]):
    """Represents the core components required to configure a model.

    This class stores the learning rate, dropout value, and dataset used for model training.

    Attributes:
        learning_rate: The learning rate for the model optimizer.
        dropout: The dropout rate applied during training.
        dataset: The dataset object used for training and evaluation.
    """
    learning_rate: float
    dropout: float
    dataset: dataset_types
    
@dataclass
class TrainerComponents:
    
    num_epochs: int
    hitter_encoder: object
    pitcher_encoder: object
    context_encoder: object
    pitch_seq_encoder: object