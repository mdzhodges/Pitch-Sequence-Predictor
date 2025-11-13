from dataclasses import dataclass
from preprocessing.fusion_dataset import FusionDataset


@dataclass
class ModelComponents():
    """Represents the core components required to configure a model.

    This class stores the learning rate, dropout value, and dataset used for model training.

    Attributes:
        learning_rate: The learning rate for the model optimizer.
        dropout: The dropout rate applied during training.
        dataset: The dataset object used for training and evaluation.
    """
    learning_rate: float
    dropout: float
    dataset: FusionDataset
    hidden_dim: int
    embed_dim: int
