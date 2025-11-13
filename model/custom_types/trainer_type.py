from preprocessing.fusion_dataset import FusionDataset
from model.pitch_sequence_encoder.pitch_sequence_encoder import PitchSequenceEncoder
from dataclasses import dataclass


@dataclass
class TrainerComponents:

    num_epochs: int
    dataset: FusionDataset
    pitch_seq_encoder: PitchSequenceEncoder
    batch_size: int
