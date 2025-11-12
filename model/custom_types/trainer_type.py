from preprocessing.context_dataset import ContextDataset
from preprocessing.hitter_dataset import HitterDataset
from preprocessing.pitcher_dataset import PitcherDataset
from preprocessing.fusion_dataset import FusionDataset
from model.pitch_sequence_encoder.pitch_sequence_encoder import PitchSequenceEncoder
from dataclasses import dataclass



@dataclass 
class TrainerComponents:
    
    num_epochs: int
    dataset: FusionDataset
    pitch_seq_encoder: PitchSequenceEncoder
    batch_size: int