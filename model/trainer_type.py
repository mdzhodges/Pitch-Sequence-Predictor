from preprocessing.context_dataset import ContextDataset
from preprocessing.hitter_dataset import HitterDataset
from preprocessing.pitcher_dataset import PitcherDataset
from model.pitch_sequence_encoder.pitch_sequence_encoder import PitchSequenceEncoder
from dataclasses import dataclass



@dataclass 
class TrainerComponents:
    
    num_epochs: int
    hitter_embeds: HitterDataset
    pitcher_embeds: PitcherDataset
    context_embeds: ContextDataset
    pitch_seq_encoder: PitchSequenceEncoder
    batch_size: int