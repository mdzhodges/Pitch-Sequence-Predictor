from model.context_encoder.context_encoder import ContextEncoder
from model.hitter_encoder.hitter_encoder import HitterEncoder
from model.pitcher_encoder.pitcher_encoder import PitcherEncoder
from model.pitch_sequence_encoder.pitch_sequence_encoder import PitchSequenceEncoder
from dataclasses import dataclass



@dataclass 
class TrainerComponents:
    
    num_epochs: int
    hitter_encoder: HitterEncoder
    pitcher_encoder: PitcherEncoder
    context_encoder: ContextEncoder
    pitch_seq_encoder: PitchSequenceEncoder
    batch_size: int