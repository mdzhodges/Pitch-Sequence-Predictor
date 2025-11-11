
from model.hitter_encoder.hitter_encoder import HitterEncoder


class PitchSequenceTrainer:
    
    def __init__(self, hitter_encoder: HitterEncoder, num_epochs: int = 10, 
                 dropout_hitter: float = .3, 
                 dropout_pitcher: float = .3,
                 dropout_context: float = .3,
                 dropout_pitch_sequence: float = .3,
                 learning_rate_hitter: float = 1e-5,
                 learning_rate_pitcher: float = 1e-5,
                 learning_rate_context: float = 1e-5,
                 learning_rate_pitch_sequence: float = 1e-5):
        
        # dropout for each encoder
        self.dropout_hitter = dropout_hitter
        self.dropout_pitcher = dropout_pitcher
        self.dropout_context = dropout_context
        self.dropout_pitch_sequence = dropout_pitch_sequence
        
        # learning rate (will be optimized by AdamW)
        self.learning_rate_hitter = learning_rate_hitter
        self.learning_rate_pitcher = learning_rate_pitcher
        self.learning_rate_context = learning_rate_context
        self.learning_rate_pitch_sequence = learning_rate_pitch_sequence
        
        
        # encoders
        self.hitter_encoder = hitter_encoder
        self.pitcher_encoder = None
        self.context_encoder = None
        self.pitch_sequence_encoder = None
        
    def train(self):
        pass