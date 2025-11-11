from model.hitter_encoder.hitter_encoder import HitterEncoder
from utils.logger import Logger


class PitchSequenceTrainer:
    
    def __init__(self, hitter_encoder: HitterEncoder, 
                 num_epochs: int = 10, 
                 sample: int = 1000,
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

        # Various training needs
        self.num_epochs = num_epochs
        self.sample = sample

        # encoders
        self.hitter_encoder = hitter_encoder
        self.pitcher_encoder = None
        self.context_encoder = None
        self.pitch_sequence_encoder = None

        # Fusions
        self.hitter_pitcher_fusion = None
        self.context_pitch_sequence_fusion = None

        # Logger
        self.logger = Logger(self.__class__.__name__)

        # Detailed initialization log
        self.logger.info(
            f"\n[Trainer Initialized]\n"
            f"Epochs: {self.num_epochs}\n"
            f"Sample Size: {self.sample}\n"
            f"\n[Dropout Rates]\n"
            f"  Hitter: {self.dropout_hitter}\n"
            f"  Pitcher: {self.dropout_pitcher}\n"
            f"  Context: {self.dropout_context}\n"
            f"  Pitch Sequence: {self.dropout_pitch_sequence}\n"
            f"\n[Learning Rates]\n"
            f"  Hitter: {self.learning_rate_hitter}\n"
            f"  Pitcher: {self.learning_rate_pitcher}\n"
            f"  Context: {self.learning_rate_context}\n"
            f"  Pitch Sequence: {self.learning_rate_pitch_sequence}\n"
            f"\n[Encoders]\n"
            f"  Hitter Encoder: {'Loaded' if self.hitter_encoder else 'None'}\n"
            f"  Pitcher Encoder: {'Loaded' if self.pitcher_encoder else 'None'}\n"
            f"  Context Encoder: {'Loaded' if self.context_encoder else 'None'}\n"
            f"  Pitch Sequence Encoder: {'Loaded' if self.pitch_sequence_encoder else 'None'}"
        )

    def train(self):
        pass
