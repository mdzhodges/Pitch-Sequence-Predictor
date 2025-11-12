from model.hitter_encoder.hitter_encoder import HitterEncoder
from model.pitcher_encoder.pitcher_encoder import PitcherEncoder
from model.context_encoder.context_encoder import ContextEncoder
from model.pitch_sequence_encoder.pitch_sequence_encoder import PitchSequenceEncoder
from utils.logger import Logger
from model.model_components import ModelComponents


class PitchSequenceTrainer:
    
    def __init__(self, model_params: ModelComponents):
        
        # dropout for each encoder
        self.dropout_hitter = model_params.dropout_hitter
        self.dropout_pitcher = model_params.dropout_pitcher
        self.dropout_context = model_params.dropout_context
        self.dropout_pitch_sequence = model_params.dropout_pitch_sequence

        # learning rate (will be optimized by AdamW)
        self.learning_rate_hitter = model_params.learning_rate_hitter
        self.learning_rate_pitcher = model_params.learning_rate_pitcher
        self.learning_rate_context = model_params.learning_rate_context
        self.learning_rate_pitch_sequence = model_params.learning_rate_pitch_sequence

        # Various training needs
        self.num_epochs = model_params.num_epochs
        self.sample = model_params.sample

        # encoders
        self.hitter_encoder = model_params.hitter_encoder
        self.pitcher_encoder =model_params.pitcher_encoder
        self.context_encoder = model_params.context_encoder
        self.pitch_sequence_encoder = model_params.pitch_sequence_encoder

        # Fusions
        self.hitter_pitcher_fusion = None
        self.context_pitch_sequence_fusion = None

        # Logger
        self.logger = Logger(self.__class__.__name__)
        

    def train(self):
        pass
