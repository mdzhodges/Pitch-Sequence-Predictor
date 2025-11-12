from model.pitch_sequence_trainer_components import PitchSequenceTrainerComponents
from utils.logger import Logger


class PitchSequenceTrainer:

    def __init__(self, pitch_sequence_trainer_components: PitchSequenceTrainerComponents):
        # dropout for each encoder
        self.dropout_hitter = pitch_sequence_trainer_components.dropout_hitter
        self.dropout_pitcher = pitch_sequence_trainer_components.dropout_pitcher
        self.dropout_context = pitch_sequence_trainer_components.dropout_context
        self.dropout_pitch_sequence = pitch_sequence_trainer_components.dropout_pitch_sequence

        # learning rate (will be optimized by AdamW)
        self.learning_rate_hitter = pitch_sequence_trainer_components.learning_rate_hitter
        self.learning_rate_pitcher = pitch_sequence_trainer_components.learning_rate_pitcher
        self.learning_rate_context = pitch_sequence_trainer_components.learning_rate_context
        self.learning_rate_pitch_sequence = pitch_sequence_trainer_components.learning_rate_pitch_sequence

        # Various training needs
        self.num_epochs = pitch_sequence_trainer_components.num_epochs
        self.sample = pitch_sequence_trainer_components.sample

        # encoders
        self.hitter_encoder = pitch_sequence_trainer_components.hitter_encoder
        self.pitcher_encoder = pitch_sequence_trainer_components.pitcher_encoder
        self.context_encoder = pitch_sequence_trainer_components.context_encoder
        self.pitch_sequence_encoder = pitch_sequence_trainer_components.pitch_sequence_encoder

        # Fusions
        self.hitter_pitcher_fusion = None
        self.context_pitch_sequence_fusion = None

        # Logger
        self.logger = Logger(self.__class__.__name__)

    def train(self):
        pass
