from utils.logger import Logger
from model.trainer_type import TrainerComponents


class PitchSequenceTrainer:
    
    def __init__(self, model_params: TrainerComponents):

        # Various training needs
        self.num_epochs = model_params.num_epochs
        self.batch_size = model_params.batch_size
        print(self.batch_size)

        # encoders
        self.hitter_encoder = model_params.hitter_encoder
        self.pitcher_encoder =model_params.pitcher_encoder
        self.context_encoder = model_params.context_encoder
        self.pitch_sequence_encoder = model_params.pitch_seq_encoder

        # Fusions
        self.hitter_pitcher_fusion = None
        self.context_pitch_sequence_fusion = None

        # Logger
        self.logger = Logger(self.__class__.__name__)

    def train(self):
        pass
