from utils.logger import Logger
from preprocessing.hitter_dataset import HitterDataset
from model.training.pitch_sequence_trainer import PitchSequenceTrainer
from model.hitter_encoder.hitter_encoder import HitterEncoder




class PitchSequencePipeline:
    
    def __init__(self,
            num_epochs: int = 10, 
            dropout_hitter: float = .3, 
            dropout_pitcher: float = .3,
            dropout_context: float = .3,
            dropout_pitch_sequence: float = .3,
            learning_rate_hitter: float = 1e-5,
            learning_rate_pitcher: float = 1e-5,
            learning_rate_context: float = 1e-5,
            learning_rate_pitch_sequence: float = 1e-5
            ):
        # Get data tensors
        self.hitter_dataset = HitterDataset("data/hitters_2025_full.parquet")
        
        # Initialize all encoders
        self.hitter_encoder = HitterEncoder(self.hitter_dataset)
        
        # Initialize trainer
        self.trainer = PitchSequenceTrainer(
            hitter_encoder=self.hitter_encoder,
            num_epochs=num_epochs,
            dropout_hitter=dropout_hitter,
            dropout_pitcher=dropout_pitcher,
            dropout_context=dropout_context,
            dropout_pitch_sequence=dropout_pitch_sequence,
            learning_rate_hitter=learning_rate_hitter,
            learning_rate_pitcher=learning_rate_pitcher,
            learning_rate_context=learning_rate_context,
            learning_rate_pitch_sequence=learning_rate_pitch_sequence,
        )
        
        
    def execute(self):
        
        history = self.trainer.train()
        