from utils.logger import Logger
from preprocessing.hitter_dataset import HitterDataset
from preprocessing.pitcher_dataset import PitcherDataset
from model.training.pitch_sequence_trainer import PitchSequenceTrainer
from model.hitter_encoder.hitter_encoder import HitterEncoder
from model.context_encoder.context_encoder import ContextEncoder
from model.pitch_sequence_encoder.pitch_sequence_encoder import PitchSequenceEncoder
from model.pitcher_encoder.pitcher_encoder import PitcherEncoder




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
            learning_rate_pitch_sequence: float = 1e-5,
            sample: int = 1000,
            ):
        # Get data tensors
        self.hitter_dataset = HitterDataset("data/hitters_2025_full.parquet")
        self.pitcher_dataset = PitcherDataset("data/pitchers_2025_full.parquet")
        
        # Various params needed
        self.sample = sample
        
        
        # Initialize all encoders
        self.hitter_encoder = HitterEncoder(self.hitter_dataset)
        self.pitcher_encoder = PitcherEncoder(self.pitcher_dataset)
        self.context_encoder = ContextEncoder(self.hitter_dataset) ##CHANGE
        self.pitch_sequence_encoder = PitchSequenceEncoder(self.hitter_dataset, sample=self.sample) ## CHANGE
        
        # Initialize trainer
        self.trainer = PitchSequenceTrainer(
            hitter_encoder=self.hitter_encoder,
            pitcher_encoder = self.pitcher_encoder,
            context_encoder = self.context_encoder,
            pitch_sequence_encoder = self.pitch_sequence_encoder,
            num_epochs=num_epochs,
            dropout_hitter=dropout_hitter,
            dropout_pitcher=dropout_pitcher,
            dropout_context=dropout_context,
            dropout_pitch_sequence=dropout_pitch_sequence,
            learning_rate_hitter=learning_rate_hitter,
            learning_rate_pitcher=learning_rate_pitcher,
            learning_rate_context=learning_rate_context,
            learning_rate_pitch_sequence=learning_rate_pitch_sequence,
            sample=sample
        )
        
        
    def execute(self):
        
        history = self.trainer.train()
        