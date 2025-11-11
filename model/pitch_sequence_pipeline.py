from utils.logger import Logger
from preprocessing.hitter_dataset import HitterDataset
from model.training.pitch_sequence_trainer import PitchSequenceTrainer
from model.hitter_encoder.hitter_encoder import HitterEncoder




class PitchSequencePipeline:
    
    def __init__(self):
        # Get data tensors
        self.hitter_dataset = HitterDataset("data/hitters_2025_full.parquet")
        
        # Initialize all encoders
        self.hitter_encoder = HitterEncoder(self.hitter_dataset)
        
        # Initialize trainer
        self.trainer = PitchSequenceTrainer(
            hitter_encoder=self.hitter_encoder
        )
        
        
    def execute(self):
        
        history = self.trainer.train()
        