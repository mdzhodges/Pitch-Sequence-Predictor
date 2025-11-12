from utils.logger import Logger
from preprocessing.hitter_dataset import HitterDataset
from preprocessing.pitcher_dataset import PitcherDataset
from preprocessing.context_dataset import ContextDataset
from preprocessing.pitch_sequence_dataset import PitchSequenceDataset
from model.training.pitch_sequence_trainer import PitchSequenceTrainer
from model.hitter_encoder.hitter_encoder import HitterEncoder
from model.context_encoder.context_encoder import ContextEncoder
from model.pitch_sequence_encoder.pitch_sequence_encoder import PitchSequenceEncoder
from model.pitcher_encoder.pitcher_encoder import PitcherEncoder
from model.data_types import ModelComponents, TrainerComponents

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
        
        self.logger = Logger(self.__class__.__name__)
        
        #Sample for the pitch sequence dataset
        self.sample = sample
        
        
        # Get data tensors
        self.hitter_dataset = HitterDataset("data/hitters_2025_full.parquet")
        self.pitcher_dataset = PitcherDataset("data/pitchers_2025_full.parquet")
        self.context_dataset = ContextDataset("data/context_2025_full.parquet")
        self.pitch_sequence_dataset = PitchSequenceDataset("data/pitch_sequence_2025.parquet", sample=self.sample)
        
        # Initialize model_params 
        self.hitter_model_params = ModelComponents(learning_rate=learning_rate_hitter, dropout=dropout_hitter, dataset=self.hitter_dataset)
        self.pitcher_model_params = ModelComponents(learning_rate=learning_rate_pitcher, dropout=dropout_pitcher, dataset=self.pitcher_dataset)
        self.context_model_params = ModelComponents(learning_rate=learning_rate_context, dropout=dropout_context, dataset=self.context_dataset)
        self.pitch_seq_model_params = ModelComponents(learning_rate=learning_rate_pitch_sequence, dropout=dropout_pitch_sequence, dataset=self.pitch_sequence_dataset)
        
        # Initialize all encoders
        self.hitter_encoder = HitterEncoder(self.hitter_model_params)
        self.pitcher_encoder = PitcherEncoder(self.pitcher_model_params)
        self.context_encoder = ContextEncoder(self.context_model_params)
        self.pitch_sequence_encoder = PitchSequenceEncoder(self.pitch_seq_model_params)
        
        # Custom Dataclass for the Trainer
        components = TrainerComponents(
            hitter_encoder=self.hitter_encoder,
            pitcher_encoder=self.pitch_sequence_encoder,
            context_encoder=self.context_dataset,
            pitch_seq_encoder=self.pitch_sequence_encoder,
            num_epochs=num_epochs,
        )

        # Initialize trainer
        self.trainer = PitchSequenceTrainer(model_params=components)
        
        self.logger.info("Pipeline Initialized")
        
    def execute(self):
        history = self.trainer.train()
        