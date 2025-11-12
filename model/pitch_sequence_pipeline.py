from model.context_encoder.context_encoder import ContextEncoder
from model.hitter_encoder.hitter_encoder import HitterEncoder
from model.pitcher_encoder.pitcher_encoder import PitcherEncoder
from model.pitch_sequence_encoder.pitch_sequence_encoder import PitchSequenceEncoder
from model.pitch_sequence_pipeline_components import PitchSequencePipelineComponents
from model.training.pitch_sequence_trainer import PitchSequenceTrainer
from preprocessing.context_dataset import ContextDataset
from preprocessing.hitter_dataset import HitterDataset
from preprocessing.pitch_sequence_dataset import PitchSequenceDataset
from preprocessing.pitcher_dataset import PitcherDataset
from utils.logger import Logger
from model.data_types import ModelComponents
from model.trainer_type import TrainerComponents

class PitchSequencePipeline:

    def __init__(self, pitch_sequence_pipeline_components: PitchSequencePipelineComponents):
        self.logger = Logger(self.__class__.__name__)
        
        #Sample for the pitch sequence dataset
        self.sample = pitch_sequence_pipeline_components.sample
        
        
        # Get data tensors
        self.hitter_dataset = HitterDataset("data/hitters_2025_full.parquet")
        self.pitcher_dataset = PitcherDataset("data/pitchers_2025_full.parquet")
        self.context_dataset = ContextDataset("data/context_2025_full.parquet")
        self.pitch_sequence_dataset = PitchSequenceDataset("data/pitch_sequence_2025.parquet", sample=self.sample)
        
        # Initialize model_params 
        self.hitter_model_params = ModelComponents(learning_rate=pitch_sequence_pipeline_components.learning_rate_hitter, dropout=pitch_sequence_pipeline_components.dropout_hitter, dataset=self.hitter_dataset, hidden_dim=256, embed_dim=128)
        self.pitcher_model_params = ModelComponents(learning_rate=pitch_sequence_pipeline_components.learning_rate_pitcher, dropout=pitch_sequence_pipeline_components.dropout_pitcher, dataset=self.pitcher_dataset, hidden_dim=256, embed_dim=128)
        self.context_model_params = ModelComponents(learning_rate=pitch_sequence_pipeline_components.learning_rate_context, dropout=pitch_sequence_pipeline_components.dropout_context, dataset=self.context_dataset, hidden_dim=256, embed_dim=128)
        self.pitch_seq_model_params = ModelComponents(learning_rate=pitch_sequence_pipeline_components.learning_rate_pitch_sequence, dropout=pitch_sequence_pipeline_components.dropout_pitch_sequence, dataset=self.pitch_sequence_dataset, hidden_dim=256, embed_dim=128)
        
        # Initialize all encoders
        self.hitter_encoder = HitterEncoder(self.hitter_model_params)
        self.pitcher_encoder = PitcherEncoder(self.pitcher_model_params)
        self.context_encoder = ContextEncoder(self.context_model_params)
        self.pitch_sequence_encoder = PitchSequenceEncoder(self.pitch_seq_model_params)
        
        # Custom Dataclass for the Trainer
        components = TrainerComponents(
            hitter_encoder=self.hitter_encoder,
            pitcher_encoder=self.pitcher_encoder,
            context_encoder=self.context_encoder,
            pitch_seq_encoder=self.pitch_sequence_encoder,
            num_epochs=pitch_sequence_pipeline_components.num_epochs,
        )

        # Initialize trainer
        self.trainer = PitchSequenceTrainer(components)

        self.logger.info("Pipeline Initialized")

    def execute(self):
        history = self.trainer.train()
