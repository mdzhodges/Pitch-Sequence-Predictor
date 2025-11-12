from model.data_types import ModelComponents
from model.pitch_sequence_encoder.pitch_sequence_encoder import PitchSequenceEncoder
from model.pitch_sequence_pipeline_components import PitchSequencePipelineComponents
from model.trainer_type import TrainerComponents
from model.training.pitch_sequence_trainer import PitchSequenceTrainer
from preprocessing.context_dataset import ContextDataset
from preprocessing.hitter_dataset import HitterDataset
from preprocessing.pitch_sequence_dataset import PitchSequenceDataset
from preprocessing.pitcher_dataset import PitcherDataset
from utils.logger import Logger


class PitchSequencePipeline:

    def __init__(self, pitch_sequence_pipeline_components: PitchSequencePipelineComponents):
        self.logger = Logger(self.__class__.__name__)

        # Sample for the pitch sequence dataset
        self.sample = pitch_sequence_pipeline_components.sample

        # Get data tensors
        self.hitter_dataset = HitterDataset()
        self.pitcher_dataset = PitcherDataset()
        self.context_dataset = ContextDataset()
        self.pitch_sequence_dataset = PitchSequenceDataset(sample=self.sample)

        # Initialize model_params 
        self.pitch_seq_model_params = ModelComponents(
            learning_rate=pitch_sequence_pipeline_components.learning_rate_pitch_sequence,
            dropout=pitch_sequence_pipeline_components.dropout_pitch_sequence, dataset=self.pitch_sequence_dataset,
            hidden_dim=256, embed_dim=128)

        # Initialize all encoders
        self.pitch_sequence_encoder = PitchSequenceEncoder(self.pitch_seq_model_params)

        # Custom Dataclass for the Trainer
        components = TrainerComponents(
            hitter_embeds=self.hitter_dataset,
            pitcher_embeds=self.pitcher_dataset,
            context_embeds=self.context_dataset,
            pitch_seq_encoder=self.pitch_sequence_encoder,
            num_epochs=pitch_sequence_pipeline_components.num_epochs,
            batch_size=pitch_sequence_pipeline_components.batch_size
        )

        # Initialize trainer
        self.trainer = PitchSequenceTrainer(components)

        self.logger.info("Pipeline Initialized")

    def execute(self):
        history = self.trainer.train()
