from model.context_encoder.context_encoder import ContextEncoder
from model.hitter_encoder.hitter_encoder import HitterEncoder
from model.pitch_sequence_encoder.pitch_sequence_encoder import PitchSequenceEncoder
from model.pitch_sequence_pipeline_components import PitchSequencePipelineComponents
from model.pitch_sequence_trainer_components import PitchSequenceTrainerComponents
from model.pitcher_encoder.pitcher_encoder import PitcherEncoder
from model.training.pitch_sequence_trainer import PitchSequenceTrainer
from preprocessing.context_dataset import ContextDataset
from preprocessing.hitter_dataset import HitterDataset
from preprocessing.pitch_sequence_dataset import PitchSequenceDataset
from preprocessing.pitcher_dataset import PitcherDataset
from utils.logger import Logger


class PitchSequencePipeline:

    def __init__(self, pitch_sequence_pipeline_components: PitchSequencePipelineComponents):
        self.logger = Logger(self.__class__.__name__)

        # Get data tensors
        self.hitter_dataset = HitterDataset("data/hitters_2025_full.parquet")
        self.pitcher_dataset = PitcherDataset("data/pitchers_2025_full.parquet")
        self.context_dataset = ContextDataset("data/context_2025_full.parquet")
        self.pitch_sequence_dataset = PitchSequenceDataset("data/pitch_sequence_2025.parquet")

        # Various params needed
        self.sample = pitch_sequence_pipeline_components.sample
        # Initialize all encoders
        self.hitter_encoder = HitterEncoder(self.hitter_dataset)
        self.pitcher_encoder = PitcherEncoder(self.pitcher_dataset)
        self.context_encoder = ContextEncoder(self.context_dataset)
        self.pitch_sequence_encoder = PitchSequenceEncoder(self.pitch_sequence_dataset)

        # Custom Dataclass for the Trainer
        components = PitchSequenceTrainerComponents(
            hitter_encoder=HitterEncoder(self.hitter_dataset),
            pitcher_encoder=PitcherEncoder(self.pitcher_dataset),
            context_encoder=ContextEncoder(self.context_dataset),
            pitch_sequence_encoder=PitchSequenceEncoder(self.pitch_sequence_dataset),
            num_epochs=pitch_sequence_pipeline_components.num_epochs,
            dropout_hitter=pitch_sequence_pipeline_components.dropout_hitter,
            dropout_pitcher=pitch_sequence_pipeline_components.dropout_pitcher,
            dropout_context=pitch_sequence_pipeline_components.dropout_context,
            dropout_pitch_sequence=pitch_sequence_pipeline_components.dropout_pitch_sequence,
            learning_rate_hitter=pitch_sequence_pipeline_components.learning_rate_hitter,
            learning_rate_pitcher=pitch_sequence_pipeline_components.learning_rate_pitcher,
            learning_rate_context=pitch_sequence_pipeline_components.learning_rate_context,
            learning_rate_pitch_sequence=pitch_sequence_pipeline_components.learning_rate_pitch_sequence,
            sample=pitch_sequence_pipeline_components.sample,
        )

        # Initialize trainer
        self.trainer = PitchSequenceTrainer(pitch_sequence_trainer_components=components)

        self.logger.info("Pipeline Initialized")

    def execute(self):
        history = self.trainer.train()
