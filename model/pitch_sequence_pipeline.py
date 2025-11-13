from model.custom_types.data_types import ModelComponents
from model.pitch_sequence_encoder.pitch_sequence_encoder import PitchSequenceEncoder
from model.custom_types.pitch_sequence_pipeline_components import PitchSequencePipelineComponents
from model.custom_types.trainer_type import TrainerComponents
from model.training.pitch_sequence_trainer import PitchSequenceTrainer
from preprocessing.fusion_dataset import FusionDataset
from utils.logger import Logger


class PitchSequencePipeline:

    def __init__(self, pitch_sequence_pipeline_components: PitchSequencePipelineComponents):
        # Sample for the pitch sequence dataset
        self.sample = pitch_sequence_pipeline_components.sample

        # Logger
        self.logger = Logger(self.__class__.__name__)

        # Check params of trainer, model, and sample
        self.dataset = FusionDataset(sample=self.sample)
        
        self.logger.info(f"Categorical: {self.dataset.categorical_cols}\n")
        self.logger.info(f"Numeric: {self.dataset.numeric_cols} \n")


        # Initialize model_params
        self.pitch_seq_model_params = ModelComponents(
            learning_rate=pitch_sequence_pipeline_components.learning_rate_pitch_sequence,
            dropout=pitch_sequence_pipeline_components.dropout_pitch_sequence, dataset=self.dataset,
            hidden_dim=256, embed_dim=128)

        # Log
        self.logger.info(f"Fusion Dataset Sample: {self.sample}")
        self.logger.info("Encoder Initialized with the Params: \n"
                         f"  learning_rate: {self.pitch_seq_model_params.learning_rate}\n"
                         f"  dropout: {self.pitch_seq_model_params.dropout}\n"
                         f"  hidden_dim: {self.pitch_seq_model_params.hidden_dim}\n"
                         f"  embed_dim: {self.pitch_seq_model_params.embed_dim}\n"
                         f"  dataset: {type(self.pitch_seq_model_params.dataset).__name__}"
                         )

        # Initialize all encoders
        self.pitch_sequence_encoder = PitchSequenceEncoder(
            self.pitch_seq_model_params)

        # Custom Dataclass for the Trainer
        components = TrainerComponents(
            dataset=self.dataset,
            pitch_seq_encoder=self.pitch_sequence_encoder,
            num_epochs=pitch_sequence_pipeline_components.num_epochs,
            batch_size=pitch_sequence_pipeline_components.batch_size
        )

        # Log
        self.logger.info("Trainer Initialized with the Params: \n"
                         f"  num_epochs: {components.num_epochs}\n"
                         f"  batch_size: {components.batch_size}\n"
                         f"  dataset: {type(components.dataset).__name__}\n"
                         f"  pitch_seq_encoder: {type(components.pitch_seq_encoder).__name__}"
                         )
        # Initialize trainer
        self.trainer = PitchSequenceTrainer(components)

        # We Good message
        self.logger.info("Pipeline is chilling, initialized")

    def execute(self):
        history = self.trainer.train()
