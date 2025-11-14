from model.custom_types.data_types import ModelComponents
from model.custom_types.pitch_sequence_pipeline_components import PitchSequencePipelineComponents
from model.custom_types.trainer_type import TrainerComponents
from model.pitch_sequence_encoder.pitch_sequence_encoder import PitchSequenceEncoder
from model.training.pitch_sequence_trainer import PitchSequenceTrainer
from preprocessing.fusion_dataset import FusionDataset
from utils.logger import Logger
from evaluation.eval import PitchSequenceEvaluator


class PitchSequencePipeline:

    def __init__(self, pitch_sequence_pipeline_components: PitchSequencePipelineComponents):
        # Sample for the pitch sequence dataset
        self.sample = pitch_sequence_pipeline_components.sample

        # Logger
        self.logger = Logger(self.__class__.__name__)

        # Check params of trainer, model, and sample
        self.dataset = FusionDataset(sample=self.sample)

        # Initialize model_params
        self.pitch_seq_model_params = ModelComponents(
            learning_rate=pitch_sequence_pipeline_components.learning_rate_pitch_sequence,
            dropout=pitch_sequence_pipeline_components.dropout_pitch_sequence, dataset=self.dataset,
            hidden_dim=512, embed_dim=256)

        # Log
        self.logger.info(f"Fusion Dataset Sample: {self.sample}")
        self._display_model_parameters()

        # Initialize all encoders
        self.pitch_sequence_encoder = PitchSequenceEncoder(
            self.pitch_seq_model_params)

        # Custom Dataclass for the Trainer
        trainer_components: TrainerComponents = TrainerComponents(
            dataset=self.dataset,
            pitch_seq_encoder=self.pitch_sequence_encoder,
            num_epochs=pitch_sequence_pipeline_components.num_epochs,
            batch_size=pitch_sequence_pipeline_components.batch_size
        )

        # Log
        self._display_trainer_params(trainer_components=trainer_components)
        # Initialize trainer
        self.trainer = PitchSequenceTrainer(model_params=trainer_components)

        # We Good message
        self.logger.info("Pipeline is chilling, initialized")

    def _display_model_parameters(self) -> None:
        self.logger.info("Encoder Initialized with the Params: \n"
                         f"  Learning Rate: {self.pitch_seq_model_params.learning_rate}\n"
                         f"  Drop Out Rate: {self.pitch_seq_model_params.dropout}\n"
                         f"  Hidden Dimensions: {self.pitch_seq_model_params.hidden_dim}\n"
                         f"  Embedded Dimensions: {self.pitch_seq_model_params.embed_dim}\n"
                         f"  Dataset Name: {type(self.pitch_seq_model_params.dataset).__name__}"
                         )

    def _display_trainer_params(self, trainer_components: TrainerComponents) -> None:
        self.logger.info("Trainer Initialized with the Params: \n"
                         f"  Num Epochs: {trainer_components.num_epochs}\n"
                         f"  Batch Size: {trainer_components.batch_size}\n"
                         f"  Dataset Name: {type(trainer_components.dataset).__name__}\n"
                         f"  Pitch Sequence Encoder Name: {type(trainer_components.pitch_seq_encoder).__name__}"
                         )

    def execute(self):
        history = self.trainer.train()