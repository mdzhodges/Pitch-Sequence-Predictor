from data_collection.context_data_collection import ContextDataCollection
from data_collection.fusion_collection import FusionCollection
from model.custom_types.pitch_sequence_pipeline_components import PitchSequencePipelineComponents
from model.pitch_sequence_pipeline import PitchSequencePipeline
from utils.logger import Logger


class Controller:

    def __init__(self, parsed_args) -> None:
        self.parsed_args = parsed_args
        self.logger = Logger(self.__class__.__name__)
        if parsed_args.gen_data:
            self.logger.info("Generating Datasets")
            self.export_context_dataset_to_parquet_file()
        self.execute_sequence_pipeline()

    def export_context_dataset_to_parquet_file(self) -> None:
        context_data_collection: ContextDataCollection = ContextDataCollection()

        context_data_collection.export_stat_cast_dataframe()
        context_data_collection.export_dataframe_to_parquet_file()

        fusion_collection: FusionCollection = FusionCollection()

        fusion_collection.preprocess_unified()

    def execute_sequence_pipeline(self) -> None:
        components = PitchSequencePipelineComponents(
            num_epochs=self.parsed_args.num_epochs,
            learning_rate_pitch_sequence=self.parsed_args.lr_pitch_seq,
            dropout_pitch_sequence=self.parsed_args.dropout_pitch_seq,
            sample=self.parsed_args.sample,
            batch_size=self.parsed_args.batch_size
        )

        PitchSequencePipeline(
            pitch_sequence_pipeline_components=components).execute()
