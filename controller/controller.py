from controller.cli_arguments import CLIArguments
from model.pitch_sequence_pipeline import PitchSequencePipeline
from model.pitch_sequence_pipeline_components import PitchSequencePipelineComponents
from utils.logger import Logger


class Controller:

    def __init__(self, parsed_args) -> None:
        self.parsed_args = parsed_args
        self._logger = Logger(self.__class__.__name__)
        
        components = PitchSequencePipelineComponents(
            num_epochs=parsed_args.num_epochs,
            learning_rate_pitch_sequence= parsed_args.lr_pitch_seq,
            dropout_pitch_sequence=parsed_args.dropout_pitch_seq,
            sample=parsed_args.sample,
            batch_size=parsed_args.batch_size
        )
        
        PitchSequencePipeline(pitch_sequence_pipeline_components=components).execute()
