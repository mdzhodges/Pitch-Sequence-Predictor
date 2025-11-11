from controller.cli_arguments import CLIArguments
from model.pitch_sequence_pipeline import PitchSequencePipeline
from utils.logger import Logger


class Controller:

    def __init__(self, parsed_args) -> None:
        self.parsed_args = parsed_args
        self._logger = Logger(self.__class__.__name__)
        
        PitchSequencePipeline(
            num_epochs=parsed_args.num_epochs,
            learning_rate_hitter=parsed_args.lr_hitter,
            learning_rate_pitcher=parsed_args.lr_pitcher,
            learning_rate_context=parsed_args.lr_context,
            learning_rate_pitch_sequence= parsed_args.lr_pitch_seq,
            dropout_hitter=parsed_args.dropout_hitter,
            dropout_pitcher=parsed_args.dropout_pitcher,
            dropout_context=parsed_args.dropout_context,
            dropout_pitch_sequence=parsed_args.dropout_pitch_seq,
            sample=parsed_args.sample).execute()
