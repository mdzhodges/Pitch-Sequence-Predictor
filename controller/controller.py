from controller.cli_arguments import CLIArguments
from model.pitch_sequence_pipeline import PitchSequencePipeline
from utils.logger import Logger


class Controller:

    def __init__(self, parsed_args: CLIArguments) -> None:
        self.parsed_args = parsed_args
        self._logger = Logger(self.__class__.__name__)
        PitchSequencePipeline()
