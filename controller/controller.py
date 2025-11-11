from utils.logger import Logger
from model.pitch_sequence_pipeline import PitchSequencePipeline
from controller.cli_arguments import CLIArguments


class Controller:

    def __init__(self, parsed_args: CLIArguments) -> None:
        self._logger = Logger(self.__class__.__name__)
        PitchSequencePipeline()
        