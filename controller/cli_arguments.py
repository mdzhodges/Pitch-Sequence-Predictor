import argparse
from typing import Any

from utils.logger import Logger


class CLIArguments:

    def __init__(self) -> None:
        self._logger = Logger(self.__class__.__name__)
        self._parser = argparse.ArgumentParser(
            description="Command-line Interface for Input Ingestion"
        )
        self._define_arguments()
        self._args = None

    def _define_arguments(self) -> None:
        self._parser.add_argument(
            "-m",
            "--cli_argument_name",
            type=str,
            required=True,
            help="CLI Argument to Be Defined Later"
        )

    def parse(self):
        self._args = self._parser.parse_args()

        return self._args

    def get(self, name: str) -> Any:
        if self._args is None:
            self._logger.error("Arguments have not been parsed yet. Call parse() first.")
            raise RuntimeError("Arguments have not been parsed yet. Call parse() first.")
        return getattr(self._args, name, None)
