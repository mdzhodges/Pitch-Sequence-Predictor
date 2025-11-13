import argparse
from typing import Any


class CLIArguments:

    def __init__(self) -> None:
        self._parser = argparse.ArgumentParser(
            description="Command-line Interface for Input Ingestion"
        )
        self._define_arguments()
        self._args = None

    def _define_arguments(self) -> None:
        self._parser.add_argument("--lr_hitter", type=float, default=1e-5)
        self._parser.add_argument("--lr_pitcher", type=float, default=1e-5)
        self._parser.add_argument("--lr_context", type=float, default=1e-5)
        self._parser.add_argument("--lr_pitch_seq", type=float, default=1e-5)
        self._parser.add_argument("--dropout_hitter", type=float, default=.3)
        self._parser.add_argument("--dropout_pitcher", type=float, default=.3)
        self._parser.add_argument("--dropout_context", type=float, default=.3)
        self._parser.add_argument(
            "--dropout_pitch_seq", type=float, default=.3)
        self._parser.add_argument("--sample", type=int, default=1000)
        self._parser.add_argument("--num_epochs", type=int, default=20)
        self._parser.add_argument("--batch_size", type=int, default=25)

    def parse(self):
        self._args = self._parser.parse_args()
        return self._args

    def get(self, name: str) -> Any:
        if self._args is None:
            raise RuntimeError(
                "Arguments have not been parsed yet. Call parse() first.")
        return getattr(self._args, name, None)
