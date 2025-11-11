import asyncio

from controller.cli_arguments import CLIArguments
from controller.controller import Controller
from data_collection.pitch_sequence_collection import PitchSequenceDataCollection
from utils.logger import Logger


async def main() -> int:
    logger: Logger = Logger(class_name=__name__)
    args: CLIArguments = CLIArguments()
    try:

        controller: Controller = Controller(parsed_args=args)


    except Exception as e:
        logger.error(f"Error Message: {e}")
        raise Exception(f"Error Message: {e}") from e

    return 0


if __name__ == "__main__":
    asyncio.run(main())
