import asyncio

from controller.cli_arguments import CLIArguments
from controller.controller import Controller
from data_collection.context_collection import ContextDataCollection
from utils.logger import Logger


async def main() -> int:
    logger: Logger = Logger(class_name=__name__)
    args: CLIArguments = CLIArguments()
    try:

        controller: Controller = Controller(parsed_args=args)

        # pitch_sequence_data_collection: PitchSequenceDataCollection = PitchSequenceDataCollection()
        #
        # pitch_sequence_data_collection.export_stat_cast_pitch_by_pitch_dataframe()
        # pitch_sequence_data_collection.export_dataframe_to_parquet_file()

        context_data_collection: ContextDataCollection = ContextDataCollection()

        context_data_collection.export_stat_cast_dataframe()
        context_data_collection.export_dataframe_to_parquet_file()



    except Exception as e:
        logger.error(f"Error Message: {e}")
        raise Exception(f"Error Message: {e}") from e

    return 0


if __name__ == "__main__":
    asyncio.run(main())
