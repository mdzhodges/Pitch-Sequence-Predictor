import asyncio
from controller.cli_arguments import CLIArguments
from utils.logger import Logger
from controller.controller import Controller


async def main() -> int:
    logger: Logger = Logger(class_name=__name__)
    args: CLIArguments = CLIArguments()
    try:
        
        logger.info(f"Hello from inside: {__name__}")
        Controller(args.parse)
    except Exception as e:
        logger.error(f"Error Message: {e}")
        raise Exception(f"Error Message: {e}") from e

    return 0


if __name__ == "__main__":
    asyncio.run(main())
