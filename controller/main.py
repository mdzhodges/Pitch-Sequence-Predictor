import asyncio

from controller.cli_arguments import CLIArguments
from controller.controller import Controller


async def main() -> int:
    args: CLIArguments = CLIArguments()
    try:

        controller: Controller = Controller(parsed_args=args.parse())


    except Exception as e:
        raise Exception(f"Error Message: {e}") from e

    return 0


if __name__ == "__main__":
    asyncio.run(main())
