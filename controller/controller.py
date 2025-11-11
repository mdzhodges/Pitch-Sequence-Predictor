from utils.logger import Logger


class Controller:

    def __init__(self) -> None:
        self._logger = Logger(self.__class__.__name__)
