import logging
from datetime import datetime

from utils.constants import Constants


class Logger:

    def __init__(self, class_name: str):
        self.class_name = class_name
        self._configure_logger()

    def _configure_logger(self) -> None:
        self.logger = logging.getLogger(self.class_name)
        self.logger.setLevel(logging.DEBUG)

        # Avoid duplicate handlers if reused
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(console_handler)

    def _log(self, level: str, color: str, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] [{self.class_name}] {message}"
        colored_message = f"{color}{full_message}{Constants.COLOR_RESET}"

        if level == "debug":
            self.logger.debug(colored_message)
        elif level == "info":
            self.logger.info(colored_message)
        elif level == "warning":
            self.logger.warning(colored_message)
        elif level == "error":
            self.logger.error(colored_message)

    def info(self, message: str) -> None:
        self._log("info", Constants.COLOR_INFO, message)

    def debug(self, message: str) -> None:
        self._log("debug", Constants.COLOR_INFO, f"DEBUG: {message}")

    def warning(self, message: str) -> None:
        self._log("warning", Constants.COLOR_WARNING, f"WARNING: {message}")

    def error(self, message: str) -> None:
        self._log("error", Constants.COLOR_ERROR, f"ERROR: {message}")
