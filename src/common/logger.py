import os
import sys
from loguru import logger


class LoguruLogger:
    def __init__(self, filename):
        self.filename = filename
        self.log_format = ("| <green>{time:YY/MM/DD}</green> | <green>{time:HH:mm:ss}</green> "
                           "| <level>{level}</level> | <cyan>{extra[task]}</cyan> - {message}")
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def print(self, level, task, text):
        logger.remove()
        logger.add(sys.stderr, format=self.log_format, colorize=True, level="INFO")
        logger.log(level, text, task=task)

    def write(self, level, task, text):
        logger.remove()
        logger.add(self.filename, format=self.log_format, colorize=False, level="INFO")
        logger.log(level, text, task=task)

    def print_and_write(self, level, task, text):
        logger.remove()
        logger.add(sys.stderr, format=self.log_format, colorize=True, level="INFO")
        logger.add(self.filename, format=self.log_format, colorize=False, level="INFO")
        logger.log(level, text, task=task)
