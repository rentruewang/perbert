import logging

from rich import pretty, traceback
from rich.logging import RichHandler


def install():
    pretty.install()
    traceback.install()
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.addHandler(RichHandler())
