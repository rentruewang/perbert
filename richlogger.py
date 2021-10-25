import logging

from rich.logging import RichHandler


def install():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.addHandler(RichHandler())
