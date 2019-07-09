import os
import logging
from logging import config


def get_project_root():
    """
    Gets project root.

    :return: str (path)
    """
    root_path = os.path.dirname(os.path.abspath(__file__))
    return root_path


def get_logger(name, log_file_name):
    """Returns a logger with the provided name, configuration path, logging path and logging level"""

    logging_conf_path = os.path.join(get_project_root(), 'configurations/logging.ini')

    try:
        if os.path.isdir(os.path.join(get_project_root(), 'logs')):
            logging_file_path = os.path.join(get_project_root(), 'logs', log_file_name)
            logging.config.fileConfig(logging_conf_path, defaults={"logging_path": logging_file_path})
        else:
            logging.basicConfig()
    except KeyError:
        logging.basicConfig()

    # create logger
    logger = logging.getLogger(name)

    return logger
