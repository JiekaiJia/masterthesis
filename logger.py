import logging
import sys


def get_logger(logger_name, file_name):
    # Create a logger.
    logger = logging.getLogger(logger_name)
    # Setting logging level to debug
    logger.setLevel(logging.DEBUG)
    # Setting logging format
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # Create a stream handler
    handler1 = logging.StreamHandler(sys.stdout)
    handler1.setLevel(logging.DEBUG)
    # Create a file handler
    handler2 = logging.FileHandler(file_name)
    handler2.setLevel(logging.DEBUG)

    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)

    # add handler into logger
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


import tensorflow as tf
print(tf.__version__)
