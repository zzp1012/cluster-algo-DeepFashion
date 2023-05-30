import os
import datetime
import argparse
import logging

def get_datetime() -> str:
    """get the date.
    Returns:
        date (str): the date.
    """
    datetime_ = datetime.datetime.now().strftime("%m%d-%H%M%S")
    return datetime_


def set_logger(save_path: str) -> None:
    """set the logger.
    Args:
        save_path(str): the path for saving logfile.txt
        name(str): the name of the logger
        verbose(bool): if true, will print to console.

    Returns:
        None
    """
    # set the logger
    logfile = os.path.join(save_path, "logfile.txt")
    logging.basicConfig(filename=logfile,
                        filemode="w+",
                        format='%(name)-12s: %(levelname)-8s %(message)s',
                        datefmt="%H:%M:%S",
                        level=logging.INFO)
    # define a Handler which writes DEBUG messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # tell the handler to use this format
    console.setFormatter(logging.Formatter(
        '%(name)-12s: %(levelname)-8s %(message)s'))
    # add the handler to the root logger
    logging.getLogger().addHandler(console)


def get_logger(name:str,
               verbose:bool = True) -> logging.Logger:
    """get the logger.
    Args:
        name (str): the name of the logger
        verbose (bool): if true, will print to console.
    Returns:
        logger (logging.Logger)
    """
    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)
    if not verbose:
        logger.setLevel(logging.INFO)
    return logger


def log_settings(args: argparse.Namespace, config: dict = {}) -> None:
    """log the settings of the program. 
    Args:
        args (argparse.Namespace): the arguments.
        config (dict): the config.
    """
    logger = get_logger(__name__)
    hyperparameters = {
        **args.__dict__, 
        **{key: value for key, value in config.items() \
            if key.isupper() and type(value) in [int, float, str, bool, dict]}
    }
    logger.info(hyperparameters)