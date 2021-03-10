from imports import *


def log(path, file):

    """[Creating a log file to record the experiment's logs]

    Arguments:

        path {string} --> path to the directory
        file {string} --> filename

    Returns
        [function] --> logger that record logs

    """

    ## Checking if the file exists 
    log_file = os.path.join(path, file)

    if not os.path.isfile(log_file):
        open(log_file, "w+").close()

    console_logging_format = "%(levelname)s %(message)s"
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

    ## Configuring the logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()

    ## creating a file handler for output file
    handler = logging.FileHandler(log_file)

    ## Creating a file handler for output file
    handler.setLevel(logging.INFO)

    ## Creating a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    ## Adding the handlers to the logger
    logger.addHandler(handler)

    return logger