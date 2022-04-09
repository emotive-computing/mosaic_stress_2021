import sys
from ..utils.singleton import Singleton
import logging
import pdb

def cm2_unhandled_exception_handler(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.exit(1) # Keyboard interrupts should exit
    else:
        Logger.getInst().error("%s: %s"%(exc_type.__name__, exc_value))
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        pdb.pm()

class LessThanFilter(logging.Filter):
    """ Custom filter for our logging StreamHandler to direct error and above messages to stderr
        and warning and below messages to stdout
    """
    def __init__(self, exclusive_maximum, name=""):
        super(LessThanFilter, self).__init__(name)
        self.max_level = exclusive_maximum

    def filter(self, record):
        #non-zero return means we log this message
        return 1 if record.levelno < self.max_level else 0

class CustomFormatter(logging.Formatter):
    """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

    grey = '\x1b[38;21m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.blue + self.fmt + self.reset,
            logging.INFO:  self.grey + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

@Singleton
class Logger(object):
    def __init__(self):
        self._loggingPrefix = ""
        # Get the root logger
        self.logger = logging.getLogger()
        # Have to set the root logger level, it defaults to logging.WARNING
        self.logger.setLevel(logging.NOTSET)
        # Define format for logs
        fmt = '%(asctime)s | %(levelname)8s | %(message)s'

        # TODO: if logging == Y
        #
        #  TODO: if console == Y
        # Create stream to log to stdout
        self.logging_handler_out = logging.StreamHandler(sys.stdout)
        # Min level for logging to stdout is INFO #TODO: add logic to take in user-specified console min level
        self.logging_handler_out.setLevel(logging.INFO)
        # ERROR and above will not log to stdout
        self.logging_handler_out.addFilter(LessThanFilter(logging.ERROR))
        self.logging_handler_out.setFormatter(CustomFormatter(fmt))
        self.logger.addHandler(self.logging_handler_out)

        # Create stream to log to stderr
        self.logging_handler_err = logging.StreamHandler(sys.stderr)
        # Min level for logging to stderr is ERROR
        self.logging_handler_err.setLevel(logging.ERROR) #TODO: add logic to take in user-specified console min level
        self.logging_handler_err.setFormatter(CustomFormatter(fmt))
        self.logger.addHandler(self.logging_handler_err)

        # Create stream to log to file
        # TODO: if file == Y
        self.logging_handler_file = logging.FileHandler("common_models.log") # TODO: set file location
        self.logging_handler_file.setLevel(logging.INFO) #TODO: add logic to take in user-specified file min level
        self.logging_handler_file.setFormatter(logging.Formatter(fmt))
        self.logger.addHandler(self.logging_handler_file)

        return

    def setLoggingPrefix(self, prefix):
        self._loggingPrefix = prefix
        return

    def debug(self, message):
        self.logger.debug("%28s | %s" % (self._loggingPrefix, message))
        return

    def info(self, message):
        self.logger.info("%28s | %s" % (self._loggingPrefix, message))
        return

    def warning(self, message):
        self.logger.warning("%28s | %s" % (self._loggingPrefix, message))
        return

    def error(self, message):
        self.logger.error("%28s | %s" % (self._loggingPrefix, message))
        return

    def critical(self, message):
        self.logger.critical("%28s | %s" % (self._loggingPrefix, message))
        return

    def exception(self, message):
        self.logger.exception("%28s | %s" % (self._loggingPrefix, message))
        return
