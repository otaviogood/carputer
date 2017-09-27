"""Helper python script to generate debug messages"""
import datetime
import logging
from logging.handlers import RotatingFileHandler

class ConsoleColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


INFO = ConsoleColors.OKGREEN + ConsoleColors.BOLD + "INFO" + ConsoleColors.ENDC
DEBUG = ConsoleColors.OKBLUE + ConsoleColors.BOLD + "DEBUG" + ConsoleColors.ENDC
WARNING = ConsoleColors.WARNING + \
    ConsoleColors.BOLD + "WARNING" + ConsoleColors.ENDC
FATAL_ERROR = ConsoleColors.FAIL + ConsoleColors.BOLD + \
    "FATAL" + ConsoleColors.ENDC

MAX_LOG_FILE_SIZE_MB = 2

class DebugMessage(object):
    """A pretty print class for debug messages"""

    def __init__(self, verbose=True, enable_logging=False):
        self.verbose = verbose
        self.enable_logging = enable_logging

    def get_current_time(self):
        """
            Returns the current system time
            @rtype: string
            @return: The current time, formatted
        """
        return ConsoleColors.BOLD + str(datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]) + ConsoleColors.ENDC

    def init_logging(self, logging_file):
        self.logging_file = logging_file
        self.print_info("Logging to file enabled")
        self.print_info("Logging to {}".format(self.logging_file))

        self.logger = logging.getLogger('carputer')
        self.logging_handler = RotatingFileHandler(self.logging_file, maxBytes=MAX_LOG_FILE_SIZE_MB*1024*1024)
        self.formatter = logging.Formatter('%(asctime)s %(message)s')

        self.logging_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.logging_handler)
        self.logger.setLevel(logging.INFO)

    def enable_verbose(self):
        self.print_info("Enabling print debug messages")
        self.verbose = True

    def disable_verbose(self):
        self.print_info("Disabling print debug messages")
        self.verbose = False

    def print_info(self,message=""):
        """
            Prints an info message to console

            Args:
                message (str): The message to be printed
        """
        print("[%-20s][%s] %s" % (INFO, self.get_current_time(), str(message)))

    def print_debug(self,message=""):
        """
            Prints a debug message to console

            Args:
                message (str): The message to be printed
        """
        if self.verbose:
            print("[%-20s][%s] %s" % (DEBUG, self.get_current_time(), str(message)))

    def print_warning(self,message=""):
        """
            Prints a Warning message to console

            Args:
                message (str): The message to be printed
        """
        print("[%-20s][%s] %s" % (WARNING, self.get_current_time(), str(message)))

    def print_fatal(self,message=""):
        """
            Prints a fatal message to console

            Args:
                message (str): The message to be printed
        """
        print("[%-20s][%s] %s" % (FATAL_ERROR, self.get_current_time(), str(message)))

    def log_info(self, message=""):
        self.logger.info(message)
    def log_warning(self, message=""):
        self.logger.warning(message)
    def log_error(self, message=""):
        self.logger.error(message)
    def log_data(self, data=""):
        self.logger.info(data)