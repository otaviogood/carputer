"""Helper python script to generate debug messages : Gil Montague"""
import datetime

#Variable to surpress debug and info messages
verbose = True

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


def get_current_time():
    """
        Returns the current system time
        @rtype: string
        @return: The current time, formatted
    """
    return ConsoleColors.BOLD + str(datetime.datetime.now().strftime("%H:%M:%S")) + ConsoleColors.ENDC

def print_info(message):
    """
        Prints an info message to console

        Args:
            message (str): The message to be printed
    """
    print("[%-20s][%s] %s" % (INFO, get_current_time(), str(message)))

def print_debug(message):
    """
        Prints a debug message to console

        Args:
            message (str): The message to be printed
    """
    if verbose:
        print("[%-20s][%s] %s" % (DEBUG, get_current_time(), str(message)))

def print_warning(message):
    """
        Prints a Warning message to console

        Args:
            message (str): The message to be printed
    """
    print("[%-20s][%s] %s" % (WARNING, get_current_time(), str(message)))

def print_fatal(message):
    """
        Prints a fatal message to console

        Args:
            message (str): The message to be printed
    """
    print("[%-20s][%s] %s" % (FATAL_ERROR, get_current_time(), str(message)))