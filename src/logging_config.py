import logging
import sys
from colorlog import ColoredFormatter

def setup_logging():
    # create color formatter for console
    console_formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s%(reset)s %(bold_white)s%(message)s%(reset)s",
        datefmt="%H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    
    # configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    
    # create logger
    logger = logging.getLogger('chessAI')
    logger.setLevel(logging.INFO)
    
    # remove any existing handlers
    logger.handlers = []
    
    # add console handler
    logger.addHandler(console_handler)
    
    return logger 