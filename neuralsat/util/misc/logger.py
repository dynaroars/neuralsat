import logging
import sys


LOGGER_LEVEL = {
    0: logging.NOTSET,
    1: logging.INFO,
    2: logging.DEBUG,
}

logging.basicConfig(
    format='%(levelname)-8s %(asctime)-12s %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)

logger = logging.getLogger(__name__)