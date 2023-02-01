from util.misc.logger import logger
import arguments

import logging

LOGGER_LEVEL = {
    0: logging.NOTSET,
    1: logging.INFO,
    2: logging.DEBUG,
}

def update_arguments(args):
    arguments.Config['device'] = args.device
    arguments.Config['attack'] = args.attack
    arguments.Config['batch'] = args.batch
    arguments.Config['pre_verify_mip_refine'] = args.refine

    logger.setLevel(LOGGER_LEVEL[args.verbosity])
    
                