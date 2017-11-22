import logging
from ...logs import setup_logger


train_logger = setup_logger('train', 'main_isot/', logging.DEBUG)
eval_logger = setup_logger('eval', 'main_isot/', logging.DEBUG)

