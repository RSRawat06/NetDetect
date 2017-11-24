from ...logs import setup_logger
import logging


train_logger = setup_logger('train', 'main/', logging.INFO)
eval_logger = setup_logger('eval', 'main/', logging.DEBUG)

