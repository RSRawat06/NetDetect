from ...logs import setup_logger
import logging


train_logger = setup_logger('train', 'main_iscx/', logging.DEBUG)
eval_logger = setup_logger('eval', 'main_iscx/', logging.DEBUG)

