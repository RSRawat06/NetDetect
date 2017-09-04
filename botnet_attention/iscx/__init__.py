"""
ISCX:

This module handles training and evaluation of
the botnet_attention model on the ISCX dataset.
Please run `python3 -m botnet_attention.iscx.train`
to run training.
To download the ISCX dataset, please run
`python3 -m botnet_attention.iscx.download`.
"""

from . import train, download, config
