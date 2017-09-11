"""
ISOT:

This module handles training and evaluation of
the models on the ISOT dataset.
Please run `python3 -m botnet_attention.isot.train`
to run training.
To download the ISOT dataset, please run
`python3 -m botnet_attention.isot.download`.
"""

from . import train, download, config, load
