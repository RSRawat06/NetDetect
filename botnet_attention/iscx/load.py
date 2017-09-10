"""
iscx.load

Load in local ISCX dataset for use.
"""

from ..utils import data
from . import config


fetch_data = data.load(config)

