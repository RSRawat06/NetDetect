"""
isot.load

Load in local ISOT dataset for use.
"""

from ..utils import data
from . import config


fetch_data = data.load(config)

