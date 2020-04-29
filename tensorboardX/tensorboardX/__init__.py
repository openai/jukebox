"""A module for visualization with tensorboard
"""

from .record_writer import RecordWriter
from .torchvis import TorchVis
from .writer import FileWriter, SummaryWriter

__version__ = "1.8"  # will be overwritten if run setup.py
