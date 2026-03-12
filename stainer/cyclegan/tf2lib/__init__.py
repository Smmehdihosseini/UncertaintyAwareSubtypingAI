import tensorflow as tf

from .data import *
from .image import *
from .ops import *
from .utils import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for d in physical_devices:
    tf.config.experimental.set_memory_growth(d, True)
