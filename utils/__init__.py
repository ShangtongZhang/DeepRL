from .config import *
from .normalizer import *
from .misc import *

try:
    from .tf_logger import Logger
except:
    from .vanilla_logger import Logger

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logger = logging.getLogger('MAIN')
logger.setLevel(logging.INFO)