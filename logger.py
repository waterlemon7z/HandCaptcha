"""프로젝트 공용 INFO·DEBUG·ERROR 로그 출력."""

import logging
import sys


logger = logging.getLogger("handcaptcha")
logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)

INFO = logger.info
DEBUG = logger.debug
ERROR = logger.error
