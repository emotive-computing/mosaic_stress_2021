import sys
from commonmodels2.log.logger import cm2_unhandled_exception_handler
sys.excepthook = cm2_unhandled_exception_handler