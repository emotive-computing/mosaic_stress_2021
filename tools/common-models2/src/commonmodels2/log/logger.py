import os
import sys
from ..utils.singleton import Singleton
import logging

@Singleton
class Logger(object):
    def __init__(self):
        #logging.basicConfig(filename='', encoding='utf-8', level=logging.DEBUG)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        return

    def debug(self, message):
        logging.debug(message)
        return

    def info(self, message):
        logging.info(message)
        return

    def warning(self, message):
        logging.warning(message)
        return

    def error(self, message):
        logging.error(message)
        return
