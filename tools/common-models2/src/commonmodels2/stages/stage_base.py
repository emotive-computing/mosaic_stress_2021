import os
import sys
from abc import ABC, abstractmethod
from ..log.logger import Logger

class StageBase(ABC):
    def __init__(self):
        self._loggingPrefix = type(self).__name__ + ": "
        self._enableCaching = False  # TODO - implement persistence and caching for faster runtime across executions
        self._inputData = None
        self._outputData = None

    def setLoggingPrefix(self, prefix):
        self._loggingPrefix = prefix + type(self).__name__ + ": "
        return

    def logDebug(self, message):
        Logger.getInst().debug(self._loggingPrefix + message)
        return

    def logInfo(self, message):
        Logger.getInst().info(self._loggingPrefix + message)
        return

    def logWarning(self, message):
        Logger.getInst().warning(self._loggingPrefix + message)
        return

    def logError(self, message):
        Logger.getInst().error(self._loggingPrefix + message)
        return

    #def setCached(self, enableCaching):
    #    self._enableCaching = False # TODO

    # This function should not be overridden in subclasses
    def execute(self, dc):
        self._validate(dc)
        dc = self._execute(dc)
        return dc

    @abstractmethod
    def _validate(self, dc):
        raise NotImplementedError()

    @abstractmethod
    def _execute(self, dc):
        raise NotImplementedError()
