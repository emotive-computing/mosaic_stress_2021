import os
import sys
from abc import ABC, abstractmethod

class StageBase(ABC):
    def __init__(self):
        self._enableCaching = False  # TODO - implement persistence and caching for faster runtime across executions
        self._inputData = None
        self._outputData = None

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
