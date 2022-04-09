import os
import sys
from .stage_base import StageBase
from ..io.data_container import DataContainer
from ..log.logger import Logger

class Pipeline(StageBase):
    def __init__(self):
        super().__init__()
        self._stages = []
        self._dc = DataContainer()

    def addStage(self, stage):
        if isinstance(stage, StageBase):
            self._stages.append(stage)
        else:
            Logger.getInst().error("addStage() called with an object which is not derived from "+ type(StageBase))
        return

    def getDC(self):
        return self._dc

    def run(self):
        self._dc = self.execute(self._dc)
        return

    def _validate(self, dc):
        return

    def _execute(self, dc):
        for stage in self._stages:
            # get name of stage to be executed
            stage_name = type(stage).__name__
            # call logger function to set prefix
            Logger.getInst().setLoggingPrefix(stage_name)
            dc = stage.execute(dc)
        return dc
