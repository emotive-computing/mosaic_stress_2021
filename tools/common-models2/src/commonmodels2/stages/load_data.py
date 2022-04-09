import os
import sys
import pandas as pd
from .stage_base import StageBase
from ..log.logger import Logger


class LoaderStageBase(StageBase):
    def __init__(self):
        super().__init__()
        return

    def _validate(self, dc):
        raise NotImplementedError()
        return

    def _execute(self, dc):
        raise NotImplementedError()
        return


class CSVLoaderStage(LoaderStageBase):
    def __init__(self):
        super().__init__()
        self._file_path = None

    def setFilePath(self, file_path):
        self._file_path = file_path
        return

    def _validate(self, dc):
        if self._file_path is None:
            raise RuntimeError("Need to call setFilePath() first")
        elif not os.path.exists(self._file_path):
            raise ValueError("File path does not exist: {}".format(self._file_path))
        return

    def _execute(self, dc):
        Logger.getInst().info("Loading CSV data from: {}".format(self._file_path))
        dc.set_item('data_file_path', self._file_path)
        df = pd.read_csv(self._file_path)
        dc.set_item('data', df)
        return dc


class DataFrameLoaderStage(LoaderStageBase):
    def __init__(self):
        super().__init__()
        self._data = None

    def setDataFrame(self, data, reset_index=False):
        if reset_index:
            self._data = data.reset_index(drop=True)
        else:
            self._data = data
        return

    def _validate(self, dc):
        if self._data is None:
            raise RuntimeError("Need to call setDataFrame() first")
        return

    def _execute(self, dc):
        Logger.getInst().info("Loading data frame")
        dc.set_item('data', self._data)
        return dc
