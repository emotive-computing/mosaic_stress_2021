from collections.abc import Iterable
from .stage_base import StageBase

class ModelPredictionStage(StageBase):
    def __init__(self, pred_idx):
        super().__init__()
        self._pred_idx = pred_idx
        self._prediction_context = None
        return

    def setPredictionContext(self, prediction_context):
        self._prediction_context = prediction_context
        return

    def _validate(self, dc):
        if not isinstance(self._prediction_context, PredictionContext):
            raise ValueError("{} requires a context of type {}".format(type(self).__name__, type(PredictionContext)))
        if "data" not in dc.get_keys():
            raise ValueError("{} needs a dataframe object named 'data' to be present in the data container".format(type(self).__name__))
        if "trained_model" not in dc.get_keys():
            raise ValueError("{} needs a trained model named 'trained_model' to be present in the data container".format(type(self).__name__))
        if min(self._pred_idx) < 0 or max(self._pred_idx) >= len(dc.get_item('data').index):
            raise ValueError("Test indices exceed bounds of the data size in {}".format(type(self).__name__))
        self._prediction_context.validate()
        return

    def _execute(self, dc):
        trained_model = dc.get_item("trained_model")
        self.logInfo("Making predictions with model {}".format(type(trained_model).__name__))

        data = dc.get_item('data')
        data_test = data[self._prediction_context.feature_cols]
        data_test = data_test.iloc[self._pred_idx, :]
        self._prediction_context.model._model = trained_model
        predictions = self._prediction_context.model.predict(data_test)
        dc.set_item('predictions', predictions)
        return dc


class PredictionContext():
    def __init__(self):
        self._feature_cols = None
        return

    def get_feature_cols(self):
        return self._feature_cols

    def set_feature_cols(self, cols):
        self._feature_cols = cols
        return

    def validate(self):
        if not isinstance(self._feature_cols, Iterable):
            raise TypeError("feature_cols must be initialized to some Iterable type")
        return

    feature_cols = property(get_feature_cols, set_feature_cols)
