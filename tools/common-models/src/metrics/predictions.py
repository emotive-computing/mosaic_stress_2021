# -*- coding: utf-8 -*-

import numpy as np

from src.configuration.settings_template import Settings
from src.io.read_data_input import Dataset
from src.metrics.metrics import Metrics
from src.metrics.output_columns import RunInfoOutputColumnNames, PROBABILITY_COLUMN_NAME_SUFFIX, \
    PredictionsOutputColumnNames


class PredictionMetrics(Metrics):
    @staticmethod
    def get(model_run_instance):
        if Settings.PREDICTION.is_regression():
            return RegressionPredictionMetrics(model_run_instance)
        elif Settings.PREDICTION.is_multiclass():
            return MulticlassPredictionMetrics(model_run_instance)
        else:
            return ClassificationPredictionMetrics(model_run_instance)

    def get_base_prediction_row(self, id, true_value, predicted_value):
        return {
            Settings.COLUMNS.IDENTIFIER: id,
            RunInfoOutputColumnNames.Model.name: self.model_run_instance.model_name,
            RunInfoOutputColumnNames.Label.name: self.model_run_instance.label,
            RunInfoOutputColumnNames.Feature_source.name: self.model_run_instance.feature_source_name,
            PredictionsOutputColumnNames.True_value.name: true_value,
            PredictionsOutputColumnNames.Predicted_value.name: predicted_value
        }


    @classmethod
    def get_output_column_names(cls, df):
        return [Settings.COLUMNS.IDENTIFIER] + \
                RunInfoOutputColumnNames.list_member_names() + PredictionsOutputColumnNames.list_member_names() + \
                [col for col in df.columns if col.endswith(PROBABILITY_COLUMN_NAME_SUFFIX)]



class ClassificationPredictionMetrics(PredictionMetrics):

    # Get the set of predictions (array of dictionaries) to be included in the predictions file that is output
    # Each row in this output file corresponds to a single prediction, and each item in the dictionary corresponds to a column of the file
    def get_prediction_rows(self, X_all, y_all, probabilities):
        predictions = np.argmax(probabilities, axis=1) if not isinstance(self.model_run_instance.label, list) else np.argmax(probabilities,axis=2)

        def get_prediction_row(id, true_value, predicted_value, probability):
            row = self.get_base_prediction_row(id, true_value, predicted_value)

            for i, c in enumerate(Dataset().get_saved_label_encoder(self.model_run_instance.label).classes_):
                row[str(c) + PROBABILITY_COLUMN_NAME_SUFFIX] = probability[i]

            return row

        le = Dataset().get_saved_label_encoder(self.model_run_instance.label)
        return [get_prediction_row(id, true_value, predicted_value, probability) for id, true_value, predicted_value, probability in
                           zip(X_all, le.inverse_transform(y_all), le.inverse_transform(predictions), probabilities)]


class MulticlassPredictionMetrics(PredictionMetrics):
    def get_prediction_rows(self, X_all, y_all, probabilities):
        predictions = probabilities

        def get_prediction_row(id, true_value, predicted_value, probability):
            row = self.get_base_prediction_row(id, true_value, predicted_value)
            return row

        le = Dataset().get_saved_label_encoder(self.model_run_instance.label)
        return [get_prediction_row(id, true_value, predicted_value, probability) for id, true_value, predicted_value, probability in
                           zip(X_all, le.inverse_transform(y_all), le.inverse_transform(predictions), probabilities)]


class RegressionPredictionMetrics(PredictionMetrics):
    def get_prediction_rows(self, X_all, y_all, probabilities):
        return [self.get_base_prediction_row(id, true_value, predicted_value) for id, true_value, predicted_value in
                               zip(X_all, y_all, probabilities)]
