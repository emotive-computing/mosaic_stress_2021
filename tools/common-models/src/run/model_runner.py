# -*- coding: utf-8 -*-
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from src.configuration.settings_template import Settings
from src.models.hierarchical_classifier import HierarchicalDataset
from src.models.utils import save_data_frame, save_feature_weights, save_model, load_model
from src.pipeline.cross_validation.cross_validation import CrossValidation
from src.pipeline.cross_validation.data_folds import DataFolds
from src.run.model_run_instance import ModelRunInstance
from src.common import utils
from src.io.read_data_input import Dataset
from src.metrics.predictions import PredictionMetrics
from src.metrics.results import ResultMetrics
from src.io.print_output import Print
from src.models.utils import load_model_for_fold


class ModelRunner(object):

    @classmethod
    def get_all_model_run_instances(cls):
        return [ModelRunInstance(model_class, label, feature_source) for model_class, label, feature_source in \
                itertools.product(Settings.MODELS_TO_RUN, Settings.COLUMNS.Y_LABELS_TO_PREDICT,
                                  Settings.FEATURE_INPUT_SOURCES_TO_RUN)]


    @classmethod
    def run(cls, model_run_instance):
        predictions= pd.DataFrame()
        results = pd.DataFrame()

        for result, prediction in cls._get_results_for_run(model_run_instance):
            results = results.append(result, ignore_index=True)
            predictions = predictions.append(prediction, ignore_index=True)

        return results, predictions

    # Main loop to run for a given model and label.
    # Splits the data into folds, finds best hyperparameters, fits the models, and makes predictions on test set, adding predictions to pool
    # Returns metrics about the predictions (scores for AUC, Accuracy, etc...) and array of predictions to be included in the predictions file that is output
    @classmethod
    def _get_results_for_run(cls, model_run_instance):
        results = []

        X, y, le = Dataset().get(model_run_instance.label, model_run_instance.feature_source)
        if not Settings.IO.USE_NEW_PREDEFINED_FOLDS_FILE and not Settings.IO.USE_PREDEFINED_FOLDS_FILE:
            print("Shuffling the data before running analysis") # BB - why shuffle without checking the settings shuffle flag?
            X, y = shuffle(X, y, random_state=Settings.RANDOM_STATE)
            X = X.reset_index(drop=True)

        # if doing one vs rest predictions for multiclass labels, create a separate label/estimator for each of the one vs rest predictions
        if le and len(le.classes_) > 2 and Settings.USE_ONE_VS_ALL_CLF_FOR_MULTICLASS:
            base_label = model_run_instance.label
            for i, c in enumerate(le.classes_):
                label = "{}_{}".format(base_label, c)
                y, le = Dataset.get_one_vs_rest_dataset(c, y) # TODO fix
                results.append(cls._run_all_folds_for_instance(model_run_instance.get_new_instance_with_label(label), X, y))
        else:
            results = [cls._run_all_folds_for_instance(model_run_instance, X, y)]

        return results


    @classmethod
    def _run_all_folds_for_instance(cls, model_run_instance, X, y):
        predicted_probabilities = np.asarray([])
        X_all, y_all = [], []
        data_folds = DataFolds.get(X, y)

        fold_num = 0
        for train_index, test_index in data_folds:
            start_time = datetime.now()
            fold_num += 1

            print("\n\nRunning fold #{}...".format(fold_num))

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Apply column filters
            X_train, y_train = Dataset().apply_column_filters(X_train, y_train, model_run_instance.label)
            X_test, y_test = Dataset().apply_column_filters(X_test, y_test, model_run_instance.label)

            # Impute missing data withing the fold
            #if Settings.IMPUTER:
            #    print("Imputing missing data in fold #%d"%(fold_num))
            #    Settings.IMPUTER.fit(X_train)
            #    X_train = Settings.IMPUTER.transform(X_train)
            #    X_test = Settings.IMPUTER.transform(X_test)

            if Settings.COLUMNS.USE_HIERARCHICAL.SCHEMA:
                hier = HierarchicalDataset.get_hierarchical(X_train, y_train, model_run_instance, train_index)
                if hier:
                    X_train, y_train = hier
                #x_test and y_test should be unaffected

            if Settings.CROSS_VALIDATION.SHUFFLE:
                X_train, y_train = shuffle(X_train, y_train, random_state=Settings.RANDOM_STATE)
                X_train = X_train.reset_index(drop=True)
                
            if hasattr(model_run_instance.model_class, 'shuffle_baseline') and getattr(model_run_instance.model_class, 'shuffle_baseline'):
                np.random.shuffle(y_train)

            # Print group by column value if used
            if Settings.SHOW_GROUP_BY_COLUMN_VALUE:
                train_group_by_value = X_train.iloc[0][Settings.COLUMNS.GROUP_BY_COLUMN]
                test_group_by_value = X_test.iloc[0][Settings.COLUMNS.GROUP_BY_COLUMN]
                print("\n\nFold #: {}, Train Set GROUP_BY_COLUMN value: {}, Test Set GROUP_BY_COLUMN value: {}".format(fold_num, train_group_by_value, test_group_by_value ))
            # X and y test are passed in for keras to continually show improving scores during training

            if fold_num == 1 and Settings.SAVE_FULL_DATA_FRAME:
                save_data_frame(pd.concat((X_train, X_test), axis=0), np.hstack((y_train, y_test)).T, model_run_instance.label)
            # you can use either Settings.LOAD_MODELS or Settings.RUN_FROM_SAVED_MODELS - both do the same. Kept for backward compatibilty.
            if (Settings.SAVE_MODELS and Settings.LOAD_MODELS) or Settings.RUN_FROM_SAVED_MODELS:
                fitted_model = load_model_for_fold(model_run_instance, fold_num)

            if (not Settings.LOAD_MODELS and not Settings.RUN_FROM_SAVED_MODELS) or fitted_model is None:
                print('Training model for fold: {}'.format(fold_num))
                fitted_model = CrossValidation.get_cross_validated_model(X_train, y_train, model_run_instance, fold_num)
            else:
                print('Loaded saved model for fold: {}'.format(fold_num))
                fitted_model = fitted_model['model']
                

            if Settings.PREDICTION.is_regression():
                predicted_probabilities_for_fold = fitted_model.predict(X_test)
            elif Settings.PREDICTION.is_multiclass():
                predicted_probabilities_for_fold = fitted_model.predict(X_test)
            elif hasattr(fitted_model, "predict_proba"):
                predicted_probabilities_for_fold = fitted_model.predict_proba(X_test)
            else:
                predicted_probabilities_for_fold = fitted_model.decision_function(X_test)


            if utils.class_has_method(fitted_model.named_steps.MODEL, 'do_after_fit'):
                fitted_model.named_steps.MODEL.do_after_fit(predicted_probabilities_for_fold)
                predicted_probabilities_for_fold= fitted_model.predict_proba(X_test)

            # print out metrics per fold
            if Settings.SHOW_GROUP_BY_COLUMN_VALUE:
                Print.print_results_per_fold(model_run_instance, fold_num=fold_num, probabilities=predicted_probabilities_for_fold, y=y_test, num_train_examples=len(X_train), num_test_examples=len(X_test), train_group_by_value=train_group_by_value, test_group_by_value=test_group_by_value)
            else:
                Print.print_results_per_fold(model_run_instance, fold_num=fold_num, probabilities=predicted_probabilities_for_fold, y=y_test, num_train_examples=len(X_train), num_test_examples=len(X_test))

            predicted_probabilities = utils.concatenate(predicted_probabilities, predicted_probabilities_for_fold)
            combined_x = utils.combine_if_list(pd.concat, X_test)
            combined_y = utils.combine_if_list(utils.flatten, y_test)

            X_all = utils.concatenate(X_all, combined_x[Settings.COLUMNS.IDENTIFIER].values)
            y_all = utils.concatenate(y_all, combined_y)

            # BB - joblib's model load fails sometimes on files *it* dumped. This feature is useful workaround
            if Settings.SAVE_FEATURE_WEIGHTS:
                save_feature_weights(fitted_model, model_run_instance, fold_num)

            if Settings.SAVE_MODELS:
                print("Saving model for fold: {}".format(fold_num))
                save_model(fitted_model, model_run_instance, fold_num)


            time_elapsed = datetime.now() - start_time
            print('Time elapsed for fold #{}: {}'.format(fold_num, time_elapsed))

        out_results, out_predictions = ResultMetrics.get(model_run_instance), PredictionMetrics.get(model_run_instance)
        return out_results.get_metrics_per_run_instance(y_true=y_all, probabilities=predicted_probabilities), out_predictions.get_prediction_rows(X_all, y_all, probabilities=predicted_probabilities)


