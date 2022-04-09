# -*- coding: utf-8 -*-
import itertools
from datetime import datetime

import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from src.configuration.settings_module_loader import SettingsModuleLoader
from src.configuration.settings_template import SettingsEnumOptions
from src.io.print_output import Print
from src.configuration.settings_template import Settings
from src.models.hierarchical_classifier import HierarchicalDataset
from src.models.utils import save_model
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
    def  run(cls, model_run_instance):
        predictions= pd.DataFrame()
        results = pd.DataFrame()
        cls._get_results_for_run(model_run_instance)

    # Main loop to run for a given model and label.
    # Splits the data into folds, finds best hyperparameters, fits the models, and makes predictions on test set, adding predictions to pool
    # Returns metrics about the predictions (scores for AUC, Accuracy, etc...) and array of predictions to be included in the predictions file that is output
    @classmethod
    def _get_results_for_run(cls, model_run_instance):
        results = []

        X, y, le = Dataset().get(model_run_instance.label, model_run_instance.feature_source)
        X, y = Dataset().apply_column_filters(X, y, model_run_instance.label)
        if not Settings.IO.USE_NEW_PREDEFINED_FOLDS_FILE:
            X, y = shuffle(X, y, random_state=Settings.RANDOM_STATE)
            X = X.reset_index()

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
        if Settings.CROSS_VALIDATION.NUM_TRAIN_TEST_FOLDS == 1:
            fold_num = 1
            X_train, X_test = X, X
            y_train, y_test = y, y
            # embed()
            dir = os.path.join(Settings.IO.RESULTS_OUTPUT_FOLDER, 'dataset', model_run_instance.label)
            if fold_num:
                dir = os.path.join(dir, str(fold_num))

            utils.ensure_directory(dir)
            x_train_name = os.path.join(dir,"{}.pkl".format('x-train'))
            y_train_name = os.path.join(dir,"{}.npy".format('y-train'))
            x_test_name = os.path.join(dir,"{}.pkl".format('x-test'))
            y_test_name = os.path.join(dir,"{}.npy".format('y-test'))

            print("\n\nSaving fold #{}...".format(fold_num))
            print("\nTrain Size: ({}, {}), Test Size: {}".format(len(X_train), len(y_train), len(X_test), len(y_test)))

            X_train.to_pickle(x_train_name)
            X_test.to_pickle(x_test_name)
            np.save(y_train_name, y_train)
            np.save(y_test_name, y_test)
            return []
        data_folds = DataFolds.get(X, y)
        print("\nTotal Dataset Instances: {}, Labels: {}".format(len(X), len(y)))
        fold_num = 0
        for train_index, test_index in data_folds:
            start_time = datetime.now()
            fold_num += 1

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # embed()
            dir = os.path.join(Settings.IO.RESULTS_OUTPUT_FOLDER, 'dataset', model_run_instance.label)
            if fold_num:
                dir = os.path.join(dir, str(fold_num))

            utils.ensure_directory(dir)
            x_train_name = os.path.join(dir,"{}.pkl".format('x-train'))
            y_train_name = os.path.join(dir,"{}.npy".format('y-train'))
            x_test_name = os.path.join(dir,"{}.pkl".format('x-test'))
            y_test_name = os.path.join(dir,"{}.npy".format('y-test'))

            print("\n\nSaving fold #{}...".format(fold_num))
            print("\nTrain Size: ({}, {}), Test Size: {}".format(len(X_train), len(y_train), len(X_test), len(y_test)))

            X_train.to_pickle(x_train_name)
            X_test.to_pickle(x_test_name)
            np.save(y_train_name, y_train)
            np.save(y_test_name, y_test)

        return []

def run_model_comparison_task():

    Print.print_run_info(SettingsModuleLoader.settings_file)

    for model_run_instance in ModelRunner.get_all_model_run_instances():

        Print.print_current_run(model_run_instance)

        ModelRunner.run(model_run_instance)

def main():
    SettingsModuleLoader.init_settings()
    # run each task
    run_model_comparison_task()


if __name__ == "__main__":
    main()


