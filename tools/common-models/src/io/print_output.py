# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import shutil
from datetime import datetime
from enum import auto

import pandas as pd

from src.common import utils
from src.common.meta import CommonEnumMeta
from src.configuration.settings_template import Settings
from src.metrics.predictions import PredictionMetrics
from src.metrics.results import ResultMetrics
from src.metrics.score import Score


class OutputFolders(CommonEnumMeta):
    results = auto()
    results_per_fold = auto()
    predictions = auto()
    hyperparameters = auto()
    confusion = auto()
    run_info = auto()



class Print:

    @staticmethod
    def print_current_run(model_run_instance):
        print("------------------------------------------")
        print("Running: ", model_run_instance)


    @staticmethod
    def print_starting_task(task):
        print("------------------------------------------------------------------------------")
        print("Running TASK: "+ str(task))

    @staticmethod
    def print_completed_task(task):
        print("Completed TASK: " + str(task))
        print("------------------------------------------------------------------------------")

    @staticmethod
    def print_hyperparameters_for_fold(model_run_instance, fold_num, grid_search, ran_cross_validation=True):

        #TODO: Clean this if/else up
        if ran_cross_validation:
            hyperparameters = grid_search.best_params_
            print("Done with cross validation for {}", model_run_instance)
            print("Best score: %0.3f" % grid_search.best_score_)
            print("Best param_grid set:")
            for param_name in sorted(hyperparameters.keys()):
                print("\t%s: %r" % (param_name, hyperparameters[param_name]))
        else:
            hyperparameters = grid_search

        directory = os.path.join(Settings.IO.RESULTS_OUTPUT_FOLDER, OutputFolders.hyperparameters.name,  model_run_instance.model_name, model_run_instance.label, str(fold_num))
        utils.ensure_directory(directory)
        file_name = "{}-{}.txt".format(model_run_instance.model_name, model_run_instance.label)
        #file_name = "{}-{}-{}.txt".format(model_run_instance.model_name, str(model_run_instance.feature_source), model_run_instance.label)
        full_file_name = os.path.join(directory, Print.sanitize_for_windows(file_name))
        f = open(full_file_name, 'w', encoding='utf-8')

        for param_name in sorted(hyperparameters.keys()):
            f.write("\t%s: %r\n" % (param_name, hyperparameters[param_name]))

        f.close()

    @staticmethod
    def print_pipeline_params(p, params):
        print("pipeline:", [name for name, _ in p.steps])

        for param_name in sorted(params.keys()):
            print("\t%s: %r" % (param_name, params[param_name]))

    @staticmethod
    def output_csv(output_folder_name, column_names, df, model_run_instance, fold_num=None, append_to_name='', use_index=False):
        dir = os.path.join(Settings.IO.RESULTS_OUTPUT_FOLDER, output_folder_name, model_run_instance.model_name)
        if fold_num:
            dir = os.path.join(dir, str(fold_num))

        utils.ensure_directory(dir)
        filename = os.path.join(dir,"{}-{}-{}.csv".format(model_run_instance.label, model_run_instance.feature_source_name, output_folder_name + append_to_name))
        try:
            df.to_csv(filename, index=use_index, na_rep="NA", columns=column_names)
        except:
            df = df.reindex(columns = column_names)
            df.to_csv(filename, index=use_index, na_rep="NA", columns=column_names)


    @staticmethod
    def output_confusion(output_folder_name, df, model_run_instance, append_to_name=""):
        y_true = pd.Series([v + "_actual" for v in df.True_value.values], name="")
        y_pred = pd.Series([v + "_predicted" for v in df.Predicted_value.values], name="Predicted")
        df_confusion = pd.crosstab(y_true, y_pred)
        Print.output_csv(output_folder_name, None, df_confusion, model_run_instance, use_index=True, append_to_name=append_to_name)

        df_confusion_normalized = df_confusion.div(df_confusion.sum(axis=1), axis=0)
        Print.output_csv(output_folder_name, None, df_confusion_normalized, model_run_instance, append_to_name=append_to_name+'-normalized', use_index=True)


    @staticmethod
    def print_predictions_and_results_from_all_folds(results, predictions, model_run_instance):

        Print.output_csv(OutputFolders.results.name, ResultMetrics.get_output_column_names(results), results,
                         model_run_instance)

        Print.output_csv(OutputFolders.predictions.name, PredictionMetrics.get_output_column_names(predictions),
                         predictions, model_run_instance)

        if not Settings.PREDICTION.is_regression():
            Print.output_confusion(OutputFolders.confusion.name, predictions, model_run_instance)


    @staticmethod
    def print_results_per_fold(model_run_instance, fold_num, probabilities, y, num_train_examples, num_test_examples, train_group_by_value=None, test_group_by_value=None):
        results = ResultMetrics.get(model_run_instance)
        df = pd.DataFrame().append(results.get_metrics_per_fold(y_true=y, probabilities=probabilities, num_train=num_train_examples, num_test=num_test_examples), ignore_index=True)
        if Settings.SHOW_GROUP_BY_COLUMN_VALUE:
            df['Train_Group_By_Value'] = train_group_by_value
            df['Test_Group_By_Value'] = test_group_by_value
            Print.output_csv(OutputFolders.results_per_fold.name, ResultMetrics.get_output_column_names(df, include_groupby=True), df, model_run_instance, fold_num)
        else:
            Print.output_csv(OutputFolders.results_per_fold.name, ResultMetrics.get_output_column_names(df), df, model_run_instance, fold_num)
        print("Scoring function and score: ", Score.evaluate_score(y, ResultMetrics.get_positive_probabilities(probabilities)))

    @staticmethod
    def print_generalizability_results_per_fold(model_run_instance, fold_num, probabilities, y, num_test_examples, train_group_by_value=None, test_group_by_value=None, file_name_suffix=""):
        results = ResultMetrics.get(model_run_instance)
        df = pd.DataFrame().append(results.get_metrics_per_fold(y_true=y, probabilities=probabilities, num_test=num_test_examples), ignore_index=True)
        if Settings.SHOW_GROUP_BY_COLUMN_VALUE and (train_group_by_value is not None or test_group_by_value is not None):
            df['Train_Group_By_Value'] = train_group_by_value
            df['Test_Group_By_Value'] = test_group_by_value
            Print.output_csv(OutputFolders.results_per_fold.name, ResultMetrics.get_output_column_names(df, include_groupby=True), df, model_run_instance, fold_num, append_to_name="_"+file_name_suffix+"_generalizability")
        else:
            Print.output_csv(OutputFolders.results_per_fold.name, ResultMetrics.get_output_column_names(df), df, model_run_instance, fold_num, append_to_name="_"+file_name_suffix+"_generalizability")
        print("Scoring function and score: ", Score.evaluate_score(y, ResultMetrics.get_positive_probabilities(probabilities)))

    @staticmethod
    def print_generalizability_results(model_run_instance, probabilities, y, num_test_examples, train_group_by_value=None, test_group_by_value=None, file_name_suffix=""):
        results = ResultMetrics.get(model_run_instance)
        df = pd.DataFrame().append(results.get_metrics_per_fold(y_true=y, probabilities=probabilities, num_test=num_test_examples), ignore_index=True)
        if Settings.SHOW_GROUP_BY_COLUMN_VALUE and (train_group_by_value is not None or test_group_by_value is not None):
            df['Train_Group_By_Value'] = train_group_by_value
            df['Test_Group_By_Value'] = test_group_by_value
            Print.output_csv(OutputFolders.results.name, ResultMetrics.get_output_column_names(df, include_groupby=True), df, model_run_instance, append_to_name="_"+file_name_suffix+"_generalizability")
        else:
            Print.output_csv(OutputFolders.results.name, ResultMetrics.get_output_column_names(df), df, model_run_instance, append_to_name="_"+file_name_suffix+"_generalizability")


    @staticmethod
    def print_run_info(settings_input_filename):
        d_date = datetime.now()
        str_time = d_date.strftime("%d %B %Y %I:%M:%S %p")

        run_info_dir = os.path.join(Settings.IO.RESULTS_OUTPUT_FOLDER, OutputFolders.run_info.name)

        settings_output_file = os.path.join(run_info_dir, "Settings-input-file.py")
        utils.ensure_directory_for_file(settings_output_file)
        with open(settings_input_filename, "r") as f1, open(settings_output_file, "w") as f2:
            shutil.copyfileobj(f1, f2)

        run_info_filename = OutputFolders.run_info.name + "_for_run_started_at_" + str_time.replace(" ",
                                                                                                    "_") + ".txt"
        run_info_output_file = os.path.join(run_info_dir, Print.sanitize_for_windows(run_info_filename))
        with open(run_info_output_file, "w") as f:
            f.write(str_time)

    @staticmethod
    def sanitize_for_windows(name):
        return name.replace("'", "").replace("<", "").replace(">", "").replace(" ", "").replace(":", "")
