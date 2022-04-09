# -*- coding: utf-8 -*-
import warnings

import os
import pandas as pd
from src.configuration.settings_module_loader import SettingsModuleLoader
from src.configuration.settings_template import SettingsEnumOptions
from src.configuration.settings_template import Settings
from src.io.print_output import Print
from src.run.model_runner import ModelRunner
from src.run.create_cv_folds_runner import CreateCrossValidationFoldsRunner

warnings.simplefilter(action='ignore', category=FutureWarning)


# Runs a given task if set to True in settings
# First prints out name of task, runs the callback task function, and then prints that the task has been completed
def run_task(task_enum):
    Print.print_starting_task(task_enum)

    if task_enum == SettingsEnumOptions.Tasks.MODEL_COMPARISON:
        run_model_comparison_task()
    elif task_enum == SettingsEnumOptions.Tasks.CREATE_TRAIN_TEST_FOLDS:
        create_cv_folds_task()

    Print.print_completed_task(task_enum)
    return

def run_model_comparison_task():
    for model_run_instance in ModelRunner.get_all_model_run_instances():
        Print.print_current_run(model_run_instance)
        results, predictions = ModelRunner.run(model_run_instance)
        Print.print_predictions_and_results_from_all_folds(results, predictions, model_run_instance)
    return

def create_cv_folds_task():
    if not os.path.isdir(Settings.IO.RESULTS_OUTPUT_FOLDER):
        os.makedirs(Settings.IO.RESULTS_OUTPUT_FOLDER)

    for create_cv_folds_run_instance in CreateCrossValidationFoldsRunner.get_all_create_cv_folds_run_instances():
        Print.print_current_run(create_cv_folds_run_instance)
        train_folds, test_folds = CreateCrossValidationFoldsRunner.run(create_cv_folds_run_instance)
        train_folds.columns = [x+'_train' for x in train_folds.columns]
        test_folds.columns = [x+'_test' for x in test_folds.columns]
        merged_folds = pd.concat((train_folds, test_folds), axis=1)

        stratified_class_names = str(create_cv_folds_run_instance.stratified_class_names)
        if len(stratified_class_names) == 0:
            stratified_class_names = "None"
            stratified_bins_str = ""
        else:
            if Settings.CROSS_VALIDATION.STRATIFIED_BINNING_EDGES is not None:
                stratified_bins_str = '_stratifiedBins-'+str(Settings.CROSS_VALIDATION.STRATIFIED_BINNING_EDGES)
            else:
                stratified_bins_str = '_percentileBins-'+str(Settings.CROSS_VALIDATION.STRATIFIED_NUMBER_PERCENTILE_BINS)
        out_file_prefix = 'stratifiedOn-'+stratified_class_names+stratified_bins_str+'_shuffle-'+str(create_cv_folds_run_instance.do_shuffle)+'_seed-'+str(create_cv_folds_run_instance.random_seed)
        merged_folds.to_csv(os.path.join(Settings.IO.RESULTS_OUTPUT_FOLDER, out_file_prefix+'_folds.csv'), index=False, header=True)
    return

def main():
    SettingsModuleLoader.init_settings()

    Print.print_run_info(SettingsModuleLoader.settings_file)

    # run each task
    for task in Settings.TASKS_TO_RUN:
        run_task(task)

    return

if __name__ == "__main__":
    main()
