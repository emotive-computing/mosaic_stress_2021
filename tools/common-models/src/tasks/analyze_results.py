import glob
import os

import numpy as np
import pandas as pd

from src.configuration.settings_template import Settings
from src.configuration.settings_module_loader import SettingsModuleLoader
from src.common import utils
from src.metrics.output_columns import CorrelationOutputColumnNames, MetricsOutputColumnNames, \
    RunInfoOutputColumnNames, AdditionalMetricsOutputColumnNames
from src.metrics.results import ResultMetrics


def concat_output_files(output_folder):
    files = glob.glob(output_folder + "/*/*.csv")
    predictions_df = pd.concat([pd.read_csv(f, index_col=None, header=0) for f in files], sort=True)
    return predictions_df

def concat_fold_output_files(output_folder, fold_num):
    files = glob.glob(output_folder + "/*/" + str(fold_num) + "/*.csv")
    df = pd.concat([pd.read_csv(f, index_col=None, header=0) for f in files])
    return df

def generate_results(results_folder):
    predictions_df = concat_output_files(os.path.join(results_folder, "predictions"))
   # results_df = concat_output_files(os.path.join(results_folder, "results"))
    # plot_model_comparison.plot_results(model_keys, results) # plot model comparison for AUC
    # plot_model_comparison.plot_results(model_keys, results, metric=run_utils.data_config.ACCURACY_COLUMN_NAME, metric_name='Accuracy') # plot model comparison for Accuracy

    # write out predictions to csv file
    # TODO: Fix (column name issue when combined) - is this even necessary?
    all_predictions = predictions_df.replace(np.nan, 'NA', regex=True)
    # all_predictions.to_csv(os.path.join(results_folder, "all-predictions.csv"),
    #                        index=False, na_rep="NA",
    #                        columns=data_utils.get_predictions_file_output_column_names(all_predictions))



    results_df = ResultMetrics.get_metrics_from_all_predictions(all_predictions)


    # write out results to csv file
    csv_output_file = os.path.join(results_folder, "all-results.csv")
    results_df.to_csv(csv_output_file, index=False, na_rep="NA",
                      columns=ResultMetrics.get_output_column_names(results_df))

    sort_key = CorrelationOutputColumnNames.Pearson_correlation.name \
        if Settings.PREDICTION.is_regression() \
        else AdditionalMetricsOutputColumnNames.AUPRC_pos.name # MetricsOutputColumnNames.AUC.name
    best_results_df = results_df.sort_values(sort_key, ascending=False).groupby(RunInfoOutputColumnNames.Label.name, sort=False).head(1)
    csv_output_file = os.path.join(results_folder, "best-results.csv")
    best_results_df.to_csv(csv_output_file, index=False, na_rep="NA",
                           columns=ResultMetrics.get_output_column_names(best_results_df))

    # if Settings.CROSS_VALIDATION.USE_PRESELECTED_FOLDS_FILE:
    #     for fold_num in range(1, Settings.CROSS_VALIDATION.NUM_TRAIN_TEST_FOLDS+1):
    #
    #         results_df_for_fold = concat_fold_output_files(os.path.join(results_folder, "results_per_fold"), fold_num)
    #         csv_output_file = os.path.join(results_folder, "fold-" + str(fold_num) + "-results.csv")
    #         results_df_for_fold.to_csv(csv_output_file, index=False, na_rep="NA",
    #                                    columns=ResultMetrics.get_output_column_names(results_df_for_fold))
    #
    #
    #         best_results_df_for_fold = results_df_for_fold.sort_values(sort_key, ascending=False).groupby('label', sort=False).head(1)
    #         csv_output_file = os.path.join(results_folder,  "best-fold-" + str(fold_num) + "-results.csv")
    #         best_results_df_for_fold.to_csv(csv_output_file, index=False, na_rep="NA",
    #                                columns=ResultMetrics.get_output_column_names(best_results_df_for_fold ))

    # try:
    #     plot_model_comparison.plot_results()
    #
    #     if Settings.setting_is_enabled(run_config.REGRESSION):
    #         plot_regression.plot_regression(all_predictions)
    #         plot_regression.plot_regression_distribution()
    #     else:
    #         plot_probability_histogram.plot_prob_dist(all_predictions)  # plot probability histogram of predictions
    #         plot_confusion_matrix.output_confusion_matrices(all_predictions)  # output confusion matrices
    # except:
    #     print("SKIPPING: Generation of plots. R not installed/configued on this computer.")

def clean_results():
    results_folder = Settings.IO.RESULTS_OUTPUT_FOLDER
    utils.try_delete_directory(os.path.join(results_folder, "distribution_plots"))
    utils.try_delete_directory(os.path.join(results_folder, "prediction_plots"))
    utils.try_delete_file(os.path.join(results_folder, "all-predictions.csv"))
    utils.try_delete_file(os.path.join(results_folder, "all-results.csv"))
    utils.try_delete_file(os.path.join(results_folder, "best-results.csv"))
    utils.try_delete_file(os.path.join(results_folder, "compare.png"))

    for fold_num in range(1, Settings.CROSS_VALIDATION.NUM_TRAIN_TEST_FOLDS+1):
        utils.try_delete_file(os.path.join(results_folder, "fold-" + str(fold_num) + "-results.csv"))
        utils.try_delete_file(os.path.join(results_folder, "best-fold-" + str(fold_num) + "-results.csv"))


# name of settings file to use must be passed as an argument to the program
#     parser = argparse.ArgumentParser(description='Generate plots and other results after models have finished running')
#    # parser.add_argument('--results_folder', type=str, help='results folder')
#     parser.add_argument('--settings', type=str, help='settings file to use')

   # args = parser.parse_args()
    #
    # if not (args.results_folder):
    #     raise ValueError(
    #         "Name of results_folder location must be passed as input. "
    #         "To run: python3 dropbox_utils.py --results_folder results_folder_name")

    #print("Using results folder: ", args.results_folder)

    # if not args.settings:
    #     raise ValueError(
    #         "Name of settings file to be used must be passed as input. To run: python3 main.py --settings settings-whatever.py")
    #
    # # print settings file being used and make the variables in the settings file available to the python program
    # print("Using settings from: ", args.settings)
    # s = __import__(args.settings.replace(".py", ""))
def main():
    SettingsModuleLoader.init_settings()

 #   clean_results()
    generate_results(Settings.IO.RESULTS_OUTPUT_FOLDER)

if __name__ == "__main__":
    main()




