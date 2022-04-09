# -*- coding: utf-8 -*-
import csv
import itertools
import os

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle

from src.common import utils
from src.configuration.settings_module_loader import SettingsModuleLoader
from src.configuration.settings_template import Settings
from src.io.read_data_input import Dataset
from src.pipeline.cross_validation.cross_validation import CrossValidation
from src.pipeline.stages import FeatureExtractionStages
from src.pipeline.stages import PipelineStages
from src.run.model_runner import ModelRunner
from src.metrics.predictions import PredictionMetrics, ClassificationPredictionMetrics
from src.metrics.results import ResultMetrics
from src.models.utils import load_models, load_model_for_fold
from src.io.print_output import Print, OutputFolders

# import matplotlib as mpl
# mpl.use('agg')  # or whatever other backend that you want
# import matplotlib.pyplot as plt

title_font = {'size': '30', 'color': 'black', 'weight': 'normal',
              'verticalalignment': 'center'}  # Bottom vertical alignment for more space
axis_font = {'size': '26'}

COEFFICIENTS = "Coefficients"
IMPORTANCES = "Importances"
CORRELATIONS = "SpearmanCorrelation"

wordcloud_parent_folder_name = 'generalizability_feature_info'
saved_model_parent_folder_name = 'saved_models'
wordcloud_values_filename_prefix = "generalizability-feature-values"


# Helper function to get all child directories under a given directory
def get_list_of_child_directories(d):
    return [item for item in os.listdir(d) if os.path.isdir(os.path.join(d, item))]


# Find top ngrams for each trait by averaging correlations across folds
def avg_ngrams_across_folds():
    for model, label in \
            itertools.product(Settings.MODELS_TO_RUN, Settings.COLUMNS.Y_LABELS_TO_PREDICT):

        # Directory holding results for a specific model (MultinomialNB, Random Forest, etc...)
        model_dir = get_directory_name(model.__name__, label,
                                       IMPORTANCES)  # os.path.join(results_dir_path, "n_gram_correlations", model)

        # There will be a directory for each of the folds under the directory for a given trait - get all of these directories
        fold_directories = get_list_of_child_directories(os.path.join(model_dir))
        for experiment in Settings.COLUMNS.GENERALIZABILITY_FILTERS:

            filename = get_file_title(wordcloud_values_filename_prefix, label, "csv", file_name_suffix=experiment["experiment_id"])

            fold_df = None

            # Read csv file for each of the folds holding correlation values
            for fold_dir in fold_directories:

                # Read in csv file holding correlation values for each of the n-grams for the current fold
                ngrams_df = pd.read_csv(
                    os.path.join(model_dir, fold_dir, filename)).set_index('feature')

                # Add the correlation values for n-grams for the current fold to the dataframe holding correlations across folds
                if fold_df is not None:
                    fold_df = fold_df.join(ngrams_df, how='outer', on='feature', lsuffix=fold_dir,
                                        rsuffix=str(int(fold_dir) + 1))
                else:
                    fold_df = ngrams_df

            if fold_df is None:
                continue

            # Calculations to find top 0 n-grams across folds for this trait
            fold_df = fold_df.fillna(0)  # Any missing values get a correlation value of 0
            fold_df['mean'] = fold_df.mean(axis=1)  # Find the average value for each n-gram across folds
            fold_df.sort_values("mean", inplace=True, ascending=False)  # Sort average values in descending order
            top_df = fold_df #.head(10)  # Find top 10 n-grams by average correlation value

            print("{}: ".format(label) + ", ".join(top_df.index.values[0:10]))

            # Write the top 10 n-grams for the given trait to a csv file
            csv_output_file = os.path.join(model_dir, filename)
            utils.ensure_directory_for_file(csv_output_file)
            top_df.to_csv(csv_output_file, index=True, na_rep=0, columns=['mean'], float_format='%.3f')


def get_directory_name(model_name, label, y_label, fold=None):
    directory = os.path.join(Settings.IO.RESULTS_OUTPUT_FOLDER, wordcloud_parent_folder_name, model_name, y_label,
                             label)
    if fold:
        directory = os.path.join(directory, str(fold))
    utils.ensure_directory(directory)
    return directory


def get_saved_model_path(model_name, y_label, fold=1):
    directory = os.path.join(Settings.IO.RESULTS_OUTPUT_FOLDER, saved_model_parent_folder_name, model_name, y_label)
    model_file = os.path.join(directory, str(fold))
    utils.ensure_directory(directory)
    return directory

def get_chart_title(model_name, label, y_label):
    model_title = "Vectorizer for {}".format(model_name) if y_label == CORRELATIONS else model_name
    model_title = "{} {} of n-grams with {}".format(model_title, y_label, label)

    return model_title


def get_file_title(file_start, label, file_type, file_name_suffix=""):
    if file_name_suffix:
        return "{}-{}-{}.{}".format(file_start, label, file_name_suffix, file_type)
    return "{}-{}.{}".format(file_start, label, file_type)


def model_is_instance_of_class(model, cls):
    return model is cls or isinstance(model, cls)


def can_find_coefficients_for_model(model):
    return model_is_instance_of_class(model, MultinomialNB)


def can_find_importances_for_model(model):
    return model_is_instance_of_class(model,
                                                  RandomForestClassifier) or model_is_instance_of_class(
        model, RandomForestRegressor)


#
# def draw_word_cloud_based_on_model(model_name, label, y_label, wscores, fold):
#
#     wordcloud = WordCloud(width=640, height=480, margin=0, colormap="rainbow", scale=8).generate_from_frequencies(wscores)
#
#     plt.axis("off")
#     plt.rcParams['ytick.labelsize'] = 26
#     plt.rcParams['xtick.labelsize'] = 26
#     fig, ax = plt.subplots(1, 1, figsize=(1600 / run.MY_DPI, 1000 / run.MY_DPI),
#                            dpi=run.MY_DPI)
#
#     fig.suptitle(get_chart_title(model_name, label, y_label), y=.95, **title_font)
#     fig.subplots_adjust(wspace=0.3)  # , top=1.1)
#
#   #  ax.legend().set_visible(False)
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     ax.imshow(wordcloud, interpolation="bilinear")
#     directory = get_directory_name(model_name, label, y_label, fold)
#     plt.savefig(os.path.join(directory, get_file_title("wordcloud", label, "png")))
#     plt.close('all')
#
# def draw_barchart(model_name, label, y_label, wscores, fold):
#     fig, ax = plt.subplots(1, 1, figsize=(1600 / run.MY_DPI, 1000 / run.MY_DPI),
#                            dpi=run.MY_DPI)
#     plt.axis("on")
#
#     N = 25
#
#     x = [k[0] for k in sorted(wscores.items(), key=lambda x: x[1])][-N:]
#     values = [wscores[k] for k in x]
#     ax.barh(x, values, align='center', alpha=0.2)
#     ax.plot(values, range(len(x)), '-o', markersize=5, alpha=0.8)
#
#     ax.set_yticks(x, values)
#
#     ax.set_xlabel(y_label, **axis_font)
#
#     fig.suptitle(get_chart_title(model_name, label, y_label), y=.95, **title_font)
#     plt.tight_layout()
#     fig.subplots_adjust(top=.9)
#
#     directory = get_directory_name(model_name, label, y_label, fold)
#     plt.savefig(os.path.join(directory, get_file_title("barchart", label, "png")))
#     plt.close('all')


def print_values_to_csv(model_run_instance, y_label, wscores, fold, file_name_suffix=""):
    directory = get_directory_name(model_run_instance.model_name, model_run_instance.label, y_label, fold)
    x = [k[0] for k in sorted(wscores.items(), key=lambda x: x[1])]
    values = [wscores[k] for k in x]
    if y_label.endswith('s'):
        y_label = y_label[:-1]
    with open(os.path.join(directory, get_file_title(wordcloud_values_filename_prefix, model_run_instance.label, "csv", file_name_suffix)), 'w',
              newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["feature", y_label])
        xs = []
        for x_, value_ in zip(reversed(x), reversed(values)):
            csvwriter.writerow([x_, value_])
            xs.append(x_)
        print(", ".join(xs[0:10]))


def output_wordclouds_barchart_csv(model_run_instance, y_label, wscores, fold, file_name_suffix=""):
    # draw_word_cloud_based_on_model(model_name, label, y_label, wscores, fold)
    # draw_barchart(model_name, label, y_label, wscores, fold)
    print_values_to_csv(model_run_instance, y_label, wscores, fold, file_name_suffix)


def get_correlations(vect, X, y, i=None):
    # Change multiclass y values to binary one vs rest format
    if i is not None:
        y = Dataset.get_one_vs_rest_dataset(i, y)

    try:
        X = vect.transform(X)
    except NotFittedError:
        X = vect.fit_transform(X)

    features_corr = [spearmanr(x, y)[0] for x in X.toarray().T]
    wscores = dict(zip(vect.get_feature_names(), features_corr))
    return wscores

def get_all_feature_importances_for_model(fitted_pipeline, model_run_instance, X, y, fold, file_name_suffix=""):
    clf = fitted_pipeline.named_steps[PipelineStages.MODEL.name]

    if can_find_importances_for_model(clf):
        importances = clf.feature_importances_
        wscores = dict(zip(model_run_instance.feature_source.regular_feature_names, importances))
        output_wordclouds_barchart_csv(model_run_instance, IMPORTANCES, wscores, fold, file_name_suffix)

def get_all_correlations_coefficients_importances_for_model(fitted_pipeline, model_run_instance, X, y, fold, file_name_suffix=""):
    clf = fitted_pipeline.named_steps[PipelineStages.MODEL.name]
    vect = fitted_pipeline.named_steps[PipelineStages.EXTRACT_FEATURES.name].transformer_list[0][1].named_steps[FeatureExtractionStages.VECTORIZER.name]

    if can_find_importances_for_model(clf):
        importances = clf.feature_importances_
        wscores = dict(zip(vect.get_feature_names(), importances))
        output_wordclouds_barchart_csv(model_run_instance, IMPORTANCES, wscores, fold, file_name_suffix)

    elif can_find_coefficients_for_model(clf):  # MultinomialNB
        importances = clf.coef_[0]
        wscores = dict(zip(vect.get_feature_names(), importances))
        output_wordclouds_barchart_csv(model_run_instance, COEFFICIENTS, wscores, fold, file_name_suffix)

    # Get correlations
   # wscores_correlations = get_correlations(vect, X, y)
  #  output_wordclouds_barchart_csv(model_run_instance, CORRELATIONS, wscores_correlations, fold)


def predict_generalizability():
    model_instances = ModelRunner.get_all_model_run_instances()
    for model_run_instance in model_instances:
        for experiment in Settings.COLUMNS.GENERALIZABILITY_FILTERS:
            print("\n\nTesting Generalizability for: {}".format(experiment["experiment_id"]))
            print("\n\nEvaluating models for label: {}".format(model_run_instance.label))
            X, y, le = Dataset().get(model_run_instance.label, model_run_instance.feature_source, custom_column_filters=experiment['filters'])
            X, y = shuffle(X, y, random_state=Settings.RANDOM_STATE)
            test_group_value = X.iloc[0][experiment["GROUPID_COLUMN"]]  if "GROUPID_COLUMN" in experiment else "NA"
            print("\nExperiment Group ID: {}".format(test_group_value))
            print("\nNum Test Examples: {}".format(len(y)))
            models = load_models(model_run_instance)
            folds = len(models)
            predicted_probabilities = np.zeros([len(y), 2])
            for model_info in models:
                fold_num = model_info['fold']
                print("\n\nRunning generalizability for fold #{}...".format(fold_num))

                # params = file_utils.read_hyperparameters_for_fold(model_key, label, fold_num)
                #  hyperparams = Settings.CROSS_VALIDATION.HYPER_PARAMS.VECTORIZER
                #hyperparams = {k: v[0] for (k, v) in hyperparams.items()}

                #params = PipelineParams(X, model_run_instance, hyperparams)
                #pipeline = Pipeline([(PipelineStages.EXTRACT_FEATURES.name, FeatureStep.get_step(params))])

                #vect = PmiCountVectorizer.get(hyperparams)
                # vect.fit(X_train[model_run_instance.feature_source.language_column_name], y_train)
                # fitted_model = CrossValidation.get_cross_validated_model(X_train, y_train, model_run_instance, fold_num)

                fitted_model = model_info['model']

                if Settings.PREDICTION.is_regression():
                    predicted_probabilities_for_fold = fitted_model.predict(X)
                elif Settings.PREDICTION.is_multiclass():
                    predicted_probabilities_for_fold = fitted_model.predict(X)
                elif hasattr(fitted_model, "predict_proba"):
                    predicted_probabilities_for_fold = fitted_model.predict_proba(X)
                else:
                    predicted_probabilities_for_fold = fitted_model.decision_function(X)


                if utils.class_has_method(fitted_model.named_steps.MODEL, 'do_after_fit'):
                    fitted_model.named_steps.MODEL.do_after_fit(predicted_probabilities_for_fold)
                    predicted_probabilities_for_fold = fitted_model.predict_proba(X)

                # print out metrics per fold
                Print.print_generalizability_results_per_fold(model_run_instance, fold_num=fold_num, probabilities=predicted_probabilities_for_fold, y=y, num_test_examples=len(X),test_group_by_value=test_group_value, file_name_suffix=experiment["experiment_id"])


                if model_run_instance.feature_source.includes_language_features:
                    get_all_correlations_coefficients_importances_for_model(fitted_model, model_run_instance, X, y, fold_num, experiment["experiment_id"])
                if model_run_instance.feature_source.includes_regular_features:
                    get_all_feature_importances_for_model(fitted_model, model_run_instance, X, y,
                                                                            fold_num, experiment["experiment_id"])
                # Add predictions across folds for average
                predicted_probabilities += predicted_probabilities_for_fold
                # Get correlations
                #wscores_correlations = get_correlations(vect, X_train[model_run_instance.feature_source.language_column_name], y_train)
            # output_wordclouds_barchart_csv(model_run_instance, CORRELATIONS, wscores_correlations, fold_num)
            predicted_probabilities /= folds

            # Print out predictions
            out_predictions = ClassificationPredictionMetrics(model_run_instance)
            predictions = out_predictions.get_prediction_rows(X[Settings.COLUMNS.IDENTIFIER].values, y, probabilities=predicted_probabilities)
            predictions_df = pd.DataFrame()
            predictions_df = predictions_df.append(predictions, ignore_index=True)
            Print.output_csv(OutputFolders.predictions.name,
                             PredictionMetrics.get_output_column_names(
                                 predictions_df),
                             predictions_df, model_run_instance,
                             append_to_name="_"+experiment["experiment_id"]+"_generalizability")

            Print.output_confusion(OutputFolders.confusion.name, predictions_df, model_run_instance, append_to_name="_"+experiment["experiment_id"]+"_generalizability")

            # print out metrics across all folds
            Print.print_generalizability_results(model_run_instance, probabilities=predicted_probabilities, y=y, num_test_examples=len(X), file_name_suffix=experiment["experiment_id"], test_group_by_value = test_group_value)


def main():
    SettingsModuleLoader.init_settings()
    predict_generalizability()
    avg_ngrams_across_folds()


if __name__ == "__main__":
    main()
