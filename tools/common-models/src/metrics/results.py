# -*- coding: utf-8 -*-
import abc
from collections import defaultdict
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, f1_score, recall_score, precision_score, \
    average_precision_score, cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from src.common import utils
from src.configuration.settings_template import Settings
from src.io.read_data_input import Dataset
from src.metrics.metrics import Metrics
from src.metrics.output_columns import RunInfoOutputColumnNames, MetricsOutputColumnNames, \
    NumExamplesOutputColumnNames, CorrelationOutputColumnNames, TrueVsPredictedNumExamplesOutputColumnNames, \
    AdditionalMetricsOutputColumnNames, PROBABILITY_COLUMN_NAME_SUFFIX, RegressionMetricsOutputColumnNames, FoldGroupByOutputColumnNames
from src.run.model_run_instance import ModelInfoOnlyInstance


class ResultMetrics(Metrics):

    @classmethod
    def get_child_type(cls):
        if Settings.PREDICTION.is_regression():
            return RegressionResultMetrics
        elif Settings.PREDICTION.is_multiclass():
            return MulticlassResultMetrics
        else:
            return ClassificationResultMetrics

    def get_child_type_from_instance(self):
        return type(self).get_child_type()

    @classmethod
    def get(cls, model_run_instance):
        child_type = cls.get_child_type()
        return child_type(model_run_instance)

    @classmethod
    def get_output_column_names(cls, df, include_groupby=False):
        if Settings.SHOW_GROUP_BY_COLUMN_VALUE:
            return cls.get_child_type().get_output_column_names(df, include_groupby)
        return cls.get_child_type().get_output_column_names(df)

    # Gets base set of metrics
    @abc.abstractmethod
    def get_metrics(self, y_true, probabilities):
        pass

    # Gets set of metrics across all folds
    @abc.abstractmethod
    def get_metrics_per_run_instance(self, y_true, probabilities):
         pass

    # Gets metrics per single fold
    def get_metrics_per_fold(self, y_true, probabilities, num_train=None, num_test=None):
        metrics = self.get_metrics(y_true, probabilities)
        if num_train is not None:
            metrics[NumExamplesOutputColumnNames.Num_train_examples.name] = num_train
        if num_test is not None:
            metrics[NumExamplesOutputColumnNames.Num_test_examples.name] = num_test
        if num_train is not None and num_test is not None:
            metrics[NumExamplesOutputColumnNames.Total_num_examples.name] = num_train + num_test

        return metrics

    @classmethod
    def get_metrics_from_all_predictions(cls, all_predictions):
        return cls.get_child_type().get_metrics_from_all_predictions(all_predictions)


    @classmethod
    def get_positive_probabilities(cls, probabilities):
        return cls.get_child_type().get_positive_probabilities(probabilities)


########################################################################################################################

class RegressionResultMetrics(ResultMetrics):

    def get_metrics_per_run_instance(self, y_true, probabilities):
        metrics = self.get_metrics(y_true, probabilities)
        print("Finished running: ", self.model_run_instance)
        return metrics


    def get_metrics(self, y_true, probabilities):
        metrics = defaultdict()

        metrics[RunInfoOutputColumnNames.Model.name] = self.model_run_instance.model_name
        metrics[RunInfoOutputColumnNames.Label.name] = self.model_run_instance.label
        metrics[RunInfoOutputColumnNames.Feature_source.name] = self.model_run_instance.feature_source_name

        metrics[RegressionMetricsOutputColumnNames.R2_score.name] = r2_score(y_true, probabilities)
        metrics[RegressionMetricsOutputColumnNames.RMSE_score.name] = sqrt(mean_squared_error(y_true, probabilities))

        metrics[CorrelationOutputColumnNames.Pearson_correlation.name], metrics[
            CorrelationOutputColumnNames.Pearson_corr_p_value.name] = pearsonr(y_true, probabilities)

        metrics[CorrelationOutputColumnNames.Spearman_correlation.name], metrics[
            CorrelationOutputColumnNames.Spearman_corr_p_value.name] = spearmanr(y_true, probabilities)

        metrics[NumExamplesOutputColumnNames.Total_num_examples.name] = len(y_true)

        print("Pearson Correlation: {}".format(metrics[CorrelationOutputColumnNames.Pearson_correlation.name]))
        print("Spearman Correlation: {}".format(metrics[CorrelationOutputColumnNames.Spearman_correlation.name]))
        return metrics

    @classmethod
    def get_metrics_from_all_predictions(cls, all_predictions):
        all_results = pd.DataFrame()
        groups = all_predictions.groupby(
            [RunInfoOutputColumnNames.Model.name, RunInfoOutputColumnNames.Feature_source.name,
             RunInfoOutputColumnNames.Label.name])
        for name, group in groups:
            model_run_instance = ModelInfoOnlyInstance(model_name=name[0], feature_source_name=name[1], label=name[2])
            all_results = all_results.append(
                RegressionResultMetrics(model_run_instance).get_metrics(group.True_value.values,
                                                                        group.Predicted_value.values),
                ignore_index=True)
        return all_results

    @classmethod
    def get_output_column_names(self, df):
        return RunInfoOutputColumnNames.list_member_names() + \
               RegressionMetricsOutputColumnNames.list_member_names() + \
               CorrelationOutputColumnNames.list_member_names() + \
               NumExamplesOutputColumnNames.get_columns_to_show_in_output(df)

    @classmethod
    def get_positive_probabilities(cls, probabilities):
        return probabilities


########################################################################################################################

# Find metrics for model / label such as AUC, Accuracy, etc...
class ClassificationResultMetrics(ResultMetrics):

    def get_predictions_from_probabilities(self, probabilities):
        if not isinstance(self.model_run_instance.label, list):
            if probabilities.ndim == 1:
                return np.argmax(probabilities)
            else:
                return np.argmax(probabilities, axis=1)
        else:
            return np.argmax(probabilities, axis=2)

    # Get metrics (like AUROC, etc... ) based on predictions and probability scores of predictions
    def get_metrics_per_run_instance(self, y_true, probabilities):

        # In the case of multi-class prediction (the y label is an array of labels)
        if isinstance(self.model_run_instance.label, list):
            metrics = []
            for idx, lbl in enumerate(self.model_run_instance.label):
                individual_metrics_for_label_class = type(self)(self.model_run_instance.get_new_instance_with_label(lbl))
                individual_metrics_for_label = individual_metrics_for_label_class.get_metrics(y_true[:, idx], probabilities[:, idx], le)
                metrics.append(individual_metrics_for_label)
        else:
            metrics = self.get_metrics(y_true, probabilities)

        print("Finished running: ", self.model_run_instance)
        return metrics

    def get_metrics(self, y_true, probabilities):

        predictions = self.get_predictions_from_probabilities(probabilities)

        metrics = defaultdict()

        metrics[RunInfoOutputColumnNames.Model.name] = self.model_run_instance.model_name
        metrics[RunInfoOutputColumnNames.Label.name] = self.model_run_instance.label
        metrics[RunInfoOutputColumnNames.Feature_source.name] = self.model_run_instance.feature_source_name
        metrics[MetricsOutputColumnNames.Accuracy.name] = accuracy_score(y_true, predictions)

        metrics[NumExamplesOutputColumnNames.Total_num_examples.name] = len(y_true)

        le = Dataset().get_saved_label_encoder(self.model_run_instance.label)

        if le.is_binary_prediction:
            # negative_class_probabilities, positive_class_probabilities = probabilities[:, 0]
            # positive_class_probabilities = probabilities[:, 1]
            negative_class_probabilities, positive_class_probabilities = list(zip(*probabilities))
            metrics[MetricsOutputColumnNames.AUC.name] = roc_auc_score(y_true, list(positive_class_probabilities))

            metrics[TrueVsPredictedNumExamplesOutputColumnNames.True_num_pos_examples.name] = len(
                [i for i in y_true if i == 1])
            metrics[TrueVsPredictedNumExamplesOutputColumnNames.True_base_rate.name] = \
                metrics[TrueVsPredictedNumExamplesOutputColumnNames.True_num_pos_examples.name] / metrics[
                    NumExamplesOutputColumnNames.Total_num_examples.name]

            metrics[TrueVsPredictedNumExamplesOutputColumnNames.Predicted_num_pos_examples.name] = len(
                [i for i in predictions if i == 1])
            metrics[TrueVsPredictedNumExamplesOutputColumnNames.Predicted_base_rate.name] = \
                metrics[TrueVsPredictedNumExamplesOutputColumnNames.Predicted_num_pos_examples.name] / metrics[
                    NumExamplesOutputColumnNames.Total_num_examples.name]

            metrics[AdditionalMetricsOutputColumnNames.F1_score_pos.name] = f1_score(y_true, predictions)
            metrics[AdditionalMetricsOutputColumnNames.Precision_pos.name] = precision_score(y_true, predictions)
            metrics[AdditionalMetricsOutputColumnNames.Recall_pos.name] = recall_score(y_true, predictions)

            y_true_neg = utils.get_inverse_binary_values(y_true)
            predictions_neg = utils.get_inverse_binary_values(predictions)
            metrics[AdditionalMetricsOutputColumnNames.F1_score_neg.name] = f1_score(y_true_neg, predictions_neg)
            metrics[AdditionalMetricsOutputColumnNames.Precision_neg.name] = precision_score(y_true_neg,
                                                                                             predictions_neg)
            metrics[AdditionalMetricsOutputColumnNames.Recall_neg.name] = recall_score(y_true_neg, predictions_neg)

            metrics[AdditionalMetricsOutputColumnNames.AUPRC_pos.name] = average_precision_score(y_true,
                                                                                                 positive_class_probabilities)


            print("AUC: {}".format(metrics[MetricsOutputColumnNames.AUC.name]))
            print("AUPRC: {}".format(metrics[AdditionalMetricsOutputColumnNames.AUPRC_pos.name]))

        else:
            y_binarized = label_binarize(le.inverse_transform(y_true), le.classes_)
            # metrics[data_config.CORREL_COLUMN_NAME], metrics[data_config.CORREL_P_VALUE_COLUMN_NAME] = pearsonr(y_binarized, probabilities)

            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            # for c in le.classes_:
            #     i = le.transform([c])[0]
            #     fpr[i], tpr[i], _ = roc_curve(y_binarized[:, i], probabilities[:, i])
            #     roc_auc[i] = auc(fpr[i], tpr[i])
            #     metrics[MetricsOutputColumnNames.AUC.name + "_" + c] = roc_auc[i]

            # metrics[MetricsOutputColumnNames.AUC.name] = roc_auc_score(y_binarized, probabilities, average="weighted")
            # metrics[CorrelationOutputColumnNames.Pearson_correlation.name], metrics[
            #     CorrelationOutputColumnNames.Pearson_corr_p_value.name] = "NA", "NA"  # pearsonr(y_binarized, probabilities) # TODO

        return metrics

    @classmethod
    def get_metrics_from_all_predictions(cls, all_predictions):
        all_results = pd.DataFrame()
        groups = all_predictions.groupby(
            [RunInfoOutputColumnNames.Label.name, RunInfoOutputColumnNames.Feature_source.name,
             RunInfoOutputColumnNames.Model.name])
        for name, group in groups:
            label = name[0]
            print(label, "   " , name[1])
            _, _, le = Dataset().get(label)  # TODO fix
            probabilities = np.asarray([group[c + PROBABILITY_COLUMN_NAME_SUFFIX] for c in le.classes_]).T
            model_run_instance = ModelInfoOnlyInstance(model_name=name[2], feature_source_name=name[1], label=name[0])
            metrics = ClassificationResultMetrics(model_run_instance).get_metrics(
                le.transform(group.True_value.values.astype('str')),
                probabilities)
            all_results = all_results.append(metrics, ignore_index=True)
            print()
        return all_results

    @classmethod
    def get_output_column_names(cls, df, include_groupby=False):

        # lists out all AUC columns in case of more than binary prediction
        auc_columns = [col for col in df.columns if MetricsOutputColumnNames.AUC.name == col]

        columns_to_print = RunInfoOutputColumnNames.list_member_names() + \
                           auc_columns + \
                           [MetricsOutputColumnNames.Accuracy.name,
                            NumExamplesOutputColumnNames.Total_num_examples.name,
                            NumExamplesOutputColumnNames.Num_train_examples.name,
                            NumExamplesOutputColumnNames.Num_test_examples.name]

        # if binary prediction
        if len(auc_columns) == 1:
            columns_to_print += TrueVsPredictedNumExamplesOutputColumnNames.list_member_names()
            columns_to_print += AdditionalMetricsOutputColumnNames.list_member_names()

        if include_groupby:
            group_by_columns = [FoldGroupByOutputColumnNames.Train_Group_By_Value.name, FoldGroupByOutputColumnNames.Test_Group_By_Value.name]
            columns_to_print += group_by_columns

        return columns_to_print

    @classmethod
    def get_positive_probabilities(cls, probabilities):
        return np.array(probabilities)[:, 1]


########################################################################################################################


# Find metrics for model / label such as AUC, Accuracy, etc...
class MulticlassResultMetrics(ResultMetrics):

    def get_predictions_from_probabilities(self, probabilities):
        return np.argmax(probabilities, axis=1) if not isinstance(self.model_run_instance.label, list) else np.argmax(
            probabilities, axis=2)

    # Get metrics (like AUROC, etc... ) based on predictions and probability scores of predictions
    def get_metrics_per_run_instance(self, y_true, probabilities):

        # In the case of multi-class prediction (the y label is an array of labels)
        if isinstance(self.model_run_instance.label, list):
            metrics = []
            for idx, lbl in enumerate(self.model_run_instance.label):
                individual_metrics_for_label_class = type(self)(self.model_run_instance.get_new_instance_with_label(lbl))
                individual_metrics_for_label = individual_metrics_for_label_class.get_metrics(y_true[:, idx], probabilities[:, idx], le)
                metrics.append(individual_metrics_for_label)
        else:
            metrics = self.get_metrics(y_true, probabilities)

        print("Finished running: ", self.model_run_instance)
        return metrics

    def get_metrics(self, y_true, probabilities):
        predictions = probabilities

        metrics = defaultdict()

        metrics[RunInfoOutputColumnNames.Model.name] = self.model_run_instance.model_name
        metrics[RunInfoOutputColumnNames.Label.name] = self.model_run_instance.label
        metrics[RunInfoOutputColumnNames.Feature_source.name] = self.model_run_instance.feature_source_name
        metrics[MetricsOutputColumnNames.Accuracy.name] = accuracy_score(y_true, predictions)

        metrics[NumExamplesOutputColumnNames.Total_num_examples.name] = len(y_true)

        le = Dataset().get_saved_label_encoder(self.model_run_instance.label)

        y_true_binarized = label_binarize(le.inverse_transform(y_true), le.classes_)
        y_pred_binarized = label_binarize(le.inverse_transform(predictions), le.classes_)

        metrics[AdditionalMetricsOutputColumnNames.F1_score.name] = f1_score(y_true, predictions, average='weighted')
        metrics[AdditionalMetricsOutputColumnNames.Kappa.name] = cohen_kappa_score(y_true, predictions, weights='linear')
        metrics[AdditionalMetricsOutputColumnNames.AUROC.name] = roc_auc_score(y_true_binarized, y_pred_binarized, average='weighted')

        return metrics

    @classmethod
    def get_metrics_from_all_predictions(cls, all_predictions):
        all_results = pd.DataFrame()
        groups = all_predictions.groupby(
            [RunInfoOutputColumnNames.Label.name, RunInfoOutputColumnNames.Feature_source.name,
             RunInfoOutputColumnNames.Model.name])
        for name, group in groups:
            label = name[0]
            print(label, "   " , name[1])
            _, _, le = Dataset().get(label)  # TODO fix
            probabilities = np.asarray([group[c + PROBABILITY_COLUMN_NAME_SUFFIX] for c in le.classes_]).T
            model_run_instance = ModelInfoOnlyInstance(model_name=name[2], feature_source_name=name[1], label=name[0])
            metrics = ClassificationResultMetrics(model_run_instance).get_metrics(
                le.transform(group.True_value.values.astype('str')),
                probabilities)
            all_results = all_results.append(metrics, ignore_index=True)
            print()
        return all_results

    @classmethod
    def get_output_column_names(cls, df):

        # lists out all AUC columns in case of more than binary prediction
        auc_columns = [col for col in df.columns if MetricsOutputColumnNames.AUC.name == col]

        columns_to_print = RunInfoOutputColumnNames.list_member_names() + \
                           auc_columns + \
                           [MetricsOutputColumnNames.Accuracy.name,
                            AdditionalMetricsOutputColumnNames.F1_score.name,
                            AdditionalMetricsOutputColumnNames.Kappa.name,
                            AdditionalMetricsOutputColumnNames.AUROC.name,
                            NumExamplesOutputColumnNames.Total_num_examples.name]

        # # if binary prediction
        # if len(auc_columns) == 1:
        #     columns_to_print += TrueVsPredictedNumExamplesOutputColumnNames.list_member_names()
        #     columns_to_print += AdditionalMetricsOutputColumnNames.list_member_names()

        return columns_to_print

    @classmethod
    def get_positive_probabilities(cls, probabilities):
        return probabilities

