from enum import auto

from src.common.meta import CommonEnumMeta

PROBABILITY_COLUMN_NAME_SUFFIX = "_probability"


class RunInfoOutputColumnNames(CommonEnumMeta):
    Model = auto()
    Label = auto()
    Feature_source = auto()

class MetricsOutputColumnNames(CommonEnumMeta):
    AUC = auto()
    Accuracy = auto()

class CorrelationOutputColumnNames(CommonEnumMeta):
    Pearson_correlation = auto()
    Pearson_corr_p_value = auto()
    Spearman_correlation = auto()
    Spearman_corr_p_value = auto()

class AdditionalMetricsOutputColumnNames(CommonEnumMeta):
    # Additional scores
    Precision_pos = auto()
    Precision_neg = auto()
    Recall_pos = auto()
    Recall_neg = auto()
    F1_score = auto()
    F1_score_pos = auto()
    F1_score_neg = auto()
    Kappa = auto()
    AUROC = auto()
    AUPRC_pos = auto()
    AUPRC_neg = auto()

class NumExamplesOutputColumnNames(CommonEnumMeta):
    Total_num_examples = auto()

    # Useful when printing per fold
    Num_train_examples = auto()
    Num_test_examples = auto()

    @classmethod
    def get_columns_to_show_in_output(cls, df):
        return cls.list_member_names() \
            if cls.Num_train_examples.name in df.columns.values and cls.Num_test_examples.name in df.columns.values \
            else [cls.Total_num_examples.name]

class TrueVsPredictedNumExamplesOutputColumnNames(CommonEnumMeta):
    True_num_pos_examples = auto()
    Predicted_num_pos_examples = auto()

    True_base_rate = auto()
    Predicted_base_rate = auto()

class PredictionsOutputColumnNames(CommonEnumMeta):
    True_value = auto()
    Predicted_value = auto()

# Regression metrics
class RegressionMetricsOutputColumnNames(CommonEnumMeta):
    R2_score = auto()
    RMSE_score = auto()

class FoldGroupByOutputColumnNames(CommonEnumMeta):
    Train_Group_By_Value = auto()
    Test_Group_By_Value = auto()