# Modify this file as needed
from src.configuration.settings_template import Settings, SettingsEnumOptions

# Settings relating to the data file used for input and columns
# ----------------------------------------------------------------------------------------------------------------------

# Path to the data file for input
Settings.IO.DATA_INPUT_FILE = "../2_FeatureCorrectionAndFilter/results/merge_all_time_summary_not_blinded_outlier_treated_08_12_with_baseline_corrected_non_survey_context_feats.csv"
Settings.IO.RESULTS_OUTPUT_FOLDER = "./results/"

Settings.COLUMNS.GROUP_BY_COLUMN = "snapshot_id"

# Cross Validation settings
Settings.CROSS_VALIDATION.NUM_TRAIN_TEST_FOLDS = 5  # Number of folds to use in outer loop to split all data into train / test
Settings.CROSS_VALIDATION.SHUFFLE = True
Settings.RANDOM_STATE = 3748
Settings.CROSS_VALIDATION.STRATIFIED_SPLIT_CLASSES = [
    "stress.d",
    "stress.d_shuffled",
    "stress.d_shuffledWithinSubject"

]
#Settings.CROSS_VALIDATION.STRATIFIED_BINNING_EDGES = [1.5,2.5,3.5,4.5]
Settings.CROSS_VALIDATION.STRATIFIED_NUMBER_PERCENTILE_BINS = 10

Settings.SAVE_MODELS = True

# Task settings - what output is desired?
# ---------------------------------------------------------------------------------------------------------------------
Settings.TASKS_TO_RUN = [SettingsEnumOptions.Tasks.CREATE_TRAIN_TEST_FOLDS]
