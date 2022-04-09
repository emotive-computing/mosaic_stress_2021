# Modify this file as needed
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model  import LinearRegression, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.configuration.settings_template import Settings, SettingsEnumOptions
from src.metrics.custom_scorers import SpearmanCorrelationScorer, CustomScorer

# Settings relating to the data file used for input and columns
# ----------------------------------------------------------------------------------------------------------------------
# Path to the data file for input
# DATA_INPUT_FILE = "/home/cat/repos/temp-data/cetd/NSF_combined_oct_18_updated.csv"
from src.pipeline.resampling import DatasetSampler

import numpy as np

Settings.IO.DATA_DIR =  "../2_FeatureCorrectionAndFilter/results"

Settings.IO.DATA_INPUT_FILE = "../2_FeatureCorrectionAndFilter/results/merge_all_time_summary_not_blinded_outlier_treated_08_12_with_baseline_corrected_non_survey_context_feats.csv"

# Folder to output results in, helpful to change if you don't want to overwrite previous results
Settings.IO.RESULTS_OUTPUT_FOLDER = "./results_ncx_feats_cross_subj_shufwtn_base_smp"

# Names of columns in spreadsheet to identify what should be the input data, and what should be the predicted labels
Settings.COLUMNS.IDENTIFIER = "id"
#Settings.COLUMNS.GROUP_BY_COLUMN = "snapshot_id"

#Settings.COLUMNS.Y_LABELS_TO_PREDICT = ['stress.d'] # stress=1 [negative class=0]; stress e [2...5] [positive class=1]
#Settings.COLUMNS.Y_LABELS_TO_PREDICT = ['stress.d_shuffled'] # stress=1 [negative class=0]; stress e [2...5] [positive class=1]
Settings.COLUMNS.Y_LABELS_TO_PREDICT = ['stress.d_shuffledWithinSubject'] # stress=1 [negative class=0]; stress e [2...5] [positive class=1]
#Settings.COLUMNS.Y_LABELS_TO_PREDICT = ['stress.d','stress.d_shuffled', 'stress.d_shuffledWithinSubject'] # stress=1 [negative class=0]; stress e [2...5] [positive class=1]

Settings.FEATURE_INPUT_SOURCES_TO_RUN = [
      SettingsEnumOptions.RegularFeatureInput.with_regular_features([
"office_at_home",
"saw_work_beacon",
"saw_home_beacon",
"saw_home_beacon_am",
"saw_work_beacon_am",
"last_home_beacon_am",
"first_work_beacon_am",
"estimated_commute_am",
"saw_home_beacon_pm",
"saw_work_beacon_pm",
"last_work_beacon_pm",
"first_home_beacon_pm",
"first_work_beacon",
"last_work_beacon",
"estimated_commute_pm",
"time_at_work",
"minutes_at_desk",
"number_desk_sessions",
"mean_desk_session_duration",
"median_desk_session_duration",
"percent_at_desk",
"percent_at_work",
"num_5min_breaks",
"num_15min_breaks",
"num_30min_breaks",
"total_dist_traveled",
"ave_dist_from_home",
"max_dist_from_home",
"ave_hrv_rmssd",
"min_hrv_rmssd",
"max_hrv_rmssd",
"median_hrv_rmssd",
"sd_hrv_rmssd",
"ave_hrv_first_hour_work_rmssd",
"ave_hrv_last_hour_work_rmssd",
"ave_hrv_15min_before_work_rmssd",
"ave_hrv_work_rmssd",
"min_hrv_work_rmssd",
"max_hrv_work_rmssd",
"ave_hrv_not_work_rmssd",
"min_hrv_not_work_rmssd",
"max_hrv_not_work_rmssd",
"ratio_sdann_work_to_not_rmssd",
"ave_hrv_8_to_6_rmssd",
"min_hrv_8_to_6_rmssd",
"max_hrv_8_to_6_rmssd",
"ave_hrv_not_8_to_6_rmssd",
"min_hrv_not_8_to_6_rmssd",
"max_hrv_not_8_to_6_rmssd",
"first_work_sighting_sdnn",
"last_work_sighting_sdnn",
"num_windows_sdnn",
"ave_lockedp_sdnn",
"ave_hrv_sdnn",
"min_hrv_sdnn",
"max_hrv_sdnn",
"median_hrv_sdnn",
"sd_hrv_sdnn",
"sdann_sdnn",
"num_first_windows_sdnn",
"ave_hrv_first_hour_work_sdnn",
"num_last_windows_sdnn",
"ave_hrv_last_hour_work_sdnn",
"num_before_windows_sdnn",
"ave_hrv_15min_before_work_sdnn",
"num_work_windows_sdnn",
"ave_hrv_work_sdnn",
"min_hrv_work_sdnn",
"max_hrv_work_sdnn",
"sdann_work_sdnn",
"num_not_work_windows_sdnn",
"ave_hrv_not_work_sdnn",
"min_hrv_not_work_sdnn",
"max_hrv_not_work_sdnn",
"sdann_not_work_sdnn",
"diff_sdann_work_to_not_sdnn",
"ratio_sdann_work_to_not_sdnn",
"num_8_to_6_windows_sdnn",
"ave_hrv_8_to_6_sdnn",
"min_hrv_8_to_6_sdnn",
"max_hrv_8_to_6_sdnn",
"sdann_8_to_6_sdnn",
"num_not_8_to_6_windows_sdnn",
"ave_hrv_not_8_to_6_sdnn",
"min_hrv_not_8_to_6_sdnn",
"max_hrv_not_8_to_6_sdnn",
"sdann_not_8_to_6_sdnn",
"diff_sdann_8_to_6_to_not_sdnn",
"ratio_sdann_8_to_6_to_not_sdnn",
"ave_hrv_baseline_rmssd",
"min_hrv_baseline_rmssd",
"max_hrv_baseline_rmssd",
"median_hrv_baseline_rmssd",
"sd_hrv_baseline_rmssd",
"ave_hrv_first_hour_work_baseline_rmssd",
"ave_hrv_last_hour_work_baseline_rmssd",
"ave_hrv_15min_before_work_baseline_rmssd",
"ave_hrv_work_baseline_rmssd",
"min_hrv_work_baseline_rmssd",
"max_hrv_work_baseline_rmssd",
"ave_hrv_not_work_baseline_rmssd",
"min_hrv_not_work_baseline_rmssd",
"max_hrv_not_work_baseline_rmssd",
"ratio_sdann_work_to_not_baseline_rmssd",
"ave_hrv_8_to_6_baseline_rmssd",
"min_hrv_8_to_6_baseline_rmssd",
"max_hrv_8_to_6_baseline_rmssd",
"ave_hrv_not_8_to_6_baseline_rmssd",
"min_hrv_not_8_to_6_baseline_rmssd",
"max_hrv_not_8_to_6_baseline_rmssd",
"ave_hrv_baseline_sdnn",
"min_hrv_baseline_sdnn",
"max_hrv_baseline_sdnn",
"median_hrv_baseline_sdnn",
"sd_hrv_baseline_sdnn",
"ave_hrv_first_hour_work_baseline_sdnn",
"ave_hrv_last_hour_work_baseline_sdnn",
"ave_hrv_15min_before_work_baseline_sdnn",
"ave_hrv_work_baseline_sdnn",
"min_hrv_work_baseline_sdnn",
"max_hrv_work_baseline_sdnn",
"ave_hrv_not_work_baseline_sdnn",
"min_hrv_not_work_baseline_sdnn",
"max_hrv_not_work_baseline_sdnn",
"ratio_sdann_work_to_not_baseline_sdnn",
"ave_hrv_8_to_6_baseline_sdnn",
"min_hrv_8_to_6_baseline_sdnn",
"max_hrv_8_to_6_baseline_sdnn",
"ave_hrv_not_8_to_6_baseline_sdnn",
"min_hrv_not_8_to_6_baseline_sdnn",
"max_hrv_not_8_to_6_baseline_sdnn",
"first_interaction_0_outliers_treated",
"last_interaction_0_outliers_treated",
"unique_participants_0_outliers_treated",
"num_interactions_0_outliers_treated",
"max_duration_0_outliers_treated",
"median_duration_0_outliers_treated",
"mean_duration_0_outliers_treated",
"percent_alone_0_outliers_treated",
"percent_one_0_outliers_treated",
"percent_one_or_more_0_outliers_treated",
"percent_two_or_more_0_outliers_treated",
"percent_three_or_more_0_outliers_treated",
"first_interaction_2_outliers_treated",
"last_interaction_2_outliers_treated",
"unique_participants_2_outliers_treated",
"num_interactions_2_outliers_treated",
"max_duration_2_outliers_treated",
"median_duration_2_outliers_treated",
"mean_duration_2_outliers_treated",
"percent_alone_2_outliers_treated",
"percent_one_2_outliers_treated",
"percent_one_or_more_2_outliers_treated",
"percent_two_or_more_2_outliers_treated",
"percent_three_or_more_2_outliers_treated",
"first_interaction_3_outliers_treated",
"last_interaction_3_outliers_treated",
"unique_participants_3_outliers_treated",
"num_interactions_3_outliers_treated",
"max_duration_3_outliers_treated",
"median_duration_3_outliers_treated",
"mean_duration_3_outliers_treated",
"percent_alone_3_outliers_treated",
"percent_one_3_outliers_treated",
"percent_one_or_more_3_outliers_treated",
"percent_two_or_more_3_outliers_treated",
"percent_three_or_more_3_outliers_treated",
"act_still_ep_2",
"act_still_ep_3",
"act_still_ep_0",
"act_still_ep_1",
"act_still_ep_4",
"call_in_num_ep_0",
"call_in_num_ep_1",
"call_in_num_ep_2",
"call_in_num_ep_3",
"call_in_num_ep_4",
"act_unknown_ep_4",
"act_on_foot_ep_0",
"act_on_foot_ep_1",
"act_on_foot_ep_2",
"act_on_foot_ep_3",
"quality_gps_on",
"quality_activity",
"act_in_vehicle_ep_4",
"locdp_median",
"act_in_vehicle_ep_1",
"call_miss_num_ep_4",
"call_miss_num_ep_1",
"call_miss_num_ep_0",
"call_miss_num_ep_3",
"call_miss_num_ep_2",
"call_out_num_ep_4",
"call_out_num_ep_0",
"call_out_num_ep_1",
"call_out_num_ep_2",
"call_out_num_ep_3",
"hrdp_median",
"call_out_duration_ep_3",
"call_out_duration_ep_2",
"call_out_duration_ep_0",
"call_out_duration_ep_4",
"act_on_bike_ep_1",
"act_on_bike_ep_0",
"act_on_bike_ep_3",
"act_on_bike_ep_2",
"act_on_bike_ep_4",
"act_on_foot_ep_4",
"loc_dist_ep_4",
"loc_dist_ep_3",
"loc_dist_ep_2",
"loc_dist_ep_1",
"loc_dist_ep_0",
"hr_active_hrs",
"act_unknown_ep_0",
"act_unknown_ep_1",
"call_in_duration_ep_3",
"call_in_duration_ep_2",
"call_in_duration_ep_1",
"call_in_duration_ep_0",
"act_unknown_ep_2",
"call_in_duration_ep_4",
"act_unknown_ep_3",
"quality_hr",
"unique_loc_count",
"light_std_ep_4",
"light_std_ep_1",
"light_std_ep_0",
"light_std_ep_3",
"light_std_ep_2",
"unlock_duration_ep_4",
"unlock_duration_ep_0",
"unlock_duration_ep_1",
"unlock_duration_ep_2",
"unlock_duration_ep_3",
"unique_act_count",
"loc_visit_num_ep_2",
"loc_visit_num_ep_3",
"loc_visit_num_ep_0",
"loc_visit_num_ep_1",
"loc_visit_num_ep_4",
"act_tilting_ep_4",
"quality_loc",
"unlock_num_ep_1",
"unlock_num_ep_0",
"unlock_num_ep_3",
"unlock_num_ep_2",
"unlock_num_ep_4",
"act_tilting_ep_3",
"act_tilting_ep_2",
"act_tilting_ep_1",
"act_tilting_ep_0",
"light_mean_ep_2",
"light_mean_ep_3",
"light_mean_ep_0",
"light_mean_ep_1",
"act_in_vehicle_ep_3",
"act_in_vehicle_ep_2",
"light_mean_ep_4",
"act_in_vehicle_ep_0",
"quality_light",
"call_out_duration_ep_1",
"step_count",
"step_goal",
#"avg_stress",
#"median_stress",
#"mode_stress",
#"min_stress",
#"max_stress",
#"range",
#"samples",
"misc_time_of_day",
"misc_work_day",
"daily_completion_time",
"response_total_duration",
"filled_in",
"acute_prior_2_hr_sent_time",
"acute_relative_to_day_sent_time",
"acute_relative_to_week_sent_time",
"episodic_relative_to_week",
"episodic_relative_to_month",
"controlling_lifetime_avg_hr",
"resilience_lifetime_avg_hrv",
"controlling_lifetime_avg_hr_healthapi",
"current_hr_sent_time",
"current_hrv_sdnn",
"current_hrv_rmssd",
"acute_prior_2_hr_started_time",
"acute_prior_2_hr_completed_time",
"current_hr_started_time",
"current_hr_completed_time",
"acute_relative_to_day_started_time",
"acute_relative_to_day_completed_time",
"acute_relative_to_week_started_time",
"acute_relative_to_week_completed_time",
"current_hrv_sdnn_start_time",
"current_hrv_rmssd_start_time",
"current_hrv_sdnn_completed_time",
"current_hrv_rmssd_completed_time",
"acute_prior_2_hr_2hr",
"acute_prior_2_hr_4hr",
"current_hr_2hr",
"current_hr_4hr",
"current_hrv_sdnn_2hr",
"current_hrv_sdnn_4hr",
"current_hrv_rmssd_2hr",
"current_hrv_rmssd_4hr",
"acute_relative_to_day_2hr",
"acute_relative_to_day_4hr",
"acute_relative_to_week_2hr",
"acute_relative_to_week_4hr",
"lifetime_avg_rmssd",
"resilience_current_steps",
"resilience_current_exercise_mins",
"saw_work_beacon_90",
"saw_home_beacon_90",
"saw_home_beacon_am_90",
"saw_work_beacon_am_90",
"last_home_beacon_am_90",
"first_work_beacon_am_90",
"estimated_commute_am_90",
"saw_home_beacon_pm_90",
"saw_work_beacon_pm_90",
"last_work_beacon_pm_90",
"first_home_beacon_pm_90",
"first_work_beacon_90",
"last_work_beacon_90",
"estimated_commute_pm_90",
"time_at_work_90",
"minutes_at_desk_90",
"number_desk_sessions_90",
"mean_desk_session_duration_90",
"median_desk_session_duration_90",
"percent_at_desk_90",
"percent_at_work_90",
"num_5min_breaks_90",
"num_15min_breaks_90",
"num_30min_breaks_90",
"usage_last_24hs",
"usage_at_home_last_24hs",
"usage_at_office_last_24hs",
"usage_same_day",
"usage_at_home_same_day",
"usage_at_office_same_day",
"usage_last2hs",
"usage_at_home_last2hs",
"usage_at_office_last2hs",
"time_spent_last_24hs",
"time_spent_using_at_home_last_24hs",
"time_spent_using_at_office_last_24hs",
"time_spent_same_day",
"time_spent_using_at_home_same_day",
"time_spent_using_at_office_same_day",
"time_spent_last_2hs",
"time_spent_using_at_home_last_2hs",
"time_spent_using_at_office_last_2hs",
"bed_time",
"wakeup_time",
"sleep_duration",
"garmin_sleep_duration",
"adjusted_bed_time",
"adjusted_wakeup_time",
"adjusted_sleep_duration",
"rolling_ideal_midsleep",
"rolling_ideal_midsleep_adjusted",
"observed_midsleep_adjusted",
"daily_sleep_debt_adjusted",
"weekend_bedtime_difference",
"weekend_wakeup_difference",
"weekend_bedtime_difference_adjusted",
"weekend_wakeup_difference_adjusted",
"weekend_duration_difference",
"weekend_duration_difference_adjusted",
"daily_sleep_debt",
"observed_midsleep",
"light_sleep_seconds",
"deep_sleep_seconds",
"awake_seconds",
"rem_sleep_seconds",
"sleep_duration_cutoff",
"adjusted_sleep_duration_cutoff",
"garmin_sleep_duration_cuttoff",
"pa_wakeup_time",
"pa_sleep_duration",
"restricted_bed_time",
"restricted_wakeup_time",
"restricted_sleep_duration",
"restricted_adjusted_bed_time",
"restricted_adjusted_wakeup_time",
"restricted_adjusted_sleep_duration",
"rolling_ideal_midsleep_restricted",
"observed_midsleep_restricted",
"daily_sleep_debt_restricted",
"rolling_ideal_midsleep_restricted_adjusted",
"observed_midsleep_restricted_adjusted",
"daily_sleep_debt_restricted_adjusted",
"weekend_bedtime_difference_restricted",
"weekend_wakeup_difference_restricted",
"weekend_bedtime_difference_restricted_adjusted",
"weekend_wakeup_difference_restricted_adjusted",
"weekend_duration_difference_restricted",
"weekend_duration_difference_restricted_adjusted",
"restricted_light_sleep_seconds",
"restricted_deep_sleep_seconds",
"restricted_awake_seconds",
"restricted_rem_sleep_seconds",
"pa_bed_time",
"beacon_bed_time",
"beacon_wakeup_time",
"beacon_sleep_duration",
"restricted_bed_time_imputed_with_mean",
"restricted_wakeup_time_imputed_with_mean",
"restricted_sleep_duration_imputed_with_mean",
"restricted_adjusted_bed_time_imputed_with_mean",
"restricted_adjusted_wakeup_time_imputed_with_mean",
"restricted_adjusted_sleep_duration_imputed_with_mean",
"rolling_ideal_midsleep_restricted_mean_imputation",
"observed_midsleep_restricted_mean_imputation",
"daily_sleep_debt_restricted_mean_imputation",
"rolling_ideal_midsleep_restricted_adjusted_mean_imputation",
"observed_midsleep_restricted_adjusted_mean_imputation",
"daily_sleep_debt_restricted_adjusted_mean_imputation",
"weekend_bedtime_difference_restricted_mean_imputation",
"weekend_wakeup_difference_restricted_mean_imputation",
"weekend_bedtime_difference_restricted_adjusted_mean_imputation",
"weekend_duration_difference_restricted_mean_imputation",
"restricted_bed_time_imputed_with_pa",
"restricted_wakeup_time_imputed_with_pa",
"restricted_sleep_duration_imputed_with_pa",
"restricted_adjusted_bed_time_imputed_with_pa",
"restricted_adjusted_wakeup_time_imputed_with_pa",
"restricted_adjusted_sleep_duration_imputed_with_pa",
"rolling_ideal_midsleep_restricted_pa_imputation",
"observed_midsleep_restricted_pa_imputation",
"daily_sleep_debt_restricted_pa_imputation",
"rolling_ideal_midsleep_restricted_adjusted_pa_imputation",
"observed_midsleep_restricted_adjusted_pa_imputation",
"daily_sleep_debt_restricted_adjusted_pa_imputation",
"sleep_periods",
"restricted_sleep_periods",
"bed_time_imputed_with_mean",
"wakeup_time_imputed_with_mean",
"sleep_duration_imputed_with_mean",
"adjusted_bed_time_imputed_with_mean",
"adjusted_wakeup_time_imputed_with_mean",
"adjusted_sleep_duration_imputed_with_mean",
"rolling_ideal_midsleep_mean_imputation",
"observed_midsleep_imputed_with_mean",
"daily_sleep_debt_imputed_with_mean",
"rolling_ideal_midsleep_adjusted_imputed_with_mean",
"observed_midsleep_adjusted_imputed_with_mean",
"daily_sleep_debt_adjusted_imputed_with_mean",
"weekend_bedtime_difference_imputed_with_mean",
"weekend_wakeup_difference_imputed_with_mean",
"weekend_bedtime_difference_adjusted_imputed_with_mean",
"weekend_wakeup_difference_adjusted_imputed_with_mean",
"weekend_duration_difference_imputed_with_mean",
"weekend_duration_difference_adjusted_imputed_with_mean",
"bed_time_imputed_with_pa",
"wakeup_time_imputed_with_pa",
"sleep_duration_imputed_with_pa",
"adjusted_bed_time_imputed_with_pa",
"adjusted_wakeup_time_imputed_with_pa",
"adjusted_sleep_duration_imputed_with_pa",
"rolling_ideal_midsleep_imputed_with_pa",
"observed_midsleep_imputed_with_pa",
"daily_sleep_debt_imputed_with_pa",
"rolling_ideal_midsleep_adjusted_imputed_with_pa",
"observed_midsleep_adjusted_imputed_with_pa",
"daily_sleep_debt_adjusted_imputed_with_pa",
"weekend_bedtime_difference_imputed_with_pa",
"weekend_wakeup_difference_imputed_with_pa",
"weekend_bedtime_difference_adjusted_imputed_with_pa",
"weekend_wakeup_difference_adjusted_imputed_with_pa",
"weekend_duration_difference_imputed_with_pa",
"weekend_duration_difference_adjusted_imputed_with_pa",
"mean_hrv_sdnn",
"ave_first_hour_work_sdnn",
"ave_last_hour_work_sdnn",
"ave_15min_before_work_sdnn",
"ave_work_total_sdnn",
"ave_8_to_6_sdnn",
"ave_lockedp_rmssd",
"mean_hrv_rmssd",
"ave_first_hour_work_rmssd",
"ave_last_hour_work_rmssd",
"ave_15min_before_work_rmssd",
"ave_work_total_rmssd",
"ave_8_to_6_rmssd",
"location_id",
"day_maxtempf",
"day_mintempf",
"day_totalsnowcm",
"tp_tempf",
"tp_weathercode",
"tp_precipmm",
"tp_humidity",
"tp_windchillf",
"tp_feelslikef",
"tp_windspeedmph",
"tp_windgustmph",
"tp_visibility",
"tp_pressure",
"tp_cloudcover",
"tp_heatidxf",
"tp_weatherdesc",
"daily_type_x",
"daily_type_y",
"agent_platform",
"platform_sdnn",
"day_sunrise_seconds_since_midnight",
"day_sunset_seconds_since_midnight"])
]

Settings.PREDICTION = SettingsEnumOptions.Prediction.REGRESSION

# Settings relating to the models to be run and the parameters to be cross validated
# ----------------------------------------------------------------------------------------------------------------------
# Classes of the models to be run

# Settings.MODELS_TO_RUN = [RandomForestRegressor,GradientBoostingRegressor,LinearRegression] #WindowRnn] # [HierarchicalClassifier |with_base| RandomForestClassifier] # models # models #WindowRnn] #, rnn.Gru] #rnn.Gru] #RandomForestClassifier] #gru12, gru15, gru31, gru44, gru67] #, rnn.Gru, rnn.Lstm]
#Settings.MODELS_TO_RUN = [RandomForestRegressor, ElasticNet]
Settings.MODELS_TO_RUN = [ElasticNet]

#Settings.IO.USE_NEW_PREDEFINED_FOLDS_FILE = "../3_PrecomputedFolds/results/stratifiedOn-stress.d_percentileBins-10_shuffle-True_seed-3748_folds.csv"
#Settings.IO.USE_NEW_PREDEFINED_FOLDS_FILE = "../3_PrecomputedFolds/results/stratifiedOn-stress.d_shuffled_percentileBins-10_shuffle-True_seed-3748_folds.csv"
Settings.IO.USE_NEW_PREDEFINED_FOLDS_FILE = "../3_PrecomputedFolds/results/stratifiedOn-stress.d_shuffledWithinSubject_percentileBins-10_shuffle-True_seed-3748_folds.csv"
Settings.CROSS_VALIDATION.NUM_CV_TRAIN_VAL_FOLDS = 5  # Number of folds to use in nested cross validation to split train data into train / validation
Settings.CROSS_VALIDATION.GROUP_BY_COLUMN = "snapshot_id"
Settings.CROSS_VALIDATION.SHUFFLE = True
Settings.RANDOM_STATE = 3748
Settings.CROSS_VALIDATION.STRATIFIED_SPLIT_CLASSES = [
    "stress.d",
    "stress.d_shuffled",
    "stress.d_shuffledWithinSubject"

]
Settings.CROSS_VALIDATION.STRATIFIED_NUMBER_PERCENTILE_BINS = 10
#Settings.CROSS_VALIDATION.NUM_TRAIN_TEST_FOLDS = 5 # Number of folds to use in outer loop to split all data into train / test

# Settings.CROSS_VALIDATION.SCORING_FUNCTION = 'roc_auc'

# Settings.CROSS_VALIDATION.SCORING_FUNCTION = 'average_precision'
class SmapeScorer(object, metaclass=CustomScorer):
    @classmethod
    def score(cls, y_true, y_score):
        return 100.0/len(y_true) * np.sum(2 * np.abs(y_score - y_true) / (np.abs(y_true) + np.abs(y_score)))
    
#Settings.CROSS_VALIDATION.SCORING_FUNCTION = SpearmanCorrelationScorer
Settings.CROSS_VALIDATION.SCORING_FUNCTION = SmapeScorer

Settings.CROSS_VALIDATION.HYPER_PARAMS.FEATURE_SCALER = [
    # no cross validation done here
    # use scikit's standard scaler
    # the with_mean parameter is False and all others are the defaults
    (StandardScaler, {'with_mean': [True], 'with_std': [True]})
]

Settings.CROSS_VALIDATION.HYPER_PARAMS.IMPUTER = [
    (SimpleImputer, {'strategy': ["mean"]})
]

# Add cross validation parameters for each model to run
Settings.CROSS_VALIDATION.HYPER_PARAMS.MODEL = {
    RandomForestRegressor.__name__: {
        # Name of the key must be the name of the parameter to be passed into the constructor of the model
        #'n_estimators': [100, 500, 800, 1200],
        'n_estimators': [500],
        #'max_depth': [10, 20, 50, 100, None]
        'max_depth': [20]
    },
    ElasticNet.__name__:{
        #'alpha': [0.5, 1.0, 2.0, 5.0, 7.0],
        #'l1_ratio': [0.2, 0.4, 0.6, 0.8]
        #'alpha': [0.001, 0.01, 0.1],
        #'l1_ratio': [0.8, 0.9, 0.95]
        'alpha': [1e-20],
        'l1_ratio': [1e-50, 1e-40, 1e-30, 1e-20, 1e-12]
    }
}

# ----------------------------------------------------------------------------------------------------------------------

# Task settings - what output is desired?
# ---------------------------------------------------------------------------------------------------------------------
Settings.TASKS_TO_RUN = [SettingsEnumOptions.Tasks.MODEL_COMPARISON]
