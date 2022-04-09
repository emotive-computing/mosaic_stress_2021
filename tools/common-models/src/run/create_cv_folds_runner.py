# -*- coding: utf-8 -*-
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.configuration.settings_template import Settings
from src.models.hierarchical_classifier import HierarchicalDataset
from src.models.utils import save_model, load_model
from src.pipeline.cross_validation.cross_validation import CrossValidation
from src.run.create_cv_folds_run_instance import CreateCVFoldsRunInstance
from src.common import utils
from src.io.read_data_input import Dataset
from src.metrics.predictions import PredictionMetrics
from src.metrics.results import ResultMetrics
from src.io.print_output import Print
from src.data.utils import get_k_folds, get_group_k_folds, get_stratified_k_folds, get_stratified_group_k_folds


class CreateCrossValidationFoldsRunner(object):

    @classmethod
    def get_all_create_cv_folds_run_instances(cls):
        n_folds = Settings.CROSS_VALIDATION.NUM_TRAIN_TEST_FOLDS
        do_shuffle = Settings.CROSS_VALIDATION.SHUFFLE
        random_seed = Settings.RANDOM_STATE
        if Settings.CROSS_VALIDATION.STRATIFIED_SPLIT_CLASSES is not None:
            return [CreateCVFoldsRunInstance(n_folds, combo[0], do_shuffle, random_seed) for combo in \
                    itertools.product(Settings.CROSS_VALIDATION.STRATIFIED_SPLIT_CLASSES)]
        else:
            return [CreateCVFoldsRunInstance(n_folds, "", do_shuffle, random_seed)]

    @classmethod
    def run(cls, create_cv_folds_run_instance):
        train_idxs = []
        test_idxs = []
        for train_fold, test_fold in cls._get_results_for_run(create_cv_folds_run_instance):
            train_idxs.append(train_fold)
            test_idxs.append(test_fold)

        # Combine results into a dataframe
        max_train_fold_len = max([len(x) for x in train_idxs])
        max_test_fold_len = max([len(x) for x in test_idxs])
        train_folds_df = pd.DataFrame(data=np.nan*np.zeros((max_train_fold_len, len(train_idxs))), columns=['Fold'+str(i) for i in range(1,len(train_idxs)+1)])
        test_folds_df = pd.DataFrame(data=np.nan*np.zeros((max_test_fold_len, len(test_idxs))), columns=['Fold'+str(i) for i in range(1,len(test_idxs)+1)])
        for fold_idx in range(len(train_idxs)):
            train_folds_df.iloc[0:len(train_idxs[fold_idx]),fold_idx] = train_idxs[fold_idx]
            test_folds_df.iloc[0:len(test_idxs[fold_idx]),fold_idx] = test_idxs[fold_idx]
        return train_folds_df, test_folds_df

    # Splits the data into folds and returns them
    @classmethod
    def _get_results_for_run(cls, create_cv_folds_run_instance):
        if len(create_cv_folds_run_instance.stratified_class_names) > 0:
            X, y, le = Dataset().get(create_cv_folds_run_instance.stratified_class_names)
            X, y = Dataset().apply_column_filters(X, y, create_cv_folds_run_instance.stratified_class_names)

            k = create_cv_folds_run_instance.num_folds
            if Settings.CROSS_VALIDATION.STRATIFIED_BINNING_EDGES is not None:
                split = Settings.CROSS_VALIDATION.STRATIFIED_BINNING_EDGES
            else:
                split = Settings.CROSS_VALIDATION.STRATIFIED_NUMBER_PERCENTILE_BINS
            do_shuffle = create_cv_folds_run_instance.do_shuffle
            rand_seed = create_cv_folds_run_instance.random_seed
            if Settings.COLUMNS.GROUP_BY_COLUMN:
                group_by = Settings.COLUMNS.GROUP_BY_COLUMN
                train_test_index_list = get_stratified_group_k_folds(X, y, k, group_by, split, do_shuffle, rand_seed)
            else:
                train_test_index_list = get_stratified_k_folds(X, y, k, split, do_shuffle, rand_seed)
        else:
            X, y, le = Dataset().get()

            if create_cv_folds_run_instance.do_shuffle:
                X = X.sample(frac=1).reset_index(drop=True)

            k = create_cv_folds_run_instance.num_folds
            if Settings.COLUMNS.GROUP_BY_COLUMN:
                group_by = Settings.COLUMNS.GROUP_BY_COLUMN
                train_test_index_list = get_group_k_folds(X, k, group_by)
            else:
                train_test_index_list = get_k_folds(X, k)
            
        return train_test_index_list
