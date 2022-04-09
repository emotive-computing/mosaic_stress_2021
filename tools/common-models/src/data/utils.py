import pandas as pd
import numpy as np
from src.io.read_data_input import Dataset
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

def get_k_folds(X, k):
    kf = KFold(n_splits=k)
    print("Performing {}-fold data split".format(k))
    return kf.split(X)

def get_group_k_folds(X, k, group_by):
    group_le = LabelEncoder()
    groups = group_le.fit_transform(X[group_by].values)
    print("Performing grouped {}-fold data split using column: {}, found {} groups".format(k, group_by, len(group_le.classes_)))
    kf = GroupKFold(n_splits=k)
    return kf.split(X, groups=groups)

def get_stratified_k_folds(X, y, k, split, do_shuffle=False, rand_seed=42):
    """Performs stratified k-fold splitting of the data X based on the distribution of values in y.

    If split is an integer, then it determines the number of percentile bins to use during
    stratification.  If split is a list, then values determine the bin edges.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=do_shuffle, random_state=rand_seed)
    if isinstance(split, int):
        print("Performing stratified {}-fold data split with {} percentile bins".format(k, split))
        # Note: argsort produces the ranks, ensuring that there are unique bin thresholds for qcut
        y = pd.qcut(y.argsort(kind='stable'), q=split, labels=range(split)).tolist()
    else: # split is an list
        print("Performing stratified {}-fold data split with bin edges: {}".format(k, split))
        y = np.digitize(y, bins=split)
    return skf.split(X, y)

def get_stratified_group_k_folds(X, y, k, group_by, split, do_shuffle=False, rand_seed=42):
    """Performs stratified grouped k-fold splitting of the data X based on the distribution of values in y and groupping by the group_by column.

    If split is an integer, then it determines the number of percentile bins to use during
    stratification.  If split is a list, then values determine the bin edges.
    """
    
    # Summarize each group by its mean prior to stratified sampling
    orig_X = X.copy()
    group_le = LabelEncoder()
    groups = group_le.fit_transform(X[group_by].values)
    X = X.groupby(by=groups, group_keys=True).mean()
    y = pd.Series(y).groupby(by=groups, group_keys=True).mean().values
        
    skf = StratifiedKFold(n_splits=k, shuffle=do_shuffle, random_state=rand_seed)
    if isinstance(split, int):
        print("Performing stratified grouped {}-fold data split using group column {} and {} percentile bins".format(k, group_by, split))
        y = pd.qcut(y.argsort(kind='stable'), q=split, labels=range(split)).tolist()
    else: # split is an list
        print("Performing stratified grouped {}-fold data split using group column {} and bin edges: {}".format(k, group_by, split))
        y = np.digitize(y, bins=split)
    train_test_index_list = skf.split(X, y)

    # Get the group IDs from the index (stored by groupby()) and recover the original train/test index set per fold
    orig_train_test_index_list = []
    for train_idxs, test_idxs in train_test_index_list:
        train_group_ids = X.index[train_idxs]
        test_group_ids = X.index[test_idxs]
        orig_train_idxs = np.where(np.sum([groups == x for x in train_group_ids], axis=0) > 0)[0]
        orig_test_idxs = np.where(np.sum([groups == x for x in test_group_ids], axis=0) > 0)[0]

        orig_train_test_index_list.append((orig_train_idxs, orig_test_idxs))
    return orig_train_test_index_list
