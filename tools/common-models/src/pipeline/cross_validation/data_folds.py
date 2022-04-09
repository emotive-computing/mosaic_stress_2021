from sklearn.model_selection import GroupKFold, KFold, PredefinedSplit, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from src.configuration.settings_template import Settings


import pandas as pd

class DataFolds:

    @staticmethod
    def get(X, y, generate_folds_file=False):
        if not generate_folds_file and Settings.IO.USE_PREDEFINED_FOLDS_FILE:
            folds = pd.read_csv(Settings.IO.USE_PREDEFINED_FOLDS_FILE)

            df3 = X.merge(folds, on=Settings.IO.USE_PREDEFINED_FOLDS_SPECIFIER)

            test_folds = df3['test_fold'].values

            ps = PredefinedSplit(test_folds)
            splits = ps.split(X, y)
        if Settings.IO.USE_NEW_PREDEFINED_FOLDS_FILE:
            folds = pd.read_csv(Settings.IO.USE_NEW_PREDEFINED_FOLDS_FILE)
            splits = []
            num_folds = int(folds.shape[1]/2)
            for fold_idx in range(1,num_folds+1):
                train_idxs = folds['Fold'+str(fold_idx)+'_train'].dropna().astype(int).values
                test_idxs = folds['Fold'+str(fold_idx)+'_test'].dropna().astype(int).values
                splits.append((train_idxs, test_idxs))

        # If need to group certain instances together to make sure they are always in the same fold
        # For instance, keep together data from same teacher, same team, etc...
        elif Settings.COLUMNS.GROUP_BY_COLUMN:
            if Settings.CROSS_VALIDATION.SHUFFLE:
                X, y = shuffle(X, y, random_state=Settings.RANDOM_STATE)
                X = X.reset_index(drop=True)

            group_le = LabelEncoder()
            groups = group_le.fit_transform(X[Settings.COLUMNS.GROUP_BY_COLUMN].values)
            print("Grouping data in folds by column: {}, found {} groups".format(Settings.COLUMNS.GROUP_BY_COLUMN, len(group_le.classes_)))
            kf = GroupKFold(n_splits=Settings.CROSS_VALIDATION.NUM_TRAIN_TEST_FOLDS)
            splits = kf.split(X, y, groups=groups)
        else:
            # Split data into training / testing folds and iterate over folds
            kf = KFold(n_splits=Settings.CROSS_VALIDATION.NUM_TRAIN_TEST_FOLDS, random_state=Settings.RANDOM_STATE)
            splits = kf.split(X, y)

        return splits
