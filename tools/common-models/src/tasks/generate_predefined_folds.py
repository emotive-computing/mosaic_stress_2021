import numpy as np
import os
from sklearn.utils import shuffle

from src.common import utils
from src.configuration.settings_module_loader import SettingsModuleLoader
from src.configuration.settings_template import Settings
from src.io.read_data_input import Dataset
from src.pipeline.cross_validation.data_folds import DataFolds


def generate_predefined_folds(label):
    results = []

    X, y, le = Dataset().get(label)
    X, y = Dataset().apply_column_filters(X, y, label)
    X, y = shuffle(X, y) #, random_state=Settings.RANDOM_STATE)
    X = X.reset_index()

    kf_splits = DataFolds.get(X, y, generate_folds_file=True)

    utils.ensure_directory_for_file(Settings.IO.USE_PREDEFINED_FOLDS_FILE)
    #file_name = os.path.join(directory, "folds.txt") #.format(model_run_instance.model_name, str(model_run_instance.feature_source),
                                   #   model_run_instance.label)
   # full_file_name = os.path.join(directory, Print.sanitize_for_windows(file_name))
    f = open(Settings.IO.USE_PREDEFINED_FOLDS_FILE, 'w', encoding='utf-8')

    f.write("{},{}\n".format(Settings.IO.USE_PREDEFINED_FOLDS_SPECIFIER, "test_fold"))

    fold_num = 0
    for train_index, test_index in kf_splits:

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        ids = X_test[Settings.IO.USE_PREDEFINED_FOLDS_SPECIFIER].values

        for val in np.unique(ids):

            f.write("{},{}\n".format(val, str(fold_num)))

        fold_num += 1

    f.close()

    return results





def main():
    SettingsModuleLoader.init_settings()

    generate_predefined_folds('Authenticity', Settings.FEATURE_INPUT_SOURCES_TO_RUN[0] )

if __name__ == "__main__":
    main()

