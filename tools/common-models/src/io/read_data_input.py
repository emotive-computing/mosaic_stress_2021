import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.common.decorators import extend
from src.configuration.settings_template import Settings
from src.configuration.settings_module_loader import SettingsModuleLoader
from src.common.singleton import Singleton

NO = 'no'
YES = 'yes'
NEGATIVE = 'negative'
POSITIVE = 'positive'
ZERO = "0"
ONE = "1"

class LabelEncoder(LabelEncoder):
    @property
    def is_binary_prediction(self):
        return self.n_classes == 2

    @property
    def n_classes(self):
        return len(self.classes_)


class Dataset(metaclass=Singleton):

    def __init__(self):
        self._data_df = self._read_data_from_file() # Only ever need to read in data once
        self._label_encoders = {}


    def get(self, label=None, feature_source=None, custom_column_filters=None):
        df = self._data_df

        if custom_column_filters:
            if label is not None and label in custom_column_filters:
                for k, v in custom_column_filters[label].items():
                    try:
                        df = df.loc[df[k].isin(v)]
                    except:
                        raise RuntimeError(
                            "Column {} could not be found in dataset provided, make sure this column exists in input file".format(
                                label))


        # Do we ever just want to drop rows instead? Make this configurable?
        # input_df = input_df.dropna(subset=get_x_columns(label, feature_source))

        # Fill text column with placeholder string - scikit doesn't handle missing values
        if feature_source and (feature_source.includes_language_features or feature_source.includes_word_embeddings):
            df[feature_source.language_column_name].fillna("NA", inplace=True)
        else:
            df = df.fillna(0)

        if label is not None:
            ys = self._get_ys_from_input_df(df, label)
            ys, le = self._get_label_encoder_and_transform_ys(label, ys)
        else:
            ys = None
            le = None
        return df, ys, le

    def apply_column_filters(self, X, y, label):
        if Settings.COLUMNS.COLUMN_FILTERS:
            new_X, new_y = self._apply_column_filters_to_data(X, y, Settings.COLUMNS.COLUMN_FILTERS, label)
            return new_X, new_y
        else:
            return X, y

    def _read_data_from_file(self):
        file_path = Settings.IO.DATA_INPUT_FILE
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.path.dirname(os.path.abspath(SettingsModuleLoader.settings_file)), os.path.normpath(file_path))

        if file_path.endswith(".csv"):
            return pd.read_csv(file_path, low_memory=False)
        elif Settings.IO.DATA_INPUT_FILE.endswith(".csv.gz"):
            return pd.read_csv(file_path, low_memory=False, compression='gzip')
        elif Settings.IO.DATA_INPUT_FILE.endswith(".xlsx"):
            return pd.read_excel(file_path, dtype=str).replace('nan', 'blank')
        else:
            raise OSError("Data input file must be of type .csv or .xlsx")

    def _apply_column_filters_to_data(self, df, y, filter_schema, label):
        if filter_schema and label in filter_schema:
            for k, v in filter_schema[label].items():
                try:
                    mask = df[k].isin(v)
                    df = df.loc[mask]
                    y = y[mask]
                    #df = df.loc[df[k].isin(v)]
                except:
                    raise RuntimeError(
                        "Column {} could not be found in dataset provided, make sure this column exists in input file".format(
                            label))

        return df, y

    def _get_ys_from_input_df(self, input_df, label):
        if hasattr(label, '__iter__') and not isinstance(label, str):
            return np.asarray([input_df[i].values for i in label]).T
        else:
            return np.asarray(input_df[label].values)

    def _get_additional_column_label_encoders(self, input_df, label):
        ys = self._get_ys_from_input_df(input_df, label)
        ys, le = self._get_label_encoder_and_transform_ys(label, ys)
        return ys

    def _get_label_encoder_and_transform_ys(self, label, ys):

        if Settings.PREDICTION is None or Settings.PREDICTION.is_regression():
            return ys, None
        else:
            ys = self._pre_modify_ys(ys, label)
            le = self.get_saved_label_encoder(label) or self.get_label_encoder(ys, label)
            return le.transform(ys), le



    def _pre_modify_ys(self, ys, label=None):
        if label and \
                Settings.COLUMNS.LABEL_DECLARATIONS and \
                label in Settings.COLUMNS.LABEL_DECLARATIONS and \
                Settings.COLUMNS.MAKE_ALL_LABELS_BINARY:
            one_label = Settings.COLUMNS.LABEL_DECLARATIONS[label][1].lower()
            ys = ["0" if str(i).lower() != one_label else str(one_label) for i in list(ys)]
        else:
            ys = [str(i) for i in list(ys)]

        return ys

    def get_saved_label_encoder(self, label, initialize_if_missing=False):
        if label in self._label_encoders:
            return self._label_encoders[label]
        elif initialize_if_missing:
            _, _, le = self.get(label)
            return le
        else:
            return None

    def get_label_encoder(self, ys, label=None):

        le = LabelEncoder()

        # Ensures that the labels are given assignments to numerical values in the preferred order
        # Important for binary class predictions where it is desired a certain label be given 0 for negative class and 1 for positive class
        if label and Settings.COLUMNS.LABEL_DECLARATIONS and label in Settings.COLUMNS.LABEL_DECLARATIONS:
            ysl = Settings.COLUMNS.LABEL_DECLARATIONS[label] + list(ys)

        # make sure no = 0 and yes = 1 in some cases
        elif NO in ys and YES in ys:
            ysl = [NO, YES] + list(ys)

        elif ZERO in ys and ONE in ys:
            ysl = [ZERO, ONE] + list(ys)

        else:
            ysl = list(ys)

        le.fit([str(i).lower() for i in ysl])

        self._label_encoders[label] = le

        return le

    @classmethod
    def get_one_vs_rest_dataset(cls, kls, y):
      #  not_cls_label = "not_{}".format(cls)
        y = [kls if i == 1 else 0 for i in y]
        le = LabelEncoder()
        le.fit(y)  # make sure no's get to be 0
        return le.transform(y), le

