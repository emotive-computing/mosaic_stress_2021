import os
import json
import pprint
import numpy as np
import pandas as pd
import tensorflow as tf
from commonmodels2.models.model import ModelBase
from commonmodels2.log.logger import Logger

class DataContainer():
    def __init__(self):
        self._keystore = {}

    def __str__(self):
        return pprint.pformat(self._keystore, indent=2)
    
    def get_item(self, key):
        if key in self.get_keys():
            return self._keystore[key]
        else:
            raise KeyError("provided key '{}' not present in {}".format(key, type(self).__name__))
    
    def set_item(self, key, obj):
        if type(key) != str:
            raise ValueError("provided key must be string type")
        self._keystore[key] = obj
    
    def get_keys(self):
        return self._keystore.keys()

    def save(self, out_dir_path):
        Logger.getInst().info("Saving data container.  This may take a while...")
        str_dict = self._make_json_compatible(self._keystore, out_dir_path)
        out_json_str = json.dumps(str_dict, sort_keys=True, indent=3, separators=(',', ': '))
        with open(os.path.join(out_dir_path, 'data_info.json'), 'w') as outfile:
            outfile.write(out_json_str)

    # Helper method for _make_json_compatible().  This function allows values
    # stored in a dict object associated with certain keys to be processed differently.
    # For example, all lists will be stored as json-compatible lists inside the output
    # json object after calling _make_json_compatible(), but this function allows certain
    # lists to be stored differently
    @classmethod
    def _special_dict_key_handler(cls, key, value, out_file_path):
        replace_val = value
        if key == 'predictions':
            if not os.path.isdir(os.path.dirname(out_file_path)):
                os.makedirs(os.path.dirname(out_file_path))
            out_file_path += '.csv'
            out_preds = np.array(value)
            num_cols = 1 if out_preds.ndim == 1 else out_preds.shape[1]
            pred_cols = ['Predictions']
            if num_cols > 1:
                pred_cols = ['Prediction_%d'%(i) for i in range(num_cols)]
            preds_df = pd.DataFrame(data=out_preds, columns=pred_cols)
            preds_df.to_csv(out_file_path, index=False, header=True)
            replace_val = out_file_path
        if key == 'cv_splits':
            if not os.path.isdir(os.path.dirname(out_file_path)):
                os.makedirs(os.path.dirname(out_file_path))
            out_file_path += '.csv'
            train_cols = []
            test_cols = []
            for fold_idx in range(len(value)):
                train_cols.append(pd.Series(value[fold_idx][0], name='Fold%d_train'%(fold_idx)))
                test_cols.append(pd.Series(value[fold_idx][1], name='Fold%d_test'%(fold_idx)))
            out_cols = train_cols+test_cols
            cv_splits_df = pd.concat(out_cols, axis=1, keys=[s.name for s in out_cols])
            cv_splits_df.to_csv(out_file_path, index=False, header=True)
            replace_val = out_file_path
        return replace_val

    # Converts the input obj to an object that can be written out as a json file.  Large
    # objects contained within this obj (e.g., dataframes inside a dict) are stored to 
    # the filesystem and replaced with a string path pointing to the object
    @classmethod
    def _make_json_compatible(cls, obj, out_dir_path):
        replace_val = None
        if isinstance(obj, dict) or isinstance(obj, DataContainer):
            dict_obj = obj
            if isinstance(obj, DataContainer):
                dict_obj = obj._keystore

            for key in dict_obj.keys():
                new_out_dir_path = os.path.join(out_dir_path, key)
                replace_val = DataContainer._make_json_compatible(dict_obj[key], new_out_dir_path)
                replace_val = cls._special_dict_key_handler(key, replace_val, new_out_dir_path)
                if replace_val is not None:
                    dict_obj[key] = replace_val
            replace_val = dict_obj
        elif isinstance(obj, ModelBase):
            dir_path = os.path.dirname(out_dir_path)
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
            file_name = os.path.basename(out_dir_path)
            obj.save(dir_path, file_name)
            replace_val = out_dir_path
        elif isinstance(obj, list):
            replace_val = obj # Triggers special handling below for lists
        elif isinstance(obj, tuple):
            replace_val = list(obj)
        elif isinstance(obj, np.ndarray):
            replace_val = obj.tolist()
        elif isinstance(obj, tf.Tensor):
            as_numpy = obj.numpy()
            replace_val = as_numpy.tolist()
        elif isinstance(obj, pd.DataFrame):
            if not os.path.isdir(os.path.dirname(out_dir_path)):
                os.makedirs(os.path.dirname(out_dir_path))
            out_file_path = out_dir_path+'.csv'
            obj.to_csv(out_file_path, index=False, header=True)
            replace_val = out_file_path
        elif isinstance(obj, pd.Series):
            if not os.path.isdir(os.path.dirname(out_dir_path)):
                os.makedirs(os.path.dirname(out_dir_path))
            out_file_path = out_dir_path+'.csv'
            obj.to_csv(out_file_path, index=False, header=True)
            replace_val = out_file_path
        elif isinstance(obj, type):
            replace_val = obj.__name__

        # Make sure lists of objects are converted to json-compatible objects
        if isinstance(replace_val, list):
            for idx in range(len(replace_val)):
                new_out_dir_path = os.path.join(out_dir_path, str(idx))
                new_item = DataContainer._make_json_compatible(replace_val[idx], new_out_dir_path)
                if new_item is not None:
                    replace_val[idx] = new_item

        return replace_val
