import json
import pprint
import numpy as np
import pandas as pd

class DataContainerEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, DataContainer):
            return obj.__dict__
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.to_list()
        else:
            try:
                ret_val = super().default(obj)
            except:
                if hasattr(obj, '__dict__'):
                    ret_val = obj.__dict__
                elif hasattr(obj, '__list__'):
                    ret_val = obj.__list__
                elif hasattr(obj, '__str__'):
                    ret_val = obj.__str__
                else:
                    ret_val = 'UNABLE TO SERIALIZE'
                
            return ret_val

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

    def to_json(self):
        #return json.dumps(self._keystore, sort_keys=True, indent=3, separators=(',', ': '))
        return json.dumps(self, cls=DataContainerEncoder, sort_keys=True, indent=3, separators=(',', ': '))
