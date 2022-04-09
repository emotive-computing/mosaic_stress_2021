from sklearn.preprocessing import StandardScaler

import numpy as np

class StandardScaler3D(StandardScaler):

    def fit(self, X, y=None, **fit_params):
       num_instances, num_time_steps, num_features = X.shape
       train_data = np.reshape(X, newshape=(-1, num_features))
       return super(StandardScaler3D, self).fit(train_data)
       # train_data = np.reshape(train_data, newshape=(num_instances, num_time_steps, num_features))
       # return train_data

    def transform(self, X, y='deprecated', copy=None):
       num_instances, num_time_steps, num_features = X.shape
       train_data = np.reshape(X, newshape=(-1, num_features))
       train_data = super(StandardScaler3D, self).transform(train_data)
       train_data = np.reshape(train_data, newshape=(num_instances, num_time_steps, num_features))
       return train_data




