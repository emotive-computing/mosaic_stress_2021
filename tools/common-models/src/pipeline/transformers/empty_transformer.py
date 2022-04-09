from sklearn.base import BaseEstimator



class EmptyTransformer(BaseEstimator):

    def __init__(self):
        pass

    def transform(self, X, y=None):
        return X

    def fit(self, X, y=None, **fit_params):
        pass

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, y)

    def get_params(self, deep=True):
       return {}

    def set_params(self, **params):
        return self

