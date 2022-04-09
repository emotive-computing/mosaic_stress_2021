from sklearn.base import BaseEstimator

# Wraps a clas
class PipelineTransformerWrapper(BaseEstimator):

    def __init__(self, kls=None):
        self.kls = kls

    def transform(self, X, y=None):
        return X

    def fit(self, X, y=None, **fit_params):
        pass

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, y)

    def get_params(self, deep=True):
       return {'kls': self.kls}

    def set_params(self, **params):
        est = params['kls']()
        est_params = {k.replace('kls__', ''): params[k] for k in params if 'kls__' in k }

        est.set_params(**est_params)

        self.__class__ = est.__class__
        self.__dict__.update(est.__dict__)

        return self

