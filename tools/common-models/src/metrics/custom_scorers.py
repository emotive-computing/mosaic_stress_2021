import abc

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer


class CustomScorer(type):

    @abc.abstractmethod
    def score(cls,_, __):
        return None

    def __call__(cls, y_true, y_score):
        return cls.score(y_true, y_score)

    @property
    def _score_func(cls):
        return cls

class PearsonCorrelationScorer(object, metaclass=CustomScorer):

    @classmethod
    def score(cls, y_true, y_score):
        return pearsonr(y_true, y_score)[0]

class SpearmanCorrelationScorer(object, metaclass=CustomScorer):
    @classmethod
    def score(cls, y_true, y_score):
        return spearmanr(y_true, y_score)[0]


class MulticlassROCScorer(object, metaclass=CustomScorer):

    @classmethod
    def score(cls, y_true, y_score):
        lb = LabelBinarizer()
        lb.fit(y_true)

        truth = lb.transform(y_true)
        pred = lb.transform(y_score)

        return roc_auc_score(truth, pred, average='macro')