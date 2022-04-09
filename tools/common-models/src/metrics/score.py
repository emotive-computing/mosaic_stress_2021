from sklearn.metrics import make_scorer, get_scorer, mean_squared_error

from src.configuration.settings_template import Settings
from src.metrics.custom_scorers import CustomScorer

class Score(object):
    @staticmethod
    def get_scoring_function():
        # Todo: Fix for multiclass
        if Settings.CROSS_VALIDATION.SCORING_FUNCTION:
            return Settings.CROSS_VALIDATION.SCORING_FUNCTION
        if Settings.PREDICTION.is_regression():
            return 'neg_mean_squared_error'
        else: #if len(y.shape) == 1: # binary prediction, not predicting multiple classes
             return 'roc_auc'
        # else:
        #     kwargs = {'average': 'weighted'}
        #     weighted_roc_auc_score = partial(roc_auc_score, **kwargs)
        #     return make_scorer(weighted_roc_auc_score)

    @staticmethod
    def _get_predicted_scores(estimator, X):
        if hasattr(estimator, "decision_function"):
            prob = estimator.decision_function(X)
        elif hasattr(estimator, "predict_proba"):
            prob = estimator.predict_proba(X)[:,1]
        else: #Settings.PREDICTION.is_regression() or Settings.PREDICTION.is_multiclass():
            prob = estimator.predict(X)
        return prob

    @staticmethod
    def _is_scoring_func_score_based(scoring_fn):
        if isinstance(scoring_fn, str):
            if scoring_fn == 'top_k_accuracy':
                return True
            elif scoring_fn == 'average_precision':
                return True
            elif scoring_fn == 'neg_brier_score':
                return True
            elif scoring_fn.startswith('roc_auc'):
                return True
            else:
                return False
        else:
            return False # TODO - better handling for custom scoring functions

    @staticmethod
    def evaluate_score(y_true, y_score):
        scoring_fn = Score.get_scoring_function()
        scorer = get_scorer(scoring_fn)._score_func

        try:
            return scorer.__name__, scorer(y_true=y_true, y_score=y_score)
        except:
            return scorer.__name__, scorer(y_true=y_true, y_pred=y_score)

    @staticmethod
    def get_scorer_for_cv():
        scoring_fn = Score.get_scoring_function()

        if type(scoring_fn) is CustomScorer:
            return make_scorer(lambda x, y: scoring_fn.score(x, y))
        elif Score._is_scoring_func_score_based(scoring_fn):
            return lambda estimator, X, y: get_scorer(scoring_fn)._score_func(y, Score._get_predicted_scores(estimator, X))
        else:
            return scoring_fn
