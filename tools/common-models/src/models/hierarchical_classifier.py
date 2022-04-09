from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from src.configuration.settings_template import Settings
from src.io.read_data_input import Dataset
from src.common.singleton import SingletonFactory


class HierarchicalDataset(object):

    @staticmethod
    def get_hierarchical(X, y, model_run_instance, train_index):
        if model_run_instance.label is Settings.COLUMNS.USE_HIERARCHICAL.START_NODE:
            return None

        schema = Settings.COLUMNS.USE_HIERARCHICAL.SCHEMA[model_run_instance.label]
        prev_name = list(schema.keys())[0]
        sel = Settings.COLUMNS.USE_HIERARCHICAL.SCHEMA[model_run_instance.label][prev_name][0].lower()

        _, k_y, le = Dataset().get(prev_name, model_run_instance.feature_source)

        k_y = k_y[train_index]

        inv_y = le.inverse_transform(k_y)
        y_match_idx = [i == sel for idx, i in enumerate(inv_y)]
        return X.loc[y_match_idx], y[y_match_idx]


class HierarchicalPredictionModelCache(object, metaclass=SingletonFactory):


    def __init__(self, label, fold_num):
        self.label = label
        self.fold_num = fold_num
        self._predictions = None

    def __hash__(self):
        return hash(self.label + str(self.fold_num))

    @property
    def predictions(self):
        return self._predictions

    @predictions.setter
    def predictions(self, pred):
        self._predictions = pred

class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
    base_cls = RandomForestClassifier
    base_cls_params = {'n_estimators': 100}

    use_hard_predictions = False
    show_top_level_model_results = False

    def __init__(self, label, fold_num):
        self.label = label
        self.fold_num = fold_num
        self.cv_completed = False

    def fit(self, X, y, **fit_params):
       # print("Fitting model with shape {} for {}, {}".format(X.shape, self.label, self.fold_num))
        m = type(self).base_cls(**type(self).base_cls_params)
        m.fit(X, y, **fit_params)
        self.model = m


    def predict_proba(self, X):

        overall_probs = self.model.predict_proba(X)

        if not type(self).show_top_level_model_results and self.cv_completed:
            curr_label = self.label

            while curr_label != Settings.COLUMNS.USE_HIERARCHICAL.START_NODE:
                schema =  Settings.COLUMNS.USE_HIERARCHICAL.SCHEMA[curr_label]
                prev_name = list(schema.keys())[0] # TODO: fix
                cache = HierarchicalPredictionModelCache(prev_name, self.fold_num)

                sel = Settings.COLUMNS.USE_HIERARCHICAL.SCHEMA[curr_label][prev_name][0].lower()
                prev_le = Dataset().get_saved_label_encoder(prev_name, True)
                needed_selection = prev_le.transform([sel])[0]

                if not type(self).use_hard_predictions:
                    proba = [[1-p[needed_selection], p[needed_selection]] for p in cache.predictions]
                else:
                    proba = [[1-round(p[needed_selection]), round(p[needed_selection])] for p in cache.predictions]

                overall_probs = [[1-((c[1] * n[1]) + (.5*(c[0] * n[1]))), (c[1] * n[1]) + (.5*(c[0] * n[1]))] for c, n in zip(overall_probs, proba)]

                curr_label = prev_name

        self.predictions = overall_probs
        return overall_probs #[[1 - i, i] for i in overall_probs]



    def do_after_fit(self, pred):
        self.cv_completed = True
        HierarchicalPredictionModelCache(self.label, self.fold_num).predictions = pred

    def predict(self, X):
        pass

    def predict_log_proba(self, X):
        pass

    def set_params(self, **params):
        return self

    def get_params(self, deep=True):
        return {}