import itertools

from src.configuration.settings_template import Settings
from src.pipeline.stages import PipelineStages, FeatureExtractionStages
from src.pipeline.transformers.empty_transformer import EmptyTransformer


class CVParamGrid:

    def __init__(self, model_run_instance):
        self.model_run_instance = model_run_instance
        self._grid = self._build_param_grid_from_settings_hyperparams()


    def _prefix_dict_keys(self, prefix, d):
        def subprefix(k):
            return "__" + k if k != 'kls' else ""

        return {str(prefix) + subprefix(k): v for k, v in d.items()}

    def split_hyper_params(self):
        hyperparams = [(name, value) for (name, value) in Settings.CROSS_VALIDATION.HYPER_PARAMS]
        keys, values = [list(t) for t in zip(*hyperparams)]
        return keys, values

    def _build_param_grid_from_settings_hyperparams(self):

        keys,values = self.split_hyper_params()

        if self.model_run_instance.base_name not in Settings.CROSS_VALIDATION.HYPER_PARAMS.MODEL.keys() :
            print("Warning: No cross validation parameters specified for model {}, using default values".format(self.model_run_instance.base_name))
            cv_params = [self._prefix_dict_keys(PipelineStages.MODEL.name, {})]
        else:
            cv_params = [self._prefix_dict_keys(PipelineStages.MODEL.name,
                                                Settings.CROSS_VALIDATION.HYPER_PARAMS.MODEL[self.model_run_instance.base_name])]

        if FeatureExtractionStages.VECTORIZER.name in keys and self.model_run_instance.feature_source.includes_language_features :
            lang_vect_key = "__".join(
                [PipelineStages.EXTRACT_FEATURES.name, FeatureExtractionStages.LANGUAGE.name,
                 FeatureExtractionStages.VECTORIZER.name])
            l = [self._prefix_dict_keys(lang_vect_key, Settings.CROSS_VALIDATION.HYPER_PARAMS.VECTORIZER)]
            [[p.update(d) for d in cv_params] for p in l]
            cv_params = l

        for key, value in list(filter(lambda x: isinstance(x[1], list), [(name, value) for (name, value) in Settings.CROSS_VALIDATION.HYPER_PARAMS])):
            l = []
            for v in value:
                if not v[0]:
                    class_dict = {'kls': [EmptyTransformer]}
                else:
                    class_dict = v[1]
                    class_dict['kls'] = [v[0]]

                l.append(self._prefix_dict_keys(key + "__kls", class_dict))

            n_l = []
            for p, d in itertools.product(l, cv_params):
                n_l.append({**p, **d})
            cv_params = n_l

        return cv_params

    def _has_no_cv_params_for_model(self, model_run_instance, keys):
        return model_run_instance.base_name not in Settings.CROSS_VALIDATION.HYPER_PARAMS.MODEL.keys() \
               or not len(Settings.CROSS_VALIDATION.HYPER_PARAMS.MODEL[model_run_instance.base_name].keys())

    @property
    def has_only_one_combination_of_parameters(self):
        if len(self._grid) > 1: return False
        return not any([len(v) > 1 for k, v in self._grid[0].items()])

    @property
    def as_list(self):
        return self._grid

    @property
    def as_single_dict(self):
        return {k: v[0] for k, v in self._grid[0].items()}

    def has_no_cv_params_for_model(self, model_run_instance):
        keys, _ = self.split_hyper_params()
        return self._has_no_cv_params_for_model(model_run_instance, keys)

