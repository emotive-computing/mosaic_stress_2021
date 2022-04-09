import os
import sys
from collections.abc import Iterable
from .stage_base import StageBase
from ..utils.utils import get_any_scoring_func

class EvaluationStage(StageBase):
    def __init__(self, eval_idx=None):
        super().__init__()
        self._eval_idx = eval_idx
        self._eval_context = None
        return

    def setEvaluationContext(self, eval_context):
        self._eval_context = eval_context
        return

    def _validate(self, dc):
        if self._eval_idx is not None:
            if min(self._eval_idx) < 0 or max(self._eval_idx) >= len(dc.get_item('data').index):
                raise ValueError("Test indices exceed bounds of the data size in {}".format(type(self).__name__))
        if not isinstance(self._eval_context, SupervisedEvaluationContext):
            raise ValueError("Evaluation context must be a subclass of {}".format(type(SupervisedEvaluationContext).__name__))
        if not isinstance(self._eval_context.y_label, str) and len(self._eval_context.y_label) > 1:
            raise ValueError("Multi-target evaluation is not supported yet. Ensure only one label is provided to {}".format(type(self).__name__))

        self._eval_context.validate()
        return

    def _execute(self, dc):
        self.logInfo("Running Model Evaluation Stage")
        data = dc.get_item('data')

        if self._eval_idx is None:
            try:
                cv_splits = dc.get_item('cv_splits')
                self._eval_idx = cv_splits[1]
            except:
                self._logInfo("Evaluation stage will be applied to entire data set.  Are you sure this is right?")
                self._eval_idx = list(range(data.shape[0]))

        eval_data = data.iloc[self._eval_idx,:]
        preds = dc.get_item('predictions')
        eval_func_names = self._eval_context.get_eval_func_names()
        eval_results = {}

        y_label = self._eval_context.y_label # TODO: support multi-target labels?

        for i in range(len(self._eval_context.eval_funcs)):
            eval_func = self._eval_context.eval_funcs[i]
            eval_labels = eval_data[y_label].values.flatten()
            eval_value = eval_func(eval_labels, preds)
            eval_results[eval_func_names[i]] = eval_value

        dc.set_item('evaluation_results', eval_results)
        return dc


class EvaluationContext():
    def __init__(self):
        self._eval_funcs = None
        return

    def get_eval_func_names(self):
        if self._eval_funcs is None:
            raise RuntimeError("get_eval_func_names() called before set_eval_funcs() in {}".format(type(self).__name__))
        ret_names = []
        for x in self._eval_funcs:
            if hasattr(x, '__name__'):
                ret_names.append(x.__name__)
            else:
                ret_names.append(type(x).__name__)
        return ret_names

    def get_eval_funcs(self):
        return self._eval_funcs

    def set_eval_funcs(self, eval_funcs):
        if callable(eval_funcs):
            self._eval_funcs = [eval_funcs]
        elif isinstance(eval_funcs, str):
            self._eval_funcs = [get_any_scoring_func(eval_funcs)]
        else:
            try:
                new_eval_funcs = []
                for eval_func in eval_funcs:
                    if callable(eval_func):
                        new_eval_funcs.append(eval_func)
                    elif isinstance(eval_func, str):
                        new_eval_funcs.append(get_any_scoring_func(eval_func))
                    else:
                        raise ValueError('eval_funcs argument must be callable type or list of callables')
                self._eval_funcs = new_eval_funcs
            except:
                raise ValueError('eval_funcs argument must be callable type or list of callables')
        return

    def validate(self):
        if self._eval_funcs is None:
            raise RuntimeError("set_eval_func() must be called before using this context")
        return

    eval_funcs = property(get_eval_funcs, set_eval_funcs)


class SupervisedEvaluationContext(EvaluationContext):
    def __init__(self):
        super().__init__()
        self._y_label = None
        return

    def get_y_label(self):
        return self._y_label

    def set_y_label(self, labels):
        if isinstance(labels, str) or isinstance(labels, Iterable):
            self._y_label = labels
        else:
            raise ValueError('labels argument must be Iterable or string type')
        return

    def validate(self):
        super().validate()
        if not isinstance(self._y_label, str) and not isinstance(self._y_label, Iterable):
            raise TypeError("y_label must be initialized to Iterable or string type")
        return

    y_label = property(get_y_label, set_y_label)
