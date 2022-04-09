
class ClassPropertyMixin(object):
    prop = None

    @classmethod
    def get_prop(cls):
        return cls.prop

    def __init__(self):
        pass


def get_mixin_class_name(mixin, value):
    return mixin.__name__ + str(value)

class UseIncrementalPredictions(type):

    def __new__(cls, *args, **kwargs):
        t = type(get_mixin_class_name(UseIncrementalPredictionsMixin, args[0]), (UseIncrementalPredictionsMixin, ClassPropertyMixin), {})
        t.use_incremental_predictions = args[0]
        t.prop = t.use_incremental_predictions
        return t

class UseIncrementalPredictionsMixin(ClassPropertyMixin):

    use_incremental_predictions = 0

    @classmethod
    def get_prev_class_name(cls):
        curr_mixin_name = get_mixin_class_name(UseIncrementalPredictionsMixin, cls.use_incremental_predictions)
        prev_mixin_name = get_mixin_class_name(UseIncrementalPredictionsMixin, cls.use_incremental_predictions - 1)
        return cls.__name__.replace(curr_mixin_name, prev_mixin_name)


class HierarchicalPredictionMixin(ClassPropertyMixin):
    pass
