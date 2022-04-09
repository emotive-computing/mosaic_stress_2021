class ModelRunInstance(object):
    def __init__(self,  model_class, label, feature_source):
        self.label = label
        self.feature_source = feature_source
        self.model_class = model_class

    def get_new_instance_with_label(self, label):
        return type(self)(self.model_class, label, self.feature_source)


    def __str__(self):
        return "model_class: {}, label: {}, feature_source: {}".format(self.model_name, self.label, self. feature_source.__name__)

    @property
    def model_name(self):
        return self.model_class.__name__

    @property
    def feature_source_name(self):
        return str(self.feature_source.__name__)

    @property
    def base_name(self):
        return self.model_name.split("-")[0]

    def model_class_is_subclass_of(self, kls):
        return issubclass(self.model_class, kls)


class ModelInfoOnlyInstance(ModelRunInstance):
    def __init__(self,  model_name, label, feature_source_name):
        self._model_name = model_name
        self._feature_source_name = feature_source_name

        super(ModelInfoOnlyInstance, self).__init__(None, label, None)

    def __str__(self):
        return "model_name: {}, label: {}, feature_source: {}".format(self._model_name, self.label, self._feature_source_name)

    @property
    def model_name(self):
        return self._model_name

    @property
    def feature_source_name(self):
        return str(self._feature_source_name)

    def model_class_is_subclass_of(self, kls):
        raise NotImplementedError("Method not valid for this class")