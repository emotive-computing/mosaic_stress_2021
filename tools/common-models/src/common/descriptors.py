from weakref import WeakValueDictionary

from src.common.error import MissingSettingError, SettingsConfigurationError, AdditionalSettingError, \
    ShowDeprecationWarning

import numpy as np

class TypeDescriptorMeta(type):

    dict = WeakValueDictionary()
    required_descriptors = []
    deprecated = []

    def ensure_all_settings_initialized(cls):
        missing = [desc for desc in cls.required_descriptors if desc not in cls.dict]
        for desc in missing: raise MissingSettingError(desc, getattr(cls, desc).type_)

        deprecated = np.unique([desc for desc in cls.deprecated])
        for depr in deprecated: ShowDeprecationWarning.display(depr, "DEPRECATION WARNING: The setting {} has been deprecated. This can be removed from the settings file.".format(depr))

    def __setattr__(cls, key, value):
        if key in cls.dict:
            td = cls.dict[key]
            if td.is_subclass_of_type(value):
                td.value = value
                cls.dict[key] = td
                return True
            else:
                raise SettingsConfigurationError(key, value, td.type_)
        else:
            raise AdditionalSettingError(key)

    def __set__(cls, label, inst):

        if not inst.is_optional:
            cls.required_descriptors.append(label)

        if inst.deprecated:
            cls.deprecated.append(label)

        cls.dict[label] = inst


    def __get__(self, obj, klass=None):
        if obj is None:
            return self
        "Emulate type_getattro() in Objects/typeobject.c"
        v = object.__getattribute__(self, klass)
        if hasattr(v, '__get__'):
            return v.__get__(None, self)
        return v

    def __getattribute__(self, item):
        attr = type.__getattribute__(self, item)
        if issubclass(type(attr), TypeDescriptor):
            return attr.value
        else:
            return attr

    def __new__(cls, name, bases, attrs):
        # find all descriptors, AUTO-set their labels
        for key, val in attrs.items():
            if issubclass(type(val), TypeDescriptor): #or isinstance(val, TypeDescriptorContainer):
                cls.__set__(cls, key, val)
        return super(TypeDescriptorMeta, cls).__new__(cls, name, bases, attrs)


    def __iter__(self):
        type(self).properties = [(a, b.value) for a, b in self.__dict__.items() if issubclass(type(b), TypeDescriptor)]
        type(self).current = -1
        return self

    def __next__(self):
        type(self).current += 1
        if type(self).current < len(type(self).properties):
            return type(self).properties[type(self).current]
        else:
            raise StopIteration

class TypeDescriptor:
    def __init__(self, type_=None, is_optional=False, default_value=None, deprecated=False):
        self.type_ = type_
        self.is_optional = is_optional
        self.value = default_value or None
        self.deprecated = deprecated

        if self.deprecated:
            self.is_optional = True
            self.value = None

    def is_subclass_of_type(self, value):
        return issubclass(type(value), self.type_)

    def __getattr__(self, item):
        v = object.__getattribute__(self.value, item)
        if hasattr(v, '__get__'):
            return v.__get__(None, self)
        if type(v) == type(self):
            return v.value
        return v

class AllowNoneTypeDescriptor(TypeDescriptor):
    def __init__(self, *args, **kwargs):
        super(AllowNoneTypeDescriptor, self).__init__(*args, **kwargs)

    def is_subclass_of_type(self, value):
        return value == None or issubclass(type(value), self.type_)


class ListTypeDescriptor(TypeDescriptor):
    def __init__(self, *args, **kwargs):
        super(ListTypeDescriptor, self).__init__(*args, **kwargs)

    def is_subclass_of_type(self, value):
        return all([issubclass(type(v), self.type_) for v in value])


    def __iter__(self):
        self.current = -1
        return self

    def __next__(self):
        self.current += 1
        if self.current < len(self.value):
            return self.value[self.current]
        else:
            raise StopIteration

# Wrapper around scikit estimators for pipeline so that if we need to use a Keras classifier or some other different classifier
# these can all be made interchangeable
class SelfTypeWrapper:
    def __init__(self, type_= None):
        self.type_ = type_
