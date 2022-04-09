from enum import Enum

from src.common import utils


class MetaUtils:
    @staticmethod
    def get_dynamic_class(cls_type, dynamic_properties=None, use_cls_name_suffix=None, mixins=None):

        if not use_cls_name_suffix:
            use_cls_name_suffix = ""
            if dynamic_properties and len(dynamic_properties):
                use_cls_name_suffix += ",".join([utils.to_camelcase(k) + str(v) for k, v in dynamic_properties.items()])
            if mixins and len(mixins):
                use_cls_name_suffix += ",".join([m.__name__ for m in mixins])

        cls_name = cls_type.__name__ + "-" + use_cls_name_suffix

        base = (cls_type,)
        bases = base if (not mixins or len(mixins) < 1) else base + tuple(mixins)

        cls = type(cls_name, bases, dynamic_properties or {})
        return cls

    @staticmethod
    def is_dynamic_class_property_enabled(instance, property, value):
        if hasattr(type(instance), property):
            return getattr(instance, property) == value



class CommonEnumMeta(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    def is_set_to(self, val):
        return self == val

    def __str__(self):
        return str(self.name)

    @classmethod
    def list_member_names(cls):
        return [name for name, member in cls.__members__.items()]





