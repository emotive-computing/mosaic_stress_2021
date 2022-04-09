
class ClassPropertyDecorator(object):

    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self

def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDecorator(func)

class InfixDecorator:
    def __init__(self, function):
        self.function = function

    def __ror__(self, other):
        return InfixDecorator(lambda x, self=self, other=other: self.function(other, x))

    def __or__(self, other):
        return self.function(other)

    def __rlshift__(self, other):
        return InfixDecorator(lambda x, self=self, other=other: self.function(other, x))

    def __rshift__(self, other):
        return self.function(other)

    def __call__(self, value1, value2):
        return self.function(value1, value2)


@InfixDecorator
def with_base(x, y):
    cls = type(x.__name__ + "_with_base_" + y.__name__, (x, ), {'base_cls': y})
    return cls


def extend(class_to_extend):
    def decorator(extending_class):
        class_to_extend.__dict__.update(extending_class.__dict__)
        return class_to_extend
    return decorator


