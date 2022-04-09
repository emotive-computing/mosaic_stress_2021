class Singleton(type):

    instance = None

    def __call__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = super(Singleton, cls).__call__(*args, **kwargs)
        return cls.instance

# Taken from https://github.com/tomerghelber/singleton-factory
class SingletonFactory(type):
    """
    Singleton Factory - keeps one object with the same hash
        of the same cls.
    Returns:
        An existing instance.
    """
    __instances = dict()

    def __call__(cls, *args, **kwargs):
        if cls not in cls.__instances:
            cls.__instances[cls] = dict()
        new_obj = super(SingletonFactory, cls).__call__(*args, **kwargs)
        if hash(new_obj) not in cls.__instances[cls]:
            cls.__instances[cls][hash(new_obj)] = new_obj
        return cls.__instances[cls][hash(new_obj)]
