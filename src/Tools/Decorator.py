# Some decorator functions used in this library

import functools


def lazy_property(function):
    """
    Property decorator that is only called once
    """
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator
