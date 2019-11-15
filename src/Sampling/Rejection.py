class Rejection(Sample):
    """
    Rejection sampling class. Uses either a know bound upper function or a
    uniform bound by default. No check is made on function is actually
    bound.
    """

    def __init__(self, distribution, bound_func=None):

        self.distribution = distribution

        if bound_func is None:
            pass
        else:
            self.bound_func = bound_func


    def generate(self):
        pass