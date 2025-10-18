from .models import ModelWrapper


def use_model_decorator(func):
    def wrapper(self: ModelWrapper,  *args, **kwargs):
        if self.model is None:
            self.__set_model()
        res = func(self, *args, **kwargs)
        self.__dispatch_model()
        return res
    return wrapper