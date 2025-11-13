from .models import ModelWrapper


def use_model_decorator(func):
    def wrapper(self: ModelWrapper,  *args, **kwargs):
        if self.model is None:
            self.set_model()
        res = func(self, *args, **kwargs)
        self.dispatch_model()
        return res
    return wrapper