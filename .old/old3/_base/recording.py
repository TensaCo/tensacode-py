import inspect


class RecorderDecorator:
    def __init__(self):
        self.records = {}

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            context = inspect.stack()[1]  # get calling context
            key = f"{context.filename}:{context.lineno}:{context.function}"
            if key in self.records:
                return self.records[key]
            else:
                result = func(*args, **kwargs)
                self.records[key] = result
                return result

        return wrapper

    @property
    def record(self):
        self.records = {}

    @property
    def replay(self):
        return self.records
