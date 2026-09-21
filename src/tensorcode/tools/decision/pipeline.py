"""A decision pipeline with caller-selected encoding and prediction."""


class Decision:
    def __init__(self, *, encode, decide):
        self.encode = encode
        self.decide = decide

    def __call__(self, value, *, context=None):
        return self.decide(self.encode(value), context=context)
