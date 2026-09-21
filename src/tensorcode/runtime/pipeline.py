"""An application pipeline using explicitly supplied models or operations."""

from ..ops import text as text_ops


class DecisionPipeline:
    def __init__(
        self,
        *,
        model=None,
        labels=None,
        instructions=None,
        encode=None,
        decide=None,
        selection_policy=None,
    ):
        ready_supplied = model is not None or labels is not None
        explicit_supplied = encode is not None or decide is not None
        self.labels = None
        if ready_supplied and explicit_supplied:
            raise ValueError("supply either model/labels or encode/decide, not both")
        if ready_supplied:
            if model is None or labels is None:
                raise ValueError("ready mode requires both model and labels")
            self.encode = text_ops.TextEncoder()
            self.labels = tuple(labels)
            self.decide = text_ops.Classify.from_model(
                model,
                labels=self.labels,
                instructions=instructions,
            )
        elif explicit_supplied:
            if encode is None or decide is None:
                raise ValueError("explicit mode requires both encode and decide")
            self.encode = encode
            self.decide = decide
        else:
            raise ValueError("supply either model/labels or encode/decide")
        if selection_policy is not None and not callable(selection_policy):
            raise TypeError("selection_policy must be callable")
        self.selection_policy = selection_policy or (lambda result: result)

    def __call__(self, value, *, context=None):
        result = self.decide(self.encode(value), context=context)
        selected = self.selection_policy(result)
        if self.labels is not None:
            choice = getattr(selected, "value", None)
            abstained = getattr(selected, "abstained", False)
            if abstained:
                if choice is not None:
                    raise ValueError("an abstained selection must have value None")
            elif choice not in self.labels:
                raise ValueError("selection policy value is not one of the configured labels")
        return selected
