from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op
from tensacode.internal.utils.misc import score_node_inheritance_distance


@BaseEngine.register_op(score_fn=score_node_inheritance_distance(inputs=SequenceNode))
def predict(
    engine: BaseEngine,
    inputs: list[Any],
    prompt: Optional[Encoded[str]] = None,
    **kwargs: Any,
) -> Any:
    """Existing docstring moved here"""

    if len(inputs) < 2:
        raise ValueError("Predict needs 2 or more examples to continue a sequence")

    for item, next_item in zip(inputs, inputs[1:]):
        item_enc = engine.encode(item)
        next_item_enc = engine.encode(next_item)
        pred_next_enc = engine.transform(item_enc, prompt="predict the next item")
        engine.info(next_item=next_item)
        similarity = engine.similarity(next_item_enc, pred_next_enc)
        engine.reward(reward=similarity, importance=engine.c_pred_reward_coef)

    last_item_enc = engine.encode(inputs[-1])
    pred_next_enc = engine.transform(last_item_enc, prompt="predict the next item")
    return engine.decode(pred_next_enc)
