"""Question answering, across knowledge, reasoning, memory, pragmatics and vision."""

from __future__ import annotations

from ..core import Task, register
from ..data import heldout
from ..judges import (any_gold_appears, asked_for_clarification, attempted_only, content_overlap,
                      majority_gold_appears)

CONTROLS = ("control:abstain", "control:echo")

register(Task(
    id="knowledge.nq_webq", area="knowledge", what="short factual questions",
    dataset=heldout.dataset("general_knowledge"), judge=any_gold_appears, controls=CONTROLS,
    splits=("dev", "calibration", "test"),
    notes="answer counted correct if any accepted answer appears in the reply"))

register(Task(
    id="reasoning.gsm8k", area="reasoning", what="grade-school maths word problems",
    dataset=heldout.dataset("arithmetic"), judge=any_gold_appears, controls=CONTROLS,
    splits=("dev", "calibration", "test"), notes="correct if the final number appears"))

register(Task(
    id="memory.longmemeval", area="memory", what="facts the user gave in earlier sessions",
    dataset=heldout.dataset("conversation_facts"), judge=content_overlap(), controls=CONTROLS,
    splits=("dev", "calibration", "test"), notes="LOOSE grader (content-word overlap); treat as an upper bound"))

register(Task(
    id="pragmatics.ambiguous", area="pragmatics", what="requests with several readings: should ask, not answer",
    dataset=heldout.dataset("ambiguous"), judge=asked_for_clarification, controls=CONTROLS,
    splits=("dev", "calibration", "test"), notes="correct == asked a clarifying question"))

register(Task(
    id="conversation.open_ended", area="conversation", what="brainstorming, advice, writing: no gold answer",
    dataset=heldout.dataset("open_ended"), judge=attempted_only, controls=CONTROLS,
    splits=("dev", "calibration", "test"), notes="only engagement is measured; quality needs a judge we do not have"))

register(Task(
    id="vision.vqa", area="vision", what="questions about photographs",
    dataset=heldout.dataset("image_questions"), judge=majority_gold_appears, controls=CONTROLS,
    splits=("dev", "calibration", "test"), notes="the image is given to the agent"))

register(Task(
    id="vision.screenqa", area="vision", what="questions about UI screenshots",
    dataset=heldout.dataset("screen_questions"), judge=any_gold_appears, controls=CONTROLS,
    splits=("dev", "calibration", "test"), notes="phone screenshots (RICO), not desktop"))

register(Task(
    id="conversation.owner_prompt", area="conversation", what="the owner's own long design prompt",
    dataset=heldout.dataset("owner"), judge=attempted_only, splits=("dev",),
    notes="one item; kept because it is the prompt that started this design"))
