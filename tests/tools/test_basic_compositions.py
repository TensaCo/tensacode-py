import pytest
from tensorcode.tools.agents import Chatbot
from tensorcode.tools.decision import Decision
from tensorcode.ops import llm, vec
import torch


def test_decision_is_a_composition_of_public_operations():
    tool = Decision(encode=vec.Transform(torch.nn.Identity()), decide=vec.Classify(torch.nn.Identity(), labels=('a','b')))
    assert tool(torch.tensor([0., 2.])).value == 'b'


def test_chatbot_commits_only_successful_turns_and_passes_history():
    received = []
    def model(messages):
        received.append(messages)
        if messages[-1].content == 'fail':
            raise RuntimeError('offline')
        return 'hello'
    bot = Chatbot(model=model)
    assert bot('first') == 'hello'
    with pytest.raises(RuntimeError):
        bot('fail')
    assert bot('second') == 'hello'
    assert [m.content for m in received[-1]] == ['first','hello','second']
    assert len(bot.history) == 4


def test_chatbots_do_not_share_conversations():
    a = Chatbot(model=lambda messages: 'yes')
    b = Chatbot(model=lambda messages: 'yes')
    a('hi')
    assert b.history == ()
