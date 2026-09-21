import importlib.util
from pathlib import Path
import runpy

import torch
from tensorcode.tools.chatbot import Chatbot


def runner():
    path = Path(__file__).parents[2] / '.development/experiments/evaluate_generator_replacement.py'
    spec = importlib.util.spec_from_file_location('replacement', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replacement_preserves_other_weights_and_roundtrips(tmp_path):
    import sys
    models = str(Path(__file__).parents[1] / 'models')
    sys.path.insert(0, models)
    try:
        config = runpy.run_path(str(Path(models) / 'test_cognitive_chatbot.py'))['config']()
    finally:
        sys.path.remove(models)
    bot = Chatbot(config).eval()
    generator = Chatbot(config['cognition']['investigator']['generator']).eval()
    before = {k: v.clone() for k, v in bot.state_dict().items() if not k.startswith('investigator.generator.')}
    runner().configure(bot, generator, scope='joint', max_tokens=512, template_version=2)
    for key, value in before.items():
        torch.testing.assert_close(bot.state_dict()[key], value, rtol=0, atol=0)
    assert bot.investigator.generator is generator
    assert bot.investigator.config['verification_scope'] == 'joint'
    assert bot.investigator.verifier.max_tokens == 512
    bot.save_pretrained(tmp_path / 'model')
    restored = Chatbot.from_pretrained(tmp_path / 'model')
    assert restored.investigator.rank.configuration() == bot.investigator.rank.configuration()
    assert restored.fingerprint == bot.fingerprint
    assert restored.investigator.config['proposal_template_version'] == 2
    assert restored.investigator.verifier.max_tokens == 512
    for key, value in bot.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
