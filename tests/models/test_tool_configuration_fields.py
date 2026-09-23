"""Tools reject unknown or obsolete configuration fields, at construction and load."""
import json

import pytest

from tensorcode.tools import Chatbot, Decision, Investigator, Planner, Scene
from tensorcode._internal.retrieval import RetrievalEncoder
from test_chatbot_model import tiny_config as chatbot_config
from test_investigation import config as investigator_config
from test_response_quality import tiny_config as foundation_config
from test_retrieval_encoder import retrieval_config


def ranking_config():
    return {'vocabulary': ['hello', 'world'], 'dimensions': 8, 'slots': 2, 'steps': 1}


def scene_config():
    return {'vocabulary': ['object', 'left'], 'dimensions': 8, 'slots': 3}


CONSTRUCTORS = [
    (Chatbot, chatbot_config, 'Chatbot'),
    (Investigator, investigator_config, 'Investigator'),
    (Decision, investigator_config, 'Decision'),
    (Planner, ranking_config, 'Planner'),
    (Scene, scene_config, 'Scene ranking'),
    (RetrievalEncoder, retrieval_config, 'RetrievalEncoder'),
]


@pytest.mark.parametrize('tool, make_config, owner', CONSTRUCTORS)
def test_construction_rejects_unknown_fields_naming_them_and_valid_fields(tool, make_config, owner):
    config = dict(make_config(), colour='blue', obsolete_head=1)
    with pytest.raises(ValueError) as error:
        tool(config)
    message = str(error.value)
    assert message.startswith(f"Unknown {owner} configuration fields: ['colour', 'obsolete_head']; valid fields: [")
    for field in make_config():
        assert repr(field) in message.split('valid fields:')[1]


def test_nested_chatbot_cognition_rejects_unknown_fields():
    config = chatbot_config()
    config['cognition'] = {'investigator': investigator_config(), 'proposal_limit': 3}
    with pytest.raises(ValueError, match=r"Unknown Chatbot cognition configuration fields: \['proposal_limit'\]"):
        Chatbot(config)


def test_scene_language_mode_rejects_ranking_fields_before_loading_assets():
    with pytest.raises(ValueError, match=r"Unknown Scene language configuration fields: \['vocabulary'\]"):
        Scene({'mode': 'language', 'vocabulary': ['object']})


def test_valid_configurations_still_construct():
    assert set(Planner(ranking_config()).configuration()) <= Planner.config_fields
    assert set(Investigator(investigator_config()).configuration()) <= Investigator.config_fields
    chatbot = Chatbot(chatbot_config())
    assert set(chatbot.configuration()) <= Chatbot.config_fields
    assert set(Scene(scene_config()).configuration()) <= Scene.ranking_fields


@pytest.mark.parametrize('tool, make_config', [
    (Planner, ranking_config), (Investigator, investigator_config),
    (Chatbot, chatbot_config), (Scene, scene_config)])
def test_from_pretrained_rejects_saved_unknown_fields(tmp_path, tool, make_config):
    directory = tool(make_config()).save_pretrained(tmp_path / 'model')
    assert type(tool.from_pretrained(directory)) is tool
    manifest_path = directory / 'tensorcode_config.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['config']['obsolete_field'] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"configuration fields: \['obsolete_field'\]; valid fields"):
        tool.from_pretrained(directory)


@pytest.mark.parametrize('model_type', ['bert', 'electra'])
@pytest.mark.parametrize('tool', [Planner, Investigator, Decision])
def test_foundation_rank_encoder_operation_configuration_is_json(tool, model_type):
    native = foundation_config(model_type)
    model = tool({key: native[key] for key in ('foundation_config', 'tokenizer_json', 'tokenizer_special_tokens')})
    encoder = model.rank.encode.configuration()
    assert encoder['module'] == model.rank.encode.module.configuration()
    assert encoder['module']['foundation_config']['model_type'] == model_type
    assert {item['name'] for item in encoder['parameters']} == {
        name for name, _ in model.rank.encode.named_parameters()}
    for operation in model.operation_bindings().values():
        json.dumps(operation.configuration(), allow_nan=False)
