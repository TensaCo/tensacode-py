from pathlib import Path
from tensorcode.agent import RefinementLibrary
"""Supplied scene models exercise integration, not learned pixel understanding."""
from dataclasses import replace
import json

import pytest

from tensorcode import ops
from tensorcode.agent import (Agent, FileSystemPlugin, InterpretationDecision, Plugin,
                             SceneGraph, SceneProposal, VisualAnchor)
from tensorcode.agent.operations import Transcript
from tensorcode.agent.understand import SentenceAlternative
from interpretation_fixtures import project_sentence
from tensorcode.records import Proposition, Ref, Var


class Scenes(Plugin):
    def __init__(self, primary=0):
        super().__init__('scene-provider')
        self.primary = primary
        self.seen = []

    def interpret_image(self, image, ref):
        self.seen.append(image)
        nodes = (Ref(ref.id + '/left'), Ref(ref.id + '/right'))
        yield SceneProposal(SceneGraph(
            ref, nodes,
            (Proposition('is_a', {'entity': nodes[0], 'kind': 'workspace'}),
             Proposition('is_a', {'entity': nodes[1], 'kind': 'workspace'}),
             Proposition('primary_workspace', {'scene': ref, 'workspace': nodes[self.primary]}),
             Proposition('left_of', {'entity': nodes[0], 'other': nodes[1]}),
             Proposition('layout', {'scene': ref, 'organization': 'split-view'})),
            (VisualAnchor(nodes[0], (0, 0, .5, 1)), VisualAnchor(nodes[1], (.5, 0, 1, 1))),
        ), provenance=('supplied test graph; no pixel inference',))



def test_scene_relations_and_global_structure_retained_without_beliefs():
    plugin = Scenes()
    agent = Agent([plugin])
    original = bytearray(b'opaque-image-evidence')
    result = agent.interpret_image(original)
    original[0] = 0
    source = agent.interpretations.get_source(result.source_id)
    assert source.modality == 'image' and source.payload == b'opaque-image-evidence'
    group = agent.interpretations.get(result.group_ids[0])
    assert group.selected_id is None
    graph = group.candidates[0].payload.graph
    assert graph.match(Proposition('layout', {'scene': result.image, 'organization': Var('layout')})) == ({'layout': 'split-view'},)
    assert graph.match(Proposition('left_of', {'entity': Var('left'), 'other': Var('right')}))
    assert agent.store.propositions() == []
    assert agent.last_image is None  # inspection does not change conversational focus
    assert not agent.turns


def test_visual_source_file_is_snapshotted(tmp_path):
    path = tmp_path / 'image.bin'
    path.write_bytes(b'original')
    plugin = Scenes()
    agent = Agent([plugin])
    result = agent.interpret_image(path)
    path.write_bytes(b'changed')
    assert agent.interpretations.get_source(result.source_id).payload == b'original'
    assert plugin.seen == [b'original']


def test_turn_records_visual_groups_without_promoting_proposals():
    agent = Agent([Scenes()])
    turn = agent.turn('', images=[b'image'])
    assert len(turn.visual_interpretation_ids) == 1
    assert agent.store.propositions() == []
    seen = next(event for event in turn.events if event['type'] == 'seen')
    assert 'claims' not in seen and 'legacy_providers' not in seen
    assert seen['interpretations'] == list(turn.visual_interpretation_ids)
    json.dumps(turn.events)


def test_abstention_retains_an_empty_unselected_group():
    class Abstains(Scenes):
        def interpret_image(self, image, ref):
            return ()
    agent = Agent([Abstains()])
    turn = agent.turn('', images=[b'image'])
    group = agent.interpretations.get(turn.visual_interpretation_ids[0])
    assert group.candidates == () and group.selected_id is None


def test_wrong_source_graph_rejected():
    class WrongSource(Scenes):
        def interpret_image(self, image, ref):
            return (SceneProposal(SceneGraph(Ref('image:another'))),)
    agent = Agent([WrongSource()])
    with pytest.raises(ValueError, match='different image'):
        agent.interpret_image(b'image')
    assert agent.interpretations.values() == ()
    assert agent.store.propositions() == []


@pytest.mark.parametrize('primary,destination', [(0, 'hello'), (1, 'demo')])
def test_scene_organization_can_change_actual_action_without_object_label_change(tmp_path, monkeypatch, primary, destination):
    scene_provider = Scenes(primary)
    agent = Agent([scene_provider, FileSystemPlugin(tmp_path, refinements=RefinementLibrary.load(
        Path(__file__).parent / "fixtures" / "project_refinements.json"))])
    sentences = [project_sentence(name) for name in ('hello', 'demo')]
    candidates = tuple(SentenceAlternative(s.reading, s.acts, s.skipped, s.guessed, 'test')
                       for s in sentences)
    sentence = replace(sentences[0], alternatives=candidates)
    monkeypatch.setattr(ops, 'parse', lambda *a, **kw: Transcript((sentence,), 'test-reader'))

    def select(group):
        visual = next(g for g in agent.interpretations.values()
                      if agent.interpretations.get_source(g.source_id).modality == 'image')
        graph = visual.candidates[0].payload.graph
        matches = graph.match(Proposition('primary_workspace', {'scene': graph.image, 'workspace': Var('workspace')}))
        index = graph.nodes.index(matches[0]['workspace'])
        return InterpretationDecision(group.candidates[index].id,
                                      f'supplied policy consulted organization in {visual.id}')

    agent.interpretation_selector = select
    turn = agent.turn('use the primary workspace', images=[b'same opaque image fixture'])
    assert turn.outcomes[0].status == 'done'
    assert (tmp_path / destination / 'main.py').is_file()
    assert not (tmp_path / ('demo' if destination == 'hello' else 'hello')).exists()


def test_providers_receive_detached_image_evidence():
    class Mutates(Scenes):
        def interpret_image(self, image, ref):
            image[0] = 0
            return ()
    ordinary = Scenes()
    agent = Agent([Mutates(), ordinary])
    result = agent.interpret_image(bytearray(b'original'))
    assert ordinary.seen == [bytearray(b'original')]
    assert agent.interpretations.get_source(result.source_id).payload == b'original'



def test_direct_image_claims_interface_is_removed():
    assert not hasattr(Plugin, 'see')


def test_none_is_not_a_legacy_fallback():
    class Invalid(Plugin):
        def interpret_image(self, image, ref):
            return None
    agent = Agent([Invalid('invalid')])
    with pytest.raises(TypeError, match='empty iterable'):
        agent.interpret_image(b'image')
