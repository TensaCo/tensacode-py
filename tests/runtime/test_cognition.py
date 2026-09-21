"""Mechanism tests with random models and explicitly authored classifier biases."""
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import BertConfig
from tensorcode.tools.investigator import Investigator
from tensorcode.runtime.cognitive_state import Evidence
from tensorcode.runtime.cognition import CognitiveSession, LearnedEpisodicMemory, SelectionPolicy


def investigator():
    tokenizer = Tokenizer(models.WordLevel({'<pad>':0,'<unk>':1,'alpha':2,'beta':3},unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    config = {'vocabulary':['alpha','beta'], 'dimensions':8,'slots':2,'steps':1,
        'verifier_config':BertConfig(vocab_size=4,hidden_size=8,num_hidden_layers=1,num_attention_heads=2,intermediate_size=16,num_labels=3).to_dict(),
        'verifier_tokenizer_json':tokenizer.to_str(),
        'verifier_tokenizer_special_tokens':{'pad_token':'<pad>','unk_token':'<unk>'},
        'verifier_labels':{'support':0,'contradiction':1,'unknown':2}}
    model = Investigator(config).eval()
    with torch.no_grad():
        model.verifier.model.classifier.weight.zero_()
        model.verifier.model.classifier.bias.copy_(torch.tensor([8.,0.,0.]))
    return model


def test_revisions_reverify_sources_and_abstain_without_promoting_hypotheses():
    tool = investigator(); session = CognitiveSession(tool)
    session.ingest([Evidence('a','alpha','document')])
    first = session.investigate('alpha',hypotheses=[{'id':'h','text':'beta'}])
    assert first['selected_id'] == 'h' and not first['abstained']
    prior = session.state
    session.revise_evidence('a','beta')
    assert session.state.selection_stale
    assert session.state.evidence[0].text == 'alpha'
    with torch.no_grad(): tool.verifier.model.classifier.bias.copy_(torch.tensor([0.,8.,0.]))
    second = session.investigate('alpha',hypotheses=[{'id':'h','text':'beta'}])
    assert second['abstained'] and second['selected_id'] is None
    assert second['evidence'][0]['text'] == 'beta'
    assert second['evidence'][0]['source_id'] == 'document'
    assert len(session.state.assessments) == 2
    assert not session.state.observations
    assert prior.evidence[0].text == 'alpha'


def test_snapshot_roundtrip_forks_and_failures_are_transactional(tmp_path):
    tool = investigator(); session = CognitiveSession(tool)
    session.ingest([Evidence('a','alpha','doc')]); session.revise_evidence('a','beta')
    fork = session.fork(); fork.ingest([Evidence('b','beta','doc2')])
    assert len(session.state.evidence) == 2
    before = session.snapshot()
    with pytest.raises(ValueError): session.investigate('alpha',hypotheses=[])
    assert session.snapshot() == before
    path = tmp_path / 'session.json'; session.save(path)
    restored = CognitiveSession.load(path,investigator=tool)
    assert restored.snapshot() == session.snapshot()
    broken = session.snapshot(); broken['active_evidence']['a'] = 'absent'
    with pytest.raises(ValueError): CognitiveSession.from_snapshot(broken,investigator=tool)


def test_authored_policy_uses_strongest_source_unknown_and_retains_conflict():
    policy = SelectionPolicy()
    assert policy.accepts([{'support':.95,'contradiction':.02,'unknown':.03}, {'support':.01,'contradiction':.01,'unknown':.98}])
    assert not policy.accepts([{'support':.95,'contradiction':.02,'unknown':.03}, {'support':.01,'contradiction':.98,'unknown':.01}])
    assert not policy.accepts([])


def test_owned_encoder_memory_detects_weight_changes_and_rebuilds():
    tool = investigator(); memory = LearnedEpisodicMemory(tool)
    memory.remember(Evidence('a','alpha','doc'),episode_id='past')
    assert memory.retrieve('alpha')[0].evidence.source_id == 'doc'
    with torch.no_grad(): tool.rank.encode.module.weight.add_(.5)
    with pytest.raises(ValueError,match='stale'): memory.retrieve('alpha')
    memory.rebuild_index()
    assert memory.retrieve('alpha')[0].score == pytest.approx(1)


def test_active_retrieval_persists_and_removed_sources_do_not_return(tmp_path):
    tool = investigator(); memory = LearnedEpisodicMemory(tool)
    memory.remember(Evidence('past','alpha','past-document'),episode_id='past',question='alpha',outcome='external feedback')
    session = CognitiveSession(tool,memory=memory)
    receipt = session.investigate('alpha',hypotheses=[{'id':'h','text':'beta'}])
    assert receipt['selected_id'] == 'h'
    assert receipt['retrieval'][0]['evidence']['source_id'] == 'past-document'
    path = tmp_path / 'with-memory.json'; session.save(path)
    restored = CognitiveSession.load(path,investigator=tool)
    assert restored.retrieve('alpha')[0].outcome == 'external feedback'
    restored.ingest([Evidence('past','alpha','past-document')])
    restored.remove_evidence('past')
    assert restored.state.selection_stale
    next_receipt = restored.investigate('alpha',hypotheses=[{'id':'h','text':'beta'}])
    assert next_receipt['evidence'] == []
    assert next_receipt['abstained']
    assert restored.state.evidence[0].text == 'alpha'


def test_configured_memory_and_revisions_exclude_archived_sources():
    tool = investigator(); session = CognitiveSession(tool,memory={'capacity':2,'top_k':1})
    session.ingest([Evidence('a','alpha','doc')])
    session.remember('a',episode_id='past')
    session.revise_evidence('a','beta')
    receipt = session.investigate('alpha',hypotheses=[{'id':'h','text':'beta'}])
    assert [e['text'] for e in receipt['evidence']] == ['beta']
    assert receipt['retrieval'] == []
    restored = CognitiveSession.from_snapshot(session.snapshot(),investigator=tool)
    assert restored.snapshot() == session.snapshot()


def test_model_weight_changes_invalidate_live_and_restored_selections():
    tool = investigator(); session = CognitiveSession(tool)
    session.ingest([Evidence('a','alpha','doc')])
    session.investigate('alpha',hypotheses=[{'id':'h','text':'beta'}])
    snapshot = session.snapshot()
    with torch.no_grad(): tool.verifier.model.classifier.bias.add_(.5)
    assert session.state.selection_stale
    assert session.state.is_stale(session.state.assessments[-1])
    restored = CognitiveSession.from_snapshot(snapshot,investigator=tool)
    assert restored.state.selection_stale
    revision = restored.state.revision
    assert restored.state.revision == revision


def test_new_episode_retrieves_past_sources_and_forked_memory_is_independent():
    tool = investigator(); session = CognitiveSession(tool,memory={'capacity':3})
    session.ingest([Evidence('a','alpha','doc')]); session.remember('a')
    fork = session.fork(copy_memory=True)
    fork.ingest([Evidence('b','beta','doc2')]); fork.remember('b')
    assert len(session.memory.memory) == 1
    session.new_episode()
    assert session.episode_id == 'episode-1' and not session.active_evidence
    receipt = session.investigate('alpha',hypotheses=[{'id':'h','text':'beta'}])
    assert receipt['retrieval'][0]['evidence']['id'] == 'a'
    session.remove_evidence('a')
    assert session.investigate('alpha',hypotheses=[{'id':'h','text':'beta'}])['abstained']
    restored = CognitiveSession.from_snapshot(session.snapshot(),investigator=tool)
    assert restored.episode_id == 'episode-1'


def test_truncated_verification_abstains_despite_authored_support_bias():
    tool = investigator(); tool.verifier.max_tokens = 2
    session = CognitiveSession(tool)
    session.ingest([Evidence('a','alpha alpha alpha alpha','doc')])
    receipt = session.investigate('alpha',hypotheses=[{'id':'h','text':'beta beta beta beta'}])
    assert receipt['candidates'][0]['verifications'][0]['input_truncated']
    assert receipt['abstained']


def test_owned_generator_outputs_remain_hypotheses(monkeypatch):
    from transformers import T5Config
    base = investigator().configuration()
    base['generator'] = {'foundation_config':T5Config(vocab_size=4,d_model=8,d_ff=16,num_layers=1,num_decoder_layers=1,num_heads=2,d_kv=4,decoder_start_token_id=0,pad_token_id=0,eos_token_id=1,dropout_rate=0).to_dict(),
        'tokenizer_json':base['verifier_tokenizer_json'],
        'tokenizer_special_tokens':base['verifier_tokenizer_special_tokens'],
        'workspace':{'slots':2,'steps':1},'max_new_tokens':2,'max_input_tokens':32}
    tool = Investigator(base).eval()
    # Authored decoder rendering tests the state plumbing, not learned ability.
    monkeypatch.setattr(tool.generator.tokenizer,'batch_decode',lambda *a,**kw:['beta'])
    session = CognitiveSession(tool)
    session.ingest([Evidence('a','alpha','doc')])
    receipt = session.investigate('alpha',count=1)
    assert session.state.hypotheses[0].origin == 'generated'
    assert session.state.hypotheses[0].text == 'beta'
    assert [e.text for e in session.state.evidence] == ['alpha']
    assert receipt['candidates'][0]['epistemic_status'] == 'hypothesis'
    assert not session.state.observations


def test_public_retrieval_excludes_revised_and_removed_sources():
    tool = investigator(); session = CognitiveSession(tool,memory={'capacity':3})
    session.ingest([Evidence('a','alpha','doc'),Evidence('b','beta','doc2')])
    session.remember('a'); session.remember('b')
    session.revise_evidence('a','beta')
    session.remove_evidence('b')
    assert session.retrieve('alpha') == ()
    assert len(session.memory.retrieve('alpha')) == 2  # Explicit raw historical API.


def test_joint_support_preserves_each_source_contradiction_veto():
    policy = SelectionPolicy()
    unknown = {'support': .01, 'contradiction': .01, 'unknown': .98}
    supported = {'support': .95, 'contradiction': .02, 'unknown': .03}
    conflict = {'support': .01, 'contradiction': .98, 'unknown': .01}
    assert not policy.accepts([unknown, unknown])
    assert policy.accepts([unknown, unknown], joint_distribution=supported)
    assert not policy.accepts([unknown, conflict], joint_distribution=supported)
    assert not policy.accepts([supported], joint_distribution=unknown)
    assert not policy.accepts([], joint_distribution=supported)


def test_joint_receipt_covers_sources_revisions_and_artifact(tmp_path):
    base = investigator()
    settings = base.configuration(); settings['verification_scope'] = 'joint'
    tool = Investigator(settings).eval(); tool.load_state_dict(base.state_dict())
    session = CognitiveSession(tool)
    session.ingest([Evidence('a', 'alpha', 'doc-a'), Evidence('b', 'beta', 'doc-b')])
    first = session.investigate('alpha', hypotheses=[{'id': 'h', 'text': 'beta'}])
    assert not first['abstained']
    joint = first['candidates'][0]['joint_verification']
    assert joint['evidence_ids'] == ['a', 'b']
    assert joint['source_ids'] == ['doc-a', 'doc-b']
    assert not joint['input_truncated']
    session.revise_evidence('a', 'alpha beta')
    assert session.state.selection_stale
    second = session.investigate('alpha', hypotheses=[{'id': 'h', 'text': 'beta'}])
    assert second['candidates'][0]['joint_verification']['evidence_ids'][0] != 'a'
    assert first['candidates'][0]['joint_verification']['evidence_ids'] == ['a', 'b']
    tool.save_pretrained(tmp_path / 'joint')
    restored = Investigator.from_pretrained(tmp_path / 'joint')
    assert restored.configuration()['verification_scope'] == 'joint'
    assert CognitiveSession(restored)._model_identity() != CognitiveSession(base)._model_identity()
    assert CognitiveSession.from_snapshot(session.snapshot(), investigator=restored).snapshot() == session.snapshot()


def test_joint_truncation_abstains_even_when_each_source_fits():
    base = investigator(); settings = base.configuration(); settings['verification_scope'] = 'joint'
    tool = Investigator(settings).eval(); tool.load_state_dict(base.state_dict())
    tool.verifier.max_tokens = 4
    session = CognitiveSession(tool)
    session.ingest([Evidence('a', 'alpha alpha', 'doc-a'), Evidence('b', 'beta beta', 'doc-b')])
    receipt = session.investigate('alpha', hypotheses=[{'id': 'h', 'text': 'beta'}])
    candidate = receipt['candidates'][0]
    assert not any(row['input_truncated'] for row in candidate['verifications'])
    assert candidate['joint_verification']['input_truncated']
    assert receipt['abstained']


def test_unknown_verification_scope_rejected():
    settings = investigator().configuration(); settings['verification_scope'] = 'automatic'
    with pytest.raises(ValueError, match='verification_scope'):
        Investigator(settings)


def test_joint_selection_uses_combined_support_not_individual_support(monkeypatch):
    settings = investigator().configuration(); settings['verification_scope'] = 'joint'
    tool = Investigator(settings)
    # Authored classifier outputs isolate evidence aggregation, not learned inference.
    def classifier(pairs, *, context=None):
        return torch.tensor([[8., 0., 0.] if '\n\n' in row['premise'] else [0., 0., 8.]
                             for row in pairs])
    monkeypatch.setattr(tool.verifier, 'forward', classifier)
    session = CognitiveSession(tool)
    session.ingest([Evidence('a', 'alpha', 'one'), Evidence('b', 'beta', 'two')])
    result = session.investigate('alpha', hypotheses=[{'id': 'h', 'text': 'beta'}])
    assert result['selected_id'] == 'h'
    assert all(row['distribution']['support'] < .01 for row in result['candidates'][0]['verifications'])
    session.remove_evidence('b')
    assert session.investigate('alpha', hypotheses=[{'id': 'h', 'text': 'beta'}])['abstained']


def test_joint_coverage_must_match_exact_order_and_ids():
    policy = SelectionPolicy()
    good = {'support': .95, 'contradiction': .02, 'unknown': .03}
    checks = [{'source_id': 'a', 'distribution': good}, {'source_id': 'b', 'distribution': good}]
    for ids in (['b', 'a'], ['a'], ['a', 'a']):
        verification = {'verifications': checks,
                        'joint_verification': {'scope': 'joint', 'source_ids': ids, 'distribution': good}}
        with pytest.raises(ValueError, match='joint verification sources'):
            policy.accepts_verification(verification, ['a', 'b'], scope='joint')


@pytest.mark.parametrize('where,field,value', [
    ('joint', 'input_truncated', 'missing'), ('joint', 'input_truncated', None),
    ('joint', 'input_truncated', 0), ('joint', 'input_truncated', 'false'),
    ('joint', 'token_count', 'missing'), ('joint', 'token_count', None),
    ('joint', 'token_count', 0), ('joint', 'token_count', True),
    ('joint', 'token_count', -1), ('joint', 'token_count', 1.5),
    ('joint', 'max_tokens', 'missing'), ('joint', 'max_tokens', None),
    ('joint', 'max_tokens', 0), ('joint', 'max_tokens', True),
    ('joint', 'max_tokens', -1), ('joint', 'max_tokens', 1.5),
    ('joint', 'max_tokens', 2), ('joint', 'token_count', 3000),
    ('joint', 'input_truncated', True),
    ('source', 'input_truncated', 'missing'), ('source', 'input_truncated', None),
    ('source', 'input_truncated', 0), ('source', 'input_truncated', 'false'),
])
def test_joint_screen_requires_explicit_complete_input_metadata(where, field, value):
    good = {'support': .95, 'contradiction': .02, 'unknown': .03}
    source = {'source_id': 'a', 'distribution': good, 'input_truncated': False}
    joint = {'scope': 'joint', 'source_ids': ['a'], 'distribution': good,
             'input_truncated': False, 'token_count': 3, 'max_tokens': 512}
    receipt = {'verifications': [source], 'joint_verification': joint}
    policy = SelectionPolicy()
    assert policy.accepts_verification(receipt, ['a'], scope='joint')
    target = joint if where == 'joint' else source
    if value == 'missing':
        target.pop(field)
    else:
        target[field] = value
    with pytest.raises(ValueError, match='coverage metadata'):
        policy.accepts_verification(receipt, ['a'], scope='joint')


def test_fresh_sessions_reuse_weight_hash_but_invalidate_on_updates(monkeypatch):
    tool = investigator()
    sessions = [CognitiveSession(tool), CognitiveSession(tool)]
    reads = []
    original = torch.Tensor.cpu
    def cpu(tensor, *args, **kwargs):
        reads.append(tensor.numel())
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'cpu', cpu)
    identity = sessions[0]._model_identity()
    first_reads = len(reads)
    assert first_reads > 0
    assert sessions[1]._model_identity() == identity
    assert len(reads) == first_reads  # No second model-sized device-to-host copy.
    with torch.no_grad():
        next(tool.parameters()).add_(1)
    changed = sessions[1]._model_identity()
    assert changed != identity and len(reads) > first_reads
    reads_after_update = len(reads)
    assert sessions[0]._model_identity() == changed
    assert len(reads) == reads_after_update
    sessions[0].invalidate_fingerprint()
    assert sessions[1]._model_identity() == changed
    assert len(reads) > reads_after_update
