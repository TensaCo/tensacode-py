import pytest
from tensorcode.runtime.cognitive_state import Evidence
from tensorcode.runtime.episodic import EpisodicMemory


def test_cosine_retrieval_source_exclusion_removal_and_capacity():
    m = EpisodicMemory(capacity=2, model_fingerprint='v1')
    m.insert(Evidence('a','A','source-a'), [1,0], episode_id='past')
    m.insert(Evidence('b','B','source-b'), [0,1], episode_id='now')
    hits = m.query([1,0], model_fingerprint='v1')
    assert hits[0].evidence.source_id == 'source-a'
    assert hits[0].score == 1
    assert len(m.query([0,1],model_fingerprint='v1',exclude_episode_id='now')) == 1
    m.insert(Evidence('c','C','source-c'), [1,1], episode_id='later')
    assert {h.evidence.id for h in m.query([1,0],model_fingerprint='v1')} == {'b','c'}
    m.remove('c')
    assert len(m.query([1,0],model_fingerprint='v1')) == 1


def test_conflicts_bad_vectors_and_stale_rebuild_are_atomic():
    m = EpisodicMemory(model_fingerprint='v1')
    e = Evidence('a','A','s')
    m.insert(e,[1,0],episode_id='past')
    m.insert(e,[1,0],episode_id='past')
    assert len(m) == 1
    with pytest.raises(ValueError): m.insert(Evidence('a','different','s'),[1,0],episode_id='past')
    with pytest.raises(ValueError): m.insert(Evidence('b','B','s'),[float('nan'),0],episode_id='past')
    with pytest.raises(ValueError): m.query([1,0],model_fingerprint='v2')
    with pytest.raises(ValueError): m.rebuild_index({}, model_fingerprint='v2')
    assert m.query([1,0],model_fingerprint='v1')[0].evidence == e
    m.rebuild_index({'a':[0,1]}, model_fingerprint='v2')
    assert m.query([0,1],model_fingerprint='v2')[0].score == 1


def test_invalid_queries_and_insert_fingerprints_do_not_corrupt_memory():
    m = EpisodicMemory(model_fingerprint='v1')
    m.insert(Evidence('a','A','s'),[1,0],episode_id='episode')
    for vector in ([0,0], [1], [float('inf'),0]):
        with pytest.raises(ValueError): m.query(vector,model_fingerprint='v1')
    with pytest.raises(ValueError):
        m.insert(Evidence('b','B','s'),[0,1],episode_id='episode',model_fingerprint='v2')
    assert len(m) == 1
    assert m.query([1,0],model_fingerprint='v1',k=0) == ()
