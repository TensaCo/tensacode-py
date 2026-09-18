"""Heterogeneous memory: episodes that decay and consolidate, told facts, real forgetting."""

from datetime import datetime, timedelta, timezone

import tensorcode as tc
from tensorcode.awareness import Awareness, AwarenessPolicy
from tensorcode.cognition import Thought
from tensorcode.frames import Frames
from tensorcode.memory import EPISODIC, SEMANTIC, Memory, MemoryPolicy

NOW = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)


def perceived(mind, subject, predicate, obj, *, at=NOW, source="obs:frame-1", method="dom-scene-graph"):
    return mind.tell(tc.Claim(tc.Ref(subject), predicate, obj), tc.Evidence(tc.Ref(source), at, method=method))


def test_a_told_fact_is_semantic_hearsay_and_answers_later():
    """'my name is Jacob' has to land somewhere, and it is not an observation."""
    mind = tc.Store()
    memory = Memory(mind)

    rec = memory.told(tc.Ref("person:user"), "name", "Jacob")

    assert rec.claim.scope == SEMANTIC
    assert [r.claim.object for r in memory.semantic(tc.Ref("person:user"), "name")] == ["Jacob"]
    assert rec.evidence[0].method == "told" and rec.evidence[0].source == tc.Ref("said:user")
    from tensorcode.frames import modality_of

    assert modality_of(rec) == ("hearsay",), "a told fact is hearsay, and says so"


def test_an_episode_records_what_happened_and_recall_is_associative():
    mind = tc.Store()
    memory = Memory(mind)
    first = perceived(mind, "path:/home/agent/Desktop/recipes", "is_a", "directory")
    memory.encode([first], summary="made the recipes folder on the desktop", salience=0.8)
    second = perceived(mind, "path:/home/agent/Documents/tax.pdf", "is_a", "regular file")
    memory.encode([second], summary="read the tax document", salience=0.4)

    got = memory.recall("recipes folder", k=2)

    assert got.episodes and "recipes" in got.episodes[0][0].summary
    assert got.episodes[0][1].kind == "similarity"
    assert any("recipes" in line for line in got.why())


def test_repeats_across_episodes_consolidate_into_a_semantic_claim_that_cites_them():
    mind = tc.Store()
    memory = Memory(mind, MemoryPolicy(consolidate_after=3))
    for i in range(3):
        rec = perceived(mind, "app:Terminal", "opens_with", "dock icon", at=NOW + timedelta(minutes=i))
        memory.encode([rec], summary=f"opened the terminal, time {i}", salience=0.5)

    made = memory.consolidate()

    assert [c.predicate for c in made] == ["opens_with"]
    semantic = memory.semantic(tc.Ref("app:Terminal"), "opens_with")
    assert semantic and semantic[0].claim.scope == SEMANTIC
    assert semantic[0].evidence[0].derived_from, "a generalization cites the episodes it came from"
    assert tc.Ref("app:Terminal") == semantic[0].claim.subject
    assert memory.consolidate() == [], "consolidating twice does not duplicate"


def test_forgetting_is_real_but_spares_what_is_aware_supported_or_told():
    mind = tc.Store()
    awareness = Awareness(mind, AwarenessPolicy(budget=4, floor=0.1))
    memory = Memory(mind, MemoryPolicy(half_life=timedelta(minutes=5), min_salience=0.2), awareness=awareness)
    # a frame from two hours ago, and one we are still looking at: awareness spreads within
    # an observation, so what was read beside something salient stays salient too
    old = perceived(mind, "ui:row#1", "reads", "stale row", at=NOW - timedelta(hours=2), source="obs:frame-old")
    aware_old = perceived(mind, "ui:row#2", "reads", "still looking at this", at=NOW - timedelta(hours=2), source="obs:frame-now")
    premise = perceived(mind, "ui:row#3", "reads", "premise", at=NOW - timedelta(hours=2), source="obs:frame-older")
    mind.tell(tc.Claim(tc.Ref("form:x"), "needs", "premise"),
              tc.Evidence(tc.Ref("rule:r"), NOW, method="derive@1", derived_from=(premise.id,)))
    told = memory.told(tc.Ref("person:user"), "name", "Jacob", at=NOW - timedelta(hours=2))
    awareness.seed(aware_old.id)
    awareness.spread()

    report = memory.forget_stale(now=NOW)

    assert old.id not in mind._claims, "stale, unsalient perception is actually gone"
    assert aware_old.id in mind._claims, "what is aware is kept"
    assert premise.id in mind._claims and report.kept_because_depended_on >= 1
    assert told.id in mind._claims and report.kept_because_protected >= 1
    assert report.claims_forgotten == 1 and report.ms >= 0.0


def test_spatial_memory_answers_where_rather_than_what():
    class Control:
        def __init__(self, box, section):
            self.box = box
            self.section = section

    mind = tc.Store()
    mind.put(tc.Ref("ui:dock/button/Terminal"), Control((10, 300, 40, 40), "Dock"))
    mind.put(tc.Ref("ui:window/Files/row#1"), Control((500, 200, 200, 20), "Files"))
    near = perceived(mind, "ui:dock/button/Terminal", "label", "Terminal")
    far = perceived(mind, "ui:window/Files/row#1", "reads", "notes.txt")
    frames = Frames(mind)
    memory = Memory(mind, frames=frames)

    in_dock = {r.id for r in memory.here(window="Dock")}
    by_pointer = {r.id for r in memory.here(near=(20, 310), slack=60)}

    assert near.id in in_dock and far.id not in in_dock
    assert near.id in by_pointer and far.id not in by_pointer


def test_procedural_recall_ranks_skills_by_how_well_a_cue_fits():
    class Proc:
        def __init__(self, id, act):
            self.id, self.act = id, act

    mind = tc.Store()
    memory = Memory(mind)
    procs = [Proc("list_dir", "list"), Proc("git_commit_everything", "git_commit"), Proc("delete_path", "delete")]

    ranked = memory.skills("commit everything in the repo", procs, k=2)

    assert ranked[0][0].id == "git_commit_everything"
    assert ranked[0][1].kind == "similarity"


def test_attending_carries_a_little_over_between_cycles():
    mind = tc.Store()
    memory = Memory(mind, MemoryPolicy(working_budget=6))
    first = perceived(mind, "ui:field#1", "value", "")
    second = perceived(mind, "ui:field#2", "value", "typed")

    memory.attend([first.id], fade=False)
    assert first.id in memory.working.aware_ids()

    memory.attend([second.id])
    assert second.id in memory.working.aware_ids()
    assert memory.working.salience(second.id) > memory.working.salience(first.id)


def test_episodes_live_in_their_own_scope_so_they_do_not_pollute_the_world():
    mind = tc.Store()
    memory = Memory(mind)
    rec = perceived(mind, "ui:x", "label", "X")

    episode = memory.encode([rec], summary="looked at X")

    assert all(r.claim.scope == EPISODIC for r in mind.claims(subject=episode.ref))
    assert mind.claims(subject=tc.Ref("ui:x"), scope=None), "the world claim is untouched"
