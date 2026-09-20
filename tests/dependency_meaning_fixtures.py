"""Explicit teaching fixtures for downstream mechanisms, not language accuracy.

A test supplies the communicative label and chooses a syntax family. Independent
synthetic lexical variants preserve that supplied structure. Their parses are
teaching fixtures, not claims that a parser inferred these artificial utterances.
"""
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace

from tensorcode.agent.understand import Act, SentenceAlternative
from tensorcode.language.deps_semantics import ProvisionalMeaning
from tensorcode.outcomes import Unknown


def teach_speech_family(agent, text, label, *, matches=lambda value: True):
    from tensorcode.agent.speech_act_learning import (
        retain_speech_act_example, fit_speech_act_model, admit_speech_act_model,
    )
    families = [act.meaning for sentence in agent.reader.read(text)
                for alternative in sentence.alternatives for act in alternative.acts
                if type(act.meaning) is ProvisionalMeaning and matches(act.meaning)]
    assert families, ('no neutral syntax family matching explicit fixture', text)
    # This is an explicit authored test choice, not a production selection policy.
    family = families[0]
    index = family.root - 1
    old_word, old_lemma = family.words[index], family.lemmas[index]
    records = []
    for variant in range(3):
        replacement = f'fixturelexeme{variant}'
        substitutions = {old_lemma: replacement, old_word: (
            replacement if old_word == old_lemma else f'Fixturelexeme{variant}')}
        def substitute(value):
            if type(value) is str:
                return substitutions.get(value, value)
            if type(value) is dict:
                return {key: substitute(item) for key, item in value.items()}
            if type(value) in (list, tuple):
                return type(value)(substitute(item) for item in value)
            if is_dataclass(value):
                return replace(value, **{field.name: substitute(getattr(value, field.name)) for field in fields(value)})
            return deepcopy(value)
        neutral = replace(family, words=tuple(substitute(word) for word in family.words),
                          lemmas=tuple(substitute(word) for word in family.lemmas), frame=substitute(family.frame))
        source_text = ' '.join(neutral.words)
        anchors, position = [], 0
        for number, word in enumerate(neutral.words, 1):
            anchors.append({'index': number, 'token': word, 'char_span': (position, position + len(word))})
            position += len(word) + 1
        source = agent.interpretations.add_source(source_text, modality='text', provider='authored-speech-teaching-fixture')
        group = agent.interpretations.create_group(source.id)
        alternative = SentenceAlternative(None, (Act('unresolved', neutral, None),),
            provenance='authored neutral syntax teaching fixture', metadata={
                'tokens': neutral.words, 'token_anchors': tuple(anchors), 'tags': neutral.tags,
                'lemmas': neutral.lemmas, 'heads': dict(neutral.heads), 'labels': dict(neutral.labels),
                'syntax_complete': True, 'sentence_span': (0, len(source_text))})
        candidate = agent.interpretations.propose(group.id, alternative)
        record = retain_speech_act_example(agent, group.id, candidate.id, 0, label,
            basis=('explicit fixture communicative label over authored lexical variants',))
        assert not isinstance(record, Unknown), record
        records.append(record)
    model = fit_speech_act_model(agent, records[:2], records[2:])
    assert not isinstance(model, Unknown), model
    handle = admit_speech_act_model(agent, model, reason='explicit downstream mechanism teaching fixture')
    assert not isinstance(handle, Unknown), handle
    agent.speech_act_model = handle
    return family


def neutral_fixture(frame, words, tags, lemmas, heads, labels):
    """An authored frame inside actual fixture syntax, with no speech-act label."""
    root = next(index for index, head in heads.items() if head == 0)
    return ProvisionalMeaning(frame, tuple(words), tuple(tags), tuple(lemmas),
                              tuple(sorted(heads.items())), tuple(sorted(labels.items())), root)
