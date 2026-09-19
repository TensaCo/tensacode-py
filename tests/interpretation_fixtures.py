"""Explicit supplied meanings for tests that isolate interpretation selection.

These fixtures provide the intended modifier attachment; no language accuracy is
claimed or parser correction installed in production.
"""
from tensorcode.agent.understand import Act, Sentence
from tensorcode.language import Entity, Frame, Request


def project_sentence(name):
    entity = Entity('description', 'python project', {
        'noun': 'project', 'quality': 'python', 'modifiers': (('compound', 'python'),),
        'name': name,
    })
    frame = Frame('make', {'object': entity}, {'mood': 'imperative'})
    return Sentence(f'supplied project meaning: {name}', (name,), None,
                    (Act('request', Request(frame), frame),))
