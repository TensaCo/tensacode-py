"""Private conversation state shared with one owned Chatbot."""
import json
import threading
from pathlib import Path
from functools import wraps


def _locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


class ChatSession:
    """Conversation evidence owned by one session, never by a weight artifact."""

    def __init__(self, model):
        self.model = model
        self._lock = threading.RLock()
        self.history = []
        self.last_result = None
        self.cognition = model._new_cognitive_session()

    @_locked
    def __call__(self, value):
        return self.model._respond(value, self)

    @_locked
    def reset(self):
        self.history.clear()
        self.last_result = None
        self.cognition = self.model._new_cognitive_session()

    @_locked
    def new_episode(self):
        if self.cognition is None:
            raise ValueError('Cognitive episodes are not configured')
        state = self.cognition.new_episode()
        self.history.clear()
        self.last_result = None
        return state

    @_locked
    def rebuild_memory(self):
        if self.cognition is None or self.cognition.memory is None:
            raise ValueError('Episodic memory is not configured')
        with self.model._lock:
            self.cognition.memory.rebuild_index()

    @_locked
    def save(self, path):
        value = {'format': 2 if self.cognition is not None else 1,
                 'model': self.model.fingerprint, 'history': self.history}
        if self.cognition is not None:
            value['cognition'] = self.cognition.snapshot()
        Path(path).write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')

    @_locked
    def load(self, path):
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        expected_format = 2 if self.cognition is not None else 1
        if value.get('format') != expected_format or value.get('model') != self.model.fingerprint:
            raise ValueError('Session is incompatible with this model configuration')
        history = value.get('history')
        if not isinstance(history, list) or len(history) > self.model.config['max_turns'] * 2:
            raise ValueError('Invalid session history capacity')
        for i, item in enumerate(history):
            if (not isinstance(item, dict) or set(item) != {'source_id', 'role', 'text'}
                    or item['role'] != ('user' if i % 2 == 0 else 'assistant')
                    or not isinstance(item['text'], str) or not isinstance(item['source_id'], str)):
                raise ValueError('Invalid session evidence')
        if len(history) % 2:
            raise ValueError('Session contains an incomplete turn')
        try:
            ids = [int(item['source_id'].removeprefix('turn-')) for item in history]
        except ValueError as exc:
            raise ValueError('Invalid session source IDs') from exc
        if (any(item['source_id'] != f'turn-{number}' or number < 0
                for item, number in zip(history, ids))
                or (ids and (ids[0] % 2 or ids != list(range(ids[0], ids[0] + len(ids)))))):
            raise ValueError('Invalid session source IDs')
        cognitive = None
        if self.cognition is not None:
            from ..cognition.session import CognitiveSession
            with self.model._lock:
                cognitive = CognitiveSession.from_snapshot(value.get('cognition'),
                                                            investigator=self.model.investigator)
            if cognitive.state.max_records != self.cognition.state.max_records:
                raise ValueError('Session cognitive capacity differs from model configuration')
            if cognitive.snapshot()['policy'] != self.cognition.snapshot()['policy']:
                raise ValueError('Session cognitive policy differs from model configuration')
        self.history = history
        self.cognition = cognitive
        self.last_result = None
        return self
