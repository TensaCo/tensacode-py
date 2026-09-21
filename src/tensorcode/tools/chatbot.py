"""A local trainable encoder/workspace/decoder model with independent sessions."""
from __future__ import annotations

import hashlib
import json
import threading
from functools import wraps
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoModelForSeq2SeqLM, GenerationConfig, PreTrainedTokenizerFast

from .._internal.pretrained import PretrainedTool
from .._internal.workspace import Workspace
from ..ops.base import Operation
from ..ops.llm.decode import Decode
from ..ops.vec.encode import SequenceEncoder


class _Objective(Operation):
    replayable = True

    def __init__(self, owner):
        import weakref
        self._owner = weakref.ref(owner)

    def forward(self, value, *, context=None):
        if context:
            raise ValueError('Objective does not consume context')
        return self._owner().loss_batch(value['inputs'], value['targets']).clone()

    def parameters(self, recurse=True):
        return self._owner().parameters(recurse=recurse)

    def configuration(self):
        return {'operation': 'tensorcode.tools.chatbot.Objective',
                'model': self._owner().configuration()}


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
            from ..runtime.cognition import CognitiveSession
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


class Chatbot(PretrainedTool):
    """Complete owned model. Construction initializes; loading supplies weights.

    No provider, callback, implicit download, or semantic seed is required.
    ``from_foundation`` is an explicit training bootstrap: its new workspace has
    random weights and must be trained before claiming useful behavior.
    """

    training_inputs_include_targets = True

    def __init__(self, config):
        config = dict(config)
        cognitive = config.get('cognition')
        if cognitive is not None:
            if not isinstance(cognitive, dict) or not isinstance(cognitive.get('investigator'), dict):
                raise ValueError('cognition requires a complete investigator configuration')
            nested = cognitive['investigator']
            if not isinstance(nested.get('generator'), dict) or 'verifier_config' not in nested:
                raise ValueError('cognition requires owned proposal generator and verifier')
            if nested['generator'].get('cognition') is not None:
                raise ValueError('Recursive cognitive generator configurations are not supported')
            count = cognitive.get('proposal_count', 3)
            if type(count) is not int or count < 1:
                raise ValueError('proposal_count must be positive')
            abstention = cognitive.get('abstention_text', 'I do not have enough supported evidence to answer.')
            if not isinstance(abstention, str) or not abstention.strip():
                raise ValueError('abstention_text must be nonempty')
        # Transformers configs contain integer id2label keys; normalize their
        # documented JSON representation before strict artifact validation.
        config['foundation_config'] = json.loads(json.dumps(config['foundation_config']))
        if cognitive is not None:
            config['cognition'] = json.loads(json.dumps(cognitive))
        for key, default in [('max_new_tokens', 64), ('max_input_tokens', 512),
                             ('max_target_tokens', 128), ('max_turns', 16)]:
            config.setdefault(key, default)
            if type(config[key]) is not int or config[key] <= 0:
                raise ValueError(f'{key} must be a positive integer')
        config.setdefault('workspace', {'slots': 8, 'steps': 2})
        config.setdefault('tokenizer_special_tokens', {})
        config.setdefault('memory_mode', 'contextualized_evidence')
        if config['memory_mode'] not in ('contextualized_evidence', 'slots'):
            raise ValueError('Unknown memory_mode')
        super().__init__(config)
        architecture = dict(config['foundation_config'])
        model_type = architecture.pop('model_type')
        foundation_config = AutoConfig.for_model(model_type, **architecture)
        self.foundation = AutoModelForSeq2SeqLM.from_config(foundation_config)
        if config.get('untied_lm_head', False):
            head = self.foundation.get_output_embeddings()
            head.weight = torch.nn.Parameter(head.weight.detach().clone())
        if 'generation_config' in config:
            self.foundation.generation_config = GenerationConfig.from_dict(config['generation_config'])
        self.config['generation_config'] = self.foundation.generation_config.to_dict()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_str(config['tokenizer_json']),
            **config['tokenizer_special_tokens'])
        if self.tokenizer.pad_token_id is None:
            raise ValueError('Tokenizer requires an explicit padding token')
        self.workspace = Workspace(foundation_config.d_model, **config['workspace'])
        self.encoder = SequenceEncoder(self.foundation, self.tokenizer,
                                       max_tokens=config['max_input_tokens'])
        self.decoder = Decode(self.foundation)
        self.memory_projection = torch.nn.Linear(foundation_config.d_model, foundation_config.d_model)
        self.memory_gate = torch.nn.Parameter(torch.tensor(0.01))
        self.investigator = None
        if cognitive is not None:
            # Investigator's proposal generator is a rankless Chatbot. Delay the
            # import to avoid cyclic module imports, and reject recursion above.
            from .investigator import Investigator
            self.investigator = Investigator(cognitive['investigator'])
        self._lock = threading.RLock()
        self._session = ChatSession(self)
        self.training_operation = _Objective(self)
        self.objective = self.training_operation

    def configuration(self):
        config = super().configuration()
        config['generation_config'] = json.loads(self.foundation.generation_config.to_json_string())
        if self.investigator is not None:
            config['cognition']['investigator'] = self.investigator.configuration()
        return config

    @property
    def capabilities(self):
        return {'language_realization': True, 'persistent_cognitive_state': self.investigator is not None,
                'hypothesis_generation': self.investigator is not None,
                'source_verification': self.investigator is not None,
                'abstention_policy': 'authored' if self.investigator is not None else None}

    def _new_cognitive_session(self):
        if self.investigator is None:
            return None
        from ..runtime.cognition import CognitiveSession
        from ..runtime.cognitive_state import CognitiveState
        state = CognitiveState(max_records=self.config['cognition'].get('max_records', 256))
        return CognitiveSession(self.investigator, state=state, policy=self.config['cognition'].get('policy'),
                                memory=self.config['cognition'].get('memory'))

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.configuration(), sort_keys=True).encode()).hexdigest()

    @property
    def last_result(self):
        return self._session.last_result

    @property
    def cognitive_state(self):
        return self._session.cognition.state if self._session.cognition is not None else None

    @property
    def history(self):
        return tuple(dict(item) for item in self._session.history)

    def new_session(self):
        return ChatSession(self)

    def reset_session(self):
        self._session.reset()

    def new_episode(self):
        return self._session.new_episode()

    def rebuild_memory(self):
        self._session.rebuild_memory()

    def save_session(self, path):
        self._session.save(path)

    def load_session(self, path):
        self._session.load(path)
        return self

    def operation_bindings(self):
        result = super().operation_bindings()
        result.update({'encoder': self.encoder, 'decoder': self.decoder,
                       'objective': self.objective})
        if self.investigator is not None:
            result.update({f'investigator.{key}': operation for key, operation
                           in self.investigator.operation_bindings().items()})
        return result

    @classmethod
    def from_foundation(cls, repo, *, revision=None, local_files_only=False, **options):
        """Initialize from external seq2seq weights plus a new, untrained workspace."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision,
                                                  local_files_only=local_files_only)
        model = AutoModelForSeq2SeqLM.from_pretrained(repo, revision=revision,
                                                     local_files_only=local_files_only)
        if not hasattr(tokenizer, 'backend_tokenizer'):
            raise ValueError('Foundation requires a serializable fast tokenizer')
        resolved = getattr(model.config, '_commit_hash', None) or revision
        if not Path(repo).is_dir() and not resolved:
            raise ValueError('Foundation provenance requires a resolved Hub revision')
        foundation_config = model.config.to_dict()
        # Preserve both forward semantics (including T5 output scaling) and
        # actual parameter aliases, which upstream metadata can disagree about.
        inputs, outputs = model.get_input_embeddings(), model.get_output_embeddings()
        untied_head = inputs is not None and outputs is not None and inputs.weight is not outputs.weight
        config = {'foundation_config': foundation_config,
                  'untied_lm_head': untied_head,
                  'generation_config': model.generation_config.to_dict(),
                  'tokenizer_json': tokenizer.backend_tokenizer.to_str(),
                  'tokenizer_special_tokens': {key: str(value) for key, value in
                                               tokenizer.special_tokens_map.items()
                                               if isinstance(value, str)},
                  'foundation': {'repository': str(repo), 'revision': resolved,
                                 'workspace_initialization': 'random'}, **options}
        result = cls(config)
        result.foundation.load_state_dict(model.state_dict())
        return result

    @classmethod
    def from_cognitive_foundations(cls, language_repo, encoder_repo, generator_repo, verifier_repo, *,
                                  language_revision=None, encoder_revision=None,
                                  generator_revision=None, verifier_revision=None,
                                  verifier_labels, local_files_only=False,
                                  investigator_options=None, cognitive_policy=None, **options):
        """Explicit bootstrap of all owned components; never a hidden download."""
        from .investigator import Investigator
        if 'cognition' in options:
            raise ValueError('Bootstrap cognition is supplied through its component arguments')
        language = cls.from_foundation(language_repo, revision=language_revision,
                                       local_files_only=local_files_only, **options)
        investigator = Investigator.from_foundations(
            encoder_repo, generator_repo, verifier_repo,
            encoder_revision=encoder_revision, generator_revision=generator_revision,
            verifier_revision=verifier_revision, verifier_labels=verifier_labels,
            local_files_only=local_files_only, **(investigator_options or {}))
        config = language.configuration()
        config['cognition'] = {'investigator': investigator.configuration()}
        if cognitive_policy is not None:
            config['cognition']['policy'] = cognitive_policy
        result = cls(config)
        missing, unexpected = result.load_state_dict(language.state_dict(), strict=False)
        if unexpected or any(not key.startswith('investigator.') for key in missing):
            raise RuntimeError('Language bootstrap state does not match the complete architecture')
        result.investigator.load_state_dict(investigator.state_dict())
        return result

    def encode_workspace(self, inputs, *, workspace_ablation=None):
        encoded = self.encoder(inputs)
        state = self.workspace(encoded['encoded'], encoded['mask'].bool())
        if self.config['memory_mode'] == 'contextualized_evidence':
            # Evidence remains source-aligned, but slot relations can revise each
            # token representation before it becomes the decoder memory.
            tokens = encoded['encoded']
            slots = state['conditioning']
            assignment = torch.softmax(tokens @ slots.transpose(-1, -2) /
                                       (tokens.shape[-1] ** 0.5), dim=-1)
            update = self.memory_projection(assignment @ slots)
            state = dict(state, conditioning=tokens + self.memory_gate * update,
                         mask=encoded['mask'])
        if workspace_ablation == 'bypass':
            state = dict(state, conditioning=encoded['encoded'], mask=encoded['mask'])
        elif workspace_ablation == 'zero':
            state = dict(state, conditioning=torch.zeros_like(state['conditioning']))
        elif workspace_ablation is not None:
            raise ValueError('Unknown workspace ablation')
        return state

    def loss_batch(self, inputs, targets, *, workspace_ablation=None):
        """Teacher-forced cross entropy; targets never enter the input encoder."""
        if (not inputs or len(inputs) != len(targets)
                or any(not isinstance(item, str) for item in [*inputs, *targets])):
            raise ValueError('Expected equally sized nonempty text input and target batches')
        state = self.encode_workspace(list(inputs), workspace_ablation=workspace_ablation)
        labels = self.tokenizer(list(targets), padding=True, truncation=True,
                                max_length=self.config['max_target_tokens'],
                                return_tensors='pt')['input_ids'].to(next(self.parameters()).device)
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
        if not (labels != -100).any():
            raise ValueError('Targets contain no supervised tokens')
        return self.decoder(dict(state, labels=labels))['loss']

    @_locked
    def generate_batch(self, inputs, *, workspace_ablation=None):
        if not inputs or any(not isinstance(item, str) for item in inputs):
            raise ValueError('Expected nonempty text batch')
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            with torch.no_grad():
                state = self.encode_workspace(list(inputs), workspace_ablation=workspace_ablation)
                tokens = self.decoder(state, context={'max_new_tokens': self.config['max_new_tokens'],
                                                       'do_sample': False})
            return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        finally:
            for module, training in modes:
                module.training = training

    def forward(self, value, *, context=None):
        if context:
            raise ValueError('Use an independent new_session() for conversation state')
        return self._session(value)

    def _respond(self, value, session):
        if session.cognition is not None:
            return self._respond_cognitive(value, session)
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Chatbot expects nonempty text')
        # Source IDs increase even when old turns are evicted.
        next_id = (int(session.history[-1]['source_id'].split('-')[-1]) + 1
                   if session.history else 0)
        pending = {'source_id': f'turn-{next_id}', 'role': 'user', 'text': value}
        evidence = [*session.history, pending]
        prompt = '\n'.join(f"{item['role']}: {item['text']}" for item in evidence)
        answer = self.generate_batch([prompt])[0]
        output = {'source_id': f'turn-{next_id + 1}', 'role': 'assistant', 'text': answer}
        # Commit only once encoding, workspace computation, and decoding succeed.
        receipt = {'text': answer, 'source_ids': [item['source_id'] for item in evidence],
                               'evidence': [dict(item) for item in evidence],
                               'input_truncated': len(self.tokenizer(prompt)['input_ids']) >
                                                  self.config['max_input_tokens']}
        session.history = [*evidence, output][-2 * self.config['max_turns']:]
        session.last_result = receipt
        return answer

    @_locked
    def _respond_cognitive(self, value, session):
        from ..runtime.cognitive_state import Evidence
        if isinstance(value, str):
            value = {'question': value}
        if not isinstance(value, dict) or not isinstance(value.get('question'), str) or not value['question'].strip():
            raise ValueError('Cognitive Chatbot expects a question and optional explicitly sourced evidence')
        if set(value) - {'question', 'evidence', 'revisions', 'remove_evidence'}:
            raise ValueError('Unknown cognitive input fields')
        proposed = session.cognition.fork(copy_memory=True)
        rows = value.get('evidence', [])
        if not isinstance(rows, list):
            raise ValueError('evidence must be a list of explicitly sourced records')
        evidence = []
        for row in rows:
            if not isinstance(row, dict) or set(row) - {'id', 'source_id', 'text'}:
                raise ValueError('Malformed evidence record')
            evidence.append(Evidence(id=row.get('id', row.get('source_id')), text=row.get('text'),
                                     source_id=row.get('source_id')))
        if evidence:
            proposed.ingest(evidence)
        revisions = value.get('revisions', [])
        if not isinstance(revisions, list):
            raise ValueError('revisions must be a list')
        for row in revisions:
            if not isinstance(row, dict) or set(row) - {'evidence_id', 'text', 'source_id'}:
                raise ValueError('Malformed evidence revision')
            proposed.revise_evidence(row.get('evidence_id'), row.get('text'), row.get('source_id'))
        removed = value.get('remove_evidence', [])
        if not isinstance(removed, list) or any(not isinstance(item, str) for item in removed):
            raise ValueError('remove_evidence must contain logical evidence IDs')
        for evidence_id in removed:
            proposed.remove_evidence(evidence_id)
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            with torch.no_grad():
                interpretation = proposed.investigate(value['question'],
                    count=self.config['cognition'].get('proposal_count', 3))
            prompt, visible_evidence, truncation = self._realization_input(value['question'], interpretation)
            decoded = self.generate_batch([prompt])[0]
            checks = []
            supported = False
            if not interpretation['abstained'] and decoded.strip() and visible_evidence:
                with torch.no_grad():
                    checks = self.investigator.verifier.verify(decoded, [
                        {'source_id': row['id'], 'text': row['text']} for row in visible_evidence])
                    supported = (not any(row.get('input_truncated', False) for row in checks)
                                 and proposed.policy.accepts([row['distribution'] for row in checks]))
                    if supported and truncation:
                        full_checks = self.investigator.verifier.verify(decoded, [
                            {'source_id': row['id'], 'text': row['text']} for row in interpretation['evidence']])
                        supported = (not any(row.get('input_truncated', False) for row in full_checks)
                                     and proposed.policy.accepts([row['distribution'] for row in full_checks]))
                        checks.extend(dict(row, scope='full-active-evidence') for row in full_checks)
            abstained = interpretation['abstained'] or not supported
            answer = (self.config['cognition'].get('abstention_text',
                       'I do not have enough supported evidence to answer.') if abstained else decoded)
            next_id = int(session.history[-1]['source_id'].split('-')[-1]) + 1 if session.history else 0
            pending = {'source_id': f'turn-{next_id}', 'role': 'user', 'text': value['question']}
            output = {'source_id': f'turn-{next_id + 1}', 'role': 'assistant', 'text': answer}
            retained = []
            if proposed.memory is not None:
                # Explicit authored retention: supplied source evidence, never
                # questions, generated hypotheses, or assistant utterances.
                logical_ids = {row.id for row in evidence} | {row['evidence_id'] for row in revisions}
                active = proposed.snapshot()['active_evidence']
                remembered = {row['evidence']['id'] for row in proposed.memory.snapshot()['records']}
                for logical_id in sorted(logical_ids):
                    if logical_id in active:
                        actual_id = active[logical_id]
                        if actual_id not in remembered:
                            proposed.remember(actual_id, question=value['question'])
                            retained.append(actual_id)
            receipt = {'text': answer, 'cognition': interpretation,
                       'response_proposal': {'text': decoded, 'origin': 'model_generation',
                                             'epistemic_status': 'unverified_proposal'},
                       'retained_evidence_ids': retained,
                       'retention_policy': 'authored: retain explicitly supplied active sources after successful turn',
                       'abstention_enforced': bool(abstained),
                       'realization_verifications': checks,
                       'realization_sources': visible_evidence,
                       'source_truncation': truncation,
                       'realization_semantics': 'Authored screening of model NLI scores; not a factual guarantee',
                       'capabilities': self.capabilities,
                       'input_truncated': len(self.tokenizer(prompt)['input_ids']) > self.config['max_input_tokens']}
        finally:
            for module, training in modes:
                module.training = training
        # A failed investigation or decoder never commits evidence or dialogue.
        session.cognition = proposed
        session.history = [*session.history, pending, output][-2 * self.config['max_turns']:]
        session.last_result = receipt
        return answer

    def _realization_input(self, question, interpretation):
        """Budget source text before audit metadata, retaining exact text prefixes."""
        selected = next((row for row in interpretation['candidates']
                         if row['id'] == interpretation['selected_id']), None)
        hypothesis = selected['text'] if selected is not None else 'No supported selection'
        prompt = (f'Answer using only the evidence below. Preserve uncertainty.\nQuestion: {question}\n'
                  f'Selected hypothesis (not an observation): {hypothesis}\nEvidence:\n')
        visible, truncated = [], []
        sources = list(interpretation['evidence'])
        if selected is not None:
            support = {row.get('evidence_id'): row['distribution']['support']
                       for row in selected.get('verifications', [])}
            sources.sort(key=lambda row: support.get(row['id'], 0), reverse=True)
        budget = self.config['max_input_tokens']
        for row in sources:
            prefix = f"[{row['source_id']}] "
            lo, hi = 0, len(row['text'])
            while lo < hi:
                middle = (lo + hi + 1) // 2
                size = len(self.tokenizer(prompt + prefix + row['text'][:middle] + '\n')['input_ids'])
                if size <= budget:
                    lo = middle
                else:
                    hi = middle - 1
            if lo and row['text'][:lo].strip():
                text = row['text'][:lo]
                prompt += prefix + text + '\n'
                visible.append(dict(row, text=text))
            if lo != len(row['text']):
                truncated.append({'evidence_id': row['id'], 'source_id': row['source_id'],
                                  'included_characters': lo, 'original_characters': len(row['text'])})
        return prompt, visible, truncated
