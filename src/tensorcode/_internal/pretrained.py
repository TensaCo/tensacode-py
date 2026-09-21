"""Versioned, non-executable model artifacts and explicit Hugging Face transport."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import uuid

import torch


class PretrainedTool(torch.nn.Module):
    """Base for owned trainable tools. Construction never performs Hub I/O.

    Subclasses construct their complete architecture from a JSON configuration.
    Only registered parameters/buffers and explicitly saved assets are published;
    conversation and optimizer state belong to separate persistence APIs.
    """

    # Stateful tool calls are not replayable; owned objective operations opt in.
    replayable = False
    artifact_format = 'tensorcode.pretrained'
    artifact_version = 1

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = self._validated_config(config)

    @staticmethod
    def _validated_config(config):
        if not isinstance(config, dict):
            raise ValueError('model config must be a JSON object')
        try:
            encoded = json.dumps(config, allow_nan=False, sort_keys=True)
            result = json.loads(encoded)
        except (TypeError, ValueError) as exc:
            raise ValueError('model config must contain finite JSON values') from exc
        if result != config:
            raise ValueError('model config must use JSON types and string keys')
        return deepcopy(result)

    @classmethod
    def _tool_identity(cls):
        return f'{cls.__module__}.{cls.__qualname__}'

    def configuration(self) -> dict[str, Any]:
        """Return an independent JSON configuration, without learned weights."""
        return self._validated_config(self.config)

    def operation_bindings(self) -> dict[str, Any]:
        """Stable paths of registered operations; compositions may extend this."""
        from tensorcode.ops.base import Operation
        return {name: module for name, module in self.named_modules()
                if name and (isinstance(module, Operation) or
                             getattr(module, 'replayable', False))}

    def _save_pretrained_assets(self, directory: Path) -> None:
        """Subclass hook to save local tokenizer/encoder assets into staging."""

    @classmethod
    def _load_pretrained_config(cls, config: dict, directory: Path) -> dict:
        """Subclass hook to bind saved assets locally before construction."""
        return config

    def save_pretrained(self, directory: str | os.PathLike) -> Path:
        """Stage a complete model then replace the destination with rollback.

        Existing unrelated files are preserved. Concurrent writers to the same
        directory are unsupported; readers should not race a directory replacement.
        """
        from safetensors.torch import save_model
        target = Path(directory).absolute()
        if target.is_symlink() or (target.exists() and not target.is_dir()):
            raise ValueError('model destination must be a directory, not a symlink')
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f'.{target.name}.stage-', dir=target.parent))
        backup = target.with_name(f'.{target.name}.backup-{uuid.uuid4().hex}')
        moved = False
        try:
            if target.exists():
                shutil.copytree(target, stage, dirs_exist_ok=True, symlinks=True)
            # Never follow preexisting artifact symlinks while writing staging.
            for name in ('model.safetensors', 'tensorcode_config.json'):
                path = stage / name
                if path.is_symlink():
                    path.unlink()
            self._save_pretrained_assets(stage)
            manifest = {'format': self.artifact_format, 'version': self.artifact_version,
                        'tool': self._tool_identity(), 'config': self.configuration()}
            (stage / 'tensorcode_config.json').write_text(
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
            save_model(self, str(stage / 'model.safetensors'))
            if target.exists():
                target.rename(backup)
                moved = True
            try:
                stage.rename(target)
            except BaseException:
                if moved:
                    backup.rename(target)
                    moved = False
                raise
            if moved:
                shutil.rmtree(backup)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        return target

    @classmethod
    def from_pretrained(cls, repo_id_or_path: str | os.PathLike, *, revision=None,
                        local_files_only=False, cache_dir=None, token=None,
                        device='cpu', **kwargs):
        """Load this known class from a local directory or pinned Hub snapshot."""
        from safetensors.torch import load_model
        path = Path(repo_id_or_path).expanduser()
        if not path.is_dir():
            if isinstance(repo_id_or_path, os.PathLike) or path.is_absolute() or str(repo_id_or_path).startswith('.'):
                raise FileNotFoundError(f'local model directory not found: {path}')
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(repo_id=str(repo_id_or_path), revision=revision,
                        local_files_only=local_files_only, cache_dir=cache_dir, token=token))
        try:
            manifest = json.loads((path / 'tensorcode_config.json').read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise ValueError(f'invalid TensorCode model manifest in {path}') from exc
        if not isinstance(manifest, dict):
            raise ValueError('model manifest must be an object')
        for name, expected in [('format', cls.artifact_format), ('version', cls.artifact_version),
                               ('tool', cls._tool_identity())]:
            if manifest.get(name) != expected or type(manifest.get(name)) is not type(expected):
                raise ValueError(f'incompatible model {name}: expected {expected!r}')
        config = cls._validated_config(manifest.get('config'))
        config = cls._load_pretrained_config(config, path)
        model = cls(config, **kwargs)
        cls._restore_artifact_dtypes(model, path / 'model.safetensors')
        load_model(model, str(path / 'model.safetensors'), strict=True, device='cpu')
        model.to(device)
        model.eval()
        return model

    @staticmethod
    def _restore_artifact_dtypes(model, filename: Path) -> None:
        """Restore tensor storage dtypes before copy-loading, including tied views.

        Safetensors carries each stored dtype in its header. Reading a CPU tensor
        here is memory mapped; we retain only its dtype, not a second weight copy.
        """
        from safetensors import safe_open
        state = model.state_dict(keep_vars=True)
        with safe_open(filename, framework='pt', device='cpu') as artifact:
            aliases = artifact.metadata() or {}
            dtypes = {name: artifact.get_tensor(name).dtype for name in artifact.keys()}
        tensors = []
        for name, tensor in state.items():
            dtype = dtypes.get(name, dtypes.get(aliases.get(name)))
            if dtype is None:
                continue  # Strict load_model reports missing keys below.
            storage = tensor.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes(), tensor.dtype, tensor.device)
            if not storage.nbytes():
                key = (*key, id(tensor))
            tensors.append((tensor, dtype, key, tensor.shape, tensor.stride(), tensor.storage_offset()))
        converted = {}
        for tensor, dtype, key, shape, stride, offset in tensors:
            if tensor.dtype == dtype:
                continue
            if key not in converted:
                flat = tensor.detach().as_strided((key[1] // tensor.element_size(),), (1,), 0)
                converted[key] = flat.to(dtype=dtype)
            storage = converted[key]
            if storage.dtype != dtype:
                raise ValueError('shared model storage has incompatible artifact dtypes')
            tensor.data = storage.as_strided(shape, stride, offset)

    def push_to_hub(self, repo_id: str, *, private=False, revision=None,
                    token=None, commit_message='Upload TensorCode model'):
        """Explicitly publish model artifacts; no sessions or optimizer state."""
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type='model')
        with tempfile.TemporaryDirectory(prefix='tensorcode-publish-') as directory:
            self.save_pretrained(Path(directory) / 'model')
            return api.upload_folder(repo_id=repo_id, repo_type='model',
                folder_path=str(Path(directory) / 'model'), revision=revision,
                commit_message=commit_message)
