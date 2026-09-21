"""Data-only configuration artifacts for parameter-free operations."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile


def validated_config(config, keys, defaults=None):
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise TypeError('Operation config must be a JSON object')
    if any(not isinstance(key, str) for key in config):
        raise TypeError('Configuration keys must be strings')
    unknown = set(config) - set(keys)
    if unknown:
        raise ValueError(f'Unknown configuration fields: {sorted(unknown)}')
    try:
        serialized = json.dumps(config, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError('Configuration must contain finite JSON data') from error
    if json.loads(serialized) != config:
        raise ValueError("Configuration must use JSON types")
    result = deepcopy(defaults or {})
    result.update(json.loads(serialized))
    return result


class ConfigOperationMixin:
    config_keys = frozenset()
    config_defaults = {}

    def __init__(self, config=None):
        self.config = validated_config(config, self.config_keys, self.config_defaults)

    def configuration(self):
        return deepcopy(self.config)

    @classmethod
    def _config_identity(cls):
        return f'{cls.__module__}.{cls.__qualname__}'

    def save_pretrained(self, directory):
        directory = Path(directory)
        if directory.is_symlink():
            raise ValueError('Artifact directory must not be a symlink')
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / 'tensorcode_config.json'
        if target.is_symlink():
            raise ValueError('Artifact manifest must not be a symlink')
        payload = {'format': 'tensorcode.operation', 'version': 1,
                   'operation': self._config_identity(), 'config': self.configuration()}
        fd, temporary = tempfile.mkstemp(dir=directory, prefix='.config-')
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(payload, stream, allow_nan=False, indent=2)
                stream.write('\n')
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return directory

    @classmethod
    def from_pretrained(cls, source, *, revision=None, local_files_only=False,
                        cache_dir=None, token=None):
        path = Path(source)
        remote = not path.exists()
        if remote:
            if path.is_absolute() or str(source).startswith('.'):
                raise FileNotFoundError(path)
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(str(source), revision=revision,
                        local_files_only=local_files_only, cache_dir=cache_dir,
                        token=token, allow_patterns=['tensorcode_config.json']))
        manifest = path / 'tensorcode_config.json'
        if not remote and (path.is_symlink() or manifest.is_symlink()):
            raise ValueError('Local configuration artifacts must not be symlinks')
        with manifest.open() as stream:
            data = json.load(stream)
        if (data.get('format') != 'tensorcode.operation' or data.get('version') != 1
                or data.get('operation') != cls._config_identity()):
            raise ValueError('Configuration artifact identity or format mismatch')
        return cls(data['config'])
