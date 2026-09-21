"""Persistent memory with caller-defined retrieval semantics."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class MemoryRecord:
    """One stored value and its stable source identity."""

    source_id: str
    kind: str
    value: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id must be a non-empty string")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("kind must be a non-empty string")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class MemorySearch:
    """Explicit operand passed to a caller-supplied retrieval policy."""

    query: Any
    candidates: tuple[MemoryRecord, ...]
    limit: int


class _MemoryTransaction:
    def __init__(self, memory: JsonMemory) -> None:
        self._memory = memory
        self._records = [memory._snapshot_record(record) for record in memory._records]
        self._next_id = memory._next_id
        self._closed = False

    def append(
        self,
        value: Any,
        *,
        kind: str,
        source_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemoryRecord:
        if self._closed:
            raise RuntimeError("memory transaction is closed")
        if source_id is None:
            occupied = {record.source_id for record in self._records}
            while True:
                source_id = f"memory-{self._next_id:08d}"
                self._next_id += 1
                if source_id not in occupied:
                    break
        if any(record.source_id == source_id for record in self._records):
            raise ValueError(f"duplicate memory source_id: {source_id}")
        record = self._memory._snapshot_record(
            MemoryRecord(source_id, kind, value, metadata or {})
        )
        # Validate before exposing a staged record. Persistence never falls back
        # to pickle or another executable format.
        self._memory._encode_record(record)
        self._records.append(record)
        return self._memory._snapshot_record(record)

    def _commit(self) -> None:
        self._memory._persist(tuple(self._records), self._next_id)
        self._memory._records = tuple(
            self._memory._snapshot_record(record) for record in self._records
        )
        self._memory._next_id = self._next_id
        self._closed = True

    def _rollback(self) -> None:
        self._closed = True


class JsonMemory:
    """A deterministic JSON store; relevance is supplied by ``retrieve``.

    Values must be JSON-compatible unless ``encode_value`` and ``decode_value``
    are supplied. The codecs are runtime configuration and are never loaded from
    the data file.
    """

    _FORMAT = "tensorcode-memory"
    _VERSION = 1

    @classmethod
    def for_messages(cls, path=None, *, retrieve):
        """Create a store using the public message-sequence JSON codec."""

        from .message_memory import decode_message_sequence, encode_message_sequence

        return cls(
            path,
            retrieve=retrieve,
            encode_value=encode_message_sequence,
            decode_value=decode_message_sequence,
        )

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        retrieve: Callable[[MemorySearch], Iterable[MemoryRecord]],
        encode_value: Callable[[Any], Any] | None = None,
        decode_value: Callable[[Any], Any] | None = None,
    ) -> None:
        if not callable(retrieve):
            raise TypeError("retrieve must be callable")
        if (encode_value is None) != (decode_value is None):
            raise ValueError("encode_value and decode_value must be supplied together")
        self.path = Path(path) if path is not None else None
        self.retrieve = retrieve
        self.encode_value = encode_value or (lambda value: value)
        self.decode_value = decode_value or (lambda value: value)
        self._lock = RLock()
        self._transaction_active = False
        self._records: tuple[MemoryRecord, ...] = ()
        self._next_id = 1
        if self.path is not None and self.path.exists():
            self._load()

    @property
    def records(self) -> tuple[MemoryRecord, ...]:
        with self._lock:
            return tuple(self._snapshot_record(record) for record in self._records)

    def append(
        self,
        value: Any,
        *,
        kind: str,
        source_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemoryRecord:
        with self.transaction() as transaction:
            return transaction.append(
                value,
                kind=kind,
                source_id=source_id,
                metadata=metadata,
            )

    @contextmanager
    def transaction(self):
        """Stage records and commit them atomically on successful exit."""

        self._lock.acquire()
        if self._transaction_active:
            self._lock.release()
            raise RuntimeError("nested memory transactions are not supported")
        self._transaction_active = True
        transaction = None
        try:
            transaction = _MemoryTransaction(self)
            yield transaction
        except BaseException:
            if transaction is not None:
                transaction._rollback()
            raise
        else:
            transaction._commit()
        finally:
            self._transaction_active = False
            self._lock.release()

    def search(
        self,
        query: Any,
        *,
        limit: int | None = None,
        kinds: Iterable[str] | None = None,
    ) -> tuple[MemoryRecord, ...]:
        if limit is not None and (not isinstance(limit, int) or limit < 0):
            raise ValueError("limit must be a non-negative integer")
        allowed_kinds = frozenset(kinds) if kinds is not None else None
        with self._lock:
            candidates = tuple(
                self._snapshot_record(record)
                for record in self._records
                if allowed_kinds is None or record.kind in allowed_kinds
            )
        resolved_limit = len(candidates) if limit is None else limit
        by_id = {
            record.source_id: self._snapshot_record(record) for record in candidates
        }
        selected = tuple(self.retrieve(MemorySearch(query, candidates, resolved_limit)))
        for record in selected:
            if not isinstance(record, MemoryRecord):
                raise TypeError("retrieve must return MemoryRecord candidates")
            if by_id.get(record.source_id) != record:
                raise ValueError("retrieve returned an item outside the supplied candidates")
        return selected[:resolved_limit]

    def _snapshot_record(self, record: MemoryRecord) -> MemoryRecord:
        encoded = self.encode_value(record.value)
        try:
            serialized_value = json.dumps(encoded, allow_nan=False)
            serialized_metadata = json.dumps(dict(record.metadata), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise TypeError("memory value and metadata must be JSON serializable") from error
        return MemoryRecord(
            record.source_id,
            record.kind,
            self.decode_value(json.loads(serialized_value)),
            json.loads(serialized_metadata),
        )

    def _encode_record(self, record: MemoryRecord) -> dict[str, Any]:
        payload = {
            "source_id": record.source_id,
            "kind": record.kind,
            "value": self.encode_value(record.value),
            "metadata": dict(record.metadata),
        }
        try:
            json.dumps(payload, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise TypeError("memory value and metadata must be JSON serializable") from error
        return payload

    def _persist(self, records: tuple[MemoryRecord, ...], next_id: int) -> None:
        if self.path is None:
            return
        payload = {
            "format": self._FORMAT,
            "version": self._VERSION,
            "next_id": next_id,
            "records": [self._encode_record(record) for record in records],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, allow_nan=False, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _load(self) -> None:
        assert self.path is not None
        with self.path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("memory file must contain an object")
        if payload.get("format") != self._FORMAT or payload.get("version") != self._VERSION:
            raise ValueError("unsupported memory file format or version")
        raw_records = payload.get("records")
        next_id = payload.get("next_id")
        if not isinstance(raw_records, list) or not isinstance(next_id, int) or next_id < 1:
            raise ValueError("malformed memory file")
        records: list[MemoryRecord] = []
        for raw in raw_records:
            if not isinstance(raw, dict) or set(raw) != {
                "source_id",
                "kind",
                "value",
                "metadata",
            }:
                raise ValueError("malformed memory record")
            records.append(
                MemoryRecord(
                    raw["source_id"],
                    raw["kind"],
                    self.decode_value(raw["value"]),
                    raw["metadata"],
                )
            )
        if len({record.source_id for record in records}) != len(records):
            raise ValueError("duplicate memory source_id in file")
        self._records = tuple(records)
        self._next_id = next_id
