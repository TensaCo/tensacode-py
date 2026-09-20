"""Durable chat transcripts and opaque attachments, independent of agent connections."""
from __future__ import annotations

import base64
import binascii
import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from uuid import uuid4

MAX_ATTACHMENT_BYTES = 32 * 1024 * 1024
MAX_REQUEST_BYTES = ((MAX_ATTACHMENT_BYTES + 2) // 3) * 4 + 65536
_ID = re.compile(r"^[a-f0-9]{32}$")


class ChatStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.directory / "chats.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS chats(id TEXT PRIMARY KEY, title TEXT, origin TEXT,
                created_at REAL, updated_at REAL, status TEXT);
            CREATE TABLE IF NOT EXISTS messages(id TEXT PRIMARY KEY, chat_id TEXT, role TEXT,
                text TEXT, created_at REAL, status TEXT, origin TEXT, attachments TEXT,
                connection_ids TEXT, metadata TEXT);
            CREATE INDEX IF NOT EXISTS messages_chat ON messages(chat_id, created_at);
            CREATE TABLE IF NOT EXISTS attachments(id TEXT PRIMARY KEY, name TEXT,
                media_type TEXT, size INTEGER, created_at REAL, data BLOB);
            CREATE TABLE IF NOT EXISTS imports(key TEXT PRIMARY KEY, chat_id TEXT);
        ''')
        # A process restart never silently replays potentially mutating turns.
        with self.db:
            self.db.execute("UPDATE messages SET status='interrupted' WHERE status IN ('queued','running')")
            self.db.execute("UPDATE chats SET status='interrupted' WHERE status IN ('queued','running')")

    def close(self):
        with self.lock:
            self.db.close()

    @staticmethod
    def validate_id(value):
        if not isinstance(value, str) or not _ID.fullmatch(value):
            raise ValueError("invalid resource id")
        return value

    def create_chat(self, title="New chat", origin="ui"):
        if not isinstance(title, str) or len(title) > 200:
            raise ValueError("title must be a string of at most 200 characters")
        now, ident = time.time(), uuid4().hex
        with self.lock, self.db:
            self.db.execute("INSERT INTO chats VALUES(?,?,?,?,?,?)", (ident, title.strip() or "New chat", origin, now, now, "idle"))
        return self.chat(ident)

    def chat(self, ident):
        self.validate_id(ident)
        with self.lock:
            row = self.db.execute("SELECT * FROM chats WHERE id=?", (ident,)).fetchone()
        if row is None:
            raise KeyError("chat not found")
        return dict(row)

    def chats(self):
        with self.lock:
            return [dict(r) for r in self.db.execute("SELECT * FROM chats ORDER BY updated_at DESC")]

    def cli_chat(self):
        with self.lock:
            row = self.db.execute("SELECT id FROM chats WHERE origin='cli' ORDER BY created_at LIMIT 1").fetchone()
            return self.chat(row[0]) if row else self.create_chat("CLI session", "cli")

    def attachment(self, ident, *, content=False):
        self.validate_id(ident)
        with self.lock:
            columns = "*" if content else "id, name, media_type, size, created_at"
            row = self.db.execute(f"SELECT {columns} FROM attachments WHERE id=?", (ident,)).fetchone()
        if row is None:
            raise KeyError("attachment not found")
        result = dict(row)
        result["content_url"] = f"/api/attachments/{ident}/content"
        return result

    def upload(self, name, media_type, encoded):
        if not isinstance(name, str) or not name.strip() or len(name) > 255:
            raise ValueError("attachment requires a name of at most 255 characters")
        if not isinstance(media_type, str) or not re.fullmatch(
                r"[A-Za-z0-9!#$%&'*+.^_`|~-]+/[A-Za-z0-9!#$%&'*+.^_`|~-]+", media_type):
            raise ValueError("invalid media type")
        # MIME types are case-insensitive. Normalize before preview and interpreter
        # dispatch; reject non-ASCII tokens that cannot be emitted as HTTP headers.
        media_type = media_type.lower()
        if not isinstance(encoded, str) or len(encoded) > ((MAX_ATTACHMENT_BYTES + 2) // 3) * 4:
            raise ValueError("attachment exceeds 32 MiB limit")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("invalid base64 attachment") from exc
        if len(data) > MAX_ATTACHMENT_BYTES:
            raise ValueError("attachment exceeds 32 MiB limit")
        ident = uuid4().hex
        with self.lock, self.db:
            self.db.execute("INSERT INTO attachments VALUES(?,?,?,?,?,?)", (ident, name, media_type, len(data), time.time(), data))
        return self.attachment(ident)

    def attachments(self):
        with self.lock:
            ids = [r[0] for r in self.db.execute("SELECT id FROM attachments ORDER BY created_at DESC")]
            return [self.attachment(i) for i in ids]

    def attachments_for_chat(self, chat_id):
        """Unique source metadata scoped to this transcript, without reading blobs."""
        with self.lock:
            self.chat(chat_id)
            attachments = {}
            for row in self.db.execute(
                    "SELECT attachments FROM messages WHERE chat_id=? ORDER BY created_at,rowid",
                    (chat_id,)):
                for attachment in json.loads(row[0]):
                    attachments.setdefault(attachment["id"], attachment)
            return list(attachments.values())

    @staticmethod
    def _validate_message_text(role, text):
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        # The inbound request limit must not truncate or reject generated replies.
        if role != "assistant" and len(text) > 20000:
            raise ValueError("input text must be at most 20000 characters")
    def add_message(self, chat_id, role, text, *, attachment_ids=(), connection_ids=(), origin="ui", status="completed", metadata=None):
        self._validate_message_text(role, text)
        if not isinstance(attachment_ids, (list, tuple)) or len(attachment_ids) > 16:
            raise ValueError("at most 16 attachments per message")
        attachments = [self.attachment(i) for i in attachment_ids]
        if not isinstance(connection_ids, (list, tuple)) or not all(isinstance(i, str) for i in connection_ids):
            raise ValueError("connection_ids must be a list of strings")
        ident, now = uuid4().hex, time.time()
        with self.lock, self.db:
            chat = self.chat(chat_id)
            self.db.execute("INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?,?)", (ident, chat_id, role, text, now, status, origin, json.dumps(attachments), json.dumps(connection_ids), json.dumps(metadata or {})))
            title = (text.strip()[:70] or (attachments[0]["name"] if attachments else "New chat")) if chat["title"] == "New chat" else chat["title"]
            self.db.execute("UPDATE chats SET title=?,updated_at=?,status=? WHERE id=?", (title, now, self._active_chat_status(chat_id, chat["status"]), chat_id))
        return self.message(ident)

    def message(self, ident):
        self.validate_id(ident)
        with self.lock:
            row = self.db.execute("SELECT * FROM messages WHERE id=?", (ident,)).fetchone()
        if row is None:
            raise KeyError("message not found")
        result = dict(row)
        for key in ("attachments", "connection_ids", "metadata"):
            result[key] = json.loads(result[key])
        return result

    def messages(self, chat_id):
        self.chat(chat_id)
        with self.lock:
            return [self.message(r[0]) for r in self.db.execute("SELECT id FROM messages WHERE chat_id=? ORDER BY created_at,rowid", (chat_id,))]

    def _active_chat_status(self, chat_id, fallback):
        """Called under the store lock; running work takes priority over its queue."""
        active = self.db.execute(
            "SELECT status FROM messages WHERE chat_id=? AND status IN ('running','queued') "
            "ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END LIMIT 1",
            (chat_id,),
        ).fetchone()
        return active[0] if active else fallback

    def status(self, chat_id, message_id, status):
        with self.lock, self.db:
            self.db.execute("UPDATE messages SET status=? WHERE id=? AND chat_id=?", (status, message_id, chat_id))
            chat_status = self._active_chat_status(chat_id, "idle" if status == "completed" else status)
            self.db.execute("UPDATE chats SET status=?,updated_at=? WHERE id=?", (chat_status, time.time(), chat_id))

    def import_history(self, events, key="previous-demo-session"):
        """Idempotently preserve available old SSE chat events without inventing lost media."""
        if isinstance(events, dict):
            events = events.get("events")
        if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
            raise ValueError("history must be an event list or an object containing events")
        with self.lock:
            existing = self.db.execute("SELECT chat_id FROM imports WHERE key=?", (key,)).fetchone()
            if existing:
                return self.chat(existing[0])
            # Prepare and validate the whole source before modifying storage. Do
            # not call create_chat/add_message here: each commits independently.
            chat_id, now = uuid4().hex, time.time()
            rows = []
            for event in events:
                if event.get("type") == "chat" and event.get("from") in ("user", "agent"):
                    role = "assistant" if event["from"] == "agent" else "user"
                    text = event.get("text", "")
                    self._validate_message_text(role, text)
                    metadata = json.dumps({"historical_event": event, "media_recovered": False})
                    rows.append((uuid4().hex, chat_id, role, text, time.time(),
                                 "completed", "imported", "[]", "[]", metadata))
            with self.db:
                self.db.execute("INSERT INTO chats VALUES(?,?,?,?,?,?)", (
                    chat_id, "Previous demo session", "imported", now,
                    rows[-1][4] if rows else now, "idle"))
                self.db.executemany("INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?,?)", rows)
                self.db.execute("INSERT INTO imports VALUES(?,?)", (key, chat_id))
            return self.chat(chat_id)
