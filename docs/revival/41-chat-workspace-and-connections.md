# Chat workspace and independent connections

2026-09-19

The demo is now a persistent chat application around the agent. Its governing
objective remains [the structured cognitive workspace](36-structured-cognitive-workspace.md):
retain unstructured source evidence, form revisable interpretations, reason within
structured models, and realize the resulting decisions. A more polished interface
and more adapters do not establish generalized understanding.

## What changed

The center pane contains the conversation and a composer with file attachment
support. The left sidebar lists stored conversations, including requests submitted
through the CLI API. The right sidebar describes available connections and uploaded
resources. Connection presentation comes from adapter descriptors, rather than
chatbot code that assumes every connection is a Computerworld desktop.

The browser and CLI use the same message ingestion path. User messages are stored
before being queued; assistant responses, errors, attachment associations, selected
connection IDs, and message origin are stored as well. SSE events update connected
pages. Message IDs identify updates to an existing message, preventing lifecycle
updates and reconnect history from appearing as duplicate chat bubbles.

Connection status and preview frames are scoped to their conversation, so activity
in a CLI chat does not replace another chat's sidebar. Transcript reads retain
message events arriving during the request before rendering the combined result.
Connection previews use their declared media type: images, video/audio players,
or a file link, rather than assuming every resource is a desktop screenshot.

The frontend concurrency and media checks are reproducible from the repository
root with an isolated optional JavaScript dependency:

```bash
npm install --prefix /tmp/tensorcode-chat-check jsdom@30.1.0
NODE_PATH=/tmp/tensorcode-chat-check/node_modules node tests/frontend/chat_workspace.cjs
```

A conversation owns an independent Agent, interpretation workspace, and mounted
adapter registry during the server's lifetime. A single worker process handles
turns sequentially because some connection engines require thread ownership.
Connections are mounted lazily when selected for a message. An unselected browser
or environment is not initialized merely because it appears in server configuration.
Changing the selected connection list changes the active plugins without replacing
the conversation's Agent. Plugin vocabulary follows the active selection.

## Running the application

From the repository root:

```bash
.venv/bin/python -m examples.general_agent.server \
  --port 8771 --no-open --reader grammar --plugin none
```

`--reader grammar` chooses the grammar reader; the default remains the registered
learned reader. Neither reader flag supplies an interpretation-selection policy.
The application preserves the explicit unresolved-meaning behavior documented in
[the removal of implicit semantic authority](39-removing-implicit-semantic-authority.md).
There is no first-reader execution fallback hidden in the new UI.

The learned reader now also requires the locally trained segmentation artifact
`~/.cache/tensorcode/models/ud_ewt_segmenter.json`. With the local UD English EWT
training data available, run `.venv/bin/python -m eval.parsing.train_segmentation`
to create it. [The segmentation report](51-learned-source-segmentation.md) gives the
full reproducible training/evaluation command and measured limits. Missing or
invalid segmentation weights leave the input unresolved; the chat server does not
silently substitute the old tokenizer or download a model.

Repeat `--plugin SPEC` to configure adapters. Without explicit plugin arguments,
the server configures `desktop` and `self`. Connections are selected from the
sidebar when composing a message. A fresh UI conversation starts without selected
connections; subsequent loads recover the latest request's selection.

Install the optional browser and Gym dependencies with
`uv pip install -e '.[connections]'`. An existing Chromium CDP endpoint supplies
the browser; disposable browser tests additionally use `playwright install chromium`.

```bash
.venv/bin/python -m examples.general_agent.server \
  --port 8771 --no-open \
  --plugin desktop \
  --plugin self \
  --plugin filesystem:/an/explicit/existing/directory \
  --plugin 'browser:http://127.0.0.1:9222#page=0' \
  --plugin gym:CartPole-v1
```

The data directory defaults to `~/.cache/tensorcode/chat`. Override it with
`--data-dir /chosen/directory` for isolated test runs or separate installations.
`chats.sqlite3` contains conversation metadata, messages, and attachment bytes;
SQLite transactions and an application lock serialize access from HTTP threads.

Stored conversation text survives restarts. The Agent's cognitive state,
interpretation selections, mounted environment state, and runtime trace do **not**
yet persist. A resumed conversation gets a fresh in-memory Agent on its next turn;
old turns are not replayed, since replay could repeat external mutations. Queued or
running messages surviving a restart become `interrupted` rather than silently
executing again. The transcript is durable history, not a cognitive-state checkpoint.

The `--import-history saved-events.json` option imports a JSON array (or an object
with an `events` array) of old demo SSE
chat events into a conversation titled **Previous demo session**. Import is
idempotent. Historical events are retained as metadata; old image counts are not
represented as recovered files when the original bytes are unavailable.

## Shared UI and CLI API

The server binds to loopback. API bodies are JSON objects, and response bodies are
JSON except attachment content and the SSE stream.

| Method and path | Request / response |
| --- | --- |
| `GET /api/chats` | `{chats: [...]}` ordered by recent activity |
| `POST /api/chats` | Optional `{title}`; returns `{chat}` |
| `GET /api/chats/{id}` | `{chat, messages}` |
| `POST /api/chats/{id}/messages` | `{text, attachment_ids, connection_ids}`; returns `{message, queued, chat_id}` |
| `POST /say` | CLI ingestion using the same message path; optional `chat_id` |
| `POST /api/attachments` | `{name, media_type, data}` with base64 bytes; returns `{attachment}` |
| `GET /api/attachments/{id}/content` | Original bytes; supports a single HTTP byte range |
| `GET /api/connections` | Configured adapter descriptors; configuration does not establish availability |
| `GET /events` | All live chat events, each carrying `chat_id` when conversation-specific |
| `GET /events?chat_id={id}` | Events restricted to one conversation |

A CLI example that appears in the same left-sidebar history:

```bash
curl http://127.0.0.1:8771/say \
  -H 'Content-Type: application/json' \
  -d '{"text":"Describe what is currently understood.","connection_ids":[]}'
```

`/say` reuses a persistent **CLI session** when `chat_id` is omitted. Supply a
specific ID to continue another conversation. Omitting `connection_ids` selects
no executable connections on either message endpoint; selecting adapters always
requires their explicit IDs. Messages submitted through `/say`
record `origin: "cli"`; messages submitted through the UI message endpoint record
`origin: "ui"`. An assistant response records `origin: "agent"`. An API client
using the UI endpoint is still using that endpoint's origin convention; this is
transport attribution, not authenticated identity.

The old `/say` `images` array remains a transport convenience: each base64 image is
stored as an attachment and then enters the same ingestion path. It does not
restore the removed image-to-claim semantics.

Concurrent submissions within one chat serialize persistence and enqueueing so
requests reach the worker in their stored order. User input text is limited to
20,000 characters. Generated assistant responses are stored in full and are not
subject to that input limit.

Submission returns HTTP 202 after persistence and enqueueing, not after cognitive
completion. Lifecycle values include `queued`, `running`, `completed`, `error`, and
`interrupted`; chat status becomes `idle` after completed work. A runtime exception
produces a stored assistant error and an error lifecycle update. SSE `message`
events carry the complete message record; `busy`, `connections`, `frame`, and
agent trace events provide additional live information. The SSE replay window is
bounded and in memory; clients recover authoritative transcripts through the API.

## Uploaded evidence and its limits

Users can attach images, video, documents, and other files. The API accepts at most
32 MiB per file and at most 16 attachments per message. Filenames are metadata;
resource lookup uses generated IDs, never user-supplied filesystem paths. Corrupt
base64, malformed IDs, oversized uploads, and invalid connection selections are
rejected. Attachment content supports byte ranges for video playback. Potentially
active documents such as HTML and SVG download as opaque bytes with sandbox and
content-sniffing protections. The server does not fall through to an unrestricted
static file handler. Cross-origin browser writes are rejected; originless CLI
requests remain supported.

Every attachment sent with a turn is retained in the Agent interpretation workspace
with its original bytes, declared media type, filename, and attachment identity.
Image bytes are additionally passed to `Agent.turn(images=...)` so installed visual
interpreters can propose interpretations. Uploading an image does not establish a
scene interpretation. Videos and other documents currently remain source evidence;
the application does not decode video events, extract document meaning, or silently
pretend that the agent has read them. The `attachment_evidence` event explicitly
reports `understood: false` for the retention operation.

The visual objective is holistic scene understanding, relational organization,
spatial and temporal structure, and grounded alternatives, as described in
[scene interpretations](38-scene-interpretations.md). File upload and preview are
transport capabilities on that path. They are not the perceptual inference itself.

## Connection boundary

`connections.py` defines transport-neutral descriptors and a registry. A descriptor
contains a stable ID, adapter-supplied kind, name, availability status, capabilities,
optional preview metadata, and explanatory text. IDs remain stable across restarts;
duplicate configured specs get distinct identities. The UI renders descriptors
without a switch over a closed list of domain types.

Configured inventory is available without initializing engines. Runtime descriptors
replace configuration assumptions after adapters are constructed; a construction
failure is reported as a turn error. An unavailable
connection cannot be selected for execution. Resource connections describe retained
uploads and are not executable plugins. `register_factory` allows additional adapter
kinds without modifying the chatbot's message protocol or page layout.

### Existing browser tabs

The browser adapter uses Playwright CDP to connect to an **explicit existing**
Chromium endpoint. The user must start a browser with remote debugging available;
the chat server does not discover arbitrary browsers or choose a tab silently.
Use `browser:http://127.0.0.1:9222#page=0` to choose the first exposed page. Omitting
`#page=N` is accepted only when exactly one page exists. The Playwright Python
package must be installed.

The adapter provides typed `navigate(url)`, `click(selector)`, and
`fill(selector, text)` actions. Navigation requires an HTTP(S) URL. Selector actions
use explicit Playwright selectors and retain its strict matching behavior. Raw
observations include URL, title, DOM HTML, screenshot bytes, and provenance.
Browser DOM and pixels are observations, not established semantic scene graphs.
The adapter does not infer selectors from arbitrary language or invent complete
action effects for model-based planning. Timeouts return an indeterminate receipt
because an external mutation may already have happened. Closing the adapter
detaches its driver without closing the user's browser or tab.

### Gymnasium environments

`gym:CartPole-v1` creates the explicitly named real Gymnasium environment. The
Gymnasium package and any dependencies required by the selected environment must
be installed. The adapter exposes typed `reset(seed)` and `step(action)` actions.
It validates actions against the environment's own action space and requires an
explicit reset before the first step and after termination or truncation. Mounting
does not automatically reset the environment or start a policy.

Raw transition observations preserve observation values, reward, info, termination,
truncation, operation, sequence, and provenance. Reward does not silently become a
belief or a user goal. The CLI factory uses the environment's default render mode,
so a preview is generally absent. The adapter supports PNG previews for environments
explicitly constructed with `render_mode="rgb_array"`. Closing it closes the owned
environment.

The browser and Gym adapters are real transports with real typed actions. Their
raw observations now enter the [action evidence pipeline](44-action-observation-evidence.md)
through `observe_evidence()`, including before/after sources linked to calls and
receipts. This retains the inputs for subsequent interpretation and learning; it
does not supply learned browser understanding, a learned environment policy, or an
inferred world model merely from the presence of DOM, pixels, and arrays.

## Validation and remaining work

The live port-8771 page was checked with its existing history: both the CLI session
and imported demo conversation appear in the left sidebar, alongside the central
composer and independent connections pane. This server currently configures only
`desktop` and `self`; browser and Gym adapters are available through the explicit
configuration examples above. The frontend regression runner passed connection
scoping, four preview media types, SSE/load races, deduplication, and failed-read
cleanup. The repository checkpoint passed 1,943 tests with five skipped.

Backend tests cover durable reload, interruption without replay, shared UI/CLI
ingestion, lifecycle updates, idempotent import, malformed uploads, limits, IDs,
video ranges, safe downloads, cross-origin writes, connection validation, SSE
scoping, per-chat isolation, and retention of file/image/video evidence. Worker tests
with an explicit test Agent isolate transport mechanics; they do not measure
understanding. Separate adapter tests exercise real Chromium and Gymnasium behavior.

The next architectural work remains evidence-guided interpretation and model
formation, not UI-specific language rules. Important outstanding product work is
explicit cognitive checkpointing, durable reasoning traces, interpretation of retained
adapter evidence, document/video interpretation, and cancellation and
resource-lifetime controls for long-lived conversations. Multiple chat runtimes
sharing an external browser tab still share that external world even though their
Agent state is isolated. The current worker processes turns sequentially rather
than providing independent parallel execution per conversation.
