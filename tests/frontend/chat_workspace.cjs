const assert = require("node:assert/strict");
const fs = require("node:fs");
const { JSDOM } = require("jsdom");
const html = fs.readFileSync(
  process.cwd() + "/examples/general_agent/chat.html",
  "utf8",
);
const script = html
  .match(/<script>([\s\S]*?)<\/script>/)[1]
  .replace(/\n\s*initialize\(\);\s*$/, "");
const dom = new JSDOM(html, {
  url: "http://127.0.0.1:8771/",
  runScripts: "outside-only",
});
const w = dom.window;
w.setTimeout = () => 0;
w.clearTimeout = () => {};
w.matchMedia = () => ({ matches: false });
w.eval(
  script +
    "\nwindow.check={state,handle,renderConnections,renderHistory,refreshChats,selectChat,syncCurrentChat,attach};",
);
const {
  state,
  handle,
  renderConnections,
  renderHistory,
  refreshChats,
  selectChat,
  syncCurrentChat,
  attach,
} = w.check;
const doc = w.document;
function descriptors(status) {
  return [
    {
      id: "shared",
      name: "Environment",
      kind: "browser",
      status,
      preview: { transport: "frames", media_type: "image/png" },
    },
  ];
}
(async () => {
  state.chat = { id: "a", title: "Chat A" };
  state.connections = descriptors("configured");
  renderConnections();
  handle({
    type: "connections",
    chat_id: "b",
    connections: descriptors("B connected"),
  });
  assert.match(doc.querySelector("#connections").textContent, /configured/);
  assert.doesNotMatch(
    doc.querySelector("#connections").textContent,
    /B connected/,
  );
  handle({
    type: "connections",
    chat_id: "a",
    connections: [
      ...descriptors("A connected"),
      ...["image/png", "video/mp4", "audio/wav", "application/pdf"].map(
        (media, i) => ({
          id: "resource" + i,
          name: "Resource " + i,
          selectable: false,
          resource: { media_type: media, url: "/resource/" + i },
          preview: { media_type: media, url: "/resource/" + i },
        }),
      ),
    ],
  });
  assert.match(doc.querySelector("#connections").textContent, /A connected/);
  assert.equal(doc.querySelectorAll("#connections video").length, 1);
  assert.equal(doc.querySelectorAll("#connections audio").length, 1);
  assert.equal(doc.querySelectorAll("#connections img").length, 2);
  assert.equal(doc.querySelectorAll("#connections a.file-chip").length, 1);
  const activeImage = state.previewImages.get("shared");
  handle({
    type: "frame",
    chat_id: "b",
    connection_id: "shared",
    data: "B_FRAME",
  });
  assert.equal(activeImage.getAttribute("src"), null);
  handle({
    type: "frame",
    chat_id: "a",
    connection_id: "shared",
    data: "A_FRAME",
  });
  assert.equal(
    activeImage.getAttribute("src"),
    "data:image/png;base64,A_FRAME",
  );
  state.chat = { id: "b", title: "Chat B" };
  renderConnections();
  assert.match(doc.querySelector("#connections").textContent, /B connected/);
  assert.equal(
    state.previewImages.get("shared").getAttribute("src"),
    "data:image/png;base64,B_FRAME",
  );
  state.chat = { id: "a", title: "Chat A" };
  let release;
  w.fetch = () =>
    new Promise((resolve) => {
      release = (data) => resolve({ ok: true, json: async () => data });
    });
  const load = selectChat("b");
  handle({
    type: "message",
    chat_id: "b",
    message: {
      id: "u",
      role: "user",
      text: "Request",
      status: "completed",
      created_at: 1,
    },
  });
  handle({
    type: "message",
    chat_id: "b",
    message: {
      id: "r",
      role: "assistant",
      text: "Fresh response",
      status: "completed",
      created_at: 2,
    },
  });
  handle({
    type: "message",
    chat_id: "a",
    message: { id: "other", role: "user", text: "Other chat", created_at: 3 },
  });
  handle({ type: "busy", chat_id: "b", busy: false });
  release({
    chat: { id: "b", title: "Chat B", status: "running" },
    messages: [
      {
        id: "u",
        role: "user",
        text: "Request",
        status: "queued",
        created_at: 1,
      },
    ],
  });
  await load;
  assert.equal(state.chat.id, "b");
  assert.equal(state.messages.size, 2);
  assert.equal(state.messages.get("u").status, "completed");
  assert.equal(state.messages.get("r").text, "Fresh response");
  assert.equal(state.messages.has("other"), false);
  assert.equal(state.busy.get("b"), false);
  assert.equal(state.transcriptReads.size, 0);
  const syncing = syncCurrentChat();
  handle({
    type: "message",
    chat_id: "b",
    message: {
      id: "r",
      role: "assistant",
      text: "Revised response",
      status: "completed",
      created_at: 2,
    },
  });
  handle({ type: "busy", chat_id: "b", busy: false });
  release({
    chat: { id: "b", title: "Chat B", status: "running" },
    messages: [
      {
        id: "r",
        role: "assistant",
        text: "Stale response",
        status: "queued",
        created_at: 2,
      },
    ],
  });
  await syncing;
  assert.equal(state.messages.get("r").text, "Revised response");
  assert.equal(state.messages.size, 2);
  assert.equal(state.busy.get("b"), false);
  assert.equal(state.syncing, false);
  assert.equal(state.transcriptReads.size, 0);
  // A rejected read must release its event buffer and allow later reconnects.
  w.fetch = async () => {
    throw new Error("test disconnect");
  };
  await assert.rejects(syncCurrentChat(), /test disconnect/);
  assert.equal(state.transcriptReads.size, 0);
  assert.equal(state.syncing, false);
  // Existing media remains mounted across status changes and arriving replies.
  const attachment = {
    id: "movie",
    name: "test.mp4",
    media_type: "video/mp4",
    content_url: "/movie",
  };
  handle({
    type: "message",
    chat_id: "b",
    message: {
      id: "u",
      role: "user",
      text: "Watch this",
      status: "running",
      created_at: 1,
      attachments: [attachment],
    },
  });
  const video = doc.querySelector("#log video");
  assert.ok(video);
  const observer = new w.MutationObserver(() => {});
  observer.observe(doc.querySelector("#log"), { childList: true });
  handle({
    type: "message",
    chat_id: "b",
    message: {
      id: "u",
      role: "user",
      text: "Watch this",
      status: "completed",
      created_at: 1,
      attachments: [{ ...attachment }],
    },
  });
  handle({
    type: "message",
    chat_id: "b",
    message: {
      id: "new-reply",
      role: "assistant",
      text: "Another reply",
      status: "completed",
      created_at: 4,
    },
  });
  assert.equal(doc.querySelector("#log video"), video);
  assert.equal(
    observer.takeRecords().some((record) => record.removedNodes.length),
    false,
  );
  handle({
    type: "message",
    chat_id: "b",
    message: {
      id: "u",
      role: "user",
      text: "Changed attachment",
      status: "completed",
      created_at: 1,
      attachments: [{ ...attachment, content_url: "/replacement" }],
    },
  });
  assert.notEqual(doc.querySelector("#log video"), video);
  assert.equal(
    doc.querySelector("#log video").getAttribute("src"),
    "/replacement",
  );
  observer.disconnect();
  // Missing historical connections remain visible and can be deselected.
  state.selected = new Set(["missing-connection"]);
  state.busy.set("b", false);
  doc.querySelector("#text").value = "Retained draft";
  renderConnections();
  assert.match(
    doc.querySelector("#connections").textContent,
    /no longer available/,
  );
  assert.equal(doc.querySelector("#send").disabled, true);
  const missing = doc.querySelector(
    'input[aria-label="Use missing-connection in this conversation"]',
  );
  assert.equal(missing.checked, true);
  assert.equal(missing.disabled, false);
  missing.click();
  assert.equal(state.selected.has("missing-connection"), false);
  assert.equal(doc.querySelector("#send").disabled, false);
  // Existing descriptors that become nonselectable also allow recovery.
  state.connections = [
    { id: "retired", name: "Retired tool", selectable: false },
  ];
  state.chatConnections.set("b", [
    {
      id: "retired",
      name: "Retired tool",
      selectable: true,
      status: "connected",
    },
  ]);
  state.selected = new Set(["retired"]);
  renderConnections();
  const retired = doc.querySelector(
    'input[aria-label="Use Retired tool in this conversation"]',
  );
  assert.equal(retired.disabled, false);
  retired.click();
  assert.equal(state.selected.size, 0);
  assert.equal(doc.querySelector("#send").disabled, false);
  // Cached connected status cannot resurrect an adapter removed from inventory.
  state.connections = [];
  state.chatConnections.set("b", descriptors("cached connected"));
  state.selected = new Set(["shared"]);
  renderConnections();
  assert.doesNotMatch(
    doc.querySelector("#connections").textContent,
    /cached connected/,
  );
  assert.match(doc.querySelector("#connections").textContent, /unavailable/);
  assert.equal(doc.querySelector("#send").disabled, true);
  doc
    .querySelector('input[aria-label="Use shared in this conversation"]')
    .click();
  assert.equal(state.selected.size, 0);
  assert.equal(doc.querySelector("#send").disabled, false);
  // CLI sessions use the same history list as UI conversations.
  state.chats = [
    { id: "cli", title: "CLI session", origin: "cli" },
    { id: "b", title: "Chat B", origin: "ui" },
  ];
  renderHistory();
  assert.match(doc.querySelector("#chat-list").textContent, /CLI session/);
  assert.equal(doc.querySelectorAll("#chat-list button").length, 2);
  // A rejected upload keeps the draft and a removable failed attachment.
  w.URL.createObjectURL = () => "blob:test-upload";
  w.URL.revokeObjectURL = () => {};
  w.fetch = async () => ({
    ok: false,
    status: 400,
    json: async () => ({ error: "Test upload rejection" }),
  });
  await attach(
    new w.File(["test fixture"], "fixture.txt", { type: "text/plain" }),
  );
  assert.equal(doc.querySelector("#text").value, "Retained draft");
  assert.equal(state.pending.length, 1);
  assert.equal(state.pending[0].error, "Test upload rejection");
  assert.equal(doc.querySelector("#send").disabled, true);
  doc.querySelector('button[aria-label="Remove fixture.txt"]').click();
  assert.equal(state.pending.length, 0);
  assert.equal(doc.querySelector("#send").disabled, false);
  // Durable chat reads reconstruct read-only resources in the connection pane.
  const resources = ["image/png", "video/mp4", "application/pdf"].map(
    (media, i) => ({
      id: "attachment:" + i,
      name: "Uploaded resource " + i,
      kind: "attachment",
      status: "available",
      selectable: false,
      resource: {
        url: "/attachment/" + i,
        media_type: media,
        size: 12,
        metadata: { attachment_id: String(i) },
      },
      preview: { url: "/attachment/" + i, media_type: media },
    }),
  );
  state.connections = descriptors("configured");
  const snapshots = {
    a: {
      chat: { id: "a", title: "Chat A", status: "idle" },
      messages: [],
      connections: descriptors("A connected"),
    },
    b: {
      chat: { id: "b", title: "Chat B", status: "idle" },
      messages: [],
      connections: [...descriptors("B connected"), ...resources],
    },
  };
  w.fetch = async (url) => ({
    ok: true,
    json: async () => snapshots[url.split("/").at(-1)],
  });
  await selectChat("b");
  assert.equal(
    doc.querySelectorAll('#connections input[type="checkbox"]').length,
    1,
  );
  assert.equal(doc.querySelectorAll("#connections video").length, 1);
  assert.equal(doc.querySelectorAll("#connections img").length, 2);
  assert.equal(doc.querySelectorAll("#connections a.file-chip").length, 1);
  for (const resource of resources)
    assert.match(
      doc.querySelector("#connections").textContent,
      new RegExp(resource.name),
    );
  assert.equal(
    (doc.querySelector("#connections").textContent.match(/Read-only/g) || [])
      .length,
    3,
  );
  const resourceVideo = doc.querySelector("#connections video");
  const resourceObserver = new w.MutationObserver(() => {});
  resourceObserver.observe(doc.querySelector("#connections"), {
    childList: true,
  });
  state.sending = true;
  renderConnections();
  state.sending = false;
  renderConnections();
  handle({
    type: "connections",
    chat_id: "b",
    connections: [...descriptors("B connected"), ...resources],
  });
  assert.equal(doc.querySelector("#connections video"), resourceVideo);
  assert.equal(
    resourceObserver
      .takeRecords()
      .some((record) =>
        [...record.removedNodes].some((n) => n.contains(resourceVideo)),
      ),
    false,
  );
  resourceObserver.disconnect();
  await selectChat("a");
  assert.equal(doc.querySelectorAll("#connections video").length, 0);
  assert.doesNotMatch(
    doc.querySelector("#connections").textContent,
    /Uploaded resource/,
  );
  await selectChat("b");
  assert.match(
    doc.querySelector("#connections").textContent,
    /Uploaded resource 2/,
  );
  handle({
    type: "connections",
    chat_id: "a",
    connections: [
      ...descriptors("A connected"),
      { ...resources[0], id: "attachment:other", name: "Other chat image" },
    ],
  });
  assert.doesNotMatch(
    doc.querySelector("#connections").textContent,
    /Other chat image/,
  );
  // New scoped resource events win over a stale concurrent transcript snapshot.
  w.fetch = () =>
    new Promise((resolve) => {
      release = (data) => resolve({ ok: true, json: async () => data });
    });
  const resourceLoad = selectChat("a");
  handle({
    type: "connections",
    chat_id: "a",
    connections: [
      ...descriptors("A connected"),
      { ...resources[0], id: "attachment:new", name: "Newly committed image" },
    ],
  });
  release(snapshots.a);
  await resourceLoad;
  assert.match(
    doc.querySelector("#connections").textContent,
    /Newly committed image/,
  );
  assert.doesNotMatch(
    doc.querySelector("#connections").textContent,
    /Uploaded resource|Other chat image/,
  );
  assert.equal(
    doc.querySelectorAll('#connections input[type="checkbox"]').length,
    1,
  );
  // API-origin messages reconcile the active header from authoritative titles.
  state.chat = { id: "a", title: "New chat" };
  w.fetch = async () => ({
    ok: true,
    json: async () => ({
      chats: [{ id: "a", title: "Custom authoritative title" }],
    }),
  });
  await refreshChats();
  assert.equal(
    doc.querySelector("#chat-title").textContent,
    "Custom authoritative title",
  );
  assert.equal(state.chat.title, "Custom authoritative title");
  // A superseded list response cannot overwrite a newer chat selection/title.
  const historyReads = [];
  w.fetch = () =>
    new Promise((resolve) =>
      historyReads.push((data) =>
        resolve({ ok: true, json: async () => data }),
      ),
    );
  const olderHistory = refreshChats();
  state.chat = { id: "b", title: "Selected B" };
  const newerHistory = refreshChats();
  historyReads[1]({
    chats: [
      { id: "a", title: "Custom authoritative title" },
      { id: "b", title: "B custom title" },
    ],
  });
  await newerHistory;
  historyReads[0]({
    chats: [
      { id: "a", title: "Stale title" },
      { id: "b", title: "Old B title" },
    ],
  });
  await olderHistory;
  assert.equal(state.chat.id, "b");
  assert.equal(state.chat.title, "B custom title");
  assert.equal(doc.querySelector("#chat-title").textContent, "B custom title");
  console.log(
    "PASS: scoped connection status and frames, 4 preview media types, selection/reconnect SSE races, message dedupe, failed-read cleanup, stable media playback DOM, unavailable connection recovery, CLI history, failed upload draft preservation, durable read-only resources, sidebar playback stability, resource scoping and active-title reconciliation",
  );
  dom.window.close();
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
  dom.window.close();
});
