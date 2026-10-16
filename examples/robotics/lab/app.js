"use strict";
const experiments = JSON.parse(
  document.getElementById("experiment-data").textContent,
).experiments;
const video = document.getElementById("video"),
  episodeSelect = document.getElementById("episode");
const tabs = document.getElementById("tabs"),
  inspector = document.getElementById("inspector");
const notesList = document.getElementById("notes"),
  noteStatus = document.getElementById("note-status");
const escapeHTML = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
let current,
  episode,
  selectedBlock = null,
  notes = [];
const key = () => `tensorcode.robotics.notes.v1:${current.id}:${episode.id}`;
function node(tag, text, cls) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (cls) el.className = cls;
  return el;
}
function save() {
  try {
    localStorage.setItem(key(), JSON.stringify(notes));
    noteStatus.textContent = "Saved in this browser.";
  } catch {
    noteStatus.textContent =
      "Browser storage unavailable. Export notes to keep them.";
  }
}
function validNote(n) {
  return (
    n &&
    typeof n.id === "string" &&
    n.id.length < 100 &&
    Number.isFinite(n.time) &&
    n.time >= 0 &&
    typeof n.text === "string" &&
    n.text.length <= 4000 &&
    (n.block === null ||
      (typeof n.block === "string" &&
        Object.hasOwn(current.graph.blocks, n.block)))
  );
}
function load() {
  notes = [];
  try {
    const data = JSON.parse(localStorage.getItem(key()) || "[]");
    if (Array.isArray(data)) notes = data.filter(validNote);
  } catch {}
  noteStatus.textContent = "Notes stay local to this browser.";
  renderNotes();
}
function renderNotes() {
  notesList.replaceChildren();
  for (const n of [...notes].sort((a, b) => a.time - b.time)) {
    const li = node("li");
    const seek = node(
      "button",
      `${n.time.toFixed(2)} s${n.block ? " · " + current.graph.blocks[n.block].title : ""}`,
    );
    seek.type = "button";
    seek.onclick = () => {
      video.currentTime = n.time;
      selectBlock(n.block);
    };
    const remove = node("button", "×", "remove");
    remove.type = "button";
    remove.setAttribute("aria-label", "Delete annotation");
    remove.onclick = () => {
      notes = notes.filter((x) => x.id !== n.id);
      save();
      renderNotes();
    };
    li.append(seek, remove, node("p", n.text));
    notesList.append(li);
  }
}
function selectBlock(id) {
  selectedBlock = id && Object.hasOwn(current.graph.blocks, id) ? id : null;
  document.querySelectorAll(".block").forEach((b) => {
    b.classList.toggle("selected", b.dataset.id === selectedBlock);
    b.setAttribute("aria-pressed", String(b.dataset.id === selectedBlock));
  });
  document.getElementById("inspector-panel").open = Boolean(selectedBlock);
  inspector.replaceChildren();
  if (!selectedBlock) {
    if (current.outcomes?.length) {
      const details = node("details");
      details.append(
        node(
          "summary",
          `Run outcomes (${current.outcomes.length})${current.outcomes_complete === false ? " · partial" : ""}`,
        ),
      );
      for (const outcome of current.outcomes)
        details.append(
          node("p", `${outcome.label} · ${outcome.split} · ${outcome.status}`),
        );
      inspector.append(details);
    }
    inspector.append(
      node("p", "Select a controller block to inspect its ports."),
    );
    return;
  }
  const b = current.graph.blocks[selectedBlock];
  inspector.append(
    node("strong", b.title),
    node(
      "p",
      b.description ||
        {
          o: "Observed input.",
          a: "Authored computation or estimated state.",
          f: "Pretrained component; weights frozen in this experiment.",
          x: "Actuation or simulated physical system.",
        }[b.kind],
    ),
  );
  const dl = node("dl");
  for (const [label, ports] of [
    ["Inputs", b.ins],
    ["Outputs", b.outs],
  ])
    dl.append(node("dt", label), node("dd", ports.join(" · ") || "None"));
  inspector.append(dl);
  updateSignals();
}
function updateSignals() {
  let el = document.getElementById("signals");
  if (el) el.remove();
  el = node("div");
  el.id = "signals";
  const t = video.currentTime;
  const latest = new Map();
  for (const s of episode.signals || [])
    if (
      s.time <= t &&
      (!s.block || !selectedBlock || s.block === selectedBlock)
    )
      latest.set(s.name, s);
  for (const s of latest.values())
    el.append(
      node(
        "p",
        `${s.name}: ${s.value}${s.unit ? " " + s.unit : ""} @ ${s.time.toFixed(2)} s`,
      ),
    );
  for (const event of episode.events || [])
    if (Math.abs(event.time - t) < 0.5) el.append(node("p", event.label));
  inspector.append(el);
}
function updateChapter() {
  const chapters = episode.chapters || [];
  let active = -1;
  chapters.forEach((chapter, i) => {
    if (chapter.time <= video.currentTime + 0.01) active = i;
  });
  document.querySelectorAll("#chapters button").forEach((button, i) => {
    button.setAttribute("aria-current", i === active ? "step" : "false");
    button.classList.toggle("complete", i < active);
  });
  const strip = document.getElementById("chapters");
  if (strip.dataset.active !== String(active)) {
    const button = strip.querySelectorAll("button")[active];
    if (button) {
      const frame = strip.getBoundingClientRect();
      const bounds = button.getBoundingClientRect();
      if (bounds.left < frame.left)
        strip.scrollLeft += bounds.left - frame.left - 8;
      else if (bounds.right > frame.right)
        strip.scrollLeft += bounds.right - frame.right + 8;
    }
    strip.dataset.active = String(active);
  }
}
function renderChapters() {
  const target = document.getElementById("chapters");
  target.replaceChildren();
  target.scrollLeft = 0;
  delete target.dataset.active;
  target.hidden = !episode.chapters?.length;
  if (target.hidden) return;
  const label = node("span", "STAGES", "chapter-label");
  label.title = episode.chapter_label || "Recorded stages";
  target.append(label);
  if (episode.chapter_tracks?.length > 1) {
    const select = node("select");
    select.setAttribute("aria-label", "Robot stage track");
    for (const [index, track] of episode.chapter_tracks.entries()) {
      const option = node("option", track.label);
      option.value = String(index);
      select.append(option);
    }
    select.value = String(episode.active_track || 0);
    select.onchange = () => {
      episode.active_track = Number(select.value);
      const track = episode.chapter_tracks[episode.active_track];
      episode.chapters = track.chapters;
      episode.chapter_label = track.label;
      renderChapters();
    };
    target.append(select);
  }
  episode.chapters.forEach((chapter) => {
    const button = node("button", chapter.label);
    button.type = "button";
    button.title = `${episode.chapter_label || "Recording"} · ${chapter.time.toFixed(2)} s`;
    button.onclick = () => {
      video.currentTime = chapter.time;
      selectBlock(chapter.block);
      updateChapter();
    };
    target.append(button);
  });
  updateChapter();
}
function evidenceTree(label, value, depth = 0) {
  if (value === null || typeof value !== "object") {
    return node("p", `${label}: ${String(value)}`, "evidence-leaf");
  }
  const details = node("details");
  details.open = false;
  details.append(node("summary", `${label} · ${Object.keys(value).length}`));
  for (const [key, child] of Object.entries(value)) {
    details.append(evidenceTree(key, child, depth + 1));
  }
  return details;
}
let previewRequest = 0;
function renderLanguagePreview() {
  previewRequest++;
  const config = current.language_preview;
  const section = document.getElementById("language-preview");
  section.hidden = !config;
  document.getElementById("language-result").replaceChildren();
  const remote =
    location.protocol === "http:" || location.protocol === "https:";
  for (const control of document.getElementById("language-form").elements)
    control.disabled = !remote;
  document.getElementById("language-status").textContent = !config
    ? ""
    : remote
      ? "Interprets the server's fixed recorded scene. Does not execute robot commands."
      : `Open through the local preview server: ${config.command}. This file cannot call the parser.`;
}
document.getElementById("language-form").onsubmit = async (event) => {
  event.preventDefault();
  if (
    !current.language_preview ||
    !["http:", "https:"].includes(location.protocol)
  )
    return;
  const request = ++previewRequest;
  const status = document.getElementById("language-status");
  const output = document.getElementById("language-result");
  status.textContent = "Interpreting recorded observation…";
  output.replaceChildren();
  try {
    const response = await fetch("/api/warehouse/interpret", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: document.getElementById("language-text").value,
        mode: document.getElementById("language-mode").value,
      }),
    });
    const result = await response.json();
    if (request !== previewRequest) return;
    if (!response.ok)
      throw new Error(
        result.reason || result.error || `Request failed (${response.status})`,
      );
    if (result.scope !== "recorded_observation_preview")
      throw new Error("Unexpected response scope; no execution requested.");
    status.textContent = `${result.status} · ${result.parser} · recorded scene ${result.scene_revision}. Preview only; no execution or motion-feasibility guarantee.`;
    const answer =
      typeof result.answer === "string" ? result.answer : result.answer?.text;
    if (answer) output.append(node("p", answer));
    if (result.reason) output.append(node("p", result.reason));
    const chips = node("div", undefined, "result-chips");
    if (Array.isArray(result.tasks))
      chips.append(node("span", `${result.tasks.length} grounded tasks`));
    output.append(chips, evidenceTree("Inspect interpretation", result));
  } catch (error) {
    if (request === previewRequest)
      status.textContent = `Preview unavailable: ${error.message}`;
  }
};
function renderInspections() {
  const target = document.getElementById("inspection-trees");
  target.replaceChildren();
  document.getElementById("representations").hidden =
    !episode.inspections?.length;
  for (const inspection of episode.inspections || []) {
    const section = node("section");
    const source = node("a", "Source JSON ↗");
    source.href = inspection.path;
    if (inspection.image || inspection.link || inspection.caption) {
      const details = node("details", undefined, "visual-evidence");
      details.append(node("summary", inspection.title));
      if (inspection.caption) details.append(node("p", inspection.caption));
      if (inspection.link) {
        const artifact = node("a", "Open diagnostic ↗");
        artifact.href = inspection.link;
        details.append(artifact);
      }
      if (inspection.image) {
        const image = node("img");
        image.src = inspection.image;
        image.alt = inspection.caption || inspection.title;
        image.loading = "lazy";
        const link = node("a");
        link.href = inspection.image;
        link.append(image);
        details.append(link);
      }
      details.append(source, evidenceTree("Evidence fields", inspection.data));
      section.append(details);
    } else
      section.append(source, evidenceTree(inspection.title, inspection.data));
    target.append(section);
  }
}
function renderMetrics() {
  const target = document.getElementById("episode-metrics");
  target.replaceChildren();
  target.hidden = !episode.metrics?.length;
  for (const metric of episode.metrics || []) {
    const item = node("span");
    item.append(
      node("strong", `${metric.value}${metric.unit ? " " + metric.unit : ""}`),
      node("span", metric.label),
    );
    if (metric.reference !== undefined) {
      item.append(
        node(
          "span",
          `vs ${metric.reference}${metric.unit ? " " + metric.unit : ""} ${metric.reference_label}`,
          "metric-reference",
        ),
      );
    }
    target.append(item);
  }
}
function selectEpisode(id) {
  episode = current.episodes.find((e) => e.id === id) || current.episodes[0];
  episodeSelect.value = episode.id;
  video.pause();
  video.src = episode.video;
  if (episode.poster) video.poster = episode.poster;
  else video.removeAttribute("poster");
  video.load();
  const still = document.getElementById("presentation-still");
  still.hidden = !episode.presentation_still;
  if (episode.presentation_still) {
    still.href = episode.presentation_still.path;
    still.title =
      "Path-traced presentation of recorded state; not the policy camera";
  } else still.removeAttribute("href");
  video.defaultPlaybackRate = Number(
    document.getElementById("playback-speed").value,
  );
  video.playbackRate = video.defaultPlaybackRate;
  document.getElementById("clock").textContent = "0.00 s";
  load();
  selectBlock(selectedBlock);
  renderChapters();
  renderMetrics();
  renderInspections();
  renderLanguagePreview();
}
function show(id, hash = false) {
  current = experiments.find((e) => e.id === id) || experiments[0];
  selectedBlock = null;
  tabs
    .querySelectorAll("button")
    .forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.id === current.id)),
    );
  const evidence = document.getElementById("evidence");
  evidence.href = current.evidence;
  evidence.textContent = "Evidence ↗";
  document.getElementById("status-run").textContent = `run: ${current.id}`;
  episodeSelect.replaceChildren(
    ...current.episodes.map((e) => {
      const o = node("option", e.title);
      o.value = e.id;
      return o;
    }),
  );
  document.getElementById("graph-title").textContent =
    current.graph_label || "CONTROLLER GRAPH";
  drawDiagram(structuredClone(current.graph));
  document.querySelectorAll(".block").forEach((g) => {
    g.setAttribute("role", "button");
    g.setAttribute("tabindex", "0");
    g.setAttribute("aria-label", current.graph.blocks[g.dataset.id].title);
    g.onclick = () =>
      selectBlock(selectedBlock === g.dataset.id ? null : g.dataset.id);
    g.onkeydown = (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        selectBlock(g.dataset.id);
      }
    };
  });
  selectEpisode(current.episodes[0].id);
  if (hash) history.replaceState(null, "", "#" + current.id);
}
const filterFields = [
  ["task", "Task"],
  ["fleet", "Humanoids"],
  ["policy", "Policy"],
  ["observation", "Observation"],
  ["status", "Outcome"],
  ["study", "Evidence study"],
];
function filterValue(experiment, field) {
  return field === "status" ? experiment.status : experiment.metadata?.[field];
}
function applyFilters() {
  let visible = 0;
  for (const button of tabs.querySelectorAll("button")) {
    const experiment = experiments.find((e) => e.id === button.dataset.id);
    const matches = [...document.querySelectorAll("#filters select")].every(
      (select) =>
        !select.value ||
        []
          .concat(filterValue(experiment, select.dataset.field) ?? [])
          .map(String)
          .includes(select.value),
    );
    button.hidden = !matches;
    if (matches) visible++;
  }
  document.getElementById("filter-empty").hidden = visible !== 0;
}
for (const [field, label] of filterFields) {
  const values = [
    ...new Set(
      experiments
        .flatMap((e) => [].concat(filterValue(e, field) ?? []))
        .filter((v) => v !== undefined),
    ),
  ];
  if (!values.length) continue;
  const wrapper = node("label", label);
  const select = node("select");
  select.dataset.field = field;
  select.setAttribute("aria-label", `Filter by ${label.toLowerCase()}`);
  const all = node("option", "All");
  all.value = "";
  select.append(all);
  for (const value of values) {
    const option = node("option", String(value));
    option.value = String(value);
    select.append(option);
  }
  select.onchange = applyFilters;
  wrapper.append(select);
  document.getElementById("filters").append(wrapper);
}
experiments.forEach((e, i) => {
  const b = node("button");
  b.type = "button";
  b.dataset.id = e.id;
  b.append(
    node("span", String(i + 1).padStart(2, "0"), "n"),
    node("span", e.title),
    node("span", e.status.toUpperCase(), `chip ${e.status}`),
  );
  b.onclick = () => show(e.id, true);
  tabs.append(b);
});
episodeSelect.onchange = () => selectEpisode(episodeSelect.value);
document.getElementById("playback-speed").onchange = (event) => {
  video.defaultPlaybackRate = Number(event.target.value);
  video.playbackRate = video.defaultPlaybackRate;
};
video.ontimeupdate = () => {
  document.getElementById("clock").textContent =
    video.currentTime.toFixed(2) + " s";
  updateSignals();
  updateChapter();
};
document.getElementById("note-form").onsubmit = (e) => {
  e.preventDefault();
  const field = document.getElementById("note"),
    text = field.value.trim();
  if (!text) return;
  notes.push({
    id: globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`,
    time: video.currentTime,
    block: selectedBlock,
    text,
  });
  field.value = "";
  save();
  renderNotes();
};
document.getElementById("export").onclick = () => {
  const blob = new Blob(
    [
      JSON.stringify(
        {
          schema_version: 1,
          experiment: current.id,
          episode: episode.id,
          video: episode.video,
          annotations: notes,
        },
        null,
        2,
      ),
    ],
    { type: "application/json" },
  );
  const url = URL.createObjectURL(blob),
    a = node("a");
  a.href = url;
  a.download = `${current.id}-${episode.id}-annotations.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
document.getElementById("import").onchange = async (e) => {
  const file = e.target.files[0];
  try {
    if (!file) return;
    if (file.size > 2000000) throw Error("File exceeds 2 MB.");
    const payload = JSON.parse(await file.text());
    if (
      payload.schema_version !== 1 ||
      payload.experiment !== current.id ||
      payload.episode !== episode.id ||
      payload.video !== episode.video
    )
      throw Error(
        "Select the matching experiment and episode before importing.",
      );
    if (
      !Array.isArray(payload.annotations) ||
      payload.annotations.length > 10000 ||
      !payload.annotations.every(validNote)
    )
      throw Error("Invalid annotation data.");
    const merged = new Map(notes.map((n) => [n.id, n]));
    for (const n of payload.annotations)
      merged.set(n.id, {
        id: n.id,
        time: n.time,
        block: n.block,
        text: n.text,
      });
    notes = [...merged.values()];
    save();
    renderNotes();
    noteStatus.textContent = `Imported ${payload.annotations.length} notes.`;
  } catch (error) {
    noteStatus.textContent = error.message;
  } finally {
    e.target.value = "";
  }
};
window.addEventListener("hashchange", () => show(location.hash.slice(1)));
show(location.hash.slice(1));
