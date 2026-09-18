// Shared helpers for the demo web apps. Nothing here is visible to agents except rendered DOM.

const params = new URLSearchParams(location.search);
export const seed = Number(params.get("seed") || 1);

export function rng(s = seed) {
  let a = s >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  next.int = (lo, hi) => lo + Math.floor(next() * (hi - lo + 1));
  next.pick = (arr) => arr[Math.floor(next() * arr.length)];
  next.chance = (p) => next() < p;
  next.shuffle = (arr) => {
    const out = [...arr];
    for (let i = out.length - 1; i > 0; i--) {
      const j = Math.floor(next() * (i + 1));
      [out[i], out[j]] = [out[j], out[i]];
    }
    return out;
  };
  return next;
}

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v == null) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "text") el.textContent = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null) el.append(c.nodeType ? c : document.createTextNode(String(c)));
  return el;
}

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Simulated server latency, announced to assistive tech (and agents) via aria-busy.
export async function server(ms) {
  document.body.setAttribute("aria-busy", "true");
  await sleep(ms);
  document.body.removeAttribute("aria-busy");
}

export function toast(text, kind = "") {
  let box = document.getElementById("toasts");
  if (!box) document.body.append((box = h("div", { id: "toasts" })));
  const t = h("div", { class: `toast ${kind}`, role: kind === "error" ? "alert" : "status", text });
  box.append(t);
  setTimeout(() => t.remove(), 2600);
}

export function modal(title, body, buttons) {
  return new Promise((resolve) => {
    const dlg = h("div", { role: "dialog", "aria-modal": "true", "aria-label": title }, h("h2", { text: title }), h("p", { text: body }));
    const actions = h("div", { class: "actions" });
    for (const [label, value, cls] of buttons) {
      actions.append(h("button", { class: cls || "", onclick: () => { back.remove(); resolve(value); } }, label));
    }
    dlg.append(actions);
    const back = h("div", { class: "backdrop" }, dlg);
    document.body.append(back);
  });
}

// Accessible custom combobox (button + listbox), operated with clicks.
export function combobox(id, label, options, onchange) {
  const lbl = h("label", { id: `${id}-label` }, label);
  const btn = h("button", { type: "button", id, role: "combobox", "aria-haspopup": "listbox", "aria-expanded": "false", "aria-labelledby": `${id}-label` }, "Select…");
  const wrap = h("div", { class: "combo" }, btn);
  let list = null;
  const close = () => { list?.remove(); list = null; btn.setAttribute("aria-expanded", "false"); };
  btn.addEventListener("click", () => {
    if (list) return close();
    list = h("div", { role: "listbox", "aria-label": label },
      options.map((o) => h("button", { type: "button", role: "option", onclick: () => { btn.textContent = o; btn.dataset.value = o; close(); onchange?.(o); } }, o)));
    wrap.append(list);
    btn.setAttribute("aria-expanded", "true");
  });
  return { el: h("div", { class: "field" }, lbl, wrap), get value() { return btn.dataset.value || ""; }, reset() { btn.textContent = "Select…"; delete btn.dataset.value; } };
}

// ---- visual agent cursor, driven only by the real mouse events the agent dispatches
function installCursor() {
  const cur = h("div", { id: "tc-cursor", "aria-hidden": "true" });
  cur.innerHTML = '<svg width="18" height="22" viewBox="0 0 18 22"><path d="M1 1 L1 17 L5.5 13 L8.5 20 L11 19 L8 12 L14 12 Z" fill="#ff3d7f" stroke="#fff" stroke-width="1.4"/></svg>';
  const cap = h("div", { id: "tc-caption", "aria-hidden": "true" });
  document.body.append(cur, cap);
  addEventListener("mousemove", (e) => { cur.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`; }, true);
  addEventListener("mousedown", (e) => {
    const r = h("div", { class: "tc-ripple", "aria-hidden": "true" });
    r.style.left = `${e.clientX}px`; r.style.top = `${e.clientY}px`;
    document.body.append(r); setTimeout(() => r.remove(), 460);
  }, true);
  window.__tcCaption = (text) => { cap.textContent = text || ""; };
}
if (document.readyState === "loading") addEventListener("DOMContentLoaded", installCursor); else installCursor();
