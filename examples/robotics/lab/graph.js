const W = 150,
  COL = 205,
  ROW = 135,
  X0 = 40,
  Y0 = 16,
  HEAD = 30,
  PORT = 18;
function drawDiagram(spec) {
  const blocks = Object.values(spec.blocks),
    minRow = Math.min(...blocks.map((b) => b.row));
  for (const b of blocks) {
    b.x = X0 + b.col * COL;
    b.y = Y0 + (b.row - minRow) * ROW;
    b.h = HEAD + Math.max(b.ins.length, b.outs.length, 1) * PORT + 6;
  }
  const rowTop = (r) => Y0 + (r - minRow) * ROW,
    rowBot = (r) =>
      Math.max(
        rowTop(r) + HEAD + PORT + 6,
        ...blocks.filter((b) => b.row === r).map((b) => b.y + b.h),
      );
  const portY = (b, i) => b.y + HEAD + 10 + i * PORT;
  const used = {},
    take = (key, base, step) => {
      const k = used[key] || 0;
      used[key] = k + 1;
      return base + k * step;
    };
  const peek = (key, base, step) => base + (used[key] || 0) * step;
  const gapX = (g) => X0 + g * COL + W + 12;
  const hits = (y, a, c) =>
    blocks.some(
      (b) =>
        y > b.y - 3 &&
        y < b.y + b.h + 3 &&
        Math.max(a, c) > b.x &&
        Math.min(a, c) < b.x + W,
    );
  let bottom = Math.max(...blocks.map((b) => b.y + b.h)) + 20,
    maxX = Math.max(...blocks.map((b) => b.x + W));
  const paths = spec.wires.map(([from, to]) => {
    const [sk, sp] = from.split(/\.(.*)/),
      [tk, tp] = to.split(/\.(.*)/),
      s = spec.blocks[sk],
      t = spec.blocks[tk];
    const x1 = s.x + W + 3,
      y1 = portY(s, s.outs.indexOf(sp)),
      x2 = t.x - 3,
      y2 = portY(t, t.ins.indexOf(tp));
    if (t.col > s.col) {
      if (t.col === s.col + 1 || (y1 === y2 && !hits(y1, x1, x2))) {
        const cx = take("g" + s.col, gapX(s.col), 7);
        return `M${x1} ${y1}H${cx}V${y2}H${x2}`;
      }
      const near = peek("g" + (t.col - 1), gapX(t.col - 1), 7);
      if (!hits(y1, x1, near)) {
        take("g" + (t.col - 1), 0, 0);
        return `M${x1} ${y1}H${near}V${y2}H${x2}`;
      }
      const far = peek("g" + s.col, gapX(s.col), 7);
      if (!hits(y2, far, x2)) {
        take("g" + s.col, 0, 0);
        return `M${x1} ${y1}H${far}V${y2}H${x2}`;
      }
      const r = y2 > y1 ? s.row : s.row - 1,
        yl = take("r" + r, (rowBot(r) + rowTop(r + 1)) / 2 - 6, 6);
      const a = take("g" + s.col, gapX(s.col), 7),
        c = take("g" + (t.col - 1), gapX(t.col - 1), 7);
      return `M${x1} ${y1}H${a}V${yl}H${c}V${y2}H${x2}`;
    }
    const a = take("g" + s.col, gapX(s.col), 7),
      c = take("g" + (t.col - 1), gapX(t.col - 1), 7),
      yl = take("bottom", bottom, 8);
    maxX = Math.max(maxX, a);
    return `M${x1} ${y1}H${a}V${yl}H${c}V${y2}H${x2}`;
  });
  const height = used.bottom ? bottom + used.bottom * 8 : bottom - 6;
  const label = (x, y, t, anchor) =>
    `<text class="port-label" x="${x}" y="${y}" text-anchor="${anchor}">${escapeHTML(t)}</text>`;
  const port = (x, y) =>
    `<rect class="port" x="${x - 3}" y="${y - 3}" width="6" height="6"/>`;
  const ends = spec.wires.map(([f, t]) => [f.split(".")[0], t.split(".")[0]]);
  const el = document.getElementById("diagram");
  el.innerHTML = `<svg viewBox="0 0 ${maxX + 30} ${height}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Controller block diagram" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="var(--wire)"/></marker></defs>
${paths.map((d, i) => `<path class="wire" data-from="${ends[i][0]}" data-to="${ends[i][1]}" d="${d}" marker-end="url(#arrow)"/>`).join("")}
${Object.entries(spec.blocks)
  .map(
    ([
      k,
      b,
    ]) => `<g class="block k-${b.kind}" data-id="${k}"><rect class="body" x="${b.x}" y="${b.y}" width="${W}" height="${b.h}"/><rect class="head" x="${b.x}" y="${b.y}" width="${W}" height="${HEAD - 4}"/>
<text class="title" x="${b.x + W / 2}" y="${b.y + 17}" text-anchor="middle">${escapeHTML(b.title)}</text>
${b.ins.map((p, i) => port(b.x, portY(b, i)) + label(b.x + 8, portY(b, i) + 4, p, "start")).join("")}
${b.outs.map((p, i) => port(b.x + W, portY(b, i)) + label(b.x + W - 8, portY(b, i) + 4, p, "end")).join("")}</g>`,
  )
  .join("")}
</svg>`;
  const svg = el.querySelector("svg");
  svg.querySelectorAll(".block").forEach((g) => {
    g.addEventListener("mouseenter", () => {
      const id = g.dataset.id,
        linked = new Set([id]);
      svg.classList.add("focus");
      svg.querySelectorAll(".wire").forEach((w) => {
        const hot = w.dataset.from === id || w.dataset.to === id;
        w.classList.toggle("hot", hot);
        if (hot) {
          linked.add(w.dataset.from);
          linked.add(w.dataset.to);
        }
      });
      svg
        .querySelectorAll(".block")
        .forEach((b) => b.classList.toggle("hot", linked.has(b.dataset.id)));
    });
    g.addEventListener("mouseleave", () => svg.classList.remove("focus"));
  });
  document.getElementById("graph-stats").textContent =
    `${blocks.length} BLOCKS · ${spec.wires.length} WIRES`;
}
