/*
 * Hand-built inline SVG charts -- no charting library. Every render function
 * clears its container and redraws at the container's current pixel width,
 * so callers can simply re-invoke the same render function on window resize
 * (App does this, debounced, for whichever page is currently visible).
 */
const Charts = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function el(tag, attrs = {}) {
    const node = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    return node;
  }

  // ---- shared tooltip singleton ----
  let tooltipEl = null;
  function tooltip() {
    if (!tooltipEl) {
      tooltipEl = document.createElement("div");
      tooltipEl.className = "chart-tooltip";
      document.body.appendChild(tooltipEl);
    }
    return tooltipEl;
  }
  function showTooltip(html, clientX, clientY) {
    const t = tooltip();
    t.innerHTML = html;
    t.style.display = "block";
    const rect = t.getBoundingClientRect();
    let left = clientX + 14;
    let top = clientY + 14;
    if (left + rect.width > window.innerWidth - 8) left = clientX - rect.width - 14;
    if (top + rect.height > window.innerHeight - 8) top = clientY - rect.height - 14;
    t.style.left = `${left}px`;
    t.style.top = `${top}px`;
  }
  function hideTooltip() {
    if (tooltipEl) tooltipEl.style.display = "none";
  }

  function roundedTopRectPath(x, y, w, h, r) {
    if (h <= 0) return "";
    r = Math.max(0, Math.min(r, w / 2, h));
    return `M${x},${y + h} L${x},${y + r} Q${x},${y} ${x + r},${y} L${x + w - r},${y} ` +
      `Q${x + w},${y} ${x + w},${y + r} L${x + w},${y + h} Z`;
  }
  function roundedSideRectPath(x, y, w, h, r, side) {
    if (w <= 0) return "";
    r = Math.max(0, Math.min(r, w, h / 2));
    if (side === "right") {
      return `M${x},${y} L${x + w - r},${y} Q${x + w},${y} ${x + w},${y + r} L${x + w},${y + h - r} ` +
        `Q${x + w},${y + h} ${x + w - r},${y + h} L${x},${y + h} Z`;
    }
    return `M${x + w},${y} L${x + r},${y} Q${x},${y} ${x},${y + r} L${x},${y + h - r} ` +
      `Q${x},${y + h} ${x + r},${y + h} L${x + w},${y + h} Z`;
  }

  function niceMax(v) {
    if (v <= 0) return 1;
    const magnitude = Math.pow(10, Math.floor(Math.log10(v)));
    const norm = v / magnitude;
    let step;
    if (norm <= 1) step = 1;
    else if (norm <= 2) step = 2;
    else if (norm <= 5) step = 5;
    else step = 10;
    return step * magnitude;
  }

  function legendRow(container, items) {
    const row = document.createElement("div");
    row.className = "chart-legend";
    items.forEach((it) => {
      const item = document.createElement("span");
      item.className = "chart-legend-item";
      item.innerHTML = `<i style="background:${it.color}"></i>${Fmt.escapeHtml(it.label)}`;
      row.appendChild(item);
    });
    container.appendChild(row);
  }

  // -------------------------------------------------------------------
  // vertical bar chart -- category on x, value on y
  // -------------------------------------------------------------------
  function renderBarChart(container, opts) {
    const { data, height = 240, valueFormatter = Fmt.int, legend = null } = opts;
    container.innerHTML = "";
    if (legend) legendRow(container, legend);

    const width = Math.max(container.clientWidth || 480, 240);
    const pad = { top: 16, right: 16, bottom: 34, left: 46 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const maxVal = niceMax(Math.max(...data.map((d) => d.value), 1) * 1.05);

    const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "chart-svg" });

    const gridline = cssVar("--gridline");
    const baseline = cssVar("--baseline");
    const muted = cssVar("--text-muted");
    const ticks = 4;
    for (let i = 0; i <= ticks; i++) {
      const val = (maxVal / ticks) * i;
      const y = pad.top + plotH - (val / maxVal) * plotH;
      svg.appendChild(el("line", {
        x1: pad.left, x2: width - pad.right, y1: y, y2: y,
        stroke: i === 0 ? baseline : gridline, "stroke-width": 1,
      }));
      const t = el("text", { x: pad.left - 8, y: y + 3, "text-anchor": "end", fill: muted, "font-size": 11, class: "tabular" });
      t.textContent = valueFormatter(val);
      svg.appendChild(t);
    }

    const slot = plotW / data.length;
    const barWidth = Math.min(46, slot * 0.5);
    data.forEach((d, i) => {
      const barHeight = (d.value / maxVal) * plotH;
      const x = pad.left + i * slot + (slot - barWidth) / 2;
      const y = pad.top + plotH - barHeight;
      const color = d.color || cssVar("--series-1-blue");

      const path = el("path", { d: roundedTopRectPath(x, y, barWidth, barHeight, 4), fill: color, class: "chart-bar" });
      svg.appendChild(path);

      const label = el("text", {
        x: pad.left + i * slot + slot / 2, y: height - pad.bottom + 18,
        "text-anchor": "middle", fill: muted, "font-size": 11,
      });
      label.textContent = d.label;
      svg.appendChild(label);

      const hit = el("rect", {
        x: pad.left + i * slot, y: pad.top, width: slot, height: plotH,
        fill: "transparent", class: "chart-hit",
      });
      hit.addEventListener("mouseenter", () => path.classList.add("chart-bar-hover"));
      hit.addEventListener("mouseleave", () => { path.classList.remove("chart-bar-hover"); hideTooltip(); });
      hit.addEventListener("mousemove", (e) => {
        showTooltip(`<strong>${Fmt.escapeHtml(d.label)}</strong><br>${valueFormatter(d.value)}`, e.clientX, e.clientY);
      });
      svg.appendChild(hit);
    });

    container.appendChild(svg);
  }

  // -------------------------------------------------------------------
  // grouped vertical bar chart -- clusters of bars per category
  // -------------------------------------------------------------------
  function renderGroupedBarChart(container, opts) {
    const { groups, height = 260, valueFormatter = Fmt.int, legend = [] } = opts;
    container.innerHTML = "";
    if (legend.length) legendRow(container, legend);

    const width = Math.max(container.clientWidth || 480, 240);
    const pad = { top: 16, right: 16, bottom: 34, left: 46 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const allVals = groups.flatMap((g) => g.bars.map((b) => b.value));
    const maxVal = niceMax(Math.max(...allVals, 1) * 1.05);

    const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "chart-svg" });
    const gridline = cssVar("--gridline");
    const baseline = cssVar("--baseline");
    const muted = cssVar("--text-muted");
    const ticks = 4;
    for (let i = 0; i <= ticks; i++) {
      const val = (maxVal / ticks) * i;
      const y = pad.top + plotH - (val / maxVal) * plotH;
      svg.appendChild(el("line", { x1: pad.left, x2: width - pad.right, y1: y, y2: y, stroke: i === 0 ? baseline : gridline, "stroke-width": 1 }));
      const t = el("text", { x: pad.left - 8, y: y + 3, "text-anchor": "end", fill: muted, "font-size": 11, class: "tabular" });
      t.textContent = valueFormatter(val);
      svg.appendChild(t);
    }

    const slot = plotW / groups.length;
    groups.forEach((g, gi) => {
      const barsN = g.bars.length;
      const clusterW = slot * 0.6;
      const barW = clusterW / barsN;
      g.bars.forEach((b, bi) => {
        const barHeight = (b.value / maxVal) * plotH;
        const x = pad.left + gi * slot + (slot - clusterW) / 2 + bi * barW;
        const y = pad.top + plotH - barHeight;
        const path = el("path", { d: roundedTopRectPath(x, y, barW - 4, barHeight, 3), fill: b.color, class: "chart-bar" });
        svg.appendChild(path);

        const hit = el("rect", { x, y: pad.top, width: barW, height: plotH, fill: "transparent", class: "chart-hit" });
        hit.addEventListener("mouseenter", () => path.classList.add("chart-bar-hover"));
        hit.addEventListener("mouseleave", () => { path.classList.remove("chart-bar-hover"); hideTooltip(); });
        hit.addEventListener("mousemove", (e) => {
          showTooltip(`<strong>${Fmt.escapeHtml(g.label)} — ${Fmt.escapeHtml(b.label)}</strong><br>${valueFormatter(b.value)}`, e.clientX, e.clientY);
        });
        svg.appendChild(hit);
      });

      const label = el("text", { x: pad.left + gi * slot + slot / 2, y: height - pad.bottom + 18, "text-anchor": "middle", fill: muted, "font-size": 11 });
      label.textContent = g.label;
      svg.appendChild(label);
    });

    container.appendChild(svg);
  }

  // -------------------------------------------------------------------
  // horizontal bar chart -- for SHAP contributions (signed, diverging)
  // and global feature importance (unsigned, single series)
  // -------------------------------------------------------------------
  function renderHBarChart(container, opts) {
    const { data, diverging = false, valueFormatter = (v) => v.toFixed(3), rowHeight = 30, legend = null } = opts;
    container.innerHTML = "";
    if (legend) legendRow(container, legend);

    const width = Math.max(container.clientWidth || 480, 260);
    const labelW = Math.min(180, width * 0.36);
    const pad = { top: 8, right: 56, bottom: 8, left: labelW };
    const plotW = width - pad.left - pad.right;
    const height = pad.top + pad.bottom + data.length * rowHeight;
    const maxAbs = niceMax(Math.max(...data.map((d) => Math.abs(d.value)), 0.001));

    const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "chart-svg" });
    const muted = cssVar("--text-muted");
    const baseline = cssVar("--baseline");
    const positiveColor = cssVar("--series-2-orange");
    const negativeColor = cssVar("--series-1-blue");
    const singleColor = cssVar("--series-1-blue");

    const zeroX = diverging ? pad.left + plotW / 2 : pad.left;
    if (diverging) {
      svg.appendChild(el("line", { x1: zeroX, x2: zeroX, y1: pad.top, y2: height - pad.bottom, stroke: baseline, "stroke-width": 1 }));
    } else {
      svg.appendChild(el("line", { x1: pad.left, x2: pad.left, y1: pad.top, y2: height - pad.bottom, stroke: baseline, "stroke-width": 1 }));
    }

    data.forEach((d, i) => {
      const y = pad.top + i * rowHeight;
      const barH = rowHeight * 0.58;
      const barY = y + (rowHeight - barH) / 2;
      const halfW = diverging ? plotW / 2 : plotW;
      const frac = d.value / maxAbs;
      const barLen = Math.abs(frac) * halfW;
      let path, x;
      const color = diverging ? (d.value >= 0 ? positiveColor : negativeColor) : singleColor;
      if (diverging) {
        if (d.value >= 0) {
          x = zeroX;
          path = roundedSideRectPath(x, barY, barLen, barH, 4, "right");
        } else {
          x = zeroX - barLen;
          path = roundedSideRectPath(x, barY, barLen, barH, 4, "left");
        }
      } else {
        x = pad.left;
        path = roundedSideRectPath(x, barY, barLen, barH, 4, "right");
      }
      svg.appendChild(el("path", { d: path, fill: color, class: "chart-bar" }));

      const label = el("text", {
        x: pad.left - 10, y: y + rowHeight / 2 + 4, "text-anchor": "end", fill: cssVar("--text-secondary"), "font-size": 12,
      });
      label.textContent = d.label.length > 24 ? d.label.slice(0, 23) + "…" : d.label;
      svg.appendChild(label);

      const valueLabelX = diverging ? (d.value >= 0 ? x + barLen + 6 : x - 6) : x + barLen + 6;
      const anchor = diverging && d.value < 0 ? "end" : "start";
      const valLabel = el("text", { x: valueLabelX, y: y + rowHeight / 2 + 4, "text-anchor": anchor, fill: muted, "font-size": 11, class: "tabular" });
      valLabel.textContent = valueFormatter(d.value);
      svg.appendChild(valLabel);

      const hit = el("rect", { x: pad.left, y, width: plotW, height: rowHeight, fill: "transparent", class: "chart-hit" });
      hit.addEventListener("mousemove", (e) => {
        showTooltip(`<strong>${Fmt.escapeHtml(d.label)}</strong><br>${valueFormatter(d.value)}`, e.clientX, e.clientY);
      });
      hit.addEventListener("mouseleave", hideTooltip);
      svg.appendChild(hit);
    });

    container.appendChild(svg);
  }

  // -------------------------------------------------------------------
  // line / area chart with crosshair tooltip -- time series & cost sweep
  // -------------------------------------------------------------------
  function renderLineChart(container, opts) {
    const {
      data, height = 260, color = cssVar("--series-1-blue"), area = true,
      xFormatter = (v) => v, yFormatter = Fmt.int, xTickCount = 6, markers = [],
    } = opts;
    container.innerHTML = "";

    const width = Math.max(container.clientWidth || 480, 300);
    const pad = { top: 20, right: 20, bottom: 30, left: 52 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const xs = data.map((d) => d.x);
    const ys = data.map((d) => d.y);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMax = niceMax(Math.max(...ys, 0.001) * 1.15);
    const yMin = 0;

    const xScale = (x) => pad.left + (xMax === xMin ? 0 : ((x - xMin) / (xMax - xMin)) * plotW);
    const yScale = (y) => pad.top + plotH - ((y - yMin) / (yMax - yMin)) * plotH;

    const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "chart-svg" });
    const gridline = cssVar("--gridline");
    const baseline = cssVar("--baseline");
    const muted = cssVar("--text-muted");

    const yTicks = 4;
    for (let i = 0; i <= yTicks; i++) {
      const val = (yMax / yTicks) * i;
      const y = yScale(val);
      svg.appendChild(el("line", { x1: pad.left, x2: width - pad.right, y1: y, y2: y, stroke: i === 0 ? baseline : gridline, "stroke-width": 1 }));
      const t = el("text", { x: pad.left - 8, y: y + 3, "text-anchor": "end", fill: muted, "font-size": 11, class: "tabular" });
      t.textContent = yFormatter(val);
      svg.appendChild(t);
    }
    for (let i = 0; i <= xTickCount; i++) {
      const val = xMin + ((xMax - xMin) / xTickCount) * i;
      const x = xScale(val);
      const t = el("text", { x, y: height - pad.bottom + 18, "text-anchor": "middle", fill: muted, "font-size": 11 });
      t.textContent = xFormatter(val);
      svg.appendChild(t);
    }

    const linePoints = data.map((d) => `${xScale(d.x)},${yScale(d.y)}`).join(" ");
    if (area) {
      const areaPoints = `${xScale(xs[0])},${yScale(0)} ${linePoints} ${xScale(xs[xs.length - 1])},${yScale(0)}`;
      svg.appendChild(el("polygon", { points: areaPoints, fill: color, opacity: 0.14, stroke: "none" }));
    }
    svg.appendChild(el("polyline", { points: linePoints, fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round" }));

    if (data.length <= 60) {
      data.forEach((d) => {
        svg.appendChild(el("circle", { cx: xScale(d.x), cy: yScale(d.y), r: 4, fill: color }));
      });
    }

    markers.forEach((m) => {
      const x = xScale(m.x);
      svg.appendChild(el("line", { x1: x, x2: x, y1: pad.top, y2: pad.top + plotH, stroke: m.color, "stroke-width": 1.5, "stroke-dasharray": "4,3" }));
      const t = el("text", { x: x + 4, y: pad.top + 12, fill: m.color, "font-size": 11, "font-weight": 600 });
      t.textContent = m.label;
      svg.appendChild(t);
    });

    // crosshair overlay
    const crosshair = el("line", { x1: 0, x2: 0, y1: pad.top, y2: pad.top + plotH, stroke: muted, "stroke-width": 1, opacity: 0 });
    const crossDot = el("circle", { r: 4.5, fill: color, stroke: cssVar("--surface-1"), "stroke-width": 2, opacity: 0 });
    svg.appendChild(crosshair);
    svg.appendChild(crossDot);

    const overlay = el("rect", { x: pad.left, y: pad.top, width: plotW, height: plotH, fill: "transparent", class: "chart-hit" });
    overlay.addEventListener("mousemove", (e) => {
      const rect = svg.getBoundingClientRect();
      const scaleX = width / rect.width;
      const mouseX = (e.clientX - rect.left) * scaleX;
      const targetVal = xMin + ((mouseX - pad.left) / plotW) * (xMax - xMin);
      let nearest = data[0];
      let bestDist = Infinity;
      for (const d of data) {
        const dist = Math.abs(d.x - targetVal);
        if (dist < bestDist) { bestDist = dist; nearest = d; }
      }
      const nx = xScale(nearest.x);
      const ny = yScale(nearest.y);
      crosshair.setAttribute("x1", nx); crosshair.setAttribute("x2", nx); crosshair.setAttribute("opacity", 1);
      crossDot.setAttribute("cx", nx); crossDot.setAttribute("cy", ny); crossDot.setAttribute("opacity", 1);
      showTooltip(`<strong>${xFormatter(nearest.x)}</strong><br>${yFormatter(nearest.y)}`, e.clientX, e.clientY);
    });
    overlay.addEventListener("mouseleave", () => {
      crosshair.setAttribute("opacity", 0);
      crossDot.setAttribute("opacity", 0);
      hideTooltip();
    });
    svg.appendChild(overlay);

    container.appendChild(svg);
  }

  return {
    renderBarChart, renderGroupedBarChart, renderHBarChart, renderLineChart,
    hideTooltip, cssVar,
  };
})();
