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

  // -------------------------------------------------------------------
  // scatter plot -- ML-focused fraud detection visualization
  // -------------------------------------------------------------------
  function renderScatterPlot(container, opts) {
    const { data, height = 400, xLabel, yLabel, colorBy = 'tier' } = opts;
    container.innerHTML = "";
    if (!data || data.length === 0) {
      container.innerHTML = '<div class="empty-state">No data available for scatter plot</div>';
      return;
    }

    // Create wrapper with ML controls
    const wrapper = document.createElement('div');
    wrapper.className = 'scatter-plot-wrapper';

    // ML Control Panel
    const controlPanel = document.createElement('div');
    controlPanel.className = 'ml-control-panel';
    controlPanel.innerHTML = `
      <div class="ml-controls-row">
        <div class="ml-control-group">
          <label class="ml-control-label">Risk Threshold</label>
          <div class="threshold-slider-container">
            <input type="range" class="threshold-slider" id="risk-threshold"
                   min="0" max="1" step="0.01" value="0.5">
            <div class="threshold-value" id="threshold-display">0.50</div>
          </div>
          <div class="threshold-stats">
            <span id="above-threshold">0</span> transactions above threshold
          </div>
        </div>

        <div class="ml-control-group">
          <label class="ml-control-label">Visualization Mode</label>
          <div class="ml-mode-buttons">
            <button class="ml-mode-btn active" data-mode="risk-stratification">Risk Stratification</button>
            <button class="ml-mode-btn" data-mode="score-based">Score Distribution</button>
            <button class="ml-mode-btn" data-mode="confidence">Model Confidence</button>
          </div>
        </div>

        <div class="ml-control-group">
          <label class="ml-control-label">Display Options</label>
          <div class="ml-toggle-group">
            <label class="ml-toggle">
              <input type="checkbox" id="show-decision-boundary" checked>
              <span>Decision Boundaries</span>
            </label>
            <label class="ml-toggle">
              <input type="checkbox" id="show-density-heatmap">
              <span>Density Heatmap</span>
            </label>
          </div>
        </div>
      </div>
    `;
    wrapper.appendChild(controlPanel);

    // Risk legend
    const riskLegend = document.createElement('div');
    riskLegend.className = 'risk-legend';
    riskLegend.innerHTML = `
      <div class="legend-item">
        <div class="legend-marker priority"></div>
        <div class="legend-text">
          <strong>Priority Review</strong>
          <span>Score ≥ 0.70 | High fraud probability</span>
        </div>
        <div class="legend-count" id="priority-count">0</div>
      </div>
      <div class="legend-item">
        <div class="legend-marker standard"></div>
        <div class="legend-text">
          <strong>Standard Review</strong>
          <span>0.50 ≤ Score < 0.70 | Moderate risk</span>
        </div>
        <div class="legend-count" id="standard-count">0</div>
      </div>
      <div class="legend-item">
        <div class="legend-marker normal"></div>
        <div class="legend-text">
          <strong>Normal</strong>
          <span>Score < 0.50 | Low risk</span>
        </div>
        <div class="legend-count" id="normal-count">0</div>
      </div>
    `;
    wrapper.appendChild(riskLegend);

    // SVG container
    const svgContainer = document.createElement('div');
    svgContainer.className = 'scatter-svg-container';
    wrapper.appendChild(svgContainer);

    container.appendChild(wrapper);

    const width = Math.max(container.clientWidth || 480, 480);
    const pad = { top: 30, right: 30, bottom: 60, left: 70 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const xs = data.map(d => d.x);
    const ys = data.map(d => d.y);
    const scores = data.map(d => d.score || 0);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const yMin = 0; // Start from 0 for risk scores
    const yMax = 1; // Risk scores go to 1

    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin;
    const xPad = xRange * 0.05;

    const xScale = (x) => pad.left + ((x - xMin + xPad) / (xRange + 2 * xPad)) * plotW;
    const yScale = (y) => pad.top + plotH - (y / yMax) * plotH;

    const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "chart-svg" });
    const gridline = cssVar("--gridline");
    const baseline = cssVar("--baseline");
    const muted = cssVar("--text-muted");

    // grid lines
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const val = yMin + (yRange / yTicks) * i;
      const y = yScale(val);
      svg.appendChild(el("line", {
        x1: pad.left, x2: width - pad.right, y1: y, y2: y,
        stroke: i === 0 ? baseline : gridline, "stroke-width": 1
      }));
      const t = el("text", { x: pad.left - 8, y: y + 3, "text-anchor": "end", fill: muted, "font-size": 10, class: "tabular" });
      t.textContent = val.toFixed(2);
      svg.appendChild(t);
    }

    const xTicks = 5;
    for (let i = 0; i <= xTicks; i++) {
      const val = xMin + (xRange / xTicks) * i;
      const x = xScale(val);
      svg.appendChild(el("line", {
        x1: x, x2: x, y1: pad.top, y2: pad.top + plotH,
        stroke: gridline, "stroke-width": 1
      }));
      const t = el("text", { x, y: height - pad.bottom + 18, "text-anchor": "middle", fill: muted, "font-size": 10 });
      t.textContent = val.toFixed(2);
      svg.appendChild(t);
    }

    // axis labels
    const xLabelEl = el("text", {
      x: pad.left + plotW / 2, y: height - 10, "text-anchor": "middle",
      fill: cssVar("--text-secondary"), "font-size": 12, "font-weight": 600
    });
    xLabelEl.textContent = xLabel || "X Axis";
    svg.appendChild(xLabelEl);

    const yLabelEl = el("text", {
      x: 15, y: pad.top + plotH / 2, "text-anchor": "middle",
      fill: cssVar("--text-secondary"), "font-size": 12, "font-weight": 600,
      transform: `rotate(-90, 15, ${pad.top + plotH / 2})`
    });
    yLabelEl.textContent = yLabel || "Y Axis";
    svg.appendChild(yLabelEl);

    // color mapping
    const colorMap = {
      priority: cssVar("--status-critical"),
      standard: cssVar("--status-warning"),
      normal: cssVar("--status-good"),
    };

    // Group data by tier
    const tierGroups = {
      priority: data.filter(d => d.tier === 'priority'),
      standard: data.filter(d => d.tier === 'standard'),
      normal: data.filter(d => d.tier === 'normal'),
    };

    // Plot points grouped by tier
    const circlesByTier = { priority: [], standard: [], normal: [] };

    Object.entries(tierGroups).forEach(([tier, tierData]) => {
      tierData.forEach((d) => {
        const cx = xScale(d.x);
        const cy = yScale(d.y);
        const color = colorMap[tier];
        const r = d.score ? 3 + (d.score * 4) : 4;

        const circle = el("circle", {
          cx, cy, r, fill: color, opacity: 0, stroke: "none",
          class: `chart-point tier-${tier}`,
          'data-tier': tier,
          'data-score': d.score || 0
        });

        // Add glow ring effect
        const glowRing = el("circle", {
          cx, cy, r: r + 4, fill: "none",
          stroke: color,
          "stroke-width": 2,
          opacity: 0,
          class: `chart-glow-ring tier-${tier}`
        });

        circle.addEventListener("mouseenter", (e) => {
          // Animate the main dot
          gsap.to(circle, {
            attr: { r: r + 3 },
            duration: 0.3,
            ease: "elastic.out(1, 0.5)"
          });
          circle.setAttribute("opacity", 1);

          // Animate the glow ring
          gsap.to(glowRing, {
            attr: { r: r + 12 },
            opacity: 0.6,
            duration: 0.4,
            ease: "power2.out"
          });

          gsap.to(glowRing, {
            opacity: 0,
            duration: 0.6,
            delay: 0.4,
            ease: "power2.in"
          });

          const html = `
            <strong>${Fmt.escapeHtml(d.id || "Transaction")}</strong><br>
            ${xLabel}: ${d.x.toFixed(4)}<br>
            ${yLabel}: ${d.y.toFixed(4)}<br>
            Risk: ${Fmt.escapeHtml(tier || "unknown")}<br>
            Score: ${d.score ? d.score.toFixed(4) : "N/A"}
          `;
          showTooltip(html, e.clientX, e.clientY);
        });

        circle.addEventListener("mouseleave", () => {
          gsap.to(circle, {
            attr: { r: r },
            duration: 0.3,
            ease: "elastic.out(1, 0.5)"
          });
          const currentOpacity = circle.getAttribute("data-visible") === "true" ? 0.7 : 0.2;
          circle.setAttribute("opacity", currentOpacity);
          hideTooltip();
        });

        if (d.clickable) {
          circle.style.cursor = "pointer";
          circle.addEventListener("click", () => {
            if (d.onClick) d.onClick(d);
          });
        }

        svg.appendChild(glowRing);
        circlesByTier[tier].push(circle);
        svg.appendChild(circle);
      });
    });

    svgContainer.appendChild(svg);

    // Size legends for each tier
    const sizeLegends = {
      priority: {
        title: "Priority Review Risk Levels",
        color: colorMap.priority,
        descriptions: [
          { size: "Small dots", range: "0.70-0.80", meaning: "Elevated risk - requires review" },
          { size: "Medium dots", range: "0.80-0.90", meaning: "High risk - priority attention" },
          { size: "Large dots", range: "0.90-1.00", meaning: "Critical risk - immediate action" }
        ]
      },
      standard: {
        title: "Standard Review Risk Levels",
        color: colorMap.standard,
        descriptions: [
          { size: "Small dots", range: "0.50-0.60", meaning: "Moderate risk - standard monitoring" },
          { size: "Medium dots", range: "0.60-0.65", meaning: "Notable risk - closer review" },
          { size: "Large dots", range: "0.65-0.70", meaning: "Concerning risk - detailed check" }
        ]
      },
      normal: {
        title: "Normal Transaction Risk Levels",
        color: colorMap.normal,
        descriptions: [
          { size: "Small dots", range: "0.00-0.30", meaning: "Low risk - typical transaction" },
          { size: "Medium dots", range: "0.30-0.45", meaning: "Slight elevation - monitor" },
          { size: "Large dots", range: "0.45-0.50", meaning: "Upper normal - watch closely" }
        ]
      }
    };

    // Function to show size legend
    function showSizeLegend(tier) {
      if (tier === 'all') {
        sizeLegendContainer.innerHTML = '';
        return;
      }

      const legend = sizeLegends[tier];
      sizeLegendContainer.innerHTML = `
        <div class="size-legend">
          <h3 style="color: ${legend.color}; margin: 0 0 12px 0;">${legend.title}</h3>
          <div class="size-legend-items">
            ${legend.descriptions.map(desc => `
              <div class="size-legend-item">
                <div class="size-indicator">
                  <svg width="80" height="30">
                    <circle cx="15" cy="15" r="${desc.size.includes('Small') ? '4' : desc.size.includes('Medium') ? '6' : '8'}"
                            fill="${legend.color}" opacity="0.8"/>
                  </svg>
                </div>
                <div class="size-legend-text">
                  <div class="size-legend-range">${desc.range} Risk Score</div>
                  <div class="size-legend-meaning">${desc.meaning}</div>
                </div>
              </div>
            `).join('')}
          </div>
        </div>
      `;

      gsap.fromTo(sizeLegendContainer.querySelector('.size-legend'),
        { opacity: 0, y: 20 },
        { opacity: 1, y: 0, duration: 0.6, ease: "power2.out" }
      );
    }

    // Create particle burst effect
    function createParticleBurst(tier) {
      const colors = {
        priority: cssVar("--status-critical"),
        standard: cssVar("--status-warning"),
        normal: cssVar("--status-good")
      };
      const color = colors[tier];
      const particleCount = 20;

      for (let i = 0; i < particleCount; i++) {
        const particle = el("circle", {
          cx: width / 2,
          cy: height / 2,
          r: 2 + Math.random() * 3,
          fill: color,
          opacity: 0.8
        });
        svg.appendChild(particle);

        const angle = (Math.PI * 2 * i) / particleCount;
        const distance = 100 + Math.random() * 150;
        const targetX = width / 2 + Math.cos(angle) * distance;
        const targetY = height / 2 + Math.sin(angle) * distance;

        gsap.to(particle, {
          attr: { cx: targetX, cy: targetY },
          opacity: 0,
          duration: 1 + Math.random() * 0.5,
          ease: "power2.out",
          onComplete: () => svg.removeChild(particle)
        });
      }
    }

    // Stage animation function
    let currentStage = -1;
    function animateToStage(stage) {
      if (currentStage === stage) return;
      currentStage = stage;

      // Update button states
      document.querySelectorAll('.stage-btn').forEach((btn, idx) => {
        btn.classList.toggle('active', idx === stage);
      });

      // Animate progress bar
      const progressBar = wrapper.querySelector('.stage-progress-bar');
      gsap.to(progressBar, {
        width: `${((stage + 1) / 4) * 100}%`,
        duration: 0.5,
        ease: "power2.inOut"
      });

      const stages = ['priority', 'standard', 'normal'];

      // Update background glow
      svgContainer.className = 'scatter-svg-container';
      if (stage < 3) {
        svgContainer.classList.add(`stage-${stages[stage]}`);
      }

      if (stage === 3) {
        // Show all
        showSizeLegend('all');
        Object.values(circlesByTier).flat().forEach(circle => {
          circle.setAttribute("data-visible", "true");
          gsap.to(circle, {
            opacity: 0.7,
            duration: 0.6,
            delay: Math.random() * 0.3,
            ease: "power2.out"
          });
        });
      } else {
        // Show specific stage with enhanced effects
        const activeTier = stages[stage];
        showSizeLegend(activeTier);

        // Trigger particle burst
        createParticleBurst(activeTier);

        stages.forEach((tier, idx) => {
          const circles = circlesByTier[tier];
          const isActive = idx === stage;

          circles.forEach((circle, circleIdx) => {
            circle.setAttribute("data-visible", isActive ? "true" : "false");

            if (isActive) {
              // Enhanced reveal animation for active tier
              gsap.fromTo(circle, {
                opacity: 0,
                attr: { r: 0 }
              }, {
                opacity: 0.7,
                attr: { r: circle.getAttribute('r') || 4 },
                duration: 0.8,
                delay: circleIdx * 0.003,
                ease: "elastic.out(1, 0.6)",
                onComplete: () => {
                  // Small pulse on complete
                  gsap.to(circle, {
                    attr: { r: parseFloat(circle.getAttribute('r')) + 1 },
                    duration: 0.2,
                    yoyo: true,
                    repeat: 1,
                    ease: "power2.inOut"
                  });
                }
              });

              // Add a flash effect
              gsap.fromTo(circle, {
                opacity: 0.9
              }, {
                opacity: 0.7,
                duration: 0.4,
                delay: circleIdx * 0.003 + 0.8,
                ease: "power2.out"
              });
            } else {
              // Fade out inactive tiers
              gsap.to(circle, {
                opacity: 0.15,
                duration: 0.4,
                ease: "power2.out"
              });
            }
          });
        });
      }
    }

    // Attach button handlers
    stageControls.querySelectorAll('.stage-btn').forEach((btn, idx) => {
      btn.addEventListener('click', () => animateToStage(idx));
    });

    // Start with first stage
    setTimeout(() => animateToStage(0), 100);
  }

  // -------------------------------------------------------------------
  // LOF (Local Outlier Factor) scatter plot -- density-based outlier viz
  // -------------------------------------------------------------------
  function renderLOFChart(container, opts) {
    const { data, height = 360 } = opts;
    container.innerHTML = "";
    if (!data || data.length === 0) {
      container.innerHTML = '<div class="empty-state">No data available</div>';
      return;
    }

    const width = Math.max(container.clientWidth || 480, 480);
    const pad = { top: 24, right: 30, bottom: 56, left: 64 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "chart-svg" });
    const gridline = cssVar("--gridline");
    const baseline = cssVar("--baseline");
    const muted = cssVar("--text-muted");

    const xs = data.map(d => d.x);
    const ys = data.map(d => d.lofScore);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    const xPad = xRange * 0.05;
    const yPad = yRange * 0.05;

    const xScale = (x) => pad.left + ((x - xMin + xPad) / (xRange + 2 * xPad)) * plotW;
    const yScale = (y) => pad.top + plotH - ((y - yMin + yPad) / (yRange + 2 * yPad)) * plotH;

    // Grid lines
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const val = yMin + (yRange / yTicks) * i;
      const y = yScale(val);
      svg.appendChild(el("line", {
        x1: pad.left, x2: width - pad.right, y1: y, y2: y,
        stroke: i === 0 ? baseline : gridline, "stroke-width": 1
      }));
      const t = el("text", { x: pad.left - 8, y: y + 3, "text-anchor": "end", fill: muted, "font-size": 10, class: "tabular" });
      t.textContent = val.toFixed(2);
      svg.appendChild(t);
    }

    const xTicks = 6;
    for (let i = 0; i <= xTicks; i++) {
      const val = xMin + (xRange / xTicks) * i;
      const x = xScale(val);
      const t = el("text", { x, y: height - pad.bottom + 18, "text-anchor": "middle", fill: muted, "font-size": 10 });
      t.textContent = val.toFixed(0);
      svg.appendChild(t);
    }

    // Axis labels
    const xLabelEl = el("text", {
      x: pad.left + plotW / 2, y: height - 8, "text-anchor": "middle",
      fill: cssVar("--text-secondary"), "font-size": 12, "font-weight": 600
    });
    xLabelEl.textContent = "Transaction Index";
    svg.appendChild(xLabelEl);

    const yLabelEl = el("text", {
      x: 14, y: pad.top + plotH / 2, "text-anchor": "middle",
      fill: cssVar("--text-secondary"), "font-size": 12, "font-weight": 600,
      transform: `rotate(-90, 14, ${pad.top + plotH / 2})`
    });
    yLabelEl.textContent = "Outlier Score";
    svg.appendChild(yLabelEl);

    // Threshold lines
    const outlierThreshold = data.length > 0 ? (() => {
      const sorted = [...ys].sort((a, b) => a - b);
      return sorted[Math.floor(sorted.length * 0.85)] || 1.5;
    })() : 1.5;

    const borderlineThreshold = data.length > 0 ? (() => {
      const sorted = [...ys].sort((a, b) => a - b);
      return sorted[Math.floor(sorted.length * 0.70)] || 1.2;
    })() : 1.2;

    // Draw threshold zones
    const outlierY = yScale(outlierThreshold);
    const borderlineY = yScale(borderlineThreshold);

    svg.appendChild(el("rect", {
      x: pad.left, y: pad.top, width: plotW, height: Math.max(0, outlierY - pad.top),
      fill: cssVar("--status-critical"), opacity: 0.04
    }));
    svg.appendChild(el("rect", {
      x: pad.left, y: outlierY, width: plotW, height: Math.max(0, borderlineY - outlierY),
      fill: cssVar("--status-warning"), opacity: 0.04
    }));

    // Threshold dashed lines
    svg.appendChild(el("line", {
      x1: pad.left, x2: width - pad.right, y1: outlierY, y2: outlierY,
      stroke: cssVar("--status-critical"), "stroke-width": 1.5, "stroke-dasharray": "6,4", opacity: 0.7
    }));
    svg.appendChild(el("line", {
      x1: pad.left, x2: width - pad.right, y1: borderlineY, y2: borderlineY,
      stroke: cssVar("--status-warning"), "stroke-width": 1.5, "stroke-dasharray": "6,4", opacity: 0.7
    }));

    // Threshold labels
    const outlierLabel = el("text", {
      x: width - pad.right - 4, y: outlierY - 5,
      "text-anchor": "end", fill: cssVar("--status-critical"), "font-size": 10, "font-weight": 600
    });
    outlierLabel.textContent = "Outlier boundary";
    svg.appendChild(outlierLabel);

    const borderlineLabel = el("text", {
      x: width - pad.right - 4, y: borderlineY - 5,
      "text-anchor": "end", fill: cssVar("--status-warning"), "font-size": 10, "font-weight": 600
    });
    borderlineLabel.textContent = "Borderline";
    svg.appendChild(borderlineLabel);

    // Plot points
    data.forEach((d) => {
      const cx = xScale(d.x);
      const cy = yScale(d.lofScore);
      const isOutlier = d.lofScore >= outlierThreshold;
      const isBorderline = d.lofScore >= borderlineThreshold && !isOutlier;
      const color = isOutlier ? cssVar("--status-critical") :
                    isBorderline ? cssVar("--status-warning") :
                    cssVar("--status-good");
      const r = isOutlier ? 5 : isBorderline ? 4 : 3;
      const opacity = isOutlier ? 0.85 : isBorderline ? 0.7 : 0.5;

      const circle = el("circle", {
        cx, cy, r, fill: color, opacity, stroke: "none", class: "chart-point"
      });

      circle.addEventListener("mouseenter", (e) => {
        circle.setAttribute("r", r + 2);
        circle.setAttribute("opacity", 1);
        const category = isOutlier ? "Outlier" : isBorderline ? "Borderline" : "Inlier";
        const html = `
          <strong>${Fmt.escapeHtml(d.id || "Transaction")}</strong><br>
          Outlier Score: ${d.lofScore.toFixed(3)}<br>
          Category: ${category}<br>
          ${d.amount ? `Amount: ${Fmt.money(d.amount)}` : ''}
        `;
        showTooltip(html, e.clientX, e.clientY);
      });

      circle.addEventListener("mouseleave", () => {
        circle.setAttribute("r", r);
        circle.setAttribute("opacity", opacity);
        hideTooltip();
      });

      svg.appendChild(circle);
    });

    container.appendChild(svg);
  }

  return {
    renderBarChart, renderGroupedBarChart, renderHBarChart, renderLineChart, renderScatterPlot, renderLOFChart,
    hideTooltip, cssVar,
  };
})();
