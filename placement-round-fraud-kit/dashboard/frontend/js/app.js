/* Bank Transaction Fraud & Anomaly Detection dashboard -- single-page app wiring. No framework, no build step.
 * Wired to the research_v2 pipeline (teammate 18-feature matrix): risk score is
 * `ensemble_percentile_average`, tiers are the Phase 13 (v2) percentile cutoffs,
 * and explanations are the precomputed Isolation Forest / Autoencoder SHAP rows. */

const NAV_META = {
  overview: { label: "Overview", icon: "overview", title: "Overview Dashboard", subtitle: "Portfolio-wide transaction risk, at a glance" },
  explorer: { label: "Transaction Explorer", icon: "explorer", title: "Transaction Explorer", subtitle: "Browse, search, and filter every scored transaction" },
  queue: { label: "Investigation Queue", icon: "queue", title: "Investigation Queue", subtitle: "Highest-risk transactions, sorted for triage" },
  comparison: { label: "Model Comparison", icon: "comparison", title: "Model Comparison", subtitle: "Twelve unsupervised models on one shared feature matrix" },
  explainability: { label: "Explainability", icon: "explainability", title: "Explainability", subtitle: "Two structurally different views of why a transaction scores high" },
  upload: { label: "Upload & Predict", icon: "upload", title: "Upload & Predict", subtitle: "Score a new CSV batch against the leakage-fixed XGBoost v1 pipeline" },
  history: { label: "Prediction History", icon: "history", title: "Prediction History", subtitle: "Every past Upload & Predict run, newest first" },
  simulator: { label: "Scenario Simulator", icon: "simulator", title: "Account Scenario Simulator", subtitle: "Secondary tool -- vary one real account's transaction" },
};

let currentPage = "overview";
let currentTab = { "data-input": "csv-upload", investigation: "flagged-cases" };
let overviewData = null;
let comparisonData = null;
let explainabilityData = null;
let lastSimResult = null;
let simOptions = null;
let currentCaseId = null;
let uploadedTransactions = []; // Store uploaded transaction results
let uploadHistory = []; // Store each upload session separately
let currentHistoryIndex = 0; // Track which upload session is being viewed

const queueState = { status: "", page: 1, page_size: 25 };
const explorerState = {
  q: "", risk_tier: "", channel: "", txn_type: "", amount_min: "", amount_max: "",
  date_start: "", date_end: "", sort_by: "date", sort_dir: "desc", page: 1, page_size: 25,
};
const explorerSelectedIds = new Set();

// Load uploaded transactions from localStorage
function loadUploadedTransactions() {
  try {
    const stored = localStorage.getItem('fraud-detection-uploads');
    if (stored) {
      uploadedTransactions = JSON.parse(stored);
    }
  } catch (err) {
    console.error("Failed to load uploaded transactions:", err);
  }
}

// Save uploaded transactions to localStorage
function saveUploadedTransactions() {
  try {
    localStorage.setItem('fraud-detection-uploads', JSON.stringify(uploadedTransactions));
    // Refresh overview if we're on that page
    if (currentPage === 'overview') {
      overviewData = null; // Clear cache
      loadOverview();
    }
  } catch (err) {
    console.error("Failed to save uploaded transactions:", err);
  }
}

// Load upload history from localStorage
function loadUploadHistory() {
  try {
    const stored = localStorage.getItem('fraud-detection-upload-history');
    if (stored) {
      uploadHistory = JSON.parse(stored);
    }
  } catch (err) {
    console.error("Failed to load upload history:", err);
  }
}

// Save upload history to localStorage
function saveUploadHistory() {
  try {
    localStorage.setItem('fraud-detection-upload-history', JSON.stringify(uploadHistory));
  } catch (err) {
    console.error("Failed to save upload history:", err);
  }
}

// Add new upload session to history
function addToUploadHistory(filename, transactions) {
  const session = {
    id: Date.now(),
    filename: filename,
    uploadDate: new Date().toISOString(),
    transactions: transactions,
    totalCount: transactions.length,
    fraudCount: transactions.filter(tx => tx.fraud_percentage >= 50).length,
    totalAmount: transactions.reduce((sum, tx) => sum + (tx.amount || 0), 0)
  };

  uploadHistory.push(session);
  saveUploadHistory();
  currentHistoryIndex = uploadHistory.length - 1; // Set to latest upload
}

// ---------------------------------------------------------------------
// small shared helpers
// ---------------------------------------------------------------------
function badgeHtml(status, label, icon) {
  return `<span class="badge badge-${status}">${icon}${Fmt.escapeHtml(label)}</span>`;
}
function riskBadge(tierCode) {
  // Priority tier gets a small restrained pulse -- it's the rare (~1%),
  // genuinely-worth-noticing tier, mirroring the "Live scoring" status-pulse.
  if (tierCode === "priority") return badgeHtml("critical", "Priority review", `<span class="badge-pulse-dot"></span>${Icons.critical}`);
  if (tierCode === "standard") return badgeHtml("warning", "Standard review", Icons.warning);
  if (tierCode === "normal") return badgeHtml("good", "Normal", Icons.good);
  return badgeHtml("neutral", "Unknown", "");
}
function riskTierLabel(tierCode) {
  if (tierCode === "priority") return "Priority review";
  if (tierCode === "standard") return "Standard review";
  if (tierCode === "normal") return "Normal";
  return "Unknown";
}
function riskTierToastType(tierCode) {
  if (tierCode === "priority") return "critical";
  if (tierCode === "standard") return "warning";
  return "good";
}
function queueStatusBadge(action) {
  if (action === "approved") return badgeHtml("good", "Approved", Icons.good);
  if (action === "escalated") return badgeHtml("warning", "Escalated", Icons.warning);
  if (action === "blocked") return badgeHtml("critical", "Blocked", Icons.critical);
  return badgeHtml("neutral", "Pending", "");
}
function skeletonRows(rows, cols) {
  let html = "";
  for (let r = 0; r < rows; r++) {
    html += "<tr>";
    for (let c = 0; c < cols; c++) html += `<td><div class="skeleton skeleton-line"></div></td>`;
    html += "</tr>";
  }
  return html;
}
function emptyRow(cols) {
  return `<tr><td colspan="${cols}"><div class="empty-state">No matching transactions.</div></td></tr>`;
}
function fillSelect(id, values) {
  const el = document.getElementById(id);
  el.innerHTML = values.map((v) => `<option value="${Fmt.escapeHtml(v)}">${Fmt.escapeHtml(v)}</option>`).join("");
}
function fillDatalist(id, values) {
  const el = document.getElementById(id);
  el.innerHTML = values.slice(0, 800).map((v) => `<option value="${Fmt.escapeHtml(v)}">`).join("");
}
// type is one of "good" (default), "warning", "critical" -- maps to the
// matching --status-* left-border color so severity is visible at a glance.
function showToast(msg, type = "good") {
  const stack = document.getElementById("toast-stack");
  const t = document.createElement("div");
  t.className = `toast toast-${type}`;
  t.textContent = msg;
  stack.appendChild(t);
  setTimeout(() => {
    t.style.transition = "opacity 200ms ease";
    t.style.opacity = "0";
    setTimeout(() => t.remove(), 200);
  }, 2600);
}
const num = (v, d = 3) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));

const AVATAR_PALETTE = [
  "--series-1-blue", "--series-2-orange", "--series-3-aqua", "--series-4-yellow",
  "--series-5-magenta", "--series-6-green", "--series-7-violet", "--series-8-red",
];
function avatarHtml(id) {
  const str = String(id || "?");
  const label = str.replace(/[^A-Za-z0-9]/g, "").slice(0, 2).toUpperCase() || "?";
  let hash = 0;
  for (let i = 0; i < str.length; i++) hash = (hash * 31 + str.charCodeAt(i)) >>> 0;
  const varName = AVATAR_PALETTE[hash % AVATAR_PALETTE.length];
  return `<span class="row-avatar" style="background:color-mix(in srgb, var(${varName}) 16%, transparent); color:var(${varName})">${Fmt.escapeHtml(label)}</span>`;
}

function txRowHtml(tx, includeType, withCheckbox) {
  const typeCell = includeType ? `<td>${Fmt.escapeHtml(tx.txn_type)}</td>` : "";
  const checkboxCell = withCheckbox
    ? `<td class="checkbox-col"><input type="checkbox" class="row-select" value="${Fmt.escapeHtml(tx.transaction_id)}" aria-label="Select transaction ${Fmt.escapeHtml(tx.transaction_id)}" /></td>`
    : "";
  return `<tr data-id="${Fmt.escapeHtml(tx.transaction_id)}">
    ${checkboxCell}
    <td>${Fmt.escapeHtml(tx.transaction_id)}</td>
    <td><div class="cell-with-avatar">${avatarHtml(tx.account_id)}<span>${Fmt.escapeHtml(tx.account_id)}</span></div></td>
    <td>${Fmt.dateTime(tx.date)}</td>
    <td class="num tabular">${Fmt.money(tx.amount)}</td>
    <td>${Fmt.escapeHtml(tx.channel)}</td>
    ${typeCell}
    <td>${riskBadge(tx.risk_tier_code)}</td>
    <td class="num tabular">${Fmt.score(tx.risk_score)}</td>
  </tr>`;
}

// ---------------------------------------------------------------------
// reusable confirm modal -- returns a Promise<boolean>
// ---------------------------------------------------------------------
function confirmModal({ title, message, confirmLabel = "Confirm", cancelLabel = "Cancel", danger = false }) {
  return new Promise((resolve) => {
    const backdrop = document.getElementById("modal-backdrop");
    const modal = document.getElementById("confirm-modal");
    document.getElementById("confirm-modal-title").textContent = title;
    document.getElementById("confirm-modal-body").textContent = message;
    const confirmBtn = document.getElementById("confirm-modal-confirm");
    const cancelBtn = document.getElementById("confirm-modal-cancel");
    confirmBtn.textContent = confirmLabel;
    cancelBtn.textContent = cancelLabel;
    confirmBtn.className = danger ? "btn btn-critical" : "btn btn-primary";
    backdrop.classList.add("visible");
    modal.classList.add("visible");

    function cleanup(result) {
      backdrop.classList.remove("visible");
      modal.classList.remove("visible");
      backdrop.removeEventListener("click", onBackdrop);
      document.removeEventListener("keydown", onKeydown);
      confirmBtn.removeEventListener("click", onConfirm);
      cancelBtn.removeEventListener("click", onCancel);
      resolve(result);
    }
    function onConfirm() { cleanup(true); }
    function onCancel() { cleanup(false); }
    function onBackdrop() { cleanup(false); }
    function onKeydown(e) {
      if (e.key === "Escape") { e.preventDefault(); cleanup(false); }
      else if (e.key === "Enter") { e.preventDefault(); cleanup(true); }
    }
    confirmBtn.addEventListener("click", onConfirm);
    cancelBtn.addEventListener("click", onCancel);
    backdrop.addEventListener("click", onBackdrop);
    document.addEventListener("keydown", onKeydown);
    confirmBtn.focus();
  });
}

// ---------------------------------------------------------------------
// KPI tiles -> Explorer, pre-filtered
// ---------------------------------------------------------------------
function goToExplorerWithFilter(riskTier, sortBy) {
  Object.assign(explorerState, {
    q: "", channel: "", txn_type: "", amount_min: "", amount_max: "", date_start: "", date_end: "",
    risk_tier: riskTier || "", sort_by: sortBy || "date", sort_dir: "desc", page: 1,
  });
  const set = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };
  set("explorer-search", "");
  set("explorer-risk-tier", riskTier || "");
  set("explorer-channel", "");
  set("explorer-txn-type", "");
  set("explorer-amount-min", "");
  set("explorer-amount-max", "");
  set("explorer-date-start", "");
  set("explorer-date-end", "");
  showPage("explorer");
}

function wireKpiTileNav() {
  document.querySelectorAll("#kpi-tiles .kpi-tile[data-kpi-nav]").forEach((tile) => {
    const nav = tile.dataset.kpiNav;
    const label = tile.querySelector(".kpi-label")?.textContent || "this metric";
    tile.setAttribute("aria-label", `View ${label} in the Transaction Explorer`);
    const go = () => {
      if (nav === "priority") goToExplorerWithFilter("priority", "risk_score");
      else if (nav === "standard") goToExplorerWithFilter("standard", "risk_score");
      else if (nav === "flagged") goToExplorerWithFilter("", "risk_score");
      else if (nav === "amount") goToExplorerWithFilter("", "amount");
      else goToExplorerWithFilter("", "date");
    };
    tile.addEventListener("click", go);
    tile.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); }
    });
  });
}

// ---------------------------------------------------------------------
// overview - shows only uploaded transaction statistics
// ---------------------------------------------------------------------
async function loadOverview() {
  const kpiEls = document.querySelectorAll("#kpi-tiles .kpi-value");
  const firstLoad = !overviewData;
  if (firstLoad) {
    try {
      overviewData = await Api.kpis();
    } catch (e) {
      showToast(`Could not load overview data: ${e.message}`, "critical");
      return;
    }
  } else {
    // Re-navigating to Overview: briefly show the skeleton again so the
    // count-up reveal replays, matching the first-load experience.
    kpiEls.forEach((el) => el.classList.add("skeleton-loading"));
  }
  const reveal = () => {
    kpiEls.forEach((el) => el.classList.remove("skeleton-loading"));
    Fmt.countUp(document.querySelector('[data-kpi="total"]'), overviewData.total_transactions, { formatter: Fmt.int });
    Fmt.countUp(document.querySelector('[data-kpi="priority"]'), overviewData.priority_count, { formatter: Fmt.int });
    Fmt.countUp(document.querySelector('[data-kpi="standard"]'), overviewData.standard_count, { formatter: Fmt.int });
    Fmt.countUp(document.querySelector('[data-kpi="flagrate"]'), overviewData.flag_rate, { formatter: Fmt.pct, decimals: 4 });
    Fmt.countUp(document.querySelector('[data-kpi="avgamount"]'), overviewData.avg_amount, { formatter: Fmt.money, decimals: 2 });
    document.getElementById("tier-distribution-subtitle").textContent =
      `${Fmt.int(overviewData.total_transactions)} transactions — priority at ensemble score ≥ ${overviewData.priority_threshold} (99th pct), ` +
      `standard at ≥ ${overviewData.standard_threshold} (95th pct). No automatic block tier.`;
    renderTrend("total", computeTrend(overviewData.timeseries, "count"), "neutral");
    renderTrend("flagrate", computeTrend(overviewData.timeseries, "flag_rate_pct"), "lower-is-better");
    renderOverviewCharts(overviewData);
    renderTopRisk(overviewData.top_risk);
  };
  // On re-navigation, let the skeleton paint for one frame before revealing
  // so the shimmer is visible instead of an instant swap.
  if (firstLoad) reveal();
  else requestAnimationFrame(() => requestAnimationFrame(reveal));
}

function renderTopRisk(rows) {
  const tbody = document.querySelector("#table-top-risk tbody");
  if (!tbody) return;
  tbody.innerHTML = (rows || []).map((tx) => txRowHtml(tx, false)).join("") || emptyRow(7);
}

function wireTopRiskTable() {
  const tbody = document.querySelector("#table-top-risk tbody");
  if (!tbody) return;
  tbody.addEventListener("click", (e) => {
    const tr = e.target.closest("tr");
    if (tr && tr.dataset.id) openDrawer(tr.dataset.id);
  });
}

// Compares the trailing window of daily values against the equal-length
// window before it -- real, derived from the same timeseries the chart
// below plots, never a fabricated number.
function computeTrend(ts, key) {
  if (!ts || ts.length < 4) return null;
  const window = Math.max(1, Math.min(7, Math.floor(ts.length / 2)));
  const recent = ts.slice(-window);
  const prior = ts.slice(-2 * window, -window);
  if (!prior.length) return null;
  const avg = (arr) => arr.reduce((a, d) => a + d[key], 0) / arr.length;
  const recentAvg = avg(recent);
  const priorAvg = avg(prior);
  if (!priorAvg) return null;
  return { pctChange: ((recentAvg - priorAvg) / priorAvg) * 100, days: window };
}

function renderTrend(kpiKey, trend, mode) {
  const el = document.querySelector(`[data-trend="${kpiKey}"]`);
  if (!el) return;
  if (!trend || !isFinite(trend.pctChange)) { el.classList.remove("visible"); return; }
  const up = trend.pctChange > 0;
  const arrow = up ? Icons.caretUp : Icons.caretDown;
  const cls = mode === "neutral" ? "trend-neutral" : (up ? "trend-up" : "trend-down");
  el.classList.remove("trend-up", "trend-down", "trend-neutral");
  el.classList.add("visible", cls);
  el.innerHTML = `${arrow} ${Math.abs(trend.pctChange).toFixed(1)}% <span class="kpi-trend-period">vs prior ${trend.days}d</span>`;
}

function renderOverviewCharts(data) {
  const tierColor = {
    priority: Charts.cssVar("--status-critical"),
    standard: Charts.cssVar("--status-warning"),
    normal: Charts.cssVar("--status-good"),
  };
  Charts.renderBarChart(document.getElementById("chart-tier-distribution"), {
    data: data.tier_distribution.map((t) => ({ label: t.tier, value: t.count, color: tierColor[t.code] })),
    valueFormatter: Fmt.int,
  });
  const tierTotal = data.tier_distribution.reduce((a, t) => a + t.count, 0) || 1;
  document.getElementById("tier-distribution-legend").innerHTML = data.tier_distribution.map((t) => `
    <div class="legend-breakdown-row">
      <span class="legend-breakdown-dot" style="background:${tierColor[t.code]}"></span>
      <span class="legend-breakdown-label">${Fmt.escapeHtml(t.tier)} (${Fmt.int(t.count)})</span>
      <span class="legend-breakdown-pct">${((t.count / tierTotal) * 100).toFixed(1)}%</span>
    </div>`).join("");
  const ts = data.timeseries.map((d) => ({ x: new Date(d.date).getTime(), alerts: d.flagged, fraud_rate: d.flag_rate_pct }));
  Charts.renderDualLineChart(document.getElementById("chart-timeseries"), {
    data: ts,
    seriesA: { key: "alerts", label: "Alerts", color: Charts.cssVar("--series-1-blue"), formatter: Fmt.int },
    seriesB: { key: "fraud_rate", label: "Fraud Rate", color: Charts.cssVar("--status-critical"), formatter: (v) => `${v.toFixed(1)}%` },
    xFormatter: (v) => Fmt.dateShort(new Date(v).toISOString()),
  });
}

// ---------------------------------------------------------------------
// explorer
// ---------------------------------------------------------------------
async function loadExplorer() {
  const tbody = document.querySelector("#table-explorer tbody");
  tbody.innerHTML = skeletonRows(8, 9);
  let resp;
  try {
    resp = await Api.transactions({
      q: explorerState.q || undefined,
      risk_tier: explorerState.risk_tier || undefined,
      channel: explorerState.channel || undefined,
      txn_type: explorerState.txn_type || undefined,
      amount_min: explorerState.amount_min || undefined,
      amount_max: explorerState.amount_max || undefined,
      date_start: explorerState.date_start || undefined,
      date_end: explorerState.date_end || undefined,
      sort_by: explorerState.sort_by,
      sort_dir: explorerState.sort_dir,
      page: explorerState.page,
      page_size: explorerState.page_size,
    });
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="9"><div class="empty-state">${Fmt.escapeHtml(e.message)}</div></td></tr>`;
    return;
  }
  tbody.innerHTML = resp.results.map((tx) => txRowHtml(tx, true, true)).join("") || emptyRow(9);
  document.getElementById("explorer-count").textContent = `${Fmt.int(resp.total)} transactions`;
  const totalPages = Math.max(1, Math.ceil(resp.total / resp.page_size));
  document.getElementById("explorer-page-label").textContent = `Page ${resp.page} of ${totalPages}`;
  document.getElementById("explorer-prev").disabled = resp.page <= 1;
  document.getElementById("explorer-next").disabled = resp.page >= totalPages;
  updateSortIndicators();
  syncExplorerRowSelectionUI();
}

// ---------------------------------------------------------------------
// explorer multi-select + bulk "add to Investigation Queue"
// ---------------------------------------------------------------------
function syncExplorerRowSelectionUI() {
  const boxes = document.querySelectorAll("#table-explorer tbody .row-select");
  let allChecked = boxes.length > 0;
  boxes.forEach((cb) => {
    const checked = explorerSelectedIds.has(cb.value);
    cb.checked = checked;
    if (!checked) allChecked = false;
  });
  const selectAll = document.getElementById("explorer-select-all");
  if (selectAll) selectAll.checked = allChecked;
  updateExplorerBulkBar();
}

function updateExplorerBulkBar() {
  const bar = document.getElementById("explorer-bulk-bar");
  if (!bar) return;
  const count = explorerSelectedIds.size;
  bar.hidden = count === 0;
  const countEl = document.getElementById("explorer-bulk-count");
  if (countEl) countEl.textContent = `${count} transaction${count === 1 ? "" : "s"} selected`;
}

async function handleExplorerBulkAddToQueue() {
  if (explorerSelectedIds.size === 0) return;
  const ids = Array.from(explorerSelectedIds);
  loadUploadedTransactions();
  const btn = document.getElementById("explorer-bulk-queue");
  if (btn) { btn.disabled = true; btn.textContent = "Adding…"; }
  let added = 0, skipped = 0, failed = 0;
  for (const id of ids) {
    if (uploadedTransactions.some((t) => t.transaction_id === id)) { skipped++; continue; }
    try {
      const d = await Api.transaction(id);
      uploadedTransactions.push({
        transaction_id: d.transaction_id,
        account_id: d.account_id,
        date: d.raw.date,
        amount: d.raw.amount,
        fraud_percentage: Math.round(d.risk.risk_score * 10000) / 100,
        risk_score: d.risk.risk_score,
        risk_tier_code: d.risk.risk_tier_code,
        status: "pending",
        uploaded_at: new Date().toISOString(),
        source: "explorer_bulk",
      });
      added++;
    } catch (e) {
      failed++;
    }
  }
  if (added > 0) saveUploadedTransactions();
  explorerSelectedIds.clear();
  syncExplorerRowSelectionUI();
  if (btn) { btn.disabled = false; btn.textContent = "Add to Investigation Queue"; }
  if (added > 0) {
    const extras = [skipped ? `${skipped} already queued` : "", failed ? `${failed} failed` : ""].filter(Boolean).join(", ");
    showToast(`${added} transaction${added === 1 ? "" : "s"} added to the Investigation Queue${extras ? ` (${extras})` : ""}.`, "good");
  } else if (skipped > 0) {
    showToast("Selected transactions are already in the Investigation Queue.", "warning");
  } else {
    showToast("Could not add the selected transactions to the queue.", "critical");
  }
}

function updateSortIndicators() {
  document.querySelectorAll("#table-explorer th[data-sort]").forEach((th) => {
    th.querySelector(".sort-caret")?.remove();
    if (th.dataset.sort === explorerState.sort_by) {
      const caret = document.createElement("span");
      caret.className = "sort-caret";
      caret.innerHTML = explorerState.sort_dir === "asc" ? Icons.caretUp : Icons.caretDown;
      th.appendChild(caret);
    }
  });
}

function wireExplorerControls() {
  document.getElementById("explorer-search").addEventListener("input", Fmt.debounce((e) => {
    explorerState.q = e.target.value; explorerState.page = 1; loadExplorer();
  }, 300));
  document.getElementById("explorer-risk-tier").addEventListener("change", (e) => { explorerState.risk_tier = e.target.value; explorerState.page = 1; loadExplorer(); });
  document.getElementById("explorer-channel").addEventListener("change", (e) => { explorerState.channel = e.target.value; explorerState.page = 1; loadExplorer(); });
  document.getElementById("explorer-txn-type").addEventListener("change", (e) => { explorerState.txn_type = e.target.value; explorerState.page = 1; loadExplorer(); });
  document.getElementById("explorer-amount-min").addEventListener("input", Fmt.debounce((e) => { explorerState.amount_min = e.target.value; explorerState.page = 1; loadExplorer(); }, 350));
  document.getElementById("explorer-amount-max").addEventListener("input", Fmt.debounce((e) => { explorerState.amount_max = e.target.value; explorerState.page = 1; loadExplorer(); }, 350));
  document.getElementById("explorer-date-start").addEventListener("change", (e) => { explorerState.date_start = e.target.value; explorerState.page = 1; loadExplorer(); });
  document.getElementById("explorer-date-end").addEventListener("change", (e) => { explorerState.date_end = e.target.value; explorerState.page = 1; loadExplorer(); });
  document.getElementById("explorer-reset").addEventListener("click", () => {
    Object.assign(explorerState, { q: "", risk_tier: "", channel: "", txn_type: "", amount_min: "", amount_max: "", date_start: "", date_end: "", sort_by: "date", sort_dir: "desc", page: 1 });
    ["explorer-search", "explorer-risk-tier", "explorer-channel", "explorer-txn-type",
     "explorer-amount-min", "explorer-amount-max", "explorer-date-start", "explorer-date-end"]
      .forEach((id) => { document.getElementById(id).value = ""; });
    loadExplorer();
  });
  document.querySelectorAll("#table-explorer th[data-sort]").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      if (explorerState.sort_by === key) explorerState.sort_dir = explorerState.sort_dir === "asc" ? "desc" : "asc";
      else { explorerState.sort_by = key; explorerState.sort_dir = "desc"; }
      loadExplorer();
    });
  });
  document.getElementById("explorer-prev").addEventListener("click", () => { if (explorerState.page > 1) { explorerState.page--; loadExplorer(); } });
  document.getElementById("explorer-next").addEventListener("click", () => { explorerState.page++; loadExplorer(); });
  document.querySelector("#table-explorer tbody").addEventListener("click", (e) => {
    if (e.target.closest(".checkbox-col")) return;
    const tr = e.target.closest("tr");
    if (tr && tr.dataset.id) openDrawer(tr.dataset.id);
  });
  document.querySelector("#table-explorer tbody").addEventListener("change", (e) => {
    if (!e.target.classList.contains("row-select")) return;
    if (e.target.checked) explorerSelectedIds.add(e.target.value);
    else explorerSelectedIds.delete(e.target.value);
    syncExplorerRowSelectionUI();
  });
  document.getElementById("explorer-select-all").addEventListener("change", (e) => {
    document.querySelectorAll("#table-explorer tbody .row-select").forEach((cb) => {
      cb.checked = e.target.checked;
      if (e.target.checked) explorerSelectedIds.add(cb.value);
      else explorerSelectedIds.delete(cb.value);
    });
    updateExplorerBulkBar();
  });
  document.getElementById("explorer-bulk-clear").addEventListener("click", () => {
    explorerSelectedIds.clear();
    syncExplorerRowSelectionUI();
  });
  document.getElementById("explorer-bulk-queue").addEventListener("click", handleExplorerBulkAddToQueue);
}

// ---------------------------------------------------------------------
// investigation queue - shows only uploaded transactions
// ---------------------------------------------------------------------
async function loadQueue() {
  const tbody = document.querySelector("#table-queue tbody");

  // Load uploaded transactions from localStorage
  loadUploadedTransactions();

  // If no uploads yet, show empty state
  if (uploadedTransactions.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8"><div class="empty-state">
      <p style="font-size:14px;margin-bottom:8px">No transactions analyzed yet</p>
      <p style="font-size:12px;color:var(--text-muted)">Upload a CSV file in the Data Input section to begin fraud analysis.</p>
    </div></td></tr>`;
    document.getElementById("queue-count").textContent = "0 transactions";
    document.getElementById("queue-page-label").textContent = "Page 0 of 0";
    document.getElementById("queue-prev").disabled = true;
    document.getElementById("queue-next").disabled = true;
    return;
  }

  // Filter by status if needed
  let filtered = uploadedTransactions;
  if (queueState.status) {
    filtered = filtered.filter(tx => (tx.status || 'pending') === queueState.status);
  }

  // Sort by fraud percentage descending
  filtered.sort((a, b) => b.fraud_percentage - a.fraud_percentage);

  // Pagination
  const total = filtered.length;
  const totalPages = Math.max(1, Math.ceil(total / queueState.page_size));
  const start = (queueState.page - 1) * queueState.page_size;
  const pageItems = filtered.slice(start, start + queueState.page_size);

  // Render table
  tbody.innerHTML = pageItems.map((tx, i) => {
    return `<tr data-tx-row="${Fmt.escapeHtml(tx.transaction_id)}">
      <td class="num tabular">${start + i + 1}</td>
      <td>${Fmt.escapeHtml(tx.transaction_id)}</td>
      <td><div class="cell-with-avatar">${avatarHtml(tx.account_id)}<span>${Fmt.escapeHtml(tx.account_id)}</span></div></td>
      <td class="num tabular">${Fmt.money(tx.amount)}</td>
      <td>${riskBadge(tx.risk_tier_code)}</td>
      <td class="num tabular">${Fmt.score(tx.risk_score)}</td>
      <td>${queueStatusBadge(tx.queue_action)}</td>
      <td>
        <button class="btn btn-sm btn-ghost action-approve" data-tx-id="${Fmt.escapeHtml(tx.transaction_id)}" data-action="approved">Approve</button>
        <button class="btn btn-sm btn-ghost action-escalate" data-tx-id="${Fmt.escapeHtml(tx.transaction_id)}" data-action="escalated">Escalate</button>
        <button class="btn btn-sm btn-ghost action-block" data-tx-id="${Fmt.escapeHtml(tx.transaction_id)}" data-action="blocked">Block</button>
      </td>
    </tr>`;
  }).join("");

  // Wire action buttons -- Block is destructive-feeling, so it's confirmed first.
  document.querySelectorAll("#table-queue [data-action]").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const txId = btn.dataset.txId;
      const action = btn.dataset.action;
      if (action === "blocked") {
        const tx = uploadedTransactions.find((t) => t.transaction_id === txId);
        const ok = await confirmModal({
          title: "Block this transaction?",
          message: `${txId}${tx && tx.account_id ? ` (account ${tx.account_id})` : ""} will be marked as blocked in the Investigation Queue. This is a manual triage label only -- it does not stop the transaction anywhere else.`,
          confirmLabel: "Block transaction",
          danger: true,
        });
        if (!ok) return;
      }
      handleUploadedQueueAction(txId, action);
    });
  });

  document.getElementById("queue-count").textContent = `${Fmt.int(total)} transactions`;
  document.getElementById("queue-page-label").textContent = `Page ${queueState.page} of ${totalPages}`;
  document.getElementById("queue-prev").disabled = queueState.page <= 1;
  document.getElementById("queue-next").disabled = queueState.page >= totalPages;
}

function handleUploadedQueueAction(txId, action) {
  const tx = uploadedTransactions.find(t => t.transaction_id === txId);
  if (!tx) return;
  tx.status = action;
  saveUploadedTransactions();
  const toastType = action === "approved" ? "good" : action === "blocked" ? "critical" : "warning";
  showToast(`${txId} marked as ${action}.`, toastType);

  // Flash the row so the state change is visually confirmed before it
  // reorders/disappears under the next status filter.
  const row = document.querySelector(`#table-queue tr[data-tx-row="${CSS.escape(txId)}"]`);
  const flashClass = action === "approved" ? "row-flash-good" : action === "blocked" ? "row-flash-critical" : "row-flash-warning";
  if (row && !Fmt.prefersReducedMotion()) {
    row.classList.add(flashClass);
    setTimeout(() => loadQueue(), 480);
  } else {
    loadQueue();
  }
}

async function handleQueueAction(transactionId, action) {
  try {
    await Api.queueAction(transactionId, action);
    showToast(`${transactionId} marked as ${action}.`, "good");
    loadQueue();
  } catch (e) {
    showToast(`Could not update ${transactionId}: ${e.message}`, "critical");
  }
}

function wireQueueControls() {
  document.getElementById("queue-status").addEventListener("change", (e) => { queueState.status = e.target.value; queueState.page = 1; loadQueue(); });
  document.getElementById("queue-prev").addEventListener("click", () => { if (queueState.page > 1) { queueState.page--; loadQueue(); } });
  document.getElementById("queue-next").addEventListener("click", () => { queueState.page++; loadQueue(); });
  document.getElementById("queue-export").addEventListener("click", exportUploadedTransactions);
  document.getElementById("queue-clear").addEventListener("click", clearUploadedTransactions);
  document.getElementById("queue-generate-report").addEventListener("click", generateFraudReport);
  document.getElementById("auto-process-btn").addEventListener("click", startAutoProcess);
}

async function clearUploadedTransactions() {
  if (uploadedTransactions.length === 0) {
    showToast("No transactions to clear.", "warning");
    return;
  }

  const ok = await confirmModal({
    title: "Clear all analyzed transactions?",
    message: `This removes all ${uploadedTransactions.length} transactions currently in the Investigation Queue. This cannot be undone.`,
    confirmLabel: "Clear all",
    danger: true,
  });
  if (!ok) return;

  uploadedTransactions = [];
  saveUploadedTransactions();
  loadQueue();
  // Refresh overview to show empty state
  overviewData = null;
  if (currentPage === 'overview') {
    loadOverview();
  }
  showToast("All transactions cleared.", "good");
}

// Auto-process all pending transactions based on ML fraud scores
async function startAutoProcess() {
  loadUploadedTransactions();

  const pendingTransactions = uploadedTransactions.filter(tx => tx.status === 'pending');

  if (pendingTransactions.length === 0) {
    showToast("No pending transactions to process.", "warning");
    return;
  }

  const btn = document.getElementById('auto-process-btn');
  const originalText = btn.innerHTML;

  const ok = await confirmModal({
    title: `Auto-process ${pendingTransactions.length} pending transaction${pendingTransactions.length === 1 ? "" : "s"}?`,
    message: "The system will automatically approve legitimate accounts (risk < 50%), escalate suspicious accounts (50-79%), and block high-risk accounts (≥ 80%). Every status can still be changed individually afterward.",
    confirmLabel: "Start auto-process",
  });
  if (!ok) return;

  // Disable button and show processing
  btn.disabled = true;
  btn.style.background = 'linear-gradient(135deg,#a0aec0 0%,#718096 100%)';
  btn.textContent = 'Processing…';

  try {
    let approved = 0, escalated = 0, blocked = 0;

    // Process transactions with simulated delay for visual feedback
    for (let i = 0; i < uploadedTransactions.length; i++) {
      const tx = uploadedTransactions[i];

      if (tx.status !== 'pending') continue;

      // Apply intelligent decision logic based on fraud percentage
      if (tx.fraud_percentage >= 80) {
        // Critical risk - Block immediately
        tx.status = 'blocked';
        blocked++;
      } else if (tx.fraud_percentage >= 60) {
        // High risk - Escalate for review
        tx.status = 'escalated';
        escalated++;
      } else if (tx.fraud_percentage >= 50) {
        // Moderate risk - Escalate for manual review
        tx.status = 'escalated';
        escalated++;
      } else {
        // Low risk - Auto-approve (legitimate)
        tx.status = 'approved';
        approved++;
      }

      // Update button text with progress
      const processed = approved + escalated + blocked;
      btn.textContent = `Processing ${processed}/${pendingTransactions.length}…`;

      // Small delay for visual feedback (every 10 transactions)
      if (i % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, 50));
      }
    }

    // Save updated transactions
    saveUploadedTransactions();

    // Show success animation
    btn.style.background = 'linear-gradient(135deg,#48bb78 0%,#38a169 100%)';
    btn.textContent = 'Complete';

    // Show detailed results
    showToast(
      `Auto-processing complete: ${approved} approved, ${escalated} escalated, ${blocked} blocked.`,
      "good"
    );

    // Reload queue to show updated statuses
    setTimeout(() => {
      loadQueue();

      // Reset button after delay
      setTimeout(() => {
        btn.disabled = false;
        btn.style.background = 'linear-gradient(135deg,#667eea 0%,#764ba2 100%)';
        btn.innerHTML = originalText;
      }, 2000);
    }, 1000);

  } catch (err) {
    console.error('Auto-process error:', err);
    showToast(`Error during auto-processing: ${err.message}`, "critical");

    // Reset button on error
    btn.disabled = false;
    btn.style.background = 'linear-gradient(135deg,#667eea 0%,#764ba2 100%)';
    btn.innerHTML = originalText;
  }
}

function exportUploadedTransactions() {
  if (uploadedTransactions.length === 0) {
    showToast("No transactions to export.", "warning");
    return;
  }

  // Create CSV content
  const headers = "Transaction ID,Account ID,Date,Amount,Fraud %,Status,Uploaded At\n";
  const rows = uploadedTransactions.map(tx =>
    `${tx.transaction_id},${tx.account_id || 'N/A'},${tx.date || 'N/A'},${tx.amount || 'N/A'},${tx.fraud_percentage.toFixed(2)},${tx.status || 'pending'},${tx.uploaded_at}`
  ).join("\n");

  // Create and download
  const blob = new Blob([headers + rows], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `fraud_analysis_${new Date().toISOString().split('T')[0]}.csv`;
  a.click();
  URL.revokeObjectURL(url);
  showToast(`Exported ${uploadedTransactions.length} transactions.`, "good");
}

// Downloadable plain-text summary of everything currently flagged
// (fraud_percentage >= the same 50% cutoff the queue table uses), grouped
// by triage status. Client-side only -- built from the same uploadedTransactions
// the Investigation Queue already renders.
function generateFraudReport() {
  loadUploadedTransactions();
  const flagged = uploadedTransactions.filter((tx) => tx.fraud_percentage >= 50);
  if (flagged.length === 0) {
    showToast("No flagged transactions to report on yet.", "warning");
    return;
  }

  const byStatus = { pending: [], approved: [], escalated: [], blocked: [] };
  flagged.forEach((tx) => { (byStatus[tx.status || "pending"] || byStatus.pending).push(tx); });
  const totalAmount = flagged.reduce((sum, tx) => sum + (Number(tx.amount) || 0), 0);
  const generatedAt = new Date();

  const lines = [];
  lines.push("BANK TRANSACTION FRAUD & ANOMALY DETECTION");
  lines.push("Investigation Queue -- Flagged Transaction Report");
  lines.push(`Generated: ${generatedAt.toLocaleString()}`);
  lines.push("=".repeat(64));
  lines.push("");
  lines.push(`Flagged transactions:  ${flagged.length} of ${uploadedTransactions.length} analyzed (>= 50% predicted fraud probability)`);
  lines.push(`Total flagged amount:  ${Fmt.money(totalAmount)}`);
  lines.push(`Pending review:        ${byStatus.pending.length}`);
  lines.push(`Approved:              ${byStatus.approved.length}`);
  lines.push(`Escalated:             ${byStatus.escalated.length}`);
  lines.push(`Blocked:               ${byStatus.blocked.length}`);
  lines.push("");
  lines.push("-".repeat(64));

  ["pending", "escalated", "blocked", "approved"].forEach((status) => {
    const rows = byStatus[status];
    if (!rows.length) return;
    lines.push("");
    lines.push(`${status.toUpperCase()} (${rows.length})`);
    rows
      .slice()
      .sort((a, b) => b.fraud_percentage - a.fraud_percentage)
      .forEach((tx) => {
        lines.push(
          `  ${(tx.transaction_id || "—").padEnd(16)} ${(tx.account_id || "—").padEnd(12)} ` +
          `${(tx.date || "—").slice(0, 10).padEnd(12)} ${String(tx.amount ?? "—").padEnd(12)} ` +
          `${tx.fraud_percentage.toFixed(1)}% fraud probability`
        );
      });
  });
  lines.push("");
  lines.push("-".repeat(64));
  lines.push("Note: this report is a manual triage summary generated in the analyst's");
  lines.push("browser from the current Investigation Queue. It is not a filed SAR/CTR");
  lines.push("and does not itself change any transaction's status.");

  const blob = new Blob([lines.join("\n")], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `fraud_report_${generatedAt.toISOString().split("T")[0]}.txt`;
  a.click();
  URL.revokeObjectURL(url);
  showToast(`Fraud report generated: ${flagged.length} flagged transactions.`, "good");
}

// ---------------------------------------------------------------------
// model comparison
// ---------------------------------------------------------------------
async function loadComparison() {
  if (!comparisonData) {
    try { comparisonData = await Api.modelComparison(); }
    catch (e) { showToast(`Could not load model comparison: ${e.message}`, "critical"); return; }
  }
  renderComparison(comparisonData);
}

const comparisonSortState = { by: null, dir: "desc" };

function sortModelsData(models) {
  if (!comparisonSortState.by) return models;
  const key = comparisonSortState.by;
  const dir = comparisonSortState.dir === "asc" ? 1 : -1;
  return models.slice().sort((a, b) => {
    let av = a[key], bv = b[key];
    if (typeof av === "string" || typeof bv === "string") {
      return String(av ?? "").localeCompare(String(bv ?? "")) * dir;
    }
    if (av === null || av === undefined) av = -Infinity;
    if (bv === null || bv === undefined) bv = -Infinity;
    return (av - bv) * dir;
  });
}

function updateModelsSortIndicators() {
  document.querySelectorAll("#table-models th[data-sort]").forEach((th) => {
    th.querySelector(".sort-caret")?.remove();
    if (th.dataset.sort === comparisonSortState.by) {
      const caret = document.createElement("span");
      caret.className = "sort-caret";
      caret.innerHTML = comparisonSortState.dir === "asc" ? Icons.caretUp : Icons.caretDown;
      th.appendChild(caret);
    }
  });
}

function wireModelsSort() {
  document.querySelectorAll("#table-models th[data-sort]").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      if (comparisonSortState.by === key) comparisonSortState.dir = comparisonSortState.dir === "asc" ? "desc" : "asc";
      else { comparisonSortState.by = key; comparisonSortState.dir = key === "label" ? "asc" : "desc"; }
      if (comparisonData) renderComparison(comparisonData);
    });
  });
}

function renderComparison(data) {
  const tbody = document.querySelector("#table-models tbody");
  tbody.innerHTML = sortModelsData(data.models).map((m) => `<tr>
      <td>${Fmt.escapeHtml(m.label)}${m.in_ensemble ? "" : ' <span class="freq-hint">not an ensemble input</span>'}</td>
      <td class="num tabular">${m.n_flagged_top5pct === null ? "—" : Fmt.int(m.n_flagged_top5pct)}</td>
      <td class="num tabular">${m.flagged_rate_pct === null || m.flagged_rate_pct === undefined ? "—" : m.flagged_rate_pct.toFixed(2) + "%"}</td>
      <td class="num tabular">${num(m.silhouette, 4)}</td>
      <td class="num tabular">${num(m.davies_bouldin, 4)}</td>
      <td class="num tabular">${m.calinski_harabasz === null ? "—" : m.calinski_harabasz.toFixed(2)}</td>
      <td class="num tabular">${num(m.mean_spearman, 3)}</td>
      <td class="num tabular">${num(m.mean_jaccard, 3)}</td>
      <td class="num tabular">${num(m.ensemble_weight, 3)}</td>
    </tr>`).join("");
  updateModelsSortIndicators();

  document.getElementById("model-table-note").textContent =
    `Flagged counts are each model's top-5%-by-score partition. LSTM-AE is restricted to the ` +
    `2,402 of 2,512 rows whose account has ≥3 transactions; the Hybrid Ensemble's row is measured on a ` +
    `269-row ≥1-vote partition, not its native 83-row ≥2-of-3 flag. Mean ρ / Jaccard are self-excluded ` +
    `pairwise means over the other 11 models.`;

  const validityModels = data.models.filter((m) => m.silhouette !== null);
  Charts.renderHBarChart(document.getElementById("chart-validity"), {
    data: validityModels.slice().reverse().map((m) => ({ label: m.label, value: m.silhouette })),
    diverging: false, valueFormatter: (v) => v.toFixed(3),
  });
  document.getElementById("validity-note").textContent =
    `Top-5%-flagged vs. rest in the shared scaled 18-feature space. Higher is better separated — but a ` +
    `top-5%-by-distance cut is structurally favoured by a distance-based index, which is why the ` +
    `reconstruction-error models sit lowest.`;

  Charts.renderBarChart(document.getElementById("chart-stability"), {
    data: data.stability.map((s, i) => ({
      label: s.label, value: s.mean_jaccard,
      color: [Charts.cssVar("--series-3-aqua"), Charts.cssVar("--series-4-yellow"), Charts.cssVar("--series-8-red")][i % 3],
    })),
    valueFormatter: (v) => v.toFixed(3),
  });
  document.getElementById("stability-note").textContent =
    data.stability.map((s) => `${s.label}: mean ${s.mean_jaccard} (min ${s.min_jaccard}, max ${s.max_jaccard})`).join(" · ")
    + ". " + data.notes.stability;

  const weighted = data.models.filter((m) => m.ensemble_weight !== null)
    .slice().sort((a, b) => a.ensemble_weight - b.ensemble_weight);
  Charts.renderHBarChart(document.getElementById("chart-weights"), {
    data: weighted.map((m) => ({ label: m.label, value: m.ensemble_weight })),
    diverging: false, valueFormatter: (v) => v.toFixed(3),
  });
  document.getElementById("dbscan-note").textContent = data.notes.dbscan;

  const stbody = document.querySelector("#table-strategies tbody");
  stbody.innerHTML = data.strategy_pairs.map((p) => `<tr>
      <td>${Fmt.escapeHtml(p.pair)}</td>
      <td class="num tabular">${p.spearman.toFixed(4)}</td>
      <td class="num tabular">${p.jaccard.toFixed(3)}</td>
    </tr>`).join("");
  document.getElementById("strategy-note").textContent =
    `${data.notes.strategies} PCA stacking's first component explains ` +
    `${(data.pc1_explained_variance * 100).toFixed(1)}% of the variance across the 11 standardised score columns. ` +
    `Recommended: ${data.recommended_strategy}.`;

  document.getElementById("leaderboard-note").textContent = data.notes.leaderboard;
  document.getElementById("ee-note").textContent = data.notes.elliptic_envelope;
  document.getElementById("agreement-note").textContent = data.notes.agreement;
  document.getElementById("hybrid-note").textContent = data.notes.hybrid;
}

// ---------------------------------------------------------------------
// explainability
// ---------------------------------------------------------------------
async function loadExplainability() {
  if (!explainabilityData) {
    try { explainabilityData = await Api.explainability(); }
    catch (e) { showToast(`Could not load explainability data: ${e.message}`, "critical"); return; }
  }
  renderExplainability(explainabilityData);
}

function renderExplainability(data) {
  const d = data.divergence;
  document.getElementById("shap-rho").textContent = d.spearman_rho.toFixed(4);
  document.getElementById("shap-overlap").textContent = `${d.top10_overlap} of 10`;
  document.getElementById("divergence-explanation").textContent = d.explanation;

  Charts.renderHBarChart(document.getElementById("chart-shap-if"), {
    data: data.global_shap.isolation_forest.slice().reverse().map((f) => ({ label: f.label, value: f.mean_abs_shap })),
    diverging: false, valueFormatter: (v) => v.toFixed(3),
  });
  Charts.renderHBarChart(document.getElementById("chart-shap-ae"), {
    data: data.global_shap.autoencoder.slice().reverse().map((f) => ({ label: f.label, value: f.mean_abs_shap })),
    diverging: false, valueFormatter: (v) => v.toFixed(4),
  });

  document.getElementById("worked-examples").innerHTML = d.worked_examples.map((w) => `
    <div class="worked-case">
      <div class="worked-case-id">${Fmt.escapeHtml(w.transaction_id)}</div>
      <div class="worked-case-note">${Fmt.escapeHtml(w.note)}</div>
    </div>`).join("");

  const sd = data.score_distribution;
  document.getElementById("score-dist-subtitle").textContent =
    `ensemble_percentile_average across all 2,512 transactions — mean ${sd.mean}, std ${sd.std}, ` +
    `min ${sd.min}, max ${sd.max}. The two markers are the Phase 13 review cutoffs.`;
  const p99 = data.percentile_thresholds.find((t) => t.method === "P99");
  const p95 = data.percentile_thresholds.find((t) => t.method === "P95");
  Charts.renderLineChart(document.getElementById("chart-score-dist"), {
    data: data.score_histogram.map((h) => ({ x: h.x, y: h.count })),
    color: Charts.cssVar("--series-1-blue"), area: true,
    xFormatter: (v) => v.toFixed(2), yFormatter: Fmt.int,
    markers: [
      { x: p95.threshold, label: `P95 ${p95.threshold}`, color: Charts.cssVar("--status-warning") },
      { x: p99.threshold, label: `P99 ${p99.threshold}`, color: Charts.cssVar("--status-critical") },
    ],
  });

  document.querySelector("#table-thresholds tbody").innerHTML = data.percentile_thresholds.map((t) => `<tr>
      <td>${Fmt.escapeHtml(t.method)}</td>
      <td class="num tabular">${t.threshold.toFixed(4)}</td>
      <td class="num tabular">${Fmt.int(t.n_flagged)}</td>
      <td class="num tabular">${t.pct_flagged.toFixed(3)}%</td>
      <td class="num tabular">${t.per_day.toFixed(3)}</td>
      <td class="num tabular">${Fmt.money(t.review_cost_ceiling)}</td>
    </tr>`).join("");
  document.getElementById("cost-note").textContent = data.cost_note;

  document.querySelector("#table-stat-thresholds tbody").innerHTML = data.statistical_thresholds.map((t) => `<tr>
      <td>${Fmt.escapeHtml(t.method)}</td>
      <td style="font-size:11.5px">${Fmt.escapeHtml(t.score)}</td>
      <td class="num tabular">${t.threshold.toFixed(4)}</td>
      <td class="num tabular"><strong>${Fmt.int(t.n_flagged)}</strong></td>
    </tr>`).join("");
  document.getElementById("stat-finding").textContent = data.statistical_finding;
}

// ---------------------------------------------------------------------
// account scenario simulator
// ---------------------------------------------------------------------
async function loadSimulatorOptions() {
  if (simOptions) return;
  try {
    simOptions = await Api.simulatorOptions();
    // No datalist/autocomplete anywhere in this form by design -- every field
    // is typed by hand. simOptions is still fetched for the note text below,
    // the high-amount threshold, and to validate typed values against real
    // data server-side on submit.
    document.getElementById("simulator-note").textContent = simOptions.note;
    document.getElementById("simulator-banner-text").textContent =
      `Secondary tool — vary one real account's transaction and see how the score moves. ` +
      `The high-amount flag fires above $${simOptions.high_amount_threshold} (the dataset's 95th percentile, frozen).`;
  } catch (e) {
    showToast(`Could not load simulator reference data: ${e.message}`, "critical");
  }
}

async function loadSimAccountDefaults(accountId) {
  if (!accountId) return;
  let acc;
  try { acc = await Api.simulatorAccount(accountId); }
  catch (e) {
    document.getElementById("sim-account-summary").textContent = e.message;
    return;
  }
  const d = acc.defaults;
  const set = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };
  set("sim-amount", d.amount);
  set("sim-balance", d.account_balance);
  set("sim-age", d.customer_age);
  set("sim-duration", d.duration_seconds);
  set("sim-login", d.login_attempts);
  set("sim-type", d.txn_type);
  set("sim-channel", d.channel);
  set("sim-occupation", d.customer_occupation);
  set("sim-location", d.location);
  set("sim-device", d.device_id);
  set("sim-ip", d.ip_address);
  set("sim-merchant", d.merchant_id);
  document.getElementById("sim-account-summary").textContent =
    `${accountId}: ${acc.n_transactions} transactions in the dataset (account_frequency = ${acc.account_frequency}). ` +
    `Fields prefilled from its most recent transaction.`;
}

function wireSimulatorForm() {
  document.getElementById("sim-account").addEventListener("change", (e) => {
    loadSimAccountDefaults(e.target.value.trim());
  });
  document.getElementById("simulator-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const payload = {
      account_id: document.getElementById("sim-account").value.trim(),
      amount: parseFloat(document.getElementById("sim-amount").value),
      account_balance: parseFloat(document.getElementById("sim-balance").value),
      txn_type: document.getElementById("sim-type").value,
      channel: document.getElementById("sim-channel").value,
      location: document.getElementById("sim-location").value,
      customer_occupation: document.getElementById("sim-occupation").value,
      device_id: document.getElementById("sim-device").value,
      ip_address: document.getElementById("sim-ip").value,
      merchant_id: document.getElementById("sim-merchant").value,
      customer_age: parseInt(document.getElementById("sim-age").value, 10),
      duration_seconds: parseInt(document.getElementById("sim-duration").value, 10),
      login_attempts: parseInt(document.getElementById("sim-login").value, 10),
    };
    try {
      lastSimResult = await Api.score(payload);
      renderSimResult(lastSimResult);
      showToast(`Scenario scored: ${riskTierLabel(lastSimResult.risk_tier_code)} (${Fmt.score(lastSimResult.two_model_percentile_average)}).`, riskTierToastType(lastSimResult.risk_tier_code));
    } catch (e2) {
      showToast(`Could not score this scenario: ${e2.message}`, "critical");
    }
  });
}

function fieldItem(label, value) {
  return `<div class="field-item"><div class="field-label">${label}</div><div class="field-value">${value}</div></div>`;
}

function renderSimResult(r) {
  document.getElementById("simulator-result-card").style.display = "block";
  document.getElementById("sim-tier-badge").innerHTML = riskBadge(r.risk_tier_code);
  document.getElementById("sim-score").textContent = Fmt.score(r.two_model_percentile_average);
  document.getElementById("sim-score-caption").textContent =
    `Two-model percentile average — Isolation Forest at the ${(r.isolation_forest.percentile * 100).toFixed(1)}th percentile ` +
    `(score ${r.isolation_forest.score}), Autoencoder at the ${(r.autoencoder.percentile * 100).toFixed(1)}th ` +
    `(reconstruction MSE ${r.autoencoder.score}). This scenario sits at the ` +
    `${(r.two_model_reference_percentile * 100).toFixed(1)}th percentile of the two-model reference distribution.`;

  const f = r.frequency_inputs_used;
  document.getElementById("sim-detail-grid").innerHTML = [
    fieldItem("Account frequency (real)", `${f.account_frequency} txns`),
    fieldItem("Device frequency (real)", `${f.device_frequency} txns`),
    fieldItem("IP frequency (real)", `${f.ip_frequency} txns`),
    fieldItem("Merchant frequency (real)", `${f.merchant_frequency} txns`),
    fieldItem("Location share (real)", `${f.location_share_pct}%`),
    fieldItem("Amount / (balance + 1)", r.derived.amount_to_balance_ratio_raw),
    fieldItem("High-amount flag", r.derived.high_amount_flag ? `Yes (> $${r.derived.high_amount_threshold})` : `No (≤ $${r.derived.high_amount_threshold})`),
  ].join("");

  document.getElementById("sim-score-note").textContent = r.score_note;
  renderSimCharts(r);
}

function renderSimCharts(r) {
  if (!r) return;
  Charts.renderHBarChart(document.getElementById("chart-sim-shap-if"), {
    data: r.shap_isolation_forest.slice().reverse().map((s) => ({ label: s.label, value: s.shap_value })),
    diverging: true, valueFormatter: (v) => v.toFixed(3),
    legend: [
      { label: "Increases anomaly score", color: Charts.cssVar("--series-2-orange") },
      { label: "Decreases anomaly score", color: Charts.cssVar("--series-1-blue") },
    ],
  });
  Charts.renderHBarChart(document.getElementById("chart-sim-shap-ae"), {
    data: r.autoencoder_error_contributions.slice().reverse().map((s) => ({
      label: `${s.label} (${(s.share_of_error * 100).toFixed(1)}%)`, value: s.shap_value,
    })),
    diverging: false, valueFormatter: (v) => v.toFixed(4),
  });
}

// ---------------------------------------------------------------------
// upload & predict
// ---------------------------------------------------------------------
function uploadRowHtml(r) {
  const highRisk = r.fraud_percentage >= 70;
  const pctStyle = highRisk ? ` style="color:var(--status-critical);font-weight:600"` : "";
  const dash = (v) => (v === null || v === undefined || v === "" ? "—" : v);
  return `<tr>
    <td>${Fmt.escapeHtml(String(dash(r.transaction_id)))}</td>
    <td>${Fmt.escapeHtml(String(dash(r.account_id)))}</td>
    <td>${Fmt.escapeHtml(String(dash(r.date)))}</td>
    <td class="num tabular">${r.amount === null || r.amount === undefined ? "—" : Fmt.money(r.amount)}</td>
    <td class="num tabular"${pctStyle}>${r.fraud_percentage.toFixed(2)}%</td>
  </tr>`;
}

function updateUploadDropzone() {
  const input = document.getElementById("upload-file-input");
  const file = input.files && input.files[0];
  const dropzone = document.getElementById("upload-dropzone");
  const chip = document.getElementById("upload-file-chip");
  if (file) {
    document.getElementById("upload-file-chip-name").textContent = file.name;
    chip.style.display = "flex";
    dropzone.style.display = "none";
  } else {
    chip.style.display = "none";
    dropzone.style.display = "flex";
  }
}

function wireUploadDropzone() {
  document.getElementById("upload-card-icon").innerHTML = Icons.upload;
  document.getElementById("upload-dropzone-icon").innerHTML = Icons.upload;
  document.getElementById("upload-file-chip-icon").innerHTML = Icons.file;
  document.getElementById("upload-file-clear").innerHTML = Icons.close;

  const input = document.getElementById("upload-file-input");
  const dropzone = document.getElementById("upload-dropzone");

  input.addEventListener("change", updateUploadDropzone);

  document.getElementById("upload-file-clear").addEventListener("click", () => {
    input.value = "";
    updateUploadDropzone();
    document.getElementById("upload-status").textContent = "";
    document.getElementById("upload-results-card").style.display = "none";
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "dragend"].forEach((evt) => {
    dropzone.addEventListener(evt, () => dropzone.classList.remove("dragover"));
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const dropped = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (dropped) {
      input.files = e.dataTransfer.files;
      updateUploadDropzone();
    }
  });
}

function wireUploadForm() {
  wireUploadDropzone();
  document.getElementById("upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = document.getElementById("upload-file-input");
    const file = input.files && input.files[0];
    const statusEl = document.getElementById("upload-status");
    const btn = document.getElementById("upload-predict-btn");
    const resultsCard = document.getElementById("upload-results-card");
    if (!file) {
      statusEl.textContent = "Choose a CSV file first.";
      return;
    }
    btn.disabled = true;
    btn.textContent = "Predicting…";
    statusEl.textContent = `Scoring ${Fmt.escapeHtml(file.name)}…`;
    resultsCard.style.display = "none";
    try {
      const resp = await Api.uploadPredict(file);

      // Save results to Investigation tab
      const timestamp = new Date().toISOString();
      const newTransactions = [];
      resp.results.forEach(tx => {
        const txData = {
          ...tx,
          uploaded_at: timestamp,
          status: 'pending'
        };
        uploadedTransactions.push(txData);
        newTransactions.push(txData);
      });
      saveUploadedTransactions();

      // Add to upload history
      addToUploadHistory(file.name, newTransactions);

      statusEl.textContent = `Analysis complete: ${Fmt.int(resp.total)} transactions processed.`;
      document.getElementById("upload-results-subtitle").textContent =
        `${Fmt.int(resp.total)} transactions analyzed — ${resp.fraud_count} flagged as potential fraud (${resp.fraud_rate_pct.toFixed(1)}%). ` +
        `Results have been saved to the Investigation tab for review.`;
      document.getElementById("upload-fraud-pct").textContent = `${resp.fraud_rate_pct.toFixed(2)}%`;
      document.getElementById("upload-not-fraud-pct").textContent = `${resp.not_fraud_rate_pct.toFixed(2)}%`;
      document.querySelector("#table-upload-results tbody").innerHTML =
        resp.results.map(uploadRowHtml).join("") || emptyRow(5);
      resultsCard.style.display = "block";

      // Show success message
      showToast(`${resp.fraud_count} suspicious transactions saved to the Investigation Queue.`, resp.fraud_count > 0 ? "warning" : "good");
    } catch (err) {
      statusEl.textContent = "";
      showToast(err.message, "critical");
    } finally {
      btn.disabled = false;
      btn.textContent = "Analyze Transactions";
    }
  });
}

// ---------------------------------------------------------------------
// prediction history
// ---------------------------------------------------------------------
let historyEntries = [];

function historyRowHtml(entry) {
  const highRisk = entry.fraud_rate_pct >= 20;
  const pctStyle = highRisk ? ` style="color:var(--status-critical);font-weight:600"` : "";
  return `<tr data-history-id="${Fmt.escapeHtml(entry.id || "")}" style="cursor:pointer">
    <td class="tabular">${Fmt.escapeHtml(new Date(entry.timestamp).toLocaleString())}</td>
    <td>${Fmt.escapeHtml(entry.filename)}</td>
    <td>${Fmt.escapeHtml(entry.model)}</td>
    <td class="num tabular">${Fmt.int(entry.total)}</td>
    <td class="num tabular">${Fmt.int(entry.fraud_count)}</td>
    <td class="num tabular">${Fmt.int(entry.not_fraud_count)}</td>
    <td class="num tabular"${pctStyle}>${entry.fraud_rate_pct.toFixed(2)}%</td>
  </tr>`;
}

async function loadHistory() {
  try {
    const resp = await Api.uploadHistory();
    historyEntries = resp.entries;
    document.getElementById("history-subtitle").textContent = historyEntries.length
      ? `${Fmt.int(historyEntries.length)} run(s) recorded, newest first. Click a row to open its full per-transaction results.`
      : "No Upload & Predict runs recorded yet -- score a CSV from the Upload & Predict page and it will show up here.";
    document.querySelector("#table-history tbody").innerHTML =
      historyEntries.map(historyRowHtml).join("") || emptyRow(7);
    document.getElementById("history-detail-card").style.display = "none";
  } catch (err) {
    showToast(err.message, "critical");
  }
}

function openHistoryDetail(id) {
  const entry = historyEntries.find((e) => e.id === id);
  const card = document.getElementById("history-detail-card");
  if (!entry || !entry.results) {
    showToast("No saved per-transaction detail for this run (it predates this feature).", "warning");
    card.style.display = "none";
    return;
  }
  document.getElementById("history-detail-title").textContent = entry.filename;
  document.getElementById("history-detail-subtitle").textContent =
    `${new Date(entry.timestamp).toLocaleString()} -- ${Fmt.int(entry.total)} transactions -- model: ${entry.model}`;
  document.getElementById("history-detail-fraud-pct").textContent = `${entry.fraud_rate_pct.toFixed(2)}%`;
  document.getElementById("history-detail-not-fraud-pct").textContent = `${entry.not_fraud_rate_pct.toFixed(2)}%`;
  document.querySelector("#table-history-detail tbody").innerHTML =
    entry.results.map(uploadRowHtml).join("") || emptyRow(5);
  card.style.display = "block";
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function wireHistoryControls() {
  document.getElementById("history-refresh-btn").addEventListener("click", loadHistory);
  document.getElementById("history-detail-close-btn").addEventListener("click", () => {
    document.getElementById("history-detail-card").style.display = "none";
  });
  document.querySelector("#table-history tbody").addEventListener("click", (e) => {
    const row = e.target.closest("tr[data-history-id]");
    if (row) openHistoryDetail(row.dataset.historyId);
  });
}

// ---------------------------------------------------------------------
// detail drawer
// ---------------------------------------------------------------------
function modelChip(m) {
  if (!m.applicable) {
    return `<span class="model-chip na" title="Not applicable to this row">${Fmt.escapeHtml(m.label)} · n/a</span>`;
  }
  const cls = m.flagged ? "model-chip flagged" : "model-chip";
  const icon = m.flagged ? Icons.serious : Icons.good;
  return `<span class="${cls}">${icon}&nbsp;${Fmt.escapeHtml(m.label)} <span class="pct">${(m.percentile * 100).toFixed(1)}%</span></span>`;
}

async function openDrawer(txId) {
  document.getElementById("drawer-backdrop").classList.add("visible");
  document.getElementById("detail-drawer").classList.add("visible");
  document.getElementById("drawer-title").textContent = txId;
  document.getElementById("drawer-subtitle").textContent = "Loading transaction detail…";
  document.getElementById("drawer-body").innerHTML = `<div class="skeleton-line skeleton" style="width:60%"></div>
    <div class="skeleton-line skeleton" style="width:80%"></div><div class="skeleton-line skeleton" style="width:40%"></div>`;
  try {
    const d = await Api.transaction(txId);
    renderDrawer(d);
  } catch (e) {
    document.getElementById("drawer-body").innerHTML = `<div class="empty-state">${Fmt.escapeHtml(e.message)}</div>`;
  }
}
function closeDrawer() {
  document.getElementById("drawer-backdrop").classList.remove("visible");
  document.getElementById("detail-drawer").classList.remove("visible");
}

function renderDrawer(d) {
  document.getElementById("drawer-title").textContent = d.transaction_id;
  document.getElementById("drawer-subtitle").textContent = `Account ${d.account_id}`;
  const r = d.raw;
  const body = document.getElementById("drawer-body");
  body.innerHTML = `
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      ${riskBadge(d.risk.risk_tier_code)}
      <span class="badge badge-neutral">Risk Score: ${(d.risk.risk_score * 100).toFixed(1)}%</span>
    </div>
    <div class="field-grid">
      ${fieldItem("Amount", Fmt.money(r.amount))}
      ${fieldItem("Date", Fmt.dateTime(r.date))}
      ${fieldItem("Type", Fmt.escapeHtml(r.txn_type))}
      ${fieldItem("Channel", Fmt.escapeHtml(r.channel))}
      ${fieldItem("Location", Fmt.escapeHtml(r.location))}
      ${fieldItem("Device ID", Fmt.escapeHtml(r.device_id))}
      ${fieldItem("IP Address", Fmt.escapeHtml(r.ip_address))}
      ${fieldItem("Merchant ID", Fmt.escapeHtml(r.merchant_id))}
      ${fieldItem("Customer Age", r.customer_age)}
      ${fieldItem("Occupation", Fmt.escapeHtml(r.customer_occupation))}
      ${fieldItem("Duration", `${r.duration_seconds}s`)}
      ${fieldItem("Login Attempts", r.login_attempts)}
      ${fieldItem("Account Balance", Fmt.money(r.account_balance))}
      ${fieldItem("Status", queueStatusBadge(d.queue_action))}
    </div>
  `;
}

function wireDrawer() {
  document.getElementById("drawer-backdrop").addEventListener("click", closeDrawer);
  document.getElementById("drawer-close-btn").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
}

// ---------------------------------------------------------------------
// command palette -- Ctrl/Cmd+K quick jump to a transaction or account
// ---------------------------------------------------------------------
let paletteActiveIndex = -1;
let paletteResults = [];

function isPaletteOpen() {
  return document.getElementById("command-palette").classList.contains("visible");
}
function openPalette() {
  document.getElementById("palette-backdrop").classList.add("visible");
  document.getElementById("command-palette").classList.add("visible");
  const input = document.getElementById("palette-input");
  input.value = "";
  renderPaletteResults([], "");
  input.focus();
}
function closePalette() {
  document.getElementById("palette-backdrop").classList.remove("visible");
  document.getElementById("command-palette").classList.remove("visible");
}

function renderPaletteResults(results, query) {
  paletteResults = results;
  paletteActiveIndex = results.length ? 0 : -1;
  const el = document.getElementById("palette-results");
  if (!query) {
    el.innerHTML = `<div class="palette-empty">Type a transaction or account ID to jump straight to it.</div>`;
    return;
  }
  if (!results.length) {
    el.innerHTML = `<div class="palette-empty">No matches for "${Fmt.escapeHtml(query)}".</div>`;
    return;
  }
  el.innerHTML = results.map((tx, i) => `
    <div class="palette-result-item${i === 0 ? " active" : ""}" data-index="${i}">
      <div class="palette-result-main">
        ${riskBadge(tx.risk_tier_code)}
        <span class="palette-result-id">${Fmt.escapeHtml(tx.transaction_id)}</span>
      </div>
      <span class="palette-result-sub">${Fmt.escapeHtml(tx.account_id)} &middot; ${Fmt.money(tx.amount)}</span>
    </div>`).join("");
}

function setPaletteActive(index) {
  const items = document.querySelectorAll(".palette-result-item");
  if (!items.length) return;
  paletteActiveIndex = ((index % items.length) + items.length) % items.length;
  items.forEach((it, i) => it.classList.toggle("active", i === paletteActiveIndex));
  items[paletteActiveIndex].scrollIntoView({ block: "nearest" });
}

function selectPaletteResult(index) {
  const tx = paletteResults[index];
  if (!tx) return;
  closePalette();
  goToExplorerWithFilter("", "date");
  explorerState.q = tx.transaction_id;
  const searchInput = document.getElementById("explorer-search");
  if (searchInput) searchInput.value = tx.transaction_id;
  loadExplorer().then(() => openDrawer(tx.transaction_id));
}

function wireCommandPalette() {
  const input = document.getElementById("palette-input");
  const searchIcon = document.getElementById("palette-search-icon");
  if (searchIcon) searchIcon.innerHTML = Icons.search;

  input.addEventListener("input", Fmt.debounce(async (e) => {
    const q = e.target.value.trim();
    if (!q) { renderPaletteResults([], ""); return; }
    try {
      const resp = await Api.transactions({ q, page: 1, page_size: 8, sort_by: "risk_score", sort_dir: "desc" });
      renderPaletteResults(resp.results, q);
    } catch (err) {
      document.getElementById("palette-results").innerHTML = `<div class="palette-empty">${Fmt.escapeHtml(err.message)}</div>`;
    }
  }, 220));

  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); setPaletteActive(paletteActiveIndex + 1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setPaletteActive(paletteActiveIndex - 1); }
    else if (e.key === "Enter") { e.preventDefault(); if (paletteActiveIndex >= 0) selectPaletteResult(paletteActiveIndex); }
  });

  document.getElementById("palette-results").addEventListener("click", (e) => {
    const item = e.target.closest(".palette-result-item");
    if (item) selectPaletteResult(Number(item.dataset.index));
  });

  document.getElementById("palette-backdrop").addEventListener("click", closePalette);
}

// ---------------------------------------------------------------------
// keyboard shortcuts help popup
// ---------------------------------------------------------------------
function isShortcutsModalOpen() {
  return document.getElementById("shortcuts-modal").classList.contains("visible");
}
function openShortcutsModal() {
  document.getElementById("shortcuts-modal-backdrop").classList.add("visible");
  document.getElementById("shortcuts-modal").classList.add("visible");
}
function closeShortcutsModal() {
  document.getElementById("shortcuts-modal-backdrop").classList.remove("visible");
  document.getElementById("shortcuts-modal").classList.remove("visible");
}
function wireShortcutsHelp() {
  document.getElementById("shortcuts-fab").addEventListener("click", openShortcutsModal);
  document.getElementById("shortcuts-modal-close").addEventListener("click", closeShortcutsModal);
  document.getElementById("shortcuts-modal-backdrop").addEventListener("click", closeShortcutsModal);
}

// ---------------------------------------------------------------------
// global keyboard shortcuts: Ctrl/Cmd+K (quick jump), ? (shortcut help),
// Escape (close whichever popup is open)
// ---------------------------------------------------------------------
function wireGlobalShortcuts() {
  document.addEventListener("keydown", (e) => {
    const mod = e.ctrlKey || e.metaKey;
    if (mod && (e.key === "k" || e.key === "K")) {
      e.preventDefault();
      if (isPaletteOpen()) document.getElementById("palette-input").focus();
      else openPalette();
      return;
    }
    if (e.key === "Escape") {
      if (isPaletteOpen()) closePalette();
      if (isShortcutsModalOpen()) closeShortcutsModal();
      return;
    }
    if (e.key === "?" && !mod) {
      const active = document.activeElement;
      const tag = active ? active.tagName : "";
      if (["INPUT", "TEXTAREA", "SELECT"].includes(tag) || (active && active.isContentEditable)) return;
      e.preventDefault();
      openShortcutsModal();
    }
  });
}

// ---------------------------------------------------------------------
// account-by-account analysis - works with ANY account data
// ---------------------------------------------------------------------
function wireAccountAnalysisForm() {
  document.getElementById("account-csv-form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const input = document.getElementById("account-csv-input");
    const file = input.files && input.files[0];
    const statusEl = document.getElementById("account-analysis-status");
    const btn = document.getElementById("analyze-account-csv-btn");
    const summaryCard = document.getElementById("account-summary-card");
    const resultsCard = document.getElementById("account-results-card");

    if (!file) {
      statusEl.textContent = "Please select a CSV file";
      return;
    }

    btn.disabled = true;
    btn.textContent = "Analyzing...";
    statusEl.textContent = `Processing ${file.name}...`;
    summaryCard.style.display = "none";
    resultsCard.style.display = "none";

    try {
      // Use the same upload endpoint that handles preprocessing
      const resp = await Api.uploadPredict(file);

      if (resp.results.length === 0) {
        statusEl.textContent = "No valid transactions found in the uploaded file";
        btn.disabled = false;
        btn.textContent = "Analyze Account";
        return;
      }

      // Extract account ID from first transaction
      const accountId = resp.results[0].account_id || "Unknown";

      // Verify all transactions are for the same account
      const uniqueAccounts = [...new Set(resp.results.map(tx => tx.account_id))];
      if (uniqueAccounts.length > 1) {
        statusEl.textContent = `Warning: CSV contains transactions from ${uniqueAccounts.length} different accounts. Showing all results.`;
      }

      // Calculate summary
      const fraudCount = resp.fraud_count;
      const avgRisk = resp.results.reduce((sum, tx) => sum + tx.fraud_percentage, 0) / resp.results.length;
      const overallRisk = avgRisk >= 70 ? "HIGH" : avgRisk >= 40 ? "MEDIUM" : "LOW";

      // Display summary
      document.getElementById("account-summary-id").textContent = accountId;
      document.getElementById("account-summary-total").textContent = resp.total;
      document.getElementById("account-summary-fraud").textContent = fraudCount;
      document.getElementById("account-summary-risk").textContent = overallRisk;
      document.getElementById("account-summary-risk").style.color =
        avgRisk >= 70 ? "var(--status-critical)" : avgRisk >= 40 ? "var(--status-warning)" : "var(--status-good)";

      // Display results table
      const tbody = document.querySelector("#table-account-results tbody");
      tbody.innerHTML = resp.results.map(tx => {
        const isFlagged = tx.fraud_percentage >= resp.fraud_cutoff_pct;
        const statusBadge = isFlagged
          ? badgeHtml("critical", "Flagged", Icons.warning)
          : badgeHtml("good", "Normal", Icons.good);

        return `<tr>
          <td>${Fmt.escapeHtml(tx.transaction_id)}</td>
          <td>${tx.date || 'N/A'}</td>
          <td class="num tabular">${tx.amount ? Fmt.money(tx.amount) : 'N/A'}</td>
          <td>${tx.channel || 'N/A'}</td>
          <td>${tx.type || 'N/A'}</td>
          <td class="num tabular" style="color:${isFlagged ? 'var(--status-critical)' : 'var(--text-primary)'}">${tx.fraud_percentage.toFixed(1)}%</td>
          <td>${statusBadge}</td>
        </tr>`;
      }).join("");

      // Save to Investigation tab
      const timestamp = new Date().toISOString();
      const newTransactions = [];
      resp.results.forEach(tx => {
        const txData = {
          ...tx,
          uploaded_at: timestamp,
          status: 'pending',
          source: 'account_analysis'
        };
        uploadedTransactions.push(txData);
        newTransactions.push(txData);
      });
      saveUploadedTransactions();

      // Add to upload history
      addToUploadHistory(file.name, newTransactions);

      // Show results
      summaryCard.style.display = "block";
      resultsCard.style.display = "block";
      document.getElementById("account-results-subtitle").textContent =
        `${resp.total} transactions analyzed for account ${accountId}. ${fraudCount} flagged as potential fraud (${resp.fraud_rate_pct.toFixed(1)}%). Results saved to Investigation tab.`;

      statusEl.textContent = "";
      showToast(`Account analysis complete: ${fraudCount} of ${resp.total} transactions flagged`);

      // Clear file input
      input.value = "";

    } catch (err) {
      statusEl.textContent = `Error: ${err.message}`;
      console.error("Account analysis error:", err);
    } finally {
      btn.disabled = false;
      btn.textContent = "Analyze Account";
    }
  });

  // Wire up account report button
  const reportBtn = document.getElementById("generate-account-report-btn");
  if (reportBtn) {
    reportBtn.addEventListener("click", generateFraudReport);
  }
}

// ---------------------------------------------------------------------
// manual entry form handling
// ---------------------------------------------------------------------
async function loadManualEntryOptions() {
  if (simOptions) {
    fillManualEntryDropdowns();
    return;
  }
  try {
    simOptions = await Api.get("/api/simulator/options");
    fillManualEntryDropdowns();
  } catch (err) {
    console.error("Failed to load manual entry options:", err);
  }
}

function fillManualEntryDropdowns() {
  // Manual Entry should be completely clean - user types their own data
  // Do not populate any dropdowns with saved data
  return;
}

function wireManualEntryForm() {
  const form = document.getElementById("manual-entry-form");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const body = {
      account_id: document.getElementById("manual-account").value,
      amount: parseFloat(document.getElementById("manual-amount").value),
      account_balance: parseFloat(document.getElementById("manual-balance").value),
      txn_type: document.getElementById("manual-type").value,
      channel: document.getElementById("manual-channel").value,
      location: document.getElementById("manual-location").value,
      device_id: document.getElementById("manual-device").value,
      ip_address: document.getElementById("manual-ip").value,
      merchant_id: document.getElementById("manual-merchant").value,
      customer_occupation: document.getElementById("manual-occupation").value,
      customer_age: parseInt(document.getElementById("manual-age").value),
      duration_seconds: parseInt(document.getElementById("manual-duration").value),
      login_attempts: parseInt(document.getElementById("manual-login").value),
    };

    try {
      const result = await Api.post("/api/score", body);
      displayManualEntryResult(result);
      document.getElementById("manual-result-card").style.display = "block";
      document.getElementById("manual-result-card").scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (err) {
      showToast(`Prediction failed: ${err.message || err}`);
    }
  });
}

function displayManualEntryResult(result) {
  document.getElementById("manual-tier-badge").innerHTML = riskBadge(result.risk_tier_code);
  document.getElementById("manual-score").textContent = (result.two_model_percentile_average * 100).toFixed(2) + "%";
  document.getElementById("manual-score-caption").textContent =
    `Percentile average of Isolation Forest (${(result.isolation_forest.percentile * 100).toFixed(1)}%) and Autoencoder (${(result.autoencoder.percentile * 100).toFixed(1)}%)`;

  // Don't display saved info details - keep it clean for manual entry
  document.getElementById("manual-detail-grid").innerHTML = "";
}

// ---------------------------------------------------------------------
// case investigation page
// ---------------------------------------------------------------------
async function openCaseInvestigation(transactionId) {
  currentCaseId = transactionId;

  try {
    const data = await Api.get(`/api/transactions/${transactionId}`);
    renderCaseInvestigation(data);

    // Show case detail tab and make it visible
    const tabBtn = document.getElementById("tab-case-detail-btn");
    tabBtn.style.display = "block";
    showTab("investigation", "case-detail");
  } catch (err) {
    showToast(`Failed to load case: ${err.message || err}`);
  }
}

function renderCaseInvestigation(d) {
  const r = d.raw;

  // Update header
  document.getElementById("case-detail-title").textContent = `Transaction ${d.transaction_id}`;
  document.getElementById("case-detail-subtitle").textContent = `Account ${d.account_id} • ${Fmt.dateTime(r.date)} • ${Fmt.money(r.amount)}`;

  // Fraud score display
  const scorePercent = (d.risk.risk_score * 100).toFixed(1);
  document.getElementById("case-score-value").textContent = scorePercent + "%";
  document.getElementById("case-score-label").textContent = d.risk.risk_tier_label.toUpperCase();
  const scoreCircle = document.getElementById("case-score-circle");
  scoreCircle.className = `score-circle ${d.risk.risk_tier_code}`;

  document.getElementById("case-score-details").innerHTML = `
    <div class="kpi-tile">
      <p class="kpi-label">Risk Rank</p>
      <div class="kpi-value">#${d.risk.score_rank}</div>
    </div>
    <div class="kpi-tile">
      <p class="kpi-label">Risk Indicators</p>
      <div class="kpi-value">${d.risk.models_flagged} of ${d.risk.models_applicable}</div>
    </div>
  `;

  // Case summary
  document.getElementById("case-summary-grid").innerHTML = `
    ${fieldItem("Transaction ID", d.transaction_id)}
    ${fieldItem("Account ID", d.account_id)}
    ${fieldItem("Date", Fmt.dateTime(r.date))}
    ${fieldItem("Amount", Fmt.money(r.amount))}
    ${fieldItem("Type", r.txn_type)}
    ${fieldItem("Channel", r.channel)}
    ${fieldItem("Location", r.location)}
    ${fieldItem("Device ID", r.device_id)}
    ${fieldItem("IP Address", r.ip_address)}
    ${fieldItem("Merchant ID", r.merchant_id)}
    ${fieldItem("Customer Age", r.customer_age)}
    ${fieldItem("Occupation", r.customer_occupation)}
    ${fieldItem("Duration", `${r.duration_seconds}s`)}
    ${fieldItem("Login Attempts", r.login_attempts)}
    ${fieldItem("Account Balance", Fmt.money(r.account_balance))}
    ${fieldItem("Amount/Balance Ratio", `${r.amount_to_balance_ratio.toFixed(3)}×`)}
  `;

  // Risk factors - simplified summary
  const riskFactorCount = d.risk.models_flagged || 0;
  const riskLevel = d.risk.risk_tier_code;
  const riskText = riskLevel === 'priority' ? 'Multiple risk factors detected' :
                   riskLevel === 'standard' ? 'Some risk factors detected' :
                   'Minimal risk factors detected';

  document.getElementById("case-models-subtitle").textContent = riskText;

  // Show simplified risk factors summary
  const riskFactorEl = document.getElementById("case-risk-factors-summary");
  if (riskFactorEl) {
    riskFactorEl.innerHTML = `
      <div style="padding:16px;background:color-mix(in srgb, var(--${riskLevel === 'priority' ? 'status-critical' : riskLevel === 'standard' ? 'status-warning' : 'status-good'}) 10%, transparent);border-radius:6px">
        <p style="font-size:14px;margin:0">${riskFactorCount} indicators flagged this transaction as ${d.risk.risk_tier_label}</p>
      </div>
    `;
  }

  // Simple explanation chart
  const explanationEl = document.getElementById("case-explanation-chart");
  if (explanationEl && d.shap_isolation_forest && d.shap_isolation_forest.length > 0) {
    const topFactors = d.shap_isolation_forest.slice(0, 5);
    explanationEl.innerHTML = `
      <div style="padding:16px">
        <p style="font-size:13px;font-weight:600;margin-bottom:12px">Top Risk Factors:</p>
        <ul style="list-style:none;padding:0;margin:0">
          ${topFactors.map(f => `
            <li style="padding:8px 0;border-bottom:1px solid var(--border);font-size:13px">
              <strong>${f.label}</strong>
              <div style="font-size:12px;color:var(--text-secondary);margin-top:4px">Value: ${f.feature_value.toFixed(2)}</div>
            </li>
          `).join('')}
        </ul>
      </div>
    `;
  } else if (explanationEl) {
    explanationEl.innerHTML = '<div style="padding:16px;color:var(--text-muted)">Detailed explanation not available for this transaction</div>';
  }

  // Status info
  document.getElementById("case-status-info").textContent =
    `Current status: ${d.queue_action} ${d.queue_updated_at ? `(updated ${new Date(d.queue_updated_at).toLocaleString()})` : ""}`;

  // Wire action buttons
  document.querySelectorAll(".case-actions button[data-action]").forEach((btn) => {
    btn.onclick = async () => {
      try {
        await Api.post("/api/queue/action", {
          transaction_id: currentCaseId,
          action: btn.dataset.action,
        });
        showToast(`Case status updated to ${btn.dataset.action}`);
        document.getElementById("case-status-info").textContent =
          `Current status: ${btn.dataset.action} (updated just now)`;
      } catch (err) {
        showToast(`Failed to update status: ${err.message || err}`);
      }
    };
  });
}

function wireCaseBackButton() {
  document.getElementById("case-back-btn").addEventListener("click", () => {
    showTab("investigation", "flagged-cases");
    document.getElementById("tab-case-detail-btn").style.display = "none";
  });
}

// ---------------------------------------------------------------------
// navigation / theme / bootstrap
// ---------------------------------------------------------------------
function populateNav() {
  document.querySelectorAll(".nav-item").forEach((btn) => {
    const meta = NAV_META[btn.dataset.page];
    if (meta) {
      btn.innerHTML = `${Icons[meta.icon]}<span>${meta.label}</span>`;
      btn.addEventListener("click", () => showPage(btn.dataset.page));
    }
  });
  const simFlaskIcon = document.getElementById("simulator-flask-icon");
  if (simFlaskIcon) simFlaskIcon.innerHTML = Icons.flask;
  document.getElementById("drawer-close-btn").innerHTML = Icons.close;
  document.getElementById("queue-prev").innerHTML = Icons.chevronLeft;
  document.getElementById("queue-next").innerHTML = Icons.chevronRight;
  document.querySelectorAll("[data-kpi-icon]").forEach((el) => { el.innerHTML = Icons[el.dataset.kpiIcon]; });
}

function showPage(page) {
  currentPage = page;
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.page === page));
  document.querySelectorAll(".page").forEach((s) => s.classList.toggle("active", s.id === `page-${page}`));
  const meta = NAV_META[page];
  document.getElementById("page-title").textContent = meta.title;
  document.getElementById("page-subtitle").textContent = meta.subtitle;
  Charts.hideTooltip();
  if (page === "overview") loadOverview();
  else if (page === "explorer") loadExplorer();
  else if (page === "queue") loadQueue();
  else if (page === "comparison") loadComparison();
  else if (page === "explainability") loadExplainability();
  else if (page === "history") loadHistory();
  else if (page === "simulator") loadSimulatorOptions();
}

function rerenderCurrentPageCharts() {
  if (currentPage === "overview" && overviewData) renderOverviewCharts(overviewData);
  else if (currentPage === "comparison" && comparisonData) renderComparison(comparisonData);
  else if (currentPage === "explainability" && explainabilityData) renderExplainability(explainabilityData);
  else if (currentPage === "simulator" && lastSimResult) renderSimCharts(lastSimResult);
}

function wireDisclosureToggle(btnId, panelId, openLabel, closeLabel) {
  const btn = document.getElementById(btnId);
  const panel = document.getElementById(panelId);
  if (!btn || !panel) return;
  btn.addEventListener("click", () => {
    const open = panel.style.display !== "none";
    panel.style.display = open ? "none" : "block";
    btn.textContent = open ? openLabel : closeLabel;
  });
}

document.addEventListener("DOMContentLoaded", () => {
  loadUploadedTransactions(); // Load uploaded transactions from localStorage
  loadUploadHistory(); // Load upload history
  populateNav();
  wireExplorerControls();
  wireQueueControls();
  wireSimulatorForm();
  wireUploadForm();
  wireHistoryControls();
  wireDrawer();
  wireTopRiskTable();
  wireKpiTileNav();
  wireModelsSort();
  wireCommandPalette();
  wireShortcutsHelp();
  wireGlobalShortcuts();
  wireDisclosureToggle("upload-format-toggle", "upload-format-panel", "Format details", "Hide format details");
  wireDisclosureToggle("about-toggle", "about-panel-detail", "Read the full methodology notes", "Hide the methodology notes");
  wireDisclosureToggle("simulator-note-toggle", "simulator-note-panel", "Why can't I enter a brand-new transaction?", "Hide explanation");
  showPage("overview");
  window.addEventListener("resize", Fmt.debounce(rerenderCurrentPageCharts, 200));
});
