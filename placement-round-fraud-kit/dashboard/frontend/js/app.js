/* Argus dashboard -- single-page app wiring. No framework, no build step. */

const NAV_META = {
  overview: { label: "Overview", icon: "overview", title: "Overview Dashboard", subtitle: "Portfolio-wide transaction risk, at a glance" },
  explorer: { label: "Transaction Explorer", icon: "explorer", title: "Transaction Explorer", subtitle: "Browse, search, and filter every scored transaction" },
  queue: { label: "Investigation Queue", icon: "queue", title: "Investigation Queue", subtitle: "Highest-risk transactions, sorted for triage" },
  comparison: { label: "Model Comparison", icon: "comparison", title: "Model Comparison", subtitle: "Detector ensemble and supervised-model performance" },
  explainability: { label: "Explainability", icon: "explainability", title: "Explainability", subtitle: "Why the model decides what it decides" },
  simulator: { label: "What-if Simulator", icon: "simulator", title: "What-if Simulator", subtitle: "Secondary tool -- hypothesize about one new transaction" },
};

const DETECTOR_LABELS = { isoforest: "Isolation Forest", lof: "LOF", ocsvm: "One-Class SVM", mcd: "Elliptic Envelope (MCD)" };

let currentPage = "overview";
let overviewData = null;
let comparisonData = null;
let explainabilityData = null;
let lastSimResult = null;

const explorerState = {
  q: "", risk_tier: "", channel: "", txn_type: "", amount_min: "", amount_max: "",
  date_start: "", date_end: "", sort_by: "date", sort_dir: "desc", page: 1, page_size: 25,
};
const queueState = { status: "", page: 1, page_size: 25 };

// ---------------------------------------------------------------------
// small shared helpers
// ---------------------------------------------------------------------
function badgeHtml(status, label, icon) {
  return `<span class="badge badge-${status}">${icon}${Fmt.escapeHtml(label)}</span>`;
}
function riskBadge(tierCode, verdictCode) {
  if (tierCode === "normal") return badgeHtml("good", "Normal", Icons.good);
  if (tierCode === "medium") return badgeHtml("warning", "Needs review", Icons.warning);
  if (tierCode === "high") {
    return verdictCode === "block"
      ? badgeHtml("critical", "Blocked", Icons.critical)
      : badgeHtml("serious", "High risk", Icons.serious);
  }
  return badgeHtml("neutral", "Unknown", "");
}
function verdictBadge(code) {
  if (code === "approve") return badgeHtml("good", "Auto-approve", Icons.good);
  if (code === "review") return badgeHtml("warning", "Manual review", Icons.warning);
  return badgeHtml("critical", "Block", Icons.critical);
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
function showToast(msg) {
  const stack = document.getElementById("toast-stack");
  const t = document.createElement("div");
  t.className = "toast";
  t.textContent = msg;
  stack.appendChild(t);
  setTimeout(() => {
    t.style.transition = "opacity 200ms ease";
    t.style.opacity = "0";
    setTimeout(() => t.remove(), 200);
  }, 2600);
}

function txRowHtml(tx, includeType) {
  const typeCell = includeType ? `<td>${Fmt.escapeHtml(tx.txn_type)}</td>` : "";
  return `<tr data-id="${Fmt.escapeHtml(tx.transaction_id)}">
    <td>${Fmt.escapeHtml(tx.transaction_id)}</td>
    <td>${Fmt.escapeHtml(tx.account_id)}</td>
    <td>${Fmt.dateTime(tx.date)}</td>
    <td class="num tabular">${Fmt.money(tx.amount)}</td>
    <td>${Fmt.escapeHtml(tx.channel)}</td>
    ${typeCell}
    <td>${riskBadge(tx.risk_tier_code, tx.verdict_code)}</td>
    <td class="num tabular">${Fmt.score(tx.risk_score)}</td>
  </tr>`;
}

// ---------------------------------------------------------------------
// overview
// ---------------------------------------------------------------------
async function loadOverview() {
  if (!overviewData) {
    try {
      overviewData = await Api.kpis();
    } catch (e) {
      showToast(`Could not load overview data: ${e.message}`);
      return;
    }
    document.querySelectorAll(".kpi-value").forEach((el) => el.classList.remove("skeleton-loading"));
    Fmt.countUp(document.querySelector('[data-kpi="total"]'), overviewData.total_transactions, { formatter: Fmt.int });
    Fmt.countUp(document.querySelector('[data-kpi="high"]'), overviewData.high_risk_count, { formatter: Fmt.int });
    Fmt.countUp(document.querySelector('[data-kpi="review"]'), overviewData.review_count, { formatter: Fmt.int });
    Fmt.countUp(document.querySelector('[data-kpi="flagrate"]'), overviewData.flag_rate, { formatter: Fmt.pct, decimals: 4 });
    Fmt.countUp(document.querySelector('[data-kpi="avgamount"]'), overviewData.avg_amount, { formatter: Fmt.money, decimals: 2 });
  }
  renderOverviewCharts(overviewData);
  const tbody = document.querySelector("#table-top-risk tbody");
  tbody.innerHTML = overviewData.top_risk.map((tx) => txRowHtml(tx, false)).join("") || emptyRow(7);
}

function renderOverviewCharts(data) {
  const tierColor = { high: Charts.cssVar("--status-critical"), medium: Charts.cssVar("--status-warning"), normal: Charts.cssVar("--status-good") };
  Charts.renderBarChart(document.getElementById("chart-tier-distribution"), {
    data: data.tier_distribution.map((t) => ({ label: t.tier, value: t.count, color: tierColor[t.code] })),
    valueFormatter: Fmt.int,
  });
  const ts = data.timeseries.map((d) => ({ x: new Date(d.date).getTime(), y: d.count }));
  Charts.renderLineChart(document.getElementById("chart-timeseries"), {
    data: ts, color: Charts.cssVar("--series-1-blue"), area: true,
    xFormatter: (v) => Fmt.dateShort(new Date(v).toISOString()), yFormatter: Fmt.int,
  });
}

// ---------------------------------------------------------------------
// explorer
// ---------------------------------------------------------------------
async function loadExplorer() {
  const tbody = document.querySelector("#table-explorer tbody");
  tbody.innerHTML = skeletonRows(8, 8);
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
    tbody.innerHTML = `<tr><td colspan="8"><div class="empty-state">${Fmt.escapeHtml(e.message)}</div></td></tr>`;
    return;
  }
  tbody.innerHTML = resp.results.map((tx) => txRowHtml(tx, true)).join("") || emptyRow(8);
  document.getElementById("explorer-count").textContent = `${Fmt.int(resp.total)} transactions`;
  const totalPages = Math.max(1, Math.ceil(resp.total / resp.page_size));
  document.getElementById("explorer-page-label").textContent = `Page ${resp.page} of ${totalPages}`;
  document.getElementById("explorer-prev").disabled = resp.page <= 1;
  document.getElementById("explorer-next").disabled = resp.page >= totalPages;
  updateSortIndicators();
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
    document.getElementById("explorer-search").value = "";
    document.getElementById("explorer-risk-tier").value = "";
    document.getElementById("explorer-channel").value = "";
    document.getElementById("explorer-txn-type").value = "";
    document.getElementById("explorer-amount-min").value = "";
    document.getElementById("explorer-amount-max").value = "";
    document.getElementById("explorer-date-start").value = "";
    document.getElementById("explorer-date-end").value = "";
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
    const tr = e.target.closest("tr");
    if (tr && tr.dataset.id) openDrawer(tr.dataset.id);
  });
}

// ---------------------------------------------------------------------
// investigation queue
// ---------------------------------------------------------------------
async function loadQueue() {
  const tbody = document.querySelector("#table-queue tbody");
  tbody.innerHTML = skeletonRows(8, 8);
  let resp;
  try {
    resp = await Api.queue({ status: queueState.status || undefined, page: queueState.page, page_size: queueState.page_size });
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="8"><div class="empty-state">${Fmt.escapeHtml(e.message)}</div></td></tr>`;
    return;
  }
  const startRank = (resp.page - 1) * resp.page_size + 1;
  tbody.innerHTML = resp.results.map((tx, i) => `<tr data-id="${Fmt.escapeHtml(tx.transaction_id)}">
      <td class="num tabular">${startRank + i}</td>
      <td>${Fmt.escapeHtml(tx.transaction_id)}</td>
      <td>${Fmt.escapeHtml(tx.account_id)}</td>
      <td class="num tabular">${Fmt.money(tx.amount)}</td>
      <td>${riskBadge(tx.risk_tier_code, tx.verdict_code)}</td>
      <td class="num tabular">${Fmt.score(tx.risk_score)}</td>
      <td>${queueStatusBadge(tx.queue_action)}</td>
      <td>
        <button class="btn btn-sm action-approve" data-action="approved">Approve</button>
        <button class="btn btn-sm action-escalate" data-action="escalated">Escalate</button>
        <button class="btn btn-sm action-block" data-action="blocked">Block</button>
      </td>
    </tr>`).join("") || emptyRow(8);
  document.getElementById("queue-count").textContent = `${Fmt.int(resp.total)} transactions`;
  const totalPages = Math.max(1, Math.ceil(resp.total / resp.page_size));
  document.getElementById("queue-page-label").textContent = `Page ${resp.page} of ${totalPages}`;
  document.getElementById("queue-prev").disabled = resp.page <= 1;
  document.getElementById("queue-next").disabled = resp.page >= totalPages;
}

async function handleQueueAction(transactionId, action) {
  try {
    await Api.queueAction(transactionId, action);
    showToast(`${transactionId} marked as ${action}.`);
    loadQueue();
  } catch (e) {
    showToast(`Could not update ${transactionId}: ${e.message}`);
  }
}

function wireQueueControls() {
  document.getElementById("queue-status").addEventListener("change", (e) => { queueState.status = e.target.value; queueState.page = 1; loadQueue(); });
  document.getElementById("queue-prev").addEventListener("click", () => { if (queueState.page > 1) { queueState.page--; loadQueue(); } });
  document.getElementById("queue-next").addEventListener("click", () => { queueState.page++; loadQueue(); });
  document.getElementById("queue-export").addEventListener("click", () => window.open(Api.queueExportUrl(), "_blank"));
  document.querySelector("#table-queue tbody").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-action]");
    const tr = e.target.closest("tr");
    if (!tr) return;
    if (btn) { handleQueueAction(tr.dataset.id, btn.dataset.action); return; }
    openDrawer(tr.dataset.id);
  });
}

// ---------------------------------------------------------------------
// model comparison
// ---------------------------------------------------------------------
async function loadComparison() {
  if (!comparisonData) {
    try { comparisonData = await Api.modelComparison(); }
    catch (e) { showToast(`Could not load model comparison: ${e.message}`); return; }
  }
  renderComparison(comparisonData);
}

function renderComparison(data) {
  const seriesColors = [Charts.cssVar("--series-1-blue"), Charts.cssVar("--series-2-orange"), Charts.cssVar("--series-3-aqua"), Charts.cssVar("--series-4-yellow")];
  const tbody = document.querySelector("#table-detectors tbody");
  tbody.innerHTML = data.detectors.map((d, i) => `<tr>
      <td><span style="display:inline-block;width:9px;height:9px;border-radius:3px;background:${seriesColors[i]};margin-right:7px"></span>${Fmt.escapeHtml(d.name)}</td>
      <td class="num tabular">${Fmt.int(d.flagged)}</td>
      <td class="num tabular">${Fmt.pct(d.rate)}</td>
    </tr>`).join("");

  Charts.renderBarChart(document.getElementById("chart-vote-distribution"), {
    data: data.vote_distribution.map((v) => ({ label: `${v.votes} vote${v.votes === 1 ? "" : "s"}`, value: v.count, color: Charts.cssVar("--series-1-blue") })),
    valueFormatter: Fmt.int,
  });

  const s1 = Charts.cssVar("--series-1-blue"), s2 = Charts.cssVar("--series-2-orange");
  const smote = data.xgboost_variants[0], cw = data.xgboost_variants[1];
  Charts.renderGroupedBarChart(document.getElementById("chart-model-variants"), {
    groups: [
      { label: "ROC-AUC", bars: [{ label: smote.name, value: smote.roc_auc, color: s1 }, { label: cw.name, value: cw.roc_auc, color: s2 }] },
      { label: "PR-AUC", bars: [{ label: smote.name, value: smote.pr_auc, color: s1 }, { label: cw.name, value: cw.pr_auc, color: s2 }] },
    ],
    valueFormatter: (v) => v.toFixed(3),
    legend: [{ label: smote.name, color: s1 }, { label: cw.name, color: s2 }],
  });
  document.getElementById("primary-model-note").textContent = data.primary_model_note;

  const cm = data.confusion_matrix;
  document.getElementById("confusion-matrix").innerHTML = `
    <div class="confusion-grid">
      <div class="cm-label"></div><div class="cm-label">Predicted: Normal</div><div class="cm-label">Predicted: Fraud</div>
      <div class="cm-label">Actual: Normal</div>
      <div class="cm-cell cm-good"><div class="cm-count tabular">${cm.tn}</div><div class="cm-name">${Icons.good} True negative</div></div>
      <div class="cm-cell cm-warning"><div class="cm-count tabular">${cm.fp}</div><div class="cm-name">${Icons.warning} False positive</div></div>
      <div class="cm-label">Actual: Fraud</div>
      <div class="cm-cell cm-critical"><div class="cm-count tabular">${cm.fn}</div><div class="cm-name">${Icons.critical} False negative</div></div>
      <div class="cm-cell cm-good"><div class="cm-count tabular">${cm.tp}</div><div class="cm-name">${Icons.good} True positive</div></div>
    </div>`;
  document.getElementById("confusion-subtitle").textContent =
    `n=${cm.n} · threshold ${cm.threshold} · precision ${cm.precision} · recall ${cm.recall} · F1 ${cm.f1} · ROC-AUC ${cm.roc_auc} · PR-AUC ${cm.pr_auc}`;

  document.getElementById("naive-accuracy").textContent = Fmt.pct(data.accuracy_contrast.naive_accuracy);
  document.getElementById("model-accuracy").textContent = Fmt.pct(data.accuracy_contrast.model_accuracy);
  document.getElementById("accuracy-explanation").textContent = data.accuracy_contrast.explanation;
}

// ---------------------------------------------------------------------
// explainability
// ---------------------------------------------------------------------
async function loadExplainability() {
  if (!explainabilityData) {
    try { explainabilityData = await Api.explainability(); }
    catch (e) { showToast(`Could not load explainability data: ${e.message}`); return; }
  }
  renderExplainability(explainabilityData);
}

function renderExplainability(data) {
  Charts.renderHBarChart(document.getElementById("chart-shap-global"), {
    data: data.global_shap_importance.map((f) => ({ label: f.label, value: f.mean_abs_shap })),
    diverging: false, valueFormatter: (v) => v.toFixed(3),
  });

  document.getElementById("rules-list").innerHTML = data.decision_tree_rules.map((r) => `
    <div class="rule-card">
      <div class="rule-if">If</div>
      <div class="rule-conditions">${r.conditions.map((c) => `<code>${Fmt.escapeHtml(c)}</code>`).join(" <strong>and</strong> ")}</div>
      <div class="rule-then">${r.outcome_code === "1" ? badgeHtml("serious", "Flag as fraud", Icons.serious) : badgeHtml("good", "Clear as normal", Icons.good)}</div>
    </div>`).join("");

  const sweep = data.cost_sweep;
  Charts.renderLineChart(document.getElementById("chart-cost-sweep"), {
    data: sweep.points.map((p) => ({ x: p.threshold, y: p.cost })),
    color: Charts.cssVar("--series-1-blue"), area: true,
    xFormatter: (v) => v.toFixed(2), yFormatter: (v) => Fmt.money(v),
    markers: [
      { x: sweep.min_threshold, label: `min-cost ${sweep.min_threshold}`, color: Charts.cssVar("--series-7-violet") },
      { x: sweep.default_threshold, label: "default 0.50", color: Charts.cssVar("--text-secondary") },
    ],
  });
  document.getElementById("cost-sweep-subtitle").textContent =
    `Illustrative costs (FP $${sweep.cost_false_positive.toFixed(0)}, FN $${sweep.cost_false_negative.toFixed(0)}) -- ` +
    `minimum total cost ${Fmt.money(sweep.min_cost)} at threshold ${sweep.min_threshold}, versus ${Fmt.money(sweep.default_cost)} at the naive default of 0.50.`;
}

// ---------------------------------------------------------------------
// what-if simulator
// ---------------------------------------------------------------------
let simOptionsLoaded = false;
async function loadSimulatorOptions() {
  if (simOptionsLoaded) return;
  try {
    const opts = await Api.simulatorOptions();
    fillDatalist("sim-account-list", opts.accounts);
    fillDatalist("sim-device-list", opts.devices);
    fillDatalist("sim-merchant-list", opts.merchants);
    fillSelect("sim-location", opts.locations);
    fillSelect("sim-occupation", opts.occupations);
    simOptionsLoaded = true;
  } catch (e) {
    showToast(`Could not load simulator reference data: ${e.message}`);
  }
}

function wireSimulatorForm() {
  document.getElementById("simulator-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const payload = {
      account_id: document.getElementById("sim-account").value.trim() || null,
      amount: parseFloat(document.getElementById("sim-amount").value),
      txn_type: document.getElementById("sim-type").value,
      channel: document.getElementById("sim-channel").value,
      location: document.getElementById("sim-location").value,
      customer_occupation: document.getElementById("sim-occupation").value,
      device_id: document.getElementById("sim-device").value.trim() || "D_NEW",
      ip_address: document.getElementById("sim-ip").value.trim() || "10.0.0.1",
      merchant_id: document.getElementById("sim-merchant").value.trim() || "M_NEW",
      customer_age: parseInt(document.getElementById("sim-age").value, 10),
      duration_seconds: parseInt(document.getElementById("sim-duration").value, 10),
      login_attempts: parseInt(document.getElementById("sim-login").value, 10),
      account_balance: parseFloat(document.getElementById("sim-balance").value),
    };
    try {
      lastSimResult = await Api.score(payload);
      renderSimResult(lastSimResult);
    } catch (e2) {
      showToast(`Could not score this transaction: ${e2.message}`);
    }
  });
}

function renderSimResult(result) {
  document.getElementById("simulator-result-card").style.display = "block";
  document.getElementById("sim-verdict-badge").innerHTML = verdictBadge(result.verdict_code);
  document.getElementById("sim-score").textContent = Fmt.score(result.risk_score);
  document.getElementById("sim-thresholds-note").textContent =
    `Review threshold ${Fmt.score(result.review_threshold)} · Block threshold ${Fmt.score(result.block_threshold)}`;
  renderSimShap(result);
}
function renderSimShap(result) {
  if (!result) return;
  Charts.renderHBarChart(document.getElementById("chart-sim-shap"), {
    data: result.shap.map((s) => ({ label: s.label, value: s.shap_value })),
    diverging: true, valueFormatter: (v) => v.toFixed(3),
    legend: [{ label: "Increases risk", color: Charts.cssVar("--series-2-orange") }, { label: "Decreases risk", color: Charts.cssVar("--series-1-blue") }],
  });
}

// ---------------------------------------------------------------------
// detail drawer
// ---------------------------------------------------------------------
function fieldItem(label, value) {
  return `<div class="field-item"><div class="field-label">${label}</div><div class="field-value">${value}</div></div>`;
}
function detectorChip(key, flagged) {
  return `<span class="detector-chip ${flagged ? "flagged" : "clear"}">${flagged ? Icons.serious : Icons.good}&nbsp;${DETECTOR_LABELS[key]}</span>`;
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
      ${riskBadge(d.risk.risk_tier_code, d.risk.verdict_code)}
      ${verdictBadge(d.risk.verdict_code)}
    </div>
    <div class="field-grid">
      ${fieldItem("Amount", Fmt.money(r.amount))}
      ${fieldItem("Date", Fmt.dateTime(r.date))}
      ${fieldItem("Type", Fmt.escapeHtml(r.txn_type))}
      ${fieldItem("Channel", Fmt.escapeHtml(r.channel))}
      ${fieldItem("Location", Fmt.escapeHtml(r.location))}
      ${fieldItem("Device ID", Fmt.escapeHtml(r.device_id))}
      ${fieldItem("IP address", Fmt.escapeHtml(r.ip_address))}
      ${fieldItem("Merchant ID", Fmt.escapeHtml(r.merchant_id))}
      ${fieldItem("Customer age", r.customer_age)}
      ${fieldItem("Occupation", Fmt.escapeHtml(r.customer_occupation))}
      ${fieldItem("Duration", `${r.duration_seconds}s`)}
      ${fieldItem("Login attempts", r.login_attempts)}
      ${fieldItem("Account balance", Fmt.money(r.account_balance))}
      ${fieldItem("Risk score", Fmt.score(d.risk.risk_score))}
      ${fieldItem("Detector votes", `${d.risk.vote_count} / 4`)}
      ${fieldItem("Queue status", queueStatusBadge(d.queue_action))}
    </div>
    <div class="drawer-section-title">Flagged by</div>
    <div class="detector-chips">
      ${detectorChip("isoforest", d.detectors.isoforest)}
      ${detectorChip("lof", d.detectors.lof)}
      ${detectorChip("ocsvm", d.detectors.ocsvm)}
      ${detectorChip("mcd", d.detectors.mcd)}
    </div>
    <div class="drawer-section-title">Top contributing features (SHAP)</div>
    <div id="drawer-shap-chart"></div>
  `;
  Charts.renderHBarChart(document.getElementById("drawer-shap-chart"), {
    data: d.shap.slice(0, 8).map((s) => ({ label: s.label, value: s.shap_value })),
    diverging: true, valueFormatter: (v) => v.toFixed(3),
    legend: [{ label: "Increases risk", color: Charts.cssVar("--series-2-orange") }, { label: "Decreases risk", color: Charts.cssVar("--series-1-blue") }],
  });
}

function wireDrawer() {
  document.getElementById("drawer-backdrop").addEventListener("click", closeDrawer);
  document.getElementById("drawer-close-btn").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
}

// ---------------------------------------------------------------------
// navigation / theme / bootstrap
// ---------------------------------------------------------------------
function populateNav() {
  document.querySelectorAll(".nav-item").forEach((btn) => {
    const meta = NAV_META[btn.dataset.page];
    btn.innerHTML = `${Icons[meta.icon]}<span>${meta.label}</span>`;
    btn.addEventListener("click", () => showPage(btn.dataset.page));
  });
  document.getElementById("explorer-search-icon").innerHTML = Icons.search;
  document.getElementById("simulator-flask-icon").innerHTML = Icons.flask;
  document.getElementById("drawer-close-btn").innerHTML = Icons.close;
  document.getElementById("explorer-prev").innerHTML = Icons.chevronLeft;
  document.getElementById("explorer-next").innerHTML = Icons.chevronRight;
  document.getElementById("queue-prev").innerHTML = Icons.chevronLeft;
  document.getElementById("queue-next").innerHTML = Icons.chevronRight;
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
  else if (page === "simulator") loadSimulatorOptions();
}

function rerenderCurrentPageCharts() {
  if (currentPage === "overview" && overviewData) renderOverviewCharts(overviewData);
  else if (currentPage === "comparison" && comparisonData) renderComparison(comparisonData);
  else if (currentPage === "explainability" && explainabilityData) renderExplainability(explainabilityData);
  else if (currentPage === "simulator" && lastSimResult) renderSimShap(lastSimResult);
}

function applyTheme(theme) {
  document.documentElement.classList.toggle("light", theme === "light");
  document.getElementById("theme-icon").innerHTML = theme === "light" ? Icons.moon : Icons.sun;
  document.getElementById("theme-label").textContent = theme === "light" ? "Dark" : "Light";
  localStorage.setItem("argus-theme", theme);
  rerenderCurrentPageCharts();
}

function wireThemeToggle() {
  let theme = localStorage.getItem("argus-theme") || "dark";
  applyTheme(theme);
  document.getElementById("theme-toggle").addEventListener("click", () => {
    theme = theme === "dark" ? "light" : "dark";
    applyTheme(theme);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  populateNav();
  wireThemeToggle();
  wireExplorerControls();
  wireQueueControls();
  wireSimulatorForm();
  wireDrawer();
  showPage("overview");
  window.addEventListener("resize", Fmt.debounce(rerenderCurrentPageCharts, 200));
});
