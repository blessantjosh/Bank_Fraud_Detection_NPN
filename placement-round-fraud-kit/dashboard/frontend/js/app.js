/* Argus dashboard -- single-page app wiring. No framework, no build step.
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
  simulator: { label: "Scenario Simulator", icon: "simulator", title: "Account Scenario Simulator", subtitle: "Secondary tool -- vary one real account's transaction" },
};

let currentPage = "overview";
let overviewData = null;
let comparisonData = null;
let explainabilityData = null;
let lastSimResult = null;
let simOptions = null;

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
function riskBadge(tierCode) {
  if (tierCode === "priority") return badgeHtml("critical", "Priority review", Icons.critical);
  if (tierCode === "standard") return badgeHtml("warning", "Standard review", Icons.warning);
  if (tierCode === "normal") return badgeHtml("good", "Normal", Icons.good);
  return badgeHtml("neutral", "Unknown", "");
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
const num = (v, d = 3) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));

function txRowHtml(tx, includeType) {
  const typeCell = includeType ? `<td>${Fmt.escapeHtml(tx.txn_type)}</td>` : "";
  return `<tr data-id="${Fmt.escapeHtml(tx.transaction_id)}">
    <td>${Fmt.escapeHtml(tx.transaction_id)}</td>
    <td>${Fmt.escapeHtml(tx.account_id)}</td>
    <td>${Fmt.dateTime(tx.date)}</td>
    <td class="num tabular">${Fmt.money(tx.amount)}</td>
    <td>${Fmt.escapeHtml(tx.channel)}</td>
    ${typeCell}
    <td>${riskBadge(tx.risk_tier_code)}</td>
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
    Fmt.countUp(document.querySelector('[data-kpi="priority"]'), overviewData.priority_count, { formatter: Fmt.int });
    Fmt.countUp(document.querySelector('[data-kpi="standard"]'), overviewData.standard_count, { formatter: Fmt.int });
    Fmt.countUp(document.querySelector('[data-kpi="flagrate"]'), overviewData.flag_rate, { formatter: Fmt.pct, decimals: 4 });
    Fmt.countUp(document.querySelector('[data-kpi="avgamount"]'), overviewData.avg_amount, { formatter: Fmt.money, decimals: 2 });
    document.getElementById("tier-distribution-subtitle").textContent =
      `${Fmt.int(overviewData.total_transactions)} transactions — priority at ensemble score ≥ ${overviewData.priority_threshold} (99th pct), ` +
      `standard at ≥ ${overviewData.standard_threshold} (95th pct). No automatic block tier.`;
  }
  renderOverviewCharts(overviewData);
  const tbody = document.querySelector("#table-top-risk tbody");
  tbody.innerHTML = overviewData.top_risk.map((tx) => txRowHtml(tx, false)).join("") || emptyRow(7);
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
      <td>${riskBadge(tx.risk_tier_code)}</td>
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
  const tbody = document.querySelector("#table-models tbody");
  tbody.innerHTML = data.models.map((m) => `<tr>
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
    catch (e) { showToast(`Could not load explainability data: ${e.message}`); return; }
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
    fillDatalist("sim-account-list", simOptions.accounts);
    fillSelect("sim-location", simOptions.locations);
    fillSelect("sim-occupation", simOptions.occupations);
    fillSelect("sim-device", simOptions.devices);
    fillSelect("sim-ip", simOptions.ip_addresses);
    fillSelect("sim-merchant", simOptions.merchants);
    document.getElementById("simulator-note").textContent = simOptions.note;
    document.getElementById("simulator-banner-text").textContent =
      `Secondary tool — vary one real account's transaction and see how the score moves. ` +
      `The high-amount flag fires above $${simOptions.high_amount_threshold} (the dataset's 95th percentile, frozen).`;
  } catch (e) {
    showToast(`Could not load simulator reference data: ${e.message}`);
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
    } catch (e2) {
      showToast(`Could not score this scenario: ${e2.message}`);
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
  return `<tr>
    <td>${Fmt.escapeHtml(r.transaction_id)}</td>
    <td>${Fmt.escapeHtml(r.account_id)}</td>
    <td>${Fmt.escapeHtml(r.date)}</td>
    <td class="num tabular">${Fmt.money(r.amount)}</td>
    <td class="num tabular"${pctStyle}>${r.fraud_percentage.toFixed(2)}%</td>
  </tr>`;
}

function wireUploadForm() {
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
      statusEl.textContent = `Scored ${Fmt.int(resp.total)} transactions with ${Fmt.escapeHtml(resp.model)}.`;
      document.getElementById("upload-results-subtitle").textContent =
        `${Fmt.int(resp.total)} transactions — model: ${resp.model}. Rows at or above 70% are highlighted.`;
      document.querySelector("#table-upload-results tbody").innerHTML =
        resp.results.map(uploadRowHtml).join("") || emptyRow(5);
      resultsCard.style.display = "block";
    } catch (err) {
      statusEl.textContent = "";
      showToast(err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Predict";
    }
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
      <span class="badge badge-neutral">Rank ${d.risk.score_rank} of 2,512</span>
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
      ${fieldItem("Amount / balance", `${r.amount_to_balance_ratio}×`)}
      ${fieldItem("Ensemble score", Fmt.score(d.risk.risk_score))}
      ${fieldItem("Score percentile", `${(d.risk.score_percentile * 100).toFixed(1)}%`)}
      ${fieldItem("Models flagging", `${d.risk.models_flagged} / ${d.risk.models_applicable}`)}
      ${fieldItem("Queue status", queueStatusBadge(d.queue_action))}
    </div>
    <div class="drawer-section-title">Per-model position (percentile of that model's own score)</div>
    <div class="model-chip-grid">${d.models.map(modelChip).join("")}</div>
    <div class="shap-dual-title">Isolation Forest — SHAP (exact, precomputed)</div>
    <div id="drawer-shap-if"></div>
    <div class="shap-dual-title">Autoencoder — SHAP (reconstruction error, precomputed)</div>
    <div id="drawer-shap-ae"></div>
    <p class="card-subtitle" style="margin-top:10px;line-height:1.55">
      Both explanations are shown because the two models attribute their scores almost oppositely
      (ρ = −0.3705 across all 18 features). Reading only one gives an incomplete picture of why this
      transaction was flagged.
    </p>
  `;
  const legend = [
    { label: "Increases anomaly score", color: Charts.cssVar("--series-2-orange") },
    { label: "Decreases anomaly score", color: Charts.cssVar("--series-1-blue") },
  ];
  Charts.renderHBarChart(document.getElementById("drawer-shap-if"), {
    data: d.shap_isolation_forest.slice(0, 6).reverse().map((s) => ({ label: s.label, value: s.shap_value })),
    diverging: true, valueFormatter: (v) => v.toFixed(3), legend,
  });
  Charts.renderHBarChart(document.getElementById("drawer-shap-ae"), {
    data: d.shap_autoencoder.slice(0, 6).reverse().map((s) => ({ label: s.label, value: s.shap_value })),
    diverging: true, valueFormatter: (v) => v.toFixed(4), legend,
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
  else if (currentPage === "simulator" && lastSimResult) renderSimCharts(lastSimResult);
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
  wireUploadForm();
  wireDrawer();
  showPage("overview");
  window.addEventListener("resize", Fmt.debounce(rerenderCurrentPageCharts, 200));
});
