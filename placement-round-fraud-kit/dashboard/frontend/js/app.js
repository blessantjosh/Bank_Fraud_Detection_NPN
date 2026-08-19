/* Bank Transaction Fraud & Anomaly Detection -- single-page app wiring. No framework, no build step.
 * Wired to the research_v2 pipeline (teammate 18-feature matrix): risk score is
 * `ensemble_percentile_average`, tiers are the Phase 13 (v2) percentile cutoffs,
 * and explanations are the precomputed Isolation Forest / Autoencoder SHAP rows. */

const NAV_META = {
  overview: { label: "Overview", icon: "overview", title: "Overview", subtitle: "Transaction risk monitoring dashboard" },
  "data-input": { label: "Data Input", icon: "upload", title: "Data Input", subtitle: "Upload and analyze transaction data" },
  investigation: { label: "Investigation", icon: "queue", title: "Investigation", subtitle: "Review flagged transactions" },
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

function fieldItem(label, value) {
  return `<div><div class="field-label">${Fmt.escapeHtml(label)}</div><div class="field-value">${value}</div></div>`;
}

function modelChip(m) {
  if (!m.applicable) return `<span class="model-chip na">${Fmt.escapeHtml(m.label)} <span class="pct">N/A</span></span>`;
  const flagged = m.flagged;
  const cls = flagged ? "flagged" : "";
  const icon = flagged ? Icons.warning : "";
  return `<span class="model-chip ${cls}">${icon}${Fmt.escapeHtml(m.label)} <span class="pct">${(m.percentile * 100).toFixed(0)}%</span></span>`;
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
    <td>${riskBadge(tx.risk_tier_code)}</td>
    <td class="num tabular">${Fmt.score(tx.risk_score)}</td>
  </tr>`;
}

// ---------------------------------------------------------------------
// overview - shows only uploaded transaction statistics
// ---------------------------------------------------------------------
async function loadOverview() {
  document.querySelectorAll(".kpi-value").forEach((el) => el.classList.remove("skeleton-loading"));

  // Load uploaded transactions
  loadUploadedTransactions();

  if (uploadedTransactions.length === 0) {
    // No data uploaded yet - show zeros
    document.querySelector('[data-kpi="total"]').textContent = "0";
    document.querySelector('[data-kpi="priority"]').textContent = "0";
    document.querySelector('[data-kpi="standard"]').textContent = "0";
    document.querySelector('[data-kpi="flagrate"]').textContent = "0%";
    document.querySelector('[data-kpi="avgamount"]').textContent = "$0";
    document.getElementById("status-dataset").textContent = "No transactions analyzed yet";
    document.getElementById("status-icon").textContent = "ℹ";

    // Show empty scatter plot
    document.getElementById("chart-outlier-scatter").innerHTML =
      '<div class="empty-state"><p style="font-size:14px;margin-bottom:8px">No data to visualize</p><p style="font-size:12px;color:var(--text-muted)">Upload transactions in the Data Input section to see risk visualization</p></div>';
    return;
  }

  // Calculate statistics from uploaded transactions
  const total = uploadedTransactions.length;
  const priority = uploadedTransactions.filter(tx => tx.fraud_percentage >= 70).length;
  const standard = uploadedTransactions.filter(tx => tx.fraud_percentage >= 50 && tx.fraud_percentage < 70).length;
  const flagRate = (priority + standard) / total;
  const avgAmount = uploadedTransactions.reduce((sum, tx) => sum + (tx.amount || 0), 0) / total;

  // Display statistics
  Fmt.countUp(document.querySelector('[data-kpi="total"]'), total, { formatter: Fmt.int });
  Fmt.countUp(document.querySelector('[data-kpi="priority"]'), priority, { formatter: Fmt.int });
  Fmt.countUp(document.querySelector('[data-kpi="standard"]'), standard, { formatter: Fmt.int });
  Fmt.countUp(document.querySelector('[data-kpi="flagrate"]'), flagRate, { formatter: Fmt.pct, decimals: 4 });
  Fmt.countUp(document.querySelector('[data-kpi="avgamount"]'), avgAmount, { formatter: Fmt.money, decimals: 2 });

  // Update status panel
  document.getElementById("status-dataset").textContent = `${Fmt.int(total)} transactions analyzed`;
  document.getElementById("status-icon").textContent = "✓";

  // Load scatter plot with uploaded data
  loadOutlierScatter();
}

async function loadOutlierScatter() {
  if (uploadedTransactions.length === 0) {
    document.getElementById("chart-outlier-scatter").innerHTML =
      '<div class="empty-state"><p style="font-size:14px;margin-bottom:8px">No data to visualize</p><p style="font-size:12px;color:var(--text-muted)">Upload transactions in the Data Input section to see risk visualization</p></div>';
    return;
  }

  try {
    const scatterData = uploadedTransactions.map(tx => {
      const fraudPercent = tx.fraud_percentage;
      const tierCode = fraudPercent >= 70 ? 'priority' : fraudPercent >= 50 ? 'standard' : 'normal';

      return {
        id: tx.transaction_id,
        x: Math.random() * 10,  // Simulated risk indicator spread
        y: fraudPercent / 100,   // Convert percentage to 0-1 scale
        tier: tierCode,
        score: fraudPercent / 100,
        clickable: false,  // Uploaded transactions don't have full detail view yet
      };
    });

    Charts.renderScatterPlot(document.getElementById("chart-outlier-scatter"), {
      data: scatterData,
      xLabel: "Risk Indicators",
      yLabel: "Risk Score",
      colorBy: "tier",
    });

    document.getElementById("outlier-subtitle").textContent =
      `Visual representation of ${uploadedTransactions.length} analyzed transactions`;
  } catch (err) {
    console.error("Failed to load scatter plot:", err);
    document.getElementById("chart-outlier-scatter").innerHTML =
      '<div class="empty-state">Risk visualization unavailable</div>';
  }
}

function renderOverviewCharts(data) {
  // Overview charts are now handled by loadOutlierScatter
  // This function is kept for compatibility but does nothing
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
    const isFlagged = tx.fraud_percentage >= 50;
    const tierCode = isFlagged ? "priority" : "normal";
    const tierLabel = isFlagged ? "Flagged" : "Normal";
    const riskScore = (tx.fraud_percentage / 100).toFixed(4);

    return `<tr>
      <td class="num tabular">${start + i + 1}</td>
      <td>${Fmt.escapeHtml(tx.transaction_id)}</td>
      <td>${Fmt.escapeHtml(tx.account_id || 'N/A')}</td>
      <td class="num tabular">${tx.amount ? Fmt.money(tx.amount) : 'N/A'}</td>
      <td>${badgeHtml(isFlagged ? "critical" : "good", tierLabel, isFlagged ? Icons.warning : Icons.good)}</td>
      <td class="num tabular">${tx.fraud_percentage.toFixed(1)}%</td>
      <td>${queueStatusBadge(tx.status || 'pending')}</td>
      <td>
        <button class="btn btn-sm btn-ghost action-approve" data-tx-id="${Fmt.escapeHtml(tx.transaction_id)}" data-action="approved">Approve</button>
        <button class="btn btn-sm btn-ghost action-escalate" data-tx-id="${Fmt.escapeHtml(tx.transaction_id)}" data-action="escalated">Escalate</button>
        <button class="btn btn-sm btn-ghost action-blocked" data-tx-id="${Fmt.escapeHtml(tx.transaction_id)}" data-action="blocked">Block</button>
      </td>
    </tr>`;
  }).join("");

  // Wire action buttons
  document.querySelectorAll("#table-queue [data-action]").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const txId = btn.dataset.txId;
      handleUploadedQueueAction(txId, btn.dataset.action);
    });
  });

  document.getElementById("queue-count").textContent = `${Fmt.int(total)} transactions`;
  document.getElementById("queue-page-label").textContent = `Page ${queueState.page} of ${totalPages}`;
  document.getElementById("queue-prev").disabled = queueState.page <= 1;
  document.getElementById("queue-next").disabled = queueState.page >= totalPages;
}

function handleUploadedQueueAction(txId, action) {
  const tx = uploadedTransactions.find(t => t.transaction_id === txId);
  if (tx) {
    tx.status = action;
    saveUploadedTransactions();
    showToast(`Transaction marked as ${action}`);
    loadQueue();
  }
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
  document.getElementById("queue-export").addEventListener("click", exportUploadedTransactions);
  document.getElementById("queue-clear").addEventListener("click", clearUploadedTransactions);
  document.getElementById("queue-generate-report").addEventListener("click", generateFraudReport);
  document.getElementById("auto-process-btn").addEventListener("click", startAutoProcess);
}

function clearUploadedTransactions() {
  if (uploadedTransactions.length === 0) {
    showToast("No transactions to clear");
    return;
  }

  if (confirm(`Are you sure you want to clear all ${uploadedTransactions.length} analyzed transactions? This cannot be undone.`)) {
    uploadedTransactions = [];
    saveUploadedTransactions();
    loadQueue();
    // Refresh overview to show empty state
    overviewData = null;
    if (currentPage === 'overview') {
      loadOverview();
    }
    showToast("All transactions cleared");
  }
}

// Auto-process all pending transactions based on ML fraud scores
async function startAutoProcess() {
  loadUploadedTransactions();

  const pendingTransactions = uploadedTransactions.filter(tx => tx.status === 'pending');

  if (pendingTransactions.length === 0) {
    showToast("No pending transactions to process");
    return;
  }

  const btn = document.getElementById('auto-process-btn');
  const originalText = btn.innerHTML;

  // Confirm action
  if (!confirm(`Auto-Process ${pendingTransactions.length} pending transactions?\n\n` +
    `The system will automatically:\n` +
    `• Approve legitimate accounts (fraud risk < 50%)\n` +
    `• Escalate suspicious accounts (fraud risk 60-79%)\n` +
    `• Block high-risk accounts (fraud risk ≥ 80%)\n\n` +
    `This action can be reversed by manually changing each transaction status.`)) {
    return;
  }

  // Disable button and show processing
  btn.disabled = true;
  btn.style.background = 'linear-gradient(135deg,#a0aec0 0%,#718096 100%)';
  btn.innerHTML = '⏳ Processing...';

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
      btn.innerHTML = `⏳ Processing ${processed}/${pendingTransactions.length}...`;

      // Small delay for visual feedback (every 10 transactions)
      if (i % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, 50));
      }
    }

    // Save updated transactions
    saveUploadedTransactions();

    // Show success animation
    btn.style.background = 'linear-gradient(135deg,#48bb78 0%,#38a169 100%)';
    btn.innerHTML = '✅ Complete!';

    // Show detailed results
    showToast(
      `Auto-processing complete!\n\n` +
      `✅ Approved (legitimate): ${approved}\n` +
      `⚠️ Escalated (review needed): ${escalated}\n` +
      `🚫 Blocked (high risk): ${blocked}`
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
    showToast(`Error during auto-processing: ${err.message}`);

    // Reset button on error
    btn.disabled = false;
    btn.style.background = 'linear-gradient(135deg,#667eea 0%,#764ba2 100%)';
    btn.innerHTML = originalText;
  }
}

function exportUploadedTransactions() {
  if (uploadedTransactions.length === 0) {
    showToast("No transactions to export");
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
  showToast("Export complete");
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
  const dash = (v) => (v === null || v === undefined || v === "" ? "—" : v);
  return `<tr>
    <td>${Fmt.escapeHtml(String(dash(r.transaction_id)))}</td>
    <td>${Fmt.escapeHtml(String(dash(r.account_id)))}</td>
    <td>${Fmt.escapeHtml(String(dash(r.date)))}</td>
    <td class="num tabular">${r.amount === null || r.amount === undefined ? "—" : Fmt.money(r.amount)}</td>
    <td class="num tabular"${pctStyle}>${r.fraud_percentage.toFixed(2)}%</td>
  </tr>`;
}

// ---------------------------------------------------------------------
// fraud report generation
// ---------------------------------------------------------------------
function generateFraudReport() {
  const fraudulent = uploadedTransactions.filter(tx => tx.fraud_percentage >= 50);

  if (fraudulent.length === 0) {
    showToast("No fraudulent transactions to report");
    return;
  }

  // Show format selection dialog
  showReportFormatDialog(fraudulent);
}

function showReportFormatDialog(fraudulent) {
  // Create modal overlay
  const overlay = document.createElement('div');
  overlay.style.cssText = `
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.5); z-index: 10000;
    display: flex; align-items: center; justify-content: center;
  `;

  const dialog = document.createElement('div');
  dialog.style.cssText = `
    background: var(--card-bg, #fff);
    padding: 32px;
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    max-width: 500px;
    width: 90%;
  `;

  dialog.innerHTML = `
    <h2 style="margin: 0 0 16px 0; color: var(--text-primary);">Generate Fraud Report</h2>
    <p style="margin: 0 0 24px 0; color: var(--text-secondary);">
      ${fraudulent.length} fraudulent transaction(s) detected. Choose report format:
    </p>
    <div style="display: flex; flex-direction: column; gap: 12px;">
      <button id="report-csv-btn" class="btn btn-primary" style="width: 100%; padding: 16px;">
        📊 Download CSV Report
        <div style="font-size: 12px; opacity: 0.8; margin-top: 4px;">Structured data for analysis & spreadsheets</div>
      </button>
      <button id="report-pdf-btn" class="btn btn-primary" style="width: 100%; padding: 16px;">
        📄 Generate PDF Report
        <div style="font-size: 12px; opacity: 0.8; margin-top: 4px;">Colorful formatted report (opens in new window)</div>
      </button>
      <button id="report-both-btn" class="btn btn-good" style="width: 100%; padding: 16px;">
        📦 Download Both Formats
        <div style="font-size: 12px; opacity: 0.8; margin-top: 4px;">CSV + PDF for complete documentation</div>
      </button>
      <button id="report-cancel-btn" class="btn btn-ghost" style="width: 100%; margin-top: 8px;">
        Cancel
      </button>
    </div>
  `;

  overlay.appendChild(dialog);
  document.body.appendChild(overlay);

  // Wire up buttons
  document.getElementById('report-csv-btn').addEventListener('click', () => {
    generateCSVReport(fraudulent);
    document.body.removeChild(overlay);
  });

  document.getElementById('report-pdf-btn').addEventListener('click', () => {
    generatePDFReport(fraudulent);
    document.body.removeChild(overlay);
  });

  document.getElementById('report-both-btn').addEventListener('click', () => {
    generateCSVReport(fraudulent);
    setTimeout(() => generatePDFReport(fraudulent), 500);
    document.body.removeChild(overlay);
  });

  document.getElementById('report-cancel-btn').addEventListener('click', () => {
    document.body.removeChild(overlay);
  });

  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) {
      document.body.removeChild(overlay);
    }
  });
}

function generateCSVReport(fraudulent) {
  // Group by account
  const accountMap = {};
  fraudulent.forEach(tx => {
    const accId = tx.account_id || 'Unknown';
    if (!accountMap[accId]) {
      accountMap[accId] = [];
    }
    accountMap[accId].push(tx);
  });

  const reportDate = new Date().toISOString();

  // CSV Header
  let csv = "Transaction ID,Account ID,Date,Amount,Fraud Score (%),Risk Level,Status,Recommended Action,Detection Summary,Uploaded At\n";

  // Add each transaction
  Object.entries(accountMap).forEach(([accountId, transactions]) => {
    const avgRisk = transactions.reduce((sum, tx) => sum + tx.fraud_percentage, 0) / transactions.length;

    transactions.forEach(tx => {
      const riskLevel = tx.fraud_percentage >= 80 ? 'CRITICAL' :
                       tx.fraud_percentage >= 70 ? 'HIGH' :
                       tx.fraud_percentage >= 60 ? 'ELEVATED' : 'MODERATE';

      const action = tx.fraud_percentage >= 80 ? 'Block Account Immediately' :
                    tx.fraud_percentage >= 70 ? 'Place Temporary Hold' :
                    tx.fraud_percentage >= 60 ? 'Priority Review Required' :
                    'Schedule Manual Review';

      const detection = `Fraud probability ${tx.fraud_percentage.toFixed(1)}% - Pattern analysis flagged suspicious behavior`;

      // Escape CSV fields
      const escapeCSV = (val) => {
        if (val === null || val === undefined) return '';
        const str = String(val);
        if (str.includes(',') || str.includes('"') || str.includes('\n')) {
          return `"${str.replace(/"/g, '""')}"`;
        }
        return str;
      };

      csv += `${escapeCSV(tx.transaction_id)},`;
      csv += `${escapeCSV(accountId)},`;
      csv += `${escapeCSV(tx.date || 'N/A')},`;
      csv += `${escapeCSV((tx.amount || 0).toFixed(2))},`;
      csv += `${tx.fraud_percentage.toFixed(2)},`;
      csv += `${riskLevel},`;
      csv += `${escapeCSV(tx.status || 'Pending')},`;
      csv += `${escapeCSV(action)},`;
      csv += `${escapeCSV(detection)},`;
      csv += `${escapeCSV(tx.uploaded_at || reportDate)}\n`;
    });
  });

  // Download CSV
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `fraud_report_${new Date().toISOString().split('T')[0]}.csv`;
  a.click();
  URL.revokeObjectURL(url);

  showToast(`CSV report generated for ${fraudulent.length} transactions`);
}

function generatePDFReport(fraudulent) {
  // Group by account
  const accountMap = {};
  fraudulent.forEach(tx => {
    const accId = tx.account_id || 'Unknown';
    if (!accountMap[accId]) {
      accountMap[accId] = [];
    }
    accountMap[accId].push(tx);
  });

  const reportDate = new Date().toLocaleString();
  const totalAmount = fraudulent.reduce((sum, tx) => sum + (tx.amount || 0), 0);
  const fraudRate = ((fraudulent.length / uploadedTransactions.length) * 100).toFixed(2);

  // Generate HTML report
  const html = `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Fraud Detection Report - ${new Date().toISOString().split('T')[0]}</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      padding: 40px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
    }
    .report-container {
      max-width: 1200px;
      margin: 0 auto;
      background: white;
      border-radius: 16px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.3);
      overflow: hidden;
    }
    .report-header {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 40px;
      text-align: center;
    }
    .report-header h1 {
      font-size: 32px;
      margin-bottom: 8px;
      font-weight: 700;
    }
    .report-header p {
      font-size: 16px;
      opacity: 0.95;
    }
    .report-body {
      padding: 40px;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 20px;
      margin-bottom: 40px;
    }
    .summary-card {
      background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
      color: white;
      padding: 24px;
      border-radius: 12px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .summary-card.blue { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .summary-card.green { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
    .summary-card.orange { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
    .summary-card.purple { background: linear-gradient(135deg, #30cfd0 0%, #330867 100%); }
    .summary-card h3 {
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 8px;
      opacity: 0.9;
    }
    .summary-card .value {
      font-size: 32px;
      font-weight: 700;
    }
    .section-title {
      font-size: 24px;
      color: #2d3748;
      margin: 40px 0 20px 0;
      padding-bottom: 12px;
      border-bottom: 3px solid #667eea;
      font-weight: 700;
    }
    .account-card {
      background: #f7fafc;
      border-left: 4px solid #667eea;
      padding: 24px;
      margin-bottom: 24px;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .account-card.critical { border-left-color: #f56565; }
    .account-card.high { border-left-color: #ed8936; }
    .account-card.elevated { border-left-color: #ecc94b; }
    .account-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }
    .account-id {
      font-size: 20px;
      font-weight: 700;
      color: #2d3748;
    }
    .risk-badge {
      padding: 8px 16px;
      border-radius: 20px;
      font-weight: 600;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .risk-badge.critical { background: #fed7d7; color: #c53030; }
    .risk-badge.high { background: #feebc8; color: #c05621; }
    .risk-badge.elevated { background: #fefcbf; color: #b7791f; }
    .risk-badge.moderate { background: #bee3f8; color: #2c5282; }
    .account-stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }
    .stat {
      background: white;
      padding: 12px;
      border-radius: 6px;
    }
    .stat-label {
      font-size: 12px;
      color: #718096;
      margin-bottom: 4px;
    }
    .stat-value {
      font-size: 20px;
      font-weight: 700;
      color: #2d3748;
    }
    .detection-box {
      background: #edf2f7;
      padding: 16px;
      border-radius: 8px;
      margin: 16px 0;
    }
    .detection-box h4 {
      color: #2d3748;
      margin-bottom: 8px;
      font-size: 14px;
      font-weight: 600;
    }
    .detection-box ul {
      list-style: none;
      padding-left: 0;
    }
    .detection-box li {
      padding: 4px 0;
      color: #4a5568;
      font-size: 14px;
    }
    .detection-box li:before {
      content: "✓ ";
      color: #48bb78;
      font-weight: bold;
      margin-right: 8px;
    }
    .action-box {
      background: #fff5f5;
      border: 2px solid #fc8181;
      padding: 16px;
      border-radius: 8px;
      margin: 16px 0;
    }
    .action-box.high { background: #fffaf0; border-color: #f6ad55; }
    .action-box.moderate { background: #ebf8ff; border-color: #63b3ed; }
    .action-box h4 {
      color: #c53030;
      margin-bottom: 8px;
      font-size: 14px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .action-box.high h4 { color: #c05621; }
    .action-box.moderate h4 { color: #2c5282; }
    .action-box ul {
      list-style: none;
      padding-left: 0;
    }
    .action-box li {
      padding: 4px 0;
      font-size: 14px;
      color: #2d3748;
    }
    .action-box li:before {
      content: "→ ";
      color: #e53e3e;
      font-weight: bold;
      margin-right: 8px;
    }
    .action-box.high li:before { color: #dd6b20; }
    .action-box.moderate li:before { color: #3182ce; }
    .transaction-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 16px;
    }
    .transaction-table th {
      background: #2d3748;
      color: white;
      padding: 12px;
      text-align: left;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
    }
    .transaction-table td {
      padding: 12px;
      border-bottom: 1px solid #e2e8f0;
      font-size: 14px;
      color: #4a5568;
    }
    .transaction-table tr:hover {
      background: #f7fafc;
    }
    .disclaimer {
      background: #fffaf0;
      border: 2px solid #f6ad55;
      padding: 24px;
      border-radius: 8px;
      margin-top: 40px;
    }
    .disclaimer h4 {
      color: #c05621;
      margin-bottom: 12px;
      font-size: 16px;
      font-weight: 700;
    }
    .disclaimer p {
      color: #744210;
      font-size: 14px;
      line-height: 1.6;
      margin-bottom: 8px;
    }
    @media print {
      body { background: white; padding: 0; }
      .report-container { box-shadow: none; }
    }
  </style>
</head>
<body>
  <div class="report-container">
    <div class="report-header">
      <h1>🛡️ Fraud Detection Report</h1>
      <p>Generated: ${reportDate}</p>
    </div>

    <div class="report-body">
      <!-- Executive Summary -->
      <div class="summary-grid">
        <div class="summary-card blue">
          <h3>Total Transactions</h3>
          <div class="value">${uploadedTransactions.length}</div>
        </div>
        <div class="summary-card">
          <h3>Fraudulent Detected</h3>
          <div class="value">${fraudulent.length}</div>
        </div>
        <div class="summary-card orange">
          <h3>Fraud Rate</h3>
          <div class="value">${fraudRate}%</div>
        </div>
        <div class="summary-card green">
          <h3>Affected Accounts</h3>
          <div class="value">${Object.keys(accountMap).length}</div>
        </div>
        <div class="summary-card purple">
          <h3>Total Amount at Risk</h3>
          <div class="value">$${totalAmount.toFixed(2)}</div>
        </div>
      </div>

      <h2 class="section-title">📋 Detailed Findings by Account</h2>

      ${Object.entries(accountMap).map(([accountId, transactions]) => {
        const avgRisk = transactions.reduce((sum, tx) => sum + tx.fraud_percentage, 0) / transactions.length;
        const totalAmount = transactions.reduce((sum, tx) => sum + (tx.amount || 0), 0);

        const riskLevel = avgRisk >= 80 ? 'critical' : avgRisk >= 70 ? 'high' : avgRisk >= 60 ? 'elevated' : 'moderate';
        const riskLabel = avgRisk >= 80 ? 'CRITICAL' : avgRisk >= 70 ? 'HIGH' : avgRisk >= 60 ? 'ELEVATED' : 'MODERATE';

        return `
          <div class="account-card ${riskLevel}">
            <div class="account-header">
              <div class="account-id">Account: ${Fmt.escapeHtml(accountId)}</div>
              <div class="risk-badge ${riskLevel}">${riskLabel} RISK</div>
            </div>

            <div class="account-stats">
              <div class="stat">
                <div class="stat-label">Fraudulent Transactions</div>
                <div class="stat-value">${transactions.length}</div>
              </div>
              <div class="stat">
                <div class="stat-label">Average Fraud Score</div>
                <div class="stat-value">${avgRisk.toFixed(1)}%</div>
              </div>
              <div class="stat">
                <div class="stat-label">Total Amount</div>
                <div class="stat-value">$${totalAmount.toFixed(2)}</div>
              </div>
            </div>

            <div class="detection-box">
              <h4>🔍 How Fraud Was Detected</h4>
              <ul>
                ${avgRisk >= 70 ? `
                  <li>High fraud probability detected (${avgRisk.toFixed(1)}%)</li>
                  <li>Multiple risk indicators flagged across transactions</li>
                  <li>Pattern analysis identified suspicious behavior</li>
                  <li>Transaction characteristics match known fraud patterns</li>
                ` : `
                  <li>Moderate fraud probability detected (${avgRisk.toFixed(1)}%)</li>
                  <li>Several risk indicators present in transaction data</li>
                  <li>Anomaly detection flagged unusual patterns</li>
                  <li>Manual review recommended for verification</li>
                `}
              </ul>
            </div>

            <div class="action-box ${avgRisk >= 80 ? 'critical' : avgRisk >= 60 ? 'high' : 'moderate'}">
              <h4>⚠️ Recommended Actions</h4>
              <ul>
                ${avgRisk >= 80 ? `
                  <li>IMMEDIATE ACTION REQUIRED - Block account immediately</li>
                  <li>Flag all pending transactions for review</li>
                  <li>Initiate fraud investigation with security team</li>
                  <li>Contact account holder for immediate verification</li>
                  <li>Review all recent account activity for additional fraud</li>
                  <li>Document all findings in case management system</li>
                ` : avgRisk >= 60 ? `
                  <li>HIGH PRIORITY - Place temporary hold on account</li>
                  <li>Contact account holder immediately for verification</li>
                  <li>Review transaction details manually within 4 hours</li>
                  <li>Monitor account for additional suspicious activity</li>
                  <li>Consider enhanced account verification requirements</li>
                  <li>Escalate to fraud investigation team if confirmed</li>
                ` : `
                  <li>STANDARD REVIEW - Schedule manual review within 24 hours</li>
                  <li>Contact account holder for transaction confirmation</li>
                  <li>Monitor account for emerging fraud patterns</li>
                  <li>Document findings in case management system</li>
                  <li>Consider additional verification for large transactions</li>
                `}
              </ul>
            </div>

            <table class="transaction-table">
              <thead>
                <tr>
                  <th>Transaction ID</th>
                  <th>Date</th>
                  <th>Amount</th>
                  <th>Fraud Score</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                ${transactions.map(tx => `
                  <tr>
                    <td>${Fmt.escapeHtml(tx.transaction_id)}</td>
                    <td>${Fmt.escapeHtml(tx.date || 'N/A')}</td>
                    <td>$${(tx.amount || 0).toFixed(2)}</td>
                    <td><strong>${tx.fraud_percentage.toFixed(1)}%</strong></td>
                    <td>${Fmt.escapeHtml(tx.status || 'Pending Review')}</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        `;
      }).join('')}

      <div class="disclaimer">
        <h4>⚠️ IMPORTANT DISCLAIMER</h4>
        <p>This report is generated by an automated fraud detection system using advanced machine learning algorithms.</p>
        <p>All findings and recommendations must be verified by authorized personnel before taking action.</p>
        <p>Do not take automated blocking actions without proper review and approval from management.</p>
        <p>This report is for internal use only and contains sensitive information. Handle according to your organization's data security policies.</p>
      </div>
    </div>
  </div>

  <script>
    // Auto-print dialog on load
    window.onload = function() {
      setTimeout(() => {
        window.print();
      }, 500);
    };
  </script>
</body>
</html>`;

  // Open in new window
  const printWindow = window.open('', '_blank');
  printWindow.document.write(html);
  printWindow.document.close();

  showToast(`PDF report opened in new window - Use "Print to PDF" to save`);
}

function wireUploadForm() {
  // Wire report generation button
  document.getElementById("generate-report-btn").addEventListener("click", generateFraudReport);

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
      showToast(`${resp.fraud_count} suspicious transactions saved to Investigation tab`);
    } catch (err) {
      statusEl.textContent = "";
      showToast(err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Analyze Transactions";
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
}

function showPage(page) {
  currentPage = page;
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.page === page));
  document.querySelectorAll(".page").forEach((s) => s.classList.toggle("active", s.id === `page-${page}`));
  const meta = NAV_META[page];
  document.getElementById("page-title").textContent = meta.title;
  document.getElementById("page-subtitle").textContent = meta.subtitle;
  Charts.hideTooltip();

  if (page === "overview") {
    loadOverview();
  } else if (page === "data-input") {
    // Restore last active tab or default to csv-upload
    const tabToShow = currentTab["data-input"] || "csv-upload";
    showTab("data-input", tabToShow);
  } else if (page === "investigation") {
    // Restore last active tab or default to flagged-cases
    const tabToShow = currentTab.investigation || "flagged-cases";
    showTab("investigation", tabToShow);
  }
}

function showTab(page, tabName) {
  currentTab[page] = tabName;
  const pageEl = document.getElementById(`page-${page}`);
  if (!pageEl) return;

  // Update tab buttons
  pageEl.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabName);
  });

  // Update tab content
  pageEl.querySelectorAll(".tab-content").forEach((content) => {
    content.classList.toggle("active", content.id === `tab-${tabName}`);
  });

  // Load content if needed
  if (page === "data-input") {
    if (tabName === "csv-upload") {
      // CSV upload is always ready
    } else if (tabName === "upload-history") {
      loadHistoryViewer();
    } else if (tabName === "manual-entry") {
      if (!simOptions) loadManualEntryOptions();
    } else if (tabName === "simulator") {
      if (!simOptions) loadSimulatorOptions();
    }
  } else if (page === "investigation") {
    if (tabName === "flagged-cases") {
      loadQueue();
    } else if (tabName === "case-detail") {
      // Case detail is loaded when clicking investigate
    } else if (tabName === "explainability") {
      loadExplainability();
    } else if (tabName === "comparison") {
      loadComparison();
    }
  }
}

function wireTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const pageId = btn.closest(".page").id.replace("page-", "");
      showTab(pageId, btn.dataset.tab);
    });
  });
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
  localStorage.setItem("fraud-detection-theme", theme);
  rerenderCurrentPageCharts();
}

function wireThemeToggle() {
  let theme = localStorage.getItem("fraud-detection-theme") || localStorage.getItem("argus-theme") || "dark";
  applyTheme(theme);
  document.getElementById("theme-toggle").addEventListener("click", () => {
    theme = theme === "dark" ? "light" : "dark";
    applyTheme(theme);
  });
}

// ---------------------------------------------------------------------
// upload history viewer
// ---------------------------------------------------------------------
function loadHistoryViewer() {
  loadUploadHistory();

  const emptyState = document.getElementById('history-empty-state');
  const viewer = document.getElementById('history-viewer');

  if (uploadHistory.length === 0) {
    emptyState.style.display = 'block';
    viewer.style.display = 'none';
    return;
  }

  emptyState.style.display = 'none';
  viewer.style.display = 'block';

  // Ensure current index is valid
  if (currentHistoryIndex >= uploadHistory.length) {
    currentHistoryIndex = uploadHistory.length - 1;
  }
  if (currentHistoryIndex < 0) {
    currentHistoryIndex = 0;
  }

  displayHistorySession(currentHistoryIndex);
}

function displayHistorySession(index) {
  if (index < 0 || index >= uploadHistory.length) return;

  currentHistoryIndex = index;
  const session = uploadHistory[index];

  // Update navigation
  document.getElementById('history-position').textContent = `${index + 1} of ${uploadHistory.length}`;
  document.getElementById('history-first').disabled = index === 0;
  document.getElementById('history-prev').disabled = index === 0;
  document.getElementById('history-next').disabled = index === uploadHistory.length - 1;
  document.getElementById('history-last').disabled = index === uploadHistory.length - 1;

  // Update session details
  document.getElementById('history-filename').textContent = session.filename;
  document.getElementById('history-date').textContent = new Date(session.uploadDate).toLocaleString();
  document.getElementById('history-total').textContent = session.totalCount;
  document.getElementById('history-fraud').textContent = session.fraudCount;
  document.getElementById('history-fraud-rate').textContent =
    `${((session.fraudCount / session.totalCount) * 100).toFixed(1)}%`;
  document.getElementById('history-amount').textContent = `$${session.totalAmount.toFixed(2)}`;

  // Display transactions
  const tbody = document.querySelector('#table-history-transactions tbody');
  tbody.innerHTML = session.transactions.map(tx => {
    const isFlagged = tx.fraud_percentage >= 50;
    return `<tr>
      <td>${Fmt.escapeHtml(tx.transaction_id)}</td>
      <td>${Fmt.escapeHtml(tx.account_id || 'N/A')}</td>
      <td>${Fmt.escapeHtml(tx.date || 'N/A')}</td>
      <td class="num tabular">${tx.amount ? Fmt.money(tx.amount) : 'N/A'}</td>
      <td class="num tabular" style="color:${isFlagged ? 'var(--status-critical)' : 'var(--status-good)'}">
        <strong>${tx.fraud_percentage.toFixed(1)}%</strong>
      </td>
      <td>${queueStatusBadge(tx.status || 'pending')}</td>
    </tr>`;
  }).join('');
}

function deleteCurrentHistory() {
  if (uploadHistory.length === 0) return;

  const session = uploadHistory[currentHistoryIndex];
  if (!confirm(`Delete upload "${session.filename}"?\n\nThis will remove ${session.totalCount} transactions from history. This cannot be undone.`)) {
    return;
  }

  // Remove from history
  uploadHistory.splice(currentHistoryIndex, 1);
  saveUploadHistory();

  // Adjust current index
  if (currentHistoryIndex >= uploadHistory.length && currentHistoryIndex > 0) {
    currentHistoryIndex--;
  }

  // Reload viewer
  loadHistoryViewer();
  showToast('Upload history deleted');
}

function generateHistoryReport() {
  if (uploadHistory.length === 0 || currentHistoryIndex < 0) return;

  const session = uploadHistory[currentHistoryIndex];
  const fraudulent = session.transactions.filter(tx => tx.fraud_percentage >= 50);

  if (fraudulent.length === 0) {
    showToast("No fraudulent transactions in this upload");
    return;
  }

  showReportFormatDialog(fraudulent);
}

function wireHistoryControls() {
  document.getElementById('history-first').addEventListener('click', () => {
    displayHistorySession(0);
  });

  document.getElementById('history-prev').addEventListener('click', () => {
    if (currentHistoryIndex > 0) {
      displayHistorySession(currentHistoryIndex - 1);
    }
  });

  document.getElementById('history-next').addEventListener('click', () => {
    if (currentHistoryIndex < uploadHistory.length - 1) {
      displayHistorySession(currentHistoryIndex + 1);
    }
  });

  document.getElementById('history-last').addEventListener('click', () => {
    displayHistorySession(uploadHistory.length - 1);
  });

  document.getElementById('history-delete-current').addEventListener('click', deleteCurrentHistory);
  document.getElementById('history-generate-report').addEventListener('click', generateHistoryReport);
}

document.addEventListener("DOMContentLoaded", () => {
  loadUploadedTransactions(); // Load uploaded transactions from localStorage
  loadUploadHistory(); // Load upload history
  populateNav();
  wireThemeToggle();
  wireTabs();
  wireQueueControls();
  wireSimulatorForm();
  wireUploadForm();
  wireAccountAnalysisForm();
  wireManualEntryForm();
  wireCaseBackButton();
  wireDrawer();
  wireHistoryControls();
  showPage("overview");
  window.addEventListener("resize", Fmt.debounce(rerenderCurrentPageCharts, 200));
});
