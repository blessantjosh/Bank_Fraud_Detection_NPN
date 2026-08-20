/* Thin fetch wrapper around the Bank Transaction Fraud & Anomaly Detection FastAPI backend. Same-origin -- the
 * backend serves this frontend directly via StaticFiles, so no base URL or
 * CORS configuration is needed. */
const Api = (() => {
  async function get(path, params = {}) {
    const url = new URL(path, window.location.origin);
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, v);
    });
    const res = await fetch(url);
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed: ${res.status}`);
    }
    return res.json();
  }

  async function post(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const b = await res.json().catch(() => ({}));
      throw new Error(b.detail || `Request failed: ${res.status}`);
    }
    return res.json();
  }

  async function postForm(path, formData) {
    const res = await fetch(path, { method: "POST", body: formData });
    if (!res.ok) {
      const b = await res.json().catch(() => ({}));
      throw new Error(b.detail || `Request failed: ${res.status}`);
    }
    return res.json();
  }

  return {
    get,
    post,
    kpis: () => get("/api/kpis"),
    transactions: (params) => get("/api/transactions", params),
    transaction: (id) => get(`/api/transactions/${encodeURIComponent(id)}`),
    queue: (params) => get("/api/queue", params),
    queueAction: (transactionId, action) => post("/api/queue/action", { transaction_id: transactionId, action }),
    modelComparison: () => get("/api/model-comparison"),
    explainability: () => get("/api/explainability"),
    simulatorOptions: () => get("/api/simulator/options"),
    simulatorAccount: (id) => get(`/api/simulator/account/${encodeURIComponent(id)}`),
    score: (payload) => post("/api/score", payload),
    meta: () => get("/api/meta"),
    uploadPredict: (file) => {
      const formData = new FormData();
      formData.append("file", file);
      return postForm("/api/upload/predict", formData);
    },
    queueExportUrl: () => "/api/queue/export",
    uploadHistory: () => get("/api/upload/history"),
  };
})();
