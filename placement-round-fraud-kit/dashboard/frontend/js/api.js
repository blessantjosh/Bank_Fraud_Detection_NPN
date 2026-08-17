/* Thin fetch wrapper around the Argus FastAPI backend. Same-origin -- the
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

  return {
    kpis: () => get("/api/kpis"),
    transactions: (params) => get("/api/transactions", params),
    transaction: (id) => get(`/api/transactions/${encodeURIComponent(id)}`),
    queue: (params) => get("/api/queue", params),
    queueAction: (transactionId, action) => post("/api/queue/action", { transaction_id: transactionId, action }),
    modelComparison: () => get("/api/model-comparison"),
    explainability: () => get("/api/explainability"),
    simulatorOptions: () => get("/api/simulator/options"),
    score: (payload) => post("/api/score", payload),
    queueExportUrl: () => "/api/queue/export",
  };
})();
