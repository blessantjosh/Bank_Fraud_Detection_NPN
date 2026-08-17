/* Shared formatting / small DOM utilities used across pages. */
const Fmt = (() => {
  const numberFmt = new Intl.NumberFormat("en-US");
  const currencyFmt = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
  const pctFmt = new Intl.NumberFormat("en-US", { style: "percent", minimumFractionDigits: 2, maximumFractionDigits: 2 });

  const prefersReducedMotion = () =>
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function int(n) { return numberFmt.format(Math.round(n)); }
  function money(n) { return currencyFmt.format(n); }
  function pct(n) { return pctFmt.format(n); }
  function score(n) { return `${(n * 100).toFixed(1)}%`; }

  function dateTime(iso) {
    const d = new Date(iso);
    return d.toLocaleString("en-US", {
      year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
    });
  }
  function dateOnly(iso) {
    const d = new Date(iso);
    return d.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });
  }
  function dateShort(iso) {
    const d = new Date(iso);
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  }

  function escapeHtml(s) {
    if (s === null || s === undefined) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function debounce(fn, wait = 250) {
    let t;
    return (...args) => {
      clearTimeout(t);
      t = setTimeout(() => fn(...args), wait);
    };
  }

  /** Animate a number counting up inside `el`, respecting reduced-motion. */
  function countUp(el, target, { duration = 800, formatter = int, decimals = 0 } = {}) {
    if (prefersReducedMotion()) {
      el.textContent = formatter(target);
      return;
    }
    const start = performance.now();
    const from = 0;
    function tick(now) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
      const value = from + (target - from) * eased;
      el.textContent = formatter(decimals ? Number(value.toFixed(decimals)) : value);
      if (t < 1) requestAnimationFrame(tick);
      else el.textContent = formatter(target);
    }
    requestAnimationFrame(tick);
  }

  return { int, money, pct, score, dateTime, dateOnly, dateShort, escapeHtml, debounce, countUp, prefersReducedMotion };
})();
