/*
 * Hand-drawn inline SVG icons. No icon font, no network fetch -- every icon
 * used anywhere in the app is defined here as a small stroke-based path so
 * the whole dashboard renders fully offline. All icons inherit color from
 * the surrounding element via stroke="currentColor".
 */
const Icons = (() => {
  const wrap = (inner, size = 18) =>
    `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" ` +
    `stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${inner}</svg>`;

  return {
    overview: wrap('<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/>' +
      '<rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>'),
    explorer: wrap('<circle cx="11" cy="11" r="6.5"/><line x1="20" y1="20" x2="15.8" y2="15.8"/>'),
    queue: wrap('<path d="M4 6h16"/><path d="M4 12h10"/><path d="M4 18h7"/><path d="M17 15l3 3 3-3" transform="translate(-2 0)"/>'),
    comparison: wrap('<path d="M4 20V10"/><path d="M11 20V4"/><path d="M18 20v-7"/><path d="M2 20h20"/>'),
    explainability: wrap('<path d="M9 18h6"/><path d="M10 22h4"/><path d="M12 2a6.5 6.5 0 0 0-4 11.6c.6.5 1 1.3 1 2.1V17h6v-1.3c0-.8.4-1.6 1-2.1A6.5 6.5 0 0 0 12 2Z"/>'),
    simulator: wrap('<path d="M9 3h6"/><path d="M10 3v5.2a2 2 0 0 1-.4 1.2L5.6 15.8A2 2 0 0 0 7.2 19h9.6a2 2 0 0 0 1.6-3.2l-4-5.4a2 2 0 0 1-.4-1.2V3"/>'),
    sun: wrap('<circle cx="12" cy="12" r="4.2"/><path d="M12 2.5v2.2M12 19.3v2.2M4.2 4.2l1.6 1.6M18.2 18.2l1.6 1.6M2.5 12h2.2M19.3 12h2.2M4.2 19.8l1.6-1.6M18.2 5.8l1.6-1.6"/>'),
    moon: wrap('<path d="M20.5 14.5A8.5 8.5 0 1 1 9.5 3.5a7 7 0 0 0 11 11Z"/>'),
    close: wrap('<path d="M5 5l14 14M19 5L5 19"/>'),
    chevronRight: wrap('<path d="M9 6l6 6-6 6"/>', 16),
    chevronLeft: wrap('<path d="M15 6l-6 6 6 6"/>', 16),
    download: wrap('<path d="M12 3v12"/><path d="M7 10l5 5 5-5"/><path d="M4 20h16"/>'),
    search: wrap('<circle cx="10.5" cy="10.5" r="6.5"/><path d="M20 20l-4.3-4.3"/>', 16),
    good: wrap('<circle cx="12" cy="12" r="9.5"/><path d="M7.5 12.5l3 3 6-6.5"/>', 15),
    warning: wrap('<path d="M12 3.5 21.5 20h-19L12 3.5Z"/><path d="M12 10v4"/><path d="M12 17h.01"/>', 15),
    serious: wrap('<circle cx="12" cy="12" r="9.5"/><path d="M12 7.5v5.5"/><path d="M12 16.3h.01"/>', 15),
    critical: wrap('<circle cx="12" cy="12" r="9.5"/><path d="M8.5 8.5l7 7M15.5 8.5l-7 7"/>', 15),
    flask: wrap('<path d="M9 3h6M10 3v6.2L4.8 18a1.6 1.6 0 0 0 1.4 2.4h11.6a1.6 1.6 0 0 0 1.4-2.4L14 9.2V3"/>'),
    upload: wrap('<path d="M12 16V4"/><path d="M7 9l5-5 5 5"/><path d="M4 16v2.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V16"/>'),
    history: wrap('<path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/><path d="M12 7v5l3.5 2"/>'),
    caretUp: wrap('<path d="M6 15l6-6 6 6"/>', 12),
    caretDown: wrap('<path d="M6 9l6 6 6-6"/>', 12),
    file: wrap('<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z"/><path d="M14 3v5h5"/>', 16),
  };
})();
