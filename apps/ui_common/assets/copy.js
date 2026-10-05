/* Copy a Plotly figure to the clipboard as a PNG, for pasting straight into slides.
 *
 * plotly.js is already on the page (dcc.Graph loads it), so Plotly.toImage renders the PNG
 * client-side — no kaleido, no server round-trip, no new dependency.
 *
 * navigator.clipboard.write with an image blob requires a secure context. localhost counts,
 * including localhost forwarded over an SSH tunnel, which is how this app is meant to be
 * reached. Firefox does not implement image writes at all, so we fall back to a download
 * rather than failing silently — the user still gets the PNG.
 */
window.phenorag = window.phenorag || {};

window.phenorag.flash = function (btn, text) {
  if (!btn) return;
  const original = btn.textContent;
  btn.textContent = text;
  setTimeout(function () { btn.textContent = original; }, 1400);
};

window.phenorag.copyPng = function (graphId, btn) {
  const gd = document.querySelector('#' + graphId + ' .js-plotly-plot') ||
             document.getElementById(graphId);
  if (!gd || !window.Plotly) {
    window.phenorag.flash(btn, 'no chart');
    return;
  }

  // Render on an opaque surface: a transparent PNG pasted onto a white slide loses every
  // dark-mode axis label.
  const isDark = document.body.getAttribute('data-theme') === 'dark';
  const surface = isDark ? '#1a1a19' : '#fcfcfb';

  window.Plotly.toImage(gd, {
    format: 'png',
    scale: 2,
    width: gd.clientWidth || 900,
    height: gd.clientHeight || 420
  }).then(function (dataUrl) {
    return fetch(dataUrl).then(function (r) { return r.blob(); });
  }).then(function (blob) {
    return window.phenorag.onOpaque(blob, surface);
  }).then(function (blob) {
    if (!navigator.clipboard || !window.ClipboardItem) throw new Error('no clipboard image support');
    return navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })])
      .then(function () { window.phenorag.flash(btn, 'copied'); });
  }).catch(function () {
    window.phenorag.downloadPng(gd, graphId, btn);
  });
};

/* Composite the (transparent) plot onto the theme surface. */
window.phenorag.onOpaque = function (blob, surface) {
  return new Promise(function (resolve) {
    const img = new Image();
    img.onload = function () {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = surface;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      canvas.toBlob(function (out) { resolve(out || blob); }, 'image/png');
    };
    img.onerror = function () { resolve(blob); };
    img.src = URL.createObjectURL(blob);
  });
};

window.phenorag.downloadPng = function (gd, graphId, btn) {
  if (!window.Plotly) return;
  window.Plotly.downloadImage(gd, { format: 'png', scale: 2, filename: graphId })
    .then(function () { window.phenorag.flash(btn, 'downloaded'); })
    .catch(function () { window.phenorag.flash(btn, 'failed'); });
};
