/* ==========================================================================
   plot.js — tiny canvas plotting + maths + widget UI helpers (no libraries)
   ========================================================================== */
(function () {
  "use strict";
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim() || "#333";
  const COL = {
    ink: () => css("--ink"), muted: () => css("--muted"), faint: () => css("--faint"), line: () => css("--line"),
    paper: () => css("--paper"), green: () => css("--c-green"), purple: () => css("--c-purple"),
    orange: () => css("--c-orange"), blue: () => css("--c-blue"), red: () => css("--c-red"),
    teal: () => css("--c-teal"), olive: () => css("--c-olive"), slate: () => css("--c-slate"), grid: () => css("--line2")
  };
  const col = (c) => (c && COL[c] ? COL[c]() : c || COL.ink());

  const lerp = (stops) => (t) => { const n = stops.length - 1, i = Math.min(n - 1, Math.floor(t * n)), f = t * n - i, a = stops[i], b = stops[i + 1]; return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]; };
  const CMAPS = {
    gray: (t) => [255 * t, 255 * t, 255 * t],
    magma: lerp([[0, 0, 4], [59, 15, 112], [140, 41, 129], [222, 73, 104], [254, 159, 109], [252, 253, 191]]),
    div: lerp([[44, 90, 160], [150, 180, 215], [247, 244, 236], [235, 160, 120], [180, 50, 40]])
  };
  /* ------------------------------------------------------------ maths */
  const M = {
    /* in-place radix-2 FFT on separate re/im arrays (length power of 2); inv = true for the inverse (scaled 1/N) */
    fft(re, im, inv = false) {
      const n = re.length; for (let i = 1, j = 0; i < n; i++) { let b = n >> 1; for (; j & b; b >>= 1) j ^= b; j ^= b; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
      for (let m = 2; m <= n; m <<= 1) { const th = (inv ? 2 : -2) * Math.PI / m; for (let k = 0; k < n; k += m) for (let j = 0; j < m / 2; j++) { const wr = Math.cos(th * j), wi = Math.sin(th * j), a = k + j, b = a + m / 2, tr = wr * re[b] - wi * im[b], ti = wr * im[b] + wi * re[b]; re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti; } }
      if (inv) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
    },
    /* 2D FFT of row-major arrays (ny rows × nx cols), in place */
    fft2(re, im, nx, ny, inv = false) {
      const r = new Array(nx), i2 = new Array(nx);
      for (let y = 0; y < ny; y++) { for (let x = 0; x < nx; x++) { r[x] = re[y * nx + x]; i2[x] = im[y * nx + x]; } M.fft(r, i2, inv); for (let x = 0; x < nx; x++) { re[y * nx + x] = r[x]; im[y * nx + x] = i2[x]; } }
      const c = new Array(ny), ci = new Array(ny);
      for (let x = 0; x < nx; x++) { for (let y = 0; y < ny; y++) { c[y] = re[y * nx + x]; ci[y] = im[y * nx + x]; } M.fft(c, ci, inv); for (let y = 0; y < ny; y++) { re[y * nx + x] = c[y]; im[y * nx + x] = ci[y]; } }
    },
    /* periodogram |FFT|^2 of a real array (zero-padded to a power of 2), first half */
    pow2(n) { let p = 1; while (p < n) p <<= 1; return p; },
    rng(seed) {                      // mulberry32: small, fast, seedable
      let a = (seed >>> 0) || 1;
      const f = () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
      let spare = null;
      f.normal = () => {             // Box–Muller with a spare
        if (spare !== null) { const s = spare; spare = null; return s; }
        const u1 = 1 - f(), u2 = f(), r = Math.sqrt(-2 * Math.log(u1));
        spare = r * Math.sin(2 * Math.PI * u2); return r * Math.cos(2 * Math.PI * u2);
      };
      f.exp = (lam = 1) => -Math.log(1 - f()) / lam;
      return f;
    },
    linspace(a, b, n) { const o = new Array(n); for (let i = 0; i < n; i++) o[i] = a + (b - a) * i / (n - 1); return o; },
    logspace(a, b, n) { return M.linspace(a, b, n).map((v) => Math.pow(10, v)); },
    sum: (a) => a.reduce((s, v) => s + v, 0),
    mean: (a) => M.sum(a) / a.length,
    std(a, ddof = 1) { const m = M.mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) * (v - m), 0) / (a.length - ddof)); },
    npdf: (x, m = 0, s = 1) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI)),
    erf(x) { const t = 1 / (1 + 0.3275911 * Math.abs(x)); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; },
    ncdf: (x, m = 0, s = 1) => 0.5 * (1 + M.erf((x - m) / (s * Math.SQRT2))),
    nppf(p) { // inverse normal CDF (Acklam)
      const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239],
        b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572],
        c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783],
        d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
      const pl = 0.02425; let q, r;
      if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
      if (p > 1 - pl) { q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
      q = p - 0.5; r = q * q;
      return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    },
    lgamma(x) { // Lanczos
      const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
      if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - M.lgamma(1 - x);
      x -= 1; let a = c[0]; const t = x + g + 0.5; for (let i = 1; i < 9; i++) a += c[i] / (x + i);
      return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
    },
    gamma: (x) => Math.exp(M.lgamma(x)),
    betaPdf(x, a, b) { if (x <= 0 || x >= 1) return 0; return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) + M.lgamma(a + b) - M.lgamma(a) - M.lgamma(b)); },
    hist(data, lo, hi, nb) {
      const counts = new Array(nb).fill(0), w = (hi - lo) / nb;
      for (const v of data) { const k = Math.floor((v - lo) / w); if (k >= 0 && k < nb) counts[k]++; }
      const edges = M.linspace(lo, hi, nb + 1), dens = counts.map((c) => c / (data.length * w));
      return { edges, counts, dens, w };
    },
    quantile(sorted, q) { const i = (sorted.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i); return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo); },
    // a discretised density on a grid -> cdf at centres, quantiles
    gridQuantile(x, p, q) { const s = M.sum(p); let c = 0; for (let i = 0; i < x.length; i++) { const c2 = c + p[i] / s; if (c2 >= q) { const f = (q - c) / (p[i] / s); return x[i] + (f - 0.5) * (i + 1 < x.length ? x[i + 1] - x[i] : x[i] - x[i - 1]); } c = c2; } return x[x.length - 1]; },
    fmt(x, d = 3) {
      if (!isFinite(x)) return x > 0 ? "∞" : (x < 0 ? "−∞" : "—");
      const a = Math.abs(x);
      if (a !== 0 && (a < 1e-3 || a >= 1e5)) return x.toExponential(Math.max(1, d - 1)).replace("e", "×10^").replace("+", "");
      return (+x.toFixed(d)).toString();
    },
    human(sec) {
      if (sec < 1e-3) return (sec * 1e6).toFixed(0) + " µs";
      if (sec < 1) return (sec * 1e3).toFixed(0) + " ms";
      if (sec < 120) return sec.toFixed(1) + " s";
      if (sec < 7200) return (sec / 60).toFixed(0) + " min";
      if (sec < 172800) return (sec / 3600).toFixed(1) + " hours";
      if (sec < 3.2e7 * 2) return (sec / 86400).toFixed(0) + " days";
      if (sec < 3.2e7 * 1e6) return M.fmt(sec / 3.156e7, 2) + " years";
      return M.fmt(sec / 3.156e7, 2) + " years (age of Universe: 1.4×10^10)";
    }
  };

  /* ------------------------------------------------------------ plot */
  function niceTicks(lo, hi, n = 5) {
    const span = hi - lo; if (!(span > 0)) return [lo];
    const step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0))), r = step0 / mag;
    const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
    const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * step; v += step) out.push(Math.abs(v) < 1e-12 * step ? 0 : v);
    return out;
  }
  function tickLabel(v) {
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) { const e = Math.floor(Math.log10(a)), m = v / Math.pow(10, e); return (Math.abs(m - 1) < 1e-9 ? "" : (+m.toFixed(1)) + "×") + "10" + sup(e); }
    return (+v.toPrecision(6)).toString();
  }
  const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  const sup = (e) => String(e).split("").map((c) => SUP[c] || c).join("");

  class Plot {
    constructor(canvas, o = {}) {
      this.c = canvas; this.ctx = canvas.getContext("2d");
      this.o = Object.assign({ h: 290, pad: [16, 16, 40, 58] }, o);
    }
    size() {
      const dpr = window.devicePixelRatio || 1, w = Math.max(200, this.c.parentElement.clientWidth);
      const h = typeof this.o.h === "function" ? this.o.h(w) : this.o.h;
      if (this.c.width !== Math.round(w * dpr) || this.c.height !== Math.round(h * dpr)) {
        this.c.width = Math.round(w * dpr); this.c.height = Math.round(h * dpr); this.c.style.height = h + "px";
      }
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0); this.W = w; this.H = h;
      const [t, r, b, l] = this.o.pad; this.x0 = l; this.x1 = w - r; this.y0 = h - b; this.y1 = t;
    }
    begin(o) {
      this.size(); Object.assign(this, { xlog: !!o.xlog, ylog: !!o.ylog });
      const tr = (v, lg) => (lg ? Math.log10(v) : v);
      this.xa = tr(o.xlim[0], this.xlog); this.xb = tr(o.xlim[1], this.xlog);
      this.ya = tr(o.ylim[0], this.ylog); this.yb = tr(o.ylim[1], this.ylog);
      const g = this.ctx; g.clearRect(0, 0, this.W, this.H);
      if (o.axes === false) return this;
      g.font = "11px JBM, monospace"; g.lineWidth = 1;
      // grid + ticks
      const xt = this.xlog ? this.logTicks(this.xa, this.xb) : niceTicks(this.xa, this.xb, o.xticks || Math.max(3, Math.round((this.x1 - this.x0) / 90)));
      const yt = this.ylog ? this.logTicks(this.ya, this.yb) : niceTicks(this.ya, this.yb, o.yticks || 5);
      g.strokeStyle = col("grid"); g.fillStyle = col("muted");
      g.textAlign = "center"; g.textBaseline = "top";
      for (const v of xt) { const px = this.px(this.xlog ? Math.pow(10, v) : v); if (px < this.x0 - 1 || px > this.x1 + 1) continue; g.beginPath(); g.moveTo(px, this.y0); g.lineTo(px, this.y1); g.stroke(); g.fillText(this.xlog ? tickLabel(Math.pow(10, v)) : tickLabel(v), px, this.y0 + 5); }
      g.textAlign = "right"; g.textBaseline = "middle";
      for (const v of yt) { const py = this.py(this.ylog ? Math.pow(10, v) : v); if (py > this.y0 + 1 || py < this.y1 - 1) continue; g.beginPath(); g.moveTo(this.x0, py); g.lineTo(this.x1, py); g.stroke(); g.fillText(this.ylog ? tickLabel(Math.pow(10, v)) : tickLabel(v), this.x0 - 6, py); }
      g.strokeStyle = col("ink"); g.lineWidth = 1.5; g.beginPath(); g.moveTo(this.x0, this.y1); g.lineTo(this.x0, this.y0); g.lineTo(this.x1, this.y0); g.stroke();
      g.fillStyle = col("ink"); g.font = "600 11.5px JBM, monospace";
      if (o.xlabel) { g.textAlign = "center"; g.textBaseline = "bottom"; g.fillText(o.xlabel, (this.x0 + this.x1) / 2, this.H - 2); }
      if (o.ylabel) { g.save(); g.translate(12, (this.y0 + this.y1) / 2); g.rotate(-Math.PI / 2); g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(o.ylabel, 0, 0); g.restore(); }
      if (o.title) { g.textAlign = "left"; g.textBaseline = "top"; g.fillText(o.title, this.x0 + 6, this.y1 + 2); }
      return this;
    }
    logTicks(a, b) { const o = []; for (let e = Math.ceil(a - 1e-9); e <= b + 1e-9; e++) o.push(e); if (o.length > 8) return o.filter((e, i) => i % Math.ceil(o.length / 7) === 0); return o; }
    px(x) { const v = this.xlog ? Math.log10(x) : x; return this.x0 + (v - this.xa) / (this.xb - this.xa) * (this.x1 - this.x0); }
    py(y) { const v = this.ylog ? Math.log10(y) : y; return this.y0 - (v - this.ya) / (this.yb - this.ya) * (this.y0 - this.y1); }
    inv(px, py) { let x = this.xa + (px - this.x0) / (this.x1 - this.x0) * (this.xb - this.xa), y = this.ya + (this.y0 - py) / (this.y0 - this.y1) * (this.yb - this.ya); if (this.xlog) x = Math.pow(10, x); if (this.ylog) y = Math.pow(10, y); return [x, y]; }
    clip(on = true) { const g = this.ctx; if (on) { g.save(); g.beginPath(); g.rect(this.x0, this.y1, this.x1 - this.x0, this.y0 - this.y1); g.clip(); } else g.restore(); }
    line(xs, ys, o = {}) {
      const g = this.ctx; this.clip(); g.strokeStyle = col(o.c); g.lineWidth = o.w || 2; g.globalAlpha = o.a ?? 1;
      g.setLineDash(o.dash || []); g.beginPath(); let pen = false;
      for (let i = 0; i < xs.length; i++) { const y = ys[i]; if (!isFinite(y) || (this.ylog && y <= 0)) { pen = false; continue; } const X = this.px(xs[i]), Y = this.py(y); pen ? g.lineTo(X, Y) : g.moveTo(X, Y); pen = true; }
      g.stroke(); g.setLineDash([]); g.globalAlpha = 1; this.clip(false); return this;
    }
    fill(xs, y0, y1, o = {}) {
      const g = this.ctx; this.clip(); g.fillStyle = col(o.c); g.globalAlpha = o.a ?? 0.25; g.beginPath();
      const lo = typeof y0 === "number" ? xs.map(() => y0) : y0;
      xs.forEach((x, i) => (i ? g.lineTo(this.px(x), this.py(y1[i])) : g.moveTo(this.px(x), this.py(y1[i]))));
      for (let i = xs.length - 1; i >= 0; i--) g.lineTo(this.px(xs[i]), this.py(lo[i]));
      g.closePath(); g.fill(); g.globalAlpha = 1; this.clip(false); return this;
    }
    pts(xs, ys, o = {}) {
      const g = this.ctx, r = o.r || 2.5; this.clip(); g.globalAlpha = o.a ?? 1;
      for (let i = 0; i < xs.length; i++) {
        const c = Array.isArray(o.c) ? o.c[i] : o.c; g.fillStyle = col(c);
        const R = Array.isArray(r) ? r[i] : r; if (!(R > 0)) continue;
        g.beginPath(); g.arc(this.px(xs[i]), this.py(ys[i]), R, 0, 7); g.fill();
        if (o.stroke) { g.strokeStyle = col(o.stroke); g.lineWidth = 1; g.stroke(); }
      }
      g.globalAlpha = 1; this.clip(false); return this;
    }
    bars(edges, h, o = {}) {
      const g = this.ctx; this.clip(); g.fillStyle = col(o.c); g.globalAlpha = o.a ?? 0.45;
      for (let i = 0; i < h.length; i++) { const X0 = this.px(edges[i]), X1 = this.px(edges[i + 1]), Y = this.py(h[i]); g.fillRect(X0, Y, Math.max(1, X1 - X0 - (o.gap ?? 0.6)), this.py(this.ylog ? Math.pow(10, this.ya) : this.ya) - Y); }
      g.globalAlpha = 1; this.clip(false); return this;
    }
    rect(xa, ya, xb, yb, o = {}) {
      const g = this.ctx; this.clip(); const X = this.px(xa), Y = this.py(yb), w = this.px(xb) - X, h = this.py(ya) - Y;
      if (o.fill !== false) { g.fillStyle = col(o.c); g.globalAlpha = o.a ?? 0.25; g.fillRect(X, Y, w, h); }
      if (o.stroke) { g.globalAlpha = 1; g.strokeStyle = col(o.stroke); g.lineWidth = o.w || 1.5; g.setLineDash(o.dash || []); g.strokeRect(X, Y, w, h); g.setLineDash([]); }
      g.globalAlpha = 1; this.clip(false); return this;
    }
    vline(x, o = {}) { const g = this.ctx; const X = this.px(x); if (X < this.x0 || X > this.x1) return this; g.strokeStyle = col(o.c); g.lineWidth = o.w || 1.5; g.setLineDash(o.dash || [5, 4]); g.beginPath(); g.moveTo(X, this.y0); g.lineTo(X, this.y1 + (o.top || 0)); g.stroke(); g.setLineDash([]); if (o.label) this.text(X + 4, this.y1 + 4 + (o.dy || 0), o.label, { c: o.c, px: true }); return this; }
    hline(y, o = {}) { const g = this.ctx; const Y = this.py(y); if (Y > this.y0 || Y < this.y1) return this; g.strokeStyle = col(o.c); g.lineWidth = o.w || 1.5; g.setLineDash(o.dash || [5, 4]); g.beginPath(); g.moveTo(this.x0, Y); g.lineTo(this.x1, Y); g.stroke(); g.setLineDash([]); if (o.label) this.text(this.x1 - 4, Y - 4, o.label, { c: o.c, px: true, align: "right", base: "bottom" }); return this; }
    text(x, y, s, o = {}) { const g = this.ctx; g.fillStyle = col(o.c); g.font = (o.bold ? "700 " : "500 ") + (o.size || 11.5) + "px JBM, monospace"; g.textAlign = o.align || "left"; g.textBaseline = o.base || "top"; const X = o.px ? x : this.px(x), Y = o.px ? y : this.py(y); if (o.bg) { const w = g.measureText(s).width; g.fillStyle = col(o.bg); g.globalAlpha = 0.85; g.fillRect(X - (o.align === "right" ? w : o.align === "center" ? w / 2 : 0) - 3, Y - 2, w + 6, (o.size || 11.5) + 5); g.globalAlpha = 1; g.fillStyle = col(o.c); } g.fillText(s, X, Y); return this; }
    /* Z: row-major ny×nx values (row 0 drawn at the TOP), placed over [xa,xb]×[ya,yb]; o.lo/o.hi clip, o.cmap: "gray"|"magma"|"div" */
    image(Z, nx, ny, xa, xb, ya, yb, o = {}) {
      let lo = o.lo, hi = o.hi; if (lo == null || hi == null) { let a = Infinity, b = -Infinity; for (const v of Z) if (isFinite(v)) { if (v < a) a = v; if (v > b) b = v; } if (lo == null) lo = a; if (hi == null) hi = b; }
      const oc = document.createElement("canvas"); oc.width = nx; oc.height = ny; const g2 = oc.getContext("2d"), im = g2.createImageData(nx, ny);
      const cm = CMAPS[o.cmap || "magma"];
      for (let i = 0; i < nx * ny; i++) { const t = Math.max(0, Math.min(1, (Z[i] - lo) / (hi - lo || 1))); const c = cm(isFinite(t) ? t : 0); im.data[4 * i] = c[0]; im.data[4 * i + 1] = c[1]; im.data[4 * i + 2] = c[2]; im.data[4 * i + 3] = 255; }
      g2.putImageData(im, 0, 0); const g = this.ctx; g.save(); g.imageSmoothingEnabled = !!o.smooth;
      const X0 = this.px(xa), X1 = this.px(xb), Y0 = this.py(yb), Y1 = this.py(ya); g.drawImage(oc, X0, Y0, X1 - X0, Y1 - Y0); g.restore(); return this;
    }
    seg(xa, ya, xb, yb, o = {}) { const g = this.ctx; this.clip(); g.strokeStyle = col(o.c); g.lineWidth = o.w || 2; g.globalAlpha = o.a ?? 1; g.setLineDash(o.dash || []); g.beginPath(); g.moveTo(this.px(xa), this.py(ya)); g.lineTo(this.px(xb), this.py(yb)); g.stroke(); g.setLineDash([]); g.globalAlpha = 1; this.clip(false); return this; }
    legendBox(items, o = {}) { // items: [[color,label,dash?]]
      const g = this.ctx; g.font = "500 11px JBM, monospace"; const w = Math.max(...items.map((i) => g.measureText(i[1]).width)) + 34, h = items.length * 16 + 8;
      const X = o.left ? this.x0 + 8 : this.x1 - w - 8, Y = o.bottom ? this.y0 - h - 8 : this.y1 + 6;
      g.fillStyle = col("paper"); g.globalAlpha = 0.9; g.fillRect(X, Y, w, h); g.globalAlpha = 1; g.strokeStyle = col("line"); g.lineWidth = 1; g.strokeRect(X, Y, w, h);
      items.forEach((it, i) => { g.strokeStyle = col(it[0]); g.lineWidth = 2.5; g.setLineDash(it[2] || []); g.beginPath(); g.moveTo(X + 6, Y + 12 + i * 16); g.lineTo(X + 24, Y + 12 + i * 16); g.stroke(); g.setLineDash([]); g.fillStyle = col("ink"); g.textAlign = "left"; g.textBaseline = "middle"; g.fillText(it[1], X + 30, Y + 12 + i * 16); });
    }
  }

  /* ------------------------------------------------------------ UI helpers */
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  const UI = {
    el,
    head(root, label, extra) { const h = el("div", "stage-h"); h.append(el("span", "lbl", label), el("span", "sp")); if (extra) h.append(extra); root.append(h); return h; },
    canvas(root, h = 290, opts = {}) { const w = el("div", "canvas-wrap"); const c = el("canvas"); w.append(c); root.append(w); return new Plot(c, Object.assign({ h }, opts)); },
    ctrls(root) { const c = el("div", "ctrls"); root.append(c); return c; },
    slider(root, o) {
      const box = el("label", "ctrl"); const top = el("div", "top"); const nm = el("span", null, o.label); const val = el("b");
      top.append(nm, val); const inp = el("input"); inp.type = "range"; inp.min = o.min; inp.max = o.max; inp.step = o.step || (o.max - o.min) / 100; inp.value = o.value;
      const show = () => (val.textContent = o.fmt ? o.fmt(+inp.value) : inp.value);
      inp.addEventListener("input", () => { show(); o.on && o.on(+inp.value); }); show();
      box.append(top, inp); root.append(box); inp.set = (v) => { inp.value = v; show(); }; return inp;
    },
    select(root, o) {
      const box = el("label", "ctrl"); box.append(el("div", "top", `<span>${o.label}</span>`)); const s = el("select");
      o.options.forEach(([v, t]) => { const op = el("option", null, t); op.value = v; s.append(op); }); s.value = o.value ?? o.options[0][0];
      s.addEventListener("change", () => o.on && o.on(s.value)); box.append(s); root.append(box); return s;
    },
    btns(root) { const r = el("div", "btnrow"); root.append(r); return r; },
    button(root, text, on, cls = "") { const b = el("button", "btn " + cls, text); b.addEventListener("click", on); root.append(b); return b; },
    kpis(root, defs) { const box = el("div", "kpis"); const map = {}; defs.forEach(([k, label]) => { const d = el("div", "kpi", `<small>${label}</small><b>—</b>`); map[k] = d.querySelector("b"); box.append(d); }); root.append(box); return { set: (k, v) => { if (map[k]) map[k].textContent = v; } }; },
    legend(root, items) { const l = el("div", "legend"); items.forEach(([c, t]) => l.append(el("span", null, `<i style="background:${col(c)}"></i>${t}`))); root.append(l); return l; },
    note(root, html) { const n = el("div", "note", html); root.append(n); return n; }
  };

  window.M = M; window.Plot = Plot; window.UI = UI; window.col = col;
})();
