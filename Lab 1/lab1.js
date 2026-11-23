/* Lab 1 — symbols and the threshold widget */
window.SYMBOLS = {
  Xk: { n: "X<sub>k</sub> — Fourier coefficient", t: "complex", s: "rfft of the whole record: one coefficient per frequency f<sub>k</sub> = k f<sub>s</sub>/N, 0 ≤ f<sub>k</sub> ≤ f<sub>s</sub>/2.", py: "X = rfft(x)", f: "call fftw_execute_dft_r2c(plan, x, X)" },
  Hk: { n: "H<sub>k</sub> — filter weight", t: "real, in [0,1]", s: "1 = keep this frequency, 0 = remove it. A real weight does not move the signal in time.", py: "H = ((f > 493) & (f < 814)).astype(float)", f: "H = merge(1.0_dp, 0.0_dp, f > 493 .and. f < 814)" },
  Yk: { n: "Y<sub>k</sub> — filtered spectrum", t: "complex", s: "H<sub>k</sub> X<sub>k</sub>; its inverse FFT is the filtered recording.", py: "y = irfft(H * X, n=N)", f: "Y = H * X" },
  sclean: { n: "s — the clean signal", t: "reference", s: "segnale.wav: given only to check the filters. In real experiments it does not exist.", py: "fs_s, s = read_wav('segnale.wav')", f: "! read with libsndfile or a raw stream" },
  Pk: { n: "P<sub>k</sub> — periodogram (one-sided density)", t: "units²/Hz", s: "2|X<sub>k</sub>|²/(f<sub>s</sub> N). Exponential around the true PSD: standard deviation = mean.", py: "Pk = np.abs(X)**2 * 2 / (fs * N)", f: "Pk = 2*abs(X)**2/(fs*N)" },
  fs: { n: "f<sub>s</sub> — sampling rate", t: "44 100 Hz", s: "Samples per second. Nyquist frequency f<sub>s</sub>/2 = 22 050 Hz.", py: "fs, x = wavfile.read(name)", f: "fs = 44100" },
  N: { n: "N — number of samples", t: "5 292 000", s: "120 s × 44 100 Hz.", py: "N = x.size", f: "N = size(x)" },
  PW: { n: "P̂<sub>W</sub>(f) — Welch estimate", t: "PSD", s: "Average of K windowed, overlapping segment periodograms: scatter ≈ 1/√K, resolution a few f<sub>s</sub>/L.", py: "fw, W = sp.welch(x, fs, window='nuttall', nperseg=2**16)", f: "! loop over segments, FFTW r2c, average |X|^2" },
  Al: { n: "A<sub>ℓ</sub> — amplitude of line ℓ", t: "sample units", s: "Projection on e<sup>−j2πf<sub>ℓ</sub>t</sup>: 60 for each note, 120 for the hum.", py: "A = 2*np.abs(np.sum(xb*np.exp(-2j*np.pi*fl*tb)))/xb.size", f: "A = 2*abs(sum(xb*exp(-2*(0,1)*pi*fl*tb)))/size(xb)" },
  Sn: { n: "S<sub>n</sub>(f) — noise PSD model", t: "smooth model", s: "A[1 + (f<sub>k</sub>/f)<sup>α</sup>], fitted to Welch with the lines masked. Needed at every frequency for z and Φ.", py: "Sn = np.exp(logS(f, *p))", f: "Sn = A*(1.0_dp + (fk/f)**alpha)" },
  A: { n: "A — white-noise level", t: "parameter", s: "Level of the flat part above the knee (≈ 3.07 for recording 1).", py: "A = np.exp(p[0])", f: "A = exp(p(1))" },
  fk: { n: "f<sub>k</sub> — knee frequency", t: "≈ 440 Hz", s: "Where the red part equals the white part.", py: "fk = np.exp(p[1])", f: "fk = exp(p(2))" },
  alpha: { n: "α — slope of the red noise", t: "≈ 2", s: "S ∝ f<sup>−α</sup> below the knee; α = 2 is a random walk (Brownian, “red”).", py: "alpha = p[2]", f: "alpha = p(3)" },
  Phi: { n: "Φ<sub>k</sub> — Wiener weight", t: "in [0,1]", s: "S<sub>s</sub>/(S<sub>s</sub> + S<sub>n</sub>) = 1 − 1/z: keeps each bin in proportion to its signal content.", py: "phi = np.clip(1 - 1/z, 0, 1)", f: "phi = max(0.0_dp, 1.0_dp - 1.0_dp/z)" },
  zk: { n: "z<sub>k</sub> — data over noise model", t: "ratio", s: "P<sub>k</sub>/S<sub>n</sub>(f<sub>k</sub>). Exponential with mean 1 in noise-only bins; ~5×10⁴ at the notes.", py: "z = Pk / Sn", f: "z = Pk / Sn" },
  M: { n: "M — number of bins tested", t: "2.64×10⁶", s: "All the frequencies above 20 Hz. More bins tested, more chances for noise to look like a line.", py: "M = np.sum(f > 20)", f: "M = count(f > 20.0_dp)" },
  zstar: { n: "z* — detection threshold", t: "19.4", s: "ln(M/p): noise alone exceeds it in p = 1% of recordings (a false-alarm correction for many trials).", py: "zstar = np.log(M / 0.01)", f: "zstar = log(M/0.01_dp)" },
  pfa: { n: "p — false alarms allowed", t: "per recording", s: "Expected number of noise bins above z*: 0.01 = one false line in a hundred recordings.", py: "p = 0.01", f: "p = 0.01_dp" },
  sum: { n: "Σ — sum", t: "operator", s: "Sum over segments, samples or bins.", py: "np.sum(a)", f: "sum(a)" }
};

(function () {
  const W = (window.WIDGETS = {});
  W.zthresh = (root) => {
    UI.head(root, "THE THRESHOLD z* = ln(M/p) · HOW MANY FALSE LINES?");
    const k = UI.kpis(root, [["z", "threshold z*"], ["fa1", "false lines expected at z > 1"], ["fa", "false lines at z > z*"], ["max", "expected largest noise z ≈ ln M"]]);
    const P = UI.canvas(root, 250);
    const c = UI.ctrls(root); let lM = Math.log10(2.6436e6), p = 0.01;
    UI.slider(c, { label: "bins tested M", min: 2, max: 8, step: 0.05, value: lM, fmt: (v) => M.fmt(Math.pow(10, v), 3), on: (v) => { lM = v; draw(); } });
    UI.slider(c, { label: "false lines allowed p", min: -4, max: 0, step: 0.05, value: -2, fmt: (v) => Math.pow(10, v).toPrecision(2), on: (v) => { p = Math.pow(10, v); draw(); } });
    function draw() {
      const Mb = Math.pow(10, lM), zs = Math.log(Mb / p), zz = M.linspace(0, 30, 300);
      P.begin({ xlim: [0, 30], ylim: [1e-6, 1e9], ylog: true, xlabel: "threshold z", ylabel: "expected noise bins above z" });
      P.line(zz, zz.map((v) => Mb * Math.exp(-v)), { c: "blue", w: 2.5 });
      P.vline(zs, { c: "red", label: "z*", dash: [] }); P.hline(p, { c: "faint", label: "p" });
      [[3544, "hum"], [41349, "notes"]].forEach(([v, l]) => P.text(29.5, 2e8, `${l}: z = ${v}`, { align: "right", c: "green", size: 11 }));
      k.set("z", zs.toFixed(1)); k.set("fa1", M.fmt(Mb * Math.exp(-1), 3)); k.set("fa", p.toPrecision(2)); k.set("max", Math.log(Mb).toFixed(1));
    }
    root._redraw = draw; draw();
  };
})();
