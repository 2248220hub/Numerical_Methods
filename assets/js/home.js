/* home page: unit cards + a small animated hero (a posterior sharpening as data arrive) */
(function () {
  const C = window.COURSE, box = document.getElementById("units");
  if (box) {
    box.innerHTML = C.units.map((u) => {
      const href = u.folder ? u.folder.split("/").map(encodeURIComponent).join("/") + "/index.html" : "#";
      return `<a class="unit${u.status === "ready" ? "" : " soon"}" href="${href}">
        <svg width="30" height="24" viewBox="0 0 24 19"><path d="M1 3.5A2.5 2.5 0 0 1 3.5 1h5l2.2 2.4h9.8A2.5 2.5 0 0 1 23 5.9v10.6a2.5 2.5 0 0 1-2.5 2.5h-17A2.5 2.5 0 0 1 1 16.5z" fill="${u.color}" stroke="rgba(0,0,0,.25)"/></svg>
        <h3>${u.label ? u.label + " " : u.id + ". "}${u.name}</h3><p>${u.blurb || "In preparation."}</p>
        <div class="row"><span class="badge ${u.status === "ready" ? "g" : ""}">${u.status === "ready" ? u.subs.length + " lessons" : "soon"}</span><span class="badge">${u.part}</span></div></a>`;
    }).join("");
  }
  const W = (window.WIDGETS = window.WIDGETS || {});
  W.hero = (root) => {
    UI.head(root, "A POSTERIOR LEARNING FROM DATA");
    const k = UI.kpis(root, [["n", "coin tosses"], ["m", "posterior mean ± σ"]]);
    const P = UI.canvas(root, 210);
    const rng = M.rng(9); let h = 0, t = 0, timer = null;
    function draw() {
      const x = M.linspace(0.001, 0.999, 300), y = x.map((v) => M.betaPdf(v, 1 + h, 1 + t));
      P.begin({ xlim: [0, 1], ylim: [0, Math.max(3, Math.max(...y) * 1.1)], xlabel: "p", ylabel: "p(p | data)" });
      P.fill(x, 0, y, { c: "purple", a: 0.2 }); P.line(x, y, { c: "purple", w: 2.5 }); P.vline(0.7, { c: "ink", label: "truth" });
      const a = 1 + h, b = 1 + t, m = a / (a + b), s = Math.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1)));
      k.set("n", h + t); k.set("m", `${m.toFixed(3)} ± ${s.toFixed(3)}`);
    }
    function tick() { if (!root.offsetParent) return; if (h + t >= 400) { h = 0; t = 0; } rng() < 0.7 ? h++ : t++; draw(); }
    timer = setInterval(tick, 120);
    root._redraw = draw; draw();
  };
  // home has no lesson sections: initialise widgets directly
  document.addEventListener("DOMContentLoaded", () => document.querySelectorAll("[data-widget]").forEach((w) => { if (!w._inited && W[w.dataset.widget]) { w._inited = true; W[w.dataset.widget](w); } }));
})();
