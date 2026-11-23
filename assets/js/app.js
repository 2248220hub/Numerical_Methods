/* ==========================================================================
   app.js — the site shell: sidebar, lesson player, hover-symbols, code tabs,
   exercises, views, search, theme, progress. Shared by every lesson folder.
   ========================================================================== */
(function () {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const store = {
    get(k, d) { try { const v = localStorage.getItem("cma." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem("cma." + k, JSON.stringify(v)); } catch (e) { /* private mode */ } }
  };
  const C = window.COURSE, body = document.body;
  body.dataset.view = "learn";
  const ROOT = body.dataset.root || "./", UNIT = body.dataset.unit || null;
  const unit = C.units.find((u) => u.id === UNIT);
  const enc = (f) => f.split("/").map(encodeURIComponent).join("/");
  const unitHref = (u) => ROOT + enc(u.folder) + "/index.html";
  const lbl = (u) => (u.label ? u.label + " " : u.id + ". ") + u.name;

  /* ------------------------------------------------------------ icons */
  const ICON = {
    folder: (c) => `<svg class="ficon" viewBox="0 0 24 19"><path d="M1 3.5A2.5 2.5 0 0 1 3.5 1h5l2.2 2.4h9.8A2.5 2.5 0 0 1 23 5.9v10.6a2.5 2.5 0 0 1-2.5 2.5h-17A2.5 2.5 0 0 1 1 16.5z" fill="${c}" stroke="rgba(0,0,0,.25)"/><path d="M1 6.5h22" stroke="rgba(255,255,255,.35)"/></svg>`,
    chev: `<svg class="chev" viewBox="0 0 10 10"><path d="M3 1.5 6.5 5 3 8.5" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>`,
    file: `<svg viewBox="0 0 16 16"><path d="M4 1.5h5l3 3v10H4z" fill="none" stroke="currentColor" stroke-width="1.3"/><path d="M9 1.5v3h3" fill="none" stroke="currentColor" stroke-width="1.3"/></svg>`,
    check: `<svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="6.5" fill="currentColor"/><path d="m5 8 2 2 4-4" stroke="#fff" stroke-width="1.6" fill="none"/></svg>`,
    spark: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6"/></svg>`
  };

  /* ------------------------------------------------------------ theme */
  const setTheme = (t) => { document.documentElement.dataset.theme = t; store.set("theme", t); window.dispatchEvent(new Event("themechange")); };
  document.documentElement.dataset.theme = store.get("theme", "light");
  $$("[data-act=theme]").forEach((b) => b.addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark")));
  $$("[data-act=burger]").forEach((b) => b.addEventListener("click", () => body.classList.toggle("nav")));

  /* ------------------------------------------------------------ progress */
  const seen = store.get("seen", {});
  const markSeen = (u, i) => { seen[u] = seen[u] || []; if (!seen[u].includes(i)) { seen[u].push(i); store.set("seen", seen); } };
  const totals = () => { let t = 0, s = 0; C.units.filter((u) => u.status === "ready").forEach((u) => { t += u.subs.length; s += (seen[u.id] || []).length; }); return [s, t]; };

  /* ------------------------------------------------------------ sidebar */
  function renderSidebar(sectionTitles) {
    const tree = $(".tree"); if (!tree) return;
    const ready = C.units.filter((u) => u.status === "ready");
    let h = `<div class="tree-head"><svg width="26" height="22" viewBox="0 0 24 19"><path d="M1 3.5A2.5 2.5 0 0 1 3.5 1h5l2.2 2.4h9.8A2.5 2.5 0 0 1 23 5.9v10.6a2.5 2.5 0 0 1-2.5 2.5h-17A2.5 2.5 0 0 1 1 16.5z" fill="currentColor"/></svg>Course<small>${ready.reduce((s, u) => s + u.subs.length, 0)} lessons</small></div>`;
    C.units.forEach((u) => {
      const cur = u.id === UNIT, nSeen = (seen[u.id] || []).length;
      if (u.status !== "ready") {
        h += `<div class="folder soon" title="Coming soon">${ICON.chev}${ICON.folder(u.color)}<span class="nm">${lbl(u)}</span><span class="cnt">soon</span></div>`;
        return;
      }
      h += `<div class="folder${cur ? " open" : ""}" data-unit="${u.id}">${ICON.chev}${ICON.folder(u.color)}<span class="nm">${lbl(u)}</span><span class="cnt">${nSeen}/${u.subs.length}</span></div><div class="subs">`;
      (cur && sectionTitles ? sectionTitles : u.subs).forEach((t, i) => {
        const done = (seen[u.id] || []).includes(i);
        const href = cur ? `#s${i + 1}` : unitHref(u) + `#s${i + 1}`;
        h += `<a class="sub${done ? " done" : ""}" data-i="${i}" href="${href}">${done ? ICON.check : ICON.file}<span>${t}</span></a>`;
      });
      h += `</div>`;
    });
    tree.innerHTML = h;
    $$(".folder[data-unit]", tree).forEach((f) => f.addEventListener("click", () => f.classList.toggle("open")));
    const [s, t] = totals(), pr = $(".progress");
    if (pr) pr.innerHTML = `PROGRESS <span class="bar"><i style="width:${t ? (100 * s) / t : 0}%"></i></span> ${s} / ${t}`;
  }

  /* ------------------------------------------------------------ maths (KaTeX) */
  const MACROS = { "\\h": "\\htmlData{s=#1}{#2}", "\\E": "\\mathbb{E}", "\\dd": "\\,\\mathrm{d}", "\\N": "\\mathcal{N}" };
  const KOPT = { macros: MACROS, trust: (c) => c.command === "\\htmlData", strict: false, throwOnError: false };
  function renderMath(root) {
    if (!window.katex) return;
    $$(".tex:not([data-done])", root).forEach((e) => { const src = e.textContent; e.dataset.done = 1; e.dataset.src = src; try { katex.render(src, e, Object.assign({ displayMode: true }, KOPT)); } catch (err) { e.textContent = src; } });
    if (window.renderMathInElement) renderMathInElement(root, Object.assign({ delimiters: [{ left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }], ignoredClasses: ["tex", "codebox", "tip"] }, KOPT));
  }

  /* ------------------------------------------------------------ hover symbols */
  const SYM = Object.assign({}, window.SYMBOLS_COMMON || {}, window.SYMBOLS || {});
  const tip = document.createElement("div"); tip.className = "tip"; body.append(tip);
  let pinned = false;
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  function showTip(key, x, y) {
    const s = SYM[key] || { n: key, s: "No description yet for this symbol." };
    let h = `<div class="tn">${s.n}${s.t ? `<i>${s.t}</i>` : ""}</div><div class="ts">${s.s}</div>`;
    if (s.py) h += `<div class="tl">PYTHON</div><pre>${hl(s.py, "py")}</pre>`;
    if (s.f) h += `<div class="tl">FORTRAN</div><pre>${hl(s.f, "f90")}</pre>`;
    h += `<div class="tf">${pinned ? "click anywhere to unpin" : "click to pin this card"}</div>`;
    tip.innerHTML = h; tip.classList.add("show"); place(x, y);
  }
  function place(x, y) {
    const r = tip.getBoundingClientRect(); let L = x + 16, T = y + 18;
    if (L + r.width > innerWidth - 8) L = Math.max(8, x - r.width - 16);
    if (T + r.height > innerHeight - 8) T = Math.max(8, y - r.height - 14);
    tip.style.left = L + "px"; tip.style.top = T + "px";
  }
  document.addEventListener("mouseover", (e) => { if (pinned) return; const t = e.target.closest && e.target.closest("[data-s]"); if (t) showTip(t.dataset.s, e.clientX, e.clientY); });
  document.addEventListener("mousemove", (e) => { if (!pinned && tip.classList.contains("show")) { if (!e.target.closest || !e.target.closest("[data-s]")) tip.classList.remove("show"); else place(e.clientX, e.clientY); } });
  document.addEventListener("click", (e) => {
    const t = e.target.closest && e.target.closest("[data-s]");
    if (t && !pinned) { pinned = true; tip.classList.add("pin"); showTip(t.dataset.s, e.clientX, e.clientY); return; }
    if (pinned && !(e.target.closest && e.target.closest(".tip"))) { pinned = false; tip.classList.remove("pin", "show"); }
  });

  /* ------------------------------------------------------------ syntax highlight */
  const KW = {
    py: "and|as|assert|break|class|continue|def|del|elif|else|except|False|finally|for|from|global|if|import|in|is|lambda|None|not|or|pass|raise|return|True|try|while|with|yield",
    f90: "program|end|implicit|none|integer|real|double|precision|complex|logical|character|parameter|allocatable|allocate|deallocate|dimension|intent|in|out|inout|do|while|if|then|else|elseif|select|case|call|subroutine|function|module|use|contains|print|write|read|open|close|return|exit|cycle|kind|pure|elemental|result|type|interface|only|save|status|file|stop"
  };
  function hl(code, lang) {
    const kw = new RegExp("^(" + (KW[lang] || KW.py) + ")$", lang === "f90" ? "i" : "");
    const cm = lang === "f90" ? "![^\\n]*" : "#[^\\n]*";
    const re = new RegExp(`(${cm})|("""[\\s\\S]*?"""|'''[\\s\\S]*?'''|"(?:\\\\.|[^"\\\\\\n])*"|'(?:\\\\.|[^'\\\\\\n])*')|(\\b\\d+\\.?\\d*(?:[eEdD][+-]?\\d+)?(?:_dp)?\\b)|([A-Za-z_][A-Za-z0-9_]*)(\\s*\\()?`, "g");
    let out = "", last = 0, m;
    while ((m = re.exec(code))) {
      out += esc(code.slice(last, m.index)); last = re.lastIndex;
      if (m[1]) out += `<span class="hl-c">${esc(m[1])}</span>`;
      else if (m[2]) out += `<span class="hl-s">${esc(m[2])}</span>`;
      else if (m[3]) out += `<span class="hl-n">${esc(m[3])}</span>`;
      else if (m[4]) { const w = m[4]; out += kw.test(w) ? `<span class="hl-k">${w}</span>` : m[5] ? `<span class="hl-f">${w}</span>` : w; if (m[5]) out += esc(m[5]); }
    }
    return out + esc(code.slice(last));
  }
  function initCode(root) {
    $$(".codebox:not([data-done])", root).forEach((cb) => {
      cb.dataset.done = 1;
      $$("pre", cb).forEach((p) => { const c = p.querySelector("code") || p; c.innerHTML = hl(c.textContent, p.dataset.lang); });
    });
  }
  document.addEventListener("click", (e) => {
    const b = e.target.closest(".cb-tabs button"); if (!b) return;
    const cb = b.closest(".codebox");
    if (b.classList.contains("copy")) {
      const p = $$("pre", cb).find((x) => !x.hidden); const txt = p ? p.textContent : "";
      (navigator.clipboard ? navigator.clipboard.writeText(txt) : Promise.reject()).then(() => { b.textContent = "copied ✓"; setTimeout(() => (b.textContent = "copy"), 1400); }).catch(() => { b.textContent = "select & copy"; });
      return;
    }
    $$(".cb-tabs button[data-lang]", cb).forEach((x) => x.classList.toggle("on", x === b));
    $$("pre", cb).forEach((p) => (p.hidden = p.dataset.lang !== b.dataset.lang));
    const fn = $(".fn", cb); if (fn) fn.textContent = b.dataset.file || "";
  });

  /* ------------------------------------------------------------ exercises */
  function parseNum(s) {
    s = String(s).trim().replace(/,/g, ".").replace(/×\s*10\^?/g, "e").replace(/\s+/g, "");
    if (!/^[-+*/().0-9eE^]+$/.test(s)) return NaN;
    try { return Function(`"use strict";return (${s.replace(/\^/g, "**")})`)(); } catch (e) { return NaN; }
  }
  const solved = store.get("solved", {});
  function checkEx(ex) {
    const inp = $("input", ex), res = $(".ex-res", ex), v = parseNum(inp.value), a = +ex.dataset.ans;
    const tol = ex.dataset.tol ? +ex.dataset.tol : Math.max(1e-9, 0.01 * Math.abs(a));
    if (!isFinite(v)) { res.className = "ex-res no"; res.textContent = "type a number (fractions like 1/3 are fine)"; return; }
    const ok = Math.abs(v - a) <= tol;
    res.className = "ex-res " + (ok ? "ok" : "no");
    res.textContent = ok ? "✓ correct" : v > a ? "✗ too high — try again or open the hint" : "✗ too low — try again or open the hint";
    ex.classList.toggle("solved", ok);
    if (ok && ex.id) { solved[ex.id] = 1; store.set("solved", solved); }
  }
  document.addEventListener("click", (e) => { const b = e.target.closest(".ex .check"); if (b) checkEx(b.closest(".ex")); });
  document.addEventListener("keydown", (e) => { if (e.key === "Enter" && e.target.closest && e.target.closest(".ex-in")) checkEx(e.target.closest(".ex")); });
  $$(".ex[id]").forEach((ex) => { if (solved[ex.id]) ex.classList.add("solved"); });

  /* ------------------------------------------------------------ lesson player */
  const sections = $$("section.lesson");
  let cur = -1;
  const W = window.WIDGETS || {};
  function initWidgets(sec) {
    $$("[data-widget]", sec).forEach((w) => {
      if (w._inited) { w._redraw && w._redraw(); return; }
      w._inited = true; const f = W[w.dataset.widget];
      if (f) { try { f(w); } catch (err) { console.error("widget", w.dataset.widget, err); w.insertAdjacentHTML("beforeend", `<div class="note">widget error: ${esc(err.message)}</div>`); } }
    });
  }
  function show(i, scroll = true) {
    if (!sections.length) return;
    i = Math.max(0, Math.min(sections.length - 1, i)); if (i === cur) return; cur = i;
    const s = sections[i];
    sections.forEach((x, k) => x.classList.toggle("active", k === i));
    const d = s.dataset;
    $("#crumb-sub").textContent = `Lesson ${i + 1} of ${sections.length}`;
    $("#title").textContent = d.title; $("#subtitle").innerHTML = d.sub || "";
    $("#b-time").lastChild.textContent = " " + (d.time || "15 min");
    $("#b-level").lastChild.textContent = " " + (d.level || "Beginner");
    $("#b-steps").lastChild.textContent = " " + (($$(".prose h2", s).length || 1) === 1 ? "1 part" : $$(".prose h2", s).length + " parts");
    // right panel
    const side = $(".side-src", s);
    $("#r-step").textContent = `STEP ${i + 1} / ${sections.length}`;
    const rb = $("#r-badge"); rb.textContent = d.badge || "CONCEPT"; rb.className = "badge " + ({ CONCEPT: "g", MATH: "p", CODE: "b", PROJECT: "y", PROBLEM: "k" }[d.badge] || "g");
    $("#r-title").textContent = d.title;
    $("#r-body").innerHTML = side ? (side.querySelector(".s-text") || side).innerHTML : "";
    const key = side && side.querySelector(".s-key");
    $("#r-key").style.display = key ? "" : "none"; if (key) $("#r-key-t").innerHTML = key.innerHTML;
    const learn = side ? $$(".s-learn li", side).map((l) => l.innerHTML) : [];
    const ck = store.get("learn", {}), kbase = `${UNIT}.${i}`;
    $("#r-learn").innerHTML = learn.map((t, k) => `<li data-k="${kbase}.${k}" class="${ck[kbase + "." + k] ? "ck" : ""}"><i>${ck[kbase + "." + k] ? "✓" : ""}</i><span>${t}</span></li>`).join("");
    $("#r-back").disabled = i === 0; $("#r-back").style.opacity = i === 0 ? 0.5 : 1;
    const nx = sections[i + 1]; $("#r-next-t").textContent = nx ? "Next: " + nx.dataset.short : "Back to course map";
    // stepper + sidebar
    $$(".dot").forEach((dt, k) => { dt.classList.toggle("on", k === i); dt.classList.toggle("seen", (seen[UNIT] || []).includes(k)); });
    $("#stepper-n").textContent = `Step ${i + 1} of ${sections.length}`;
    markSeen(UNIT, i); renderSidebar(sections.map((x) => x.dataset.title));
    $$(".sub").forEach((a) => a.classList.toggle("on", a.closest(".subs").previousElementSibling.dataset.unit === UNIT && +a.dataset.i === i));
    history.replaceState(null, "", "#s" + (i + 1));
    renderMath(s); initCode(s); initWidgets(s); renderMath($(".right"));
    if (scroll) window.scrollTo({ top: 0, behavior: "smooth" });
    body.classList.remove("nav");
  }
  if (sections.length) {
    const dots = $(".dots");
    dots.innerHTML = sections.map((s, k) => `<button class="dot" data-i="${k}"><i>${k + 1}</i>${s.dataset.short}</button>`).join("");
    dots.addEventListener("click", (e) => { const d = e.target.closest(".dot"); if (d) { setView("learn"); show(+d.dataset.i); } });
    $("#r-back").addEventListener("click", () => show(cur - 1));
    $("#r-next").addEventListener("click", () => (cur < sections.length - 1 ? show(cur + 1) : (location.href = ROOT + "index.html")));
    $("#r-learn").addEventListener("click", (e) => { const li = e.target.closest("li"); if (!li) return; const ck = store.get("learn", {}); ck[li.dataset.k] = !ck[li.dataset.k]; store.set("learn", ck); li.classList.toggle("ck"); li.querySelector("i").textContent = ck[li.dataset.k] ? "✓" : ""; });
    const fromHash = () => { const m = location.hash.match(/^#s(\d+)/); return m ? +m[1] - 1 : 0; };
    window.addEventListener("hashchange", () => { setView("learn"); show(fromHash()); });
    renderSidebar(sections.map((x) => x.dataset.title));
    show(fromHash(), false);
  } else renderSidebar();

  /* ------------------------------------------------------------ views */
  const VIEWS = { learn: null, practice: ".ex", code: ".codebox, .task", sheet: ".formula" };
  function buildView(name) {
    const box = $(".v-" + name); if (!box || box.dataset.built) return; box.dataset.built = 1;
    sections.forEach((s, k) => {
      const items = $$(VIEWS[name], s).filter((x) => !x.closest(".tip")); if (!items.length) return;
      box.insertAdjacentHTML("beforeend", `<div class="view-from">LESSON ${k + 1} · ${s.dataset.title.toUpperCase()}</div>`);
      const wrap = document.createElement("div"); if (name === "sheet") wrap.className = "sheet-grid";
      items.forEach((it) => { const c = it.cloneNode(true); c.removeAttribute("id"); $$("[data-done]", c).forEach((x) => x.removeAttribute("data-done")); if (c.matches(".tex")) c.removeAttribute("data-done"); wrap.append(c); });
      box.append(wrap);
    });
    // re-render maths in clones from their saved source
    $$(".tex", box).forEach((e) => { if (e.dataset.src) { e.textContent = e.dataset.src; e.removeAttribute("data-done"); } });
    renderMath(box); initCode(box);
  }
  function setView(v) {
    body.dataset.view = v; $$(".tab[data-view]").forEach((t) => t.classList.toggle("on", t.dataset.view === v));
    if (v !== "learn") { sections.forEach((s) => { renderMath(s); }); buildView(v); window.scrollTo({ top: 0 }); }
  }
  $$(".tab[data-view]").forEach((t) => t.addEventListener("click", (e) => { if (!sections.length) return; e.preventDefault(); setView(t.dataset.view); }));

  /* ------------------------------------------------------------ search */
  const ov = document.createElement("div"); ov.className = "overlay";
  ov.innerHTML = `<div class="sbox"><input placeholder="Search lessons, formulas, symbols…"><div class="sres"></div></div>`; body.append(ov);
  const sin = $("input", ov), sres = $(".sres", ov);
  function index() {
    const out = [];
    C.units.filter((u) => u.status === "ready").forEach((u) => u.subs.forEach((t, i) => out.push({ t, k: `${u.label || u.id} · lesson`, href: (u.id === UNIT ? "" : unitHref(u)) + `#s${i + 1}` })));
    sections.forEach((s, i) => {
      $$(".formula", s).forEach((f) => out.push({ t: f.dataset.name || "formula", k: "formula", href: `#s${i + 1}` }));
      $$(".prose h2", s).forEach((h) => out.push({ t: h.textContent.replace(/^(BASIC|CORE|EXPERT|CODE|PROJECT)/, ""), k: `section · ${i + 1}`, href: `#s${i + 1}` }));
    });
    Object.entries(SYM).forEach(([k, s]) => out.push({ t: s.n.replace(/<[^>]+>/g, ""), k: "symbol", href: null, sym: k }));
    return out;
  }
  let IDX = null, sel = 0;
  function runSearch() {
    IDX = IDX || index(); const q = sin.value.toLowerCase().trim();
    const r = (q ? IDX.filter((x) => x.t.toLowerCase().includes(q)) : IDX.slice(0, 30)).slice(0, 40);
    sres.innerHTML = r.map((x, i) => `<a href="${x.href || "#"}" data-sym="${x.sym || ""}" class="${i === sel ? "sel" : ""}">${esc(x.t)}<small>${x.k}</small></a>`).join("") || `<a>No match</a>`;
  }
  const openS = () => { ov.classList.add("show"); sin.value = ""; sel = 0; runSearch(); setTimeout(() => sin.focus(), 10); };
  $$("[data-act=search]").forEach((b) => b.addEventListener("click", openS));
  sin.addEventListener("input", () => { sel = 0; runSearch(); });
  ov.addEventListener("click", (e) => {
    if (e.target === ov) return ov.classList.remove("show");
    const a = e.target.closest("a"); if (!a) return;
    if (a.dataset.sym) { e.preventDefault(); const r = a.getBoundingClientRect(); ov.classList.remove("show"); pinned = true; tip.classList.add("pin"); showTip(a.dataset.sym, innerWidth / 2 - 200, r.top); return; }
    ov.classList.remove("show");
  });
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); openS(); return; }
    if (ov.classList.contains("show")) {
      const as = $$("a", sres);
      if (e.key === "Escape") ov.classList.remove("show");
      if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); sel = (sel + (e.key === "ArrowDown" ? 1 : -1) + as.length) % as.length; as.forEach((a, i) => a.classList.toggle("sel", i === sel)); as[sel] && as[sel].scrollIntoView({ block: "nearest" }); }
      if (e.key === "Enter" && as[sel]) as[sel].click();
      return;
    }
    if (e.target.closest && e.target.closest("input, textarea, select")) return;
    if (!sections.length || body.dataset.view !== "learn") return;
    if (e.key === "ArrowRight") show(cur + 1); else if (e.key === "ArrowLeft") show(cur - 1);
  });

  /* ------------------------------------------------------------ redraw on resize / theme */
  let rt; const redraw = () => $$("[data-widget]").forEach((w) => w._redraw && w.offsetParent && w._redraw());
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(redraw, 120); });
  window.addEventListener("themechange", redraw);
  renderMath(document.body); initCode(document.body);
  window.APP = { show, hl, renderMath, store };
})();
