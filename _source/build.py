"""Assemble the study site: shell + lesson bodies + code files (HTML-escaped)."""
import html
import os
import re
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SITE = os.environ.get("SITE_DIR") or (os.path.dirname(SRC_DIR) if os.path.basename(SRC_DIR) == "_source" else "/home/claude/site")
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

ICONS = {
    "learn": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 5.5C5 4 8.5 4 12 6c3.5-2 7-2 10-.5V19c-3-1.5-6.5-1.5-10 .5-3.5-2-7-2-10-.5z"/><path d="M12 6v13.5"/></svg>',
    "practice": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M4.9 19.1 7 17M17 7l2.1-2.1"/></svg>',
    "code": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2.5" y="4" width="19" height="16" rx="2"/><path d="M2.5 8.5h19M7 13l-2 2 2 2M11 17h5"/></svg>',
    "sheet": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><path d="M14 17.5h7M17.5 14v7"/></svg>',
    "search": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></svg>',
    "moon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/></svg>',
    "menu": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h16M4 18h16"/></svg>',
    "clock": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
    "level": '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="3" y="15" width="3.2" height="6" rx="1"/><rect x="8.5" y="11" width="3.2" height="10" rx="1"/><rect x="14" y="7" width="3.2" height="14" rx="1"/><rect x="19.5" y="3" width="2.5" height="18" rx="1" opacity=".35"/></svg>',
    "steps": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M9 6h12M9 12h12M9 18h12M4 6h.01M4 12h.01M4 18h.01"/></svg>',
    "arrowr": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><path d="M5 12h14M13 6l6 6-6 6"/></svg>',
    "arrowl": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><path d="M19 12H5M11 6l-6 6 6 6"/></svg>',
    "spark": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M12 3c.5 4 2 5.5 6 6-4 .5-5.5 2-6 6-.5-4-2-5.5-6-6 4-.5 5.5-2 6-6zM18.5 15.5c.2 1.6.9 2.3 2.5 2.5-1.6.2-2.3.9-2.5 2.5-.2-1.6-.9-2.3-2.5-2.5 1.6-.2 2.3-.9 2.5-2.5z"/></svg>',
}

LOGO = '''<svg viewBox="0 0 40 40"><circle cx="20" cy="20" r="15.5" fill="none" stroke="currentColor" stroke-width="2.5"/>
<ellipse cx="20" cy="20" rx="19" ry="6.5" fill="none" stroke="currentColor" stroke-width="2.2" transform="rotate(-24 20 20)"/>
<circle cx="20" cy="20" r="6" fill="#cfe3b2" stroke="currentColor" stroke-width="2.2"/><circle cx="35" cy="11.5" r="2.6" fill="#e6824c" stroke="currentColor" stroke-width="1.5"/></svg>'''


def topbar(root, active="learn"):
    tabs = [("learn", "Learn"), ("practice", "Practice"), ("code", "Code Lab"), ("sheet", "Cheatsheet")]
    t = "".join(f'<a class="tab{" on" if k == active else ""}" data-view="{k}" href="{root}index.html">{ICONS[k]}<span>{n}</span></a>'
                for k, n in tabs)
    return f'''<header class="topbar">
  <button class="iconbtn burger" data-act="burger" aria-label="menu">{ICONS["menu"]}</button>
  <a class="brand" href="{root}index.html">{LOGO}<span>Comp <mark>Astro</mark></span></a>
  <nav class="tabs">{t}</nav>
  <div class="search" data-act="search">{ICONS["search"]}<span>Search lessons &amp; formulas…</span><kbd>Ctrl K</kbd></div>
  <button class="iconbtn" data-act="theme" aria-label="dark mode">{ICONS["moon"]}</button>
  <div class="credit"><div class="av">∑</div><div>open study guide<b>numerical methods</b></div></div>
</header>'''


def head(title, root):
    return f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 40'%3E%3Ccircle cx='20' cy='20' r='14' fill='%23cfe3b2' stroke='%231d1b18' stroke-width='3'/%3E%3C/svg%3E">
<link rel="stylesheet" href="{root}assets/katex/katex.min.css">
<link rel="stylesheet" href="{root}assets/css/style.css">
<script>try{{document.documentElement.dataset.theme=JSON.parse(localStorage.getItem("cma.theme"))||"light"}}catch(e){{}}</script>
</head>'''


# ------------------------------------------------------------------ macros
def esc(s):
    return html.escape(s, quote=False)


def dedent(s):
    lines = s.strip("\n").split("\n")
    ind = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
    return "\n".join(l[ind:] for l in lines).rstrip()


def codebox(blocks, files=None):
    """blocks: list of (lang, code, filename)."""
    tabs = "".join(f'<button data-lang="{l}" data-file="{fn or ""}" class="{"on" if i == 0 else ""}">{ {"py": "Python", "f90": "Fortran"}.get(l, l)}</button>'
                   for i, (l, c, fn) in enumerate(blocks))
    first_fn = blocks[0][2] or ""
    pres = "".join(f'<pre data-lang="{l}"{"" if i == 0 else " hidden"}><code>{esc(c)}</code></pre>' for i, (l, c, fn) in enumerate(blocks))
    return f'<div class="codebox"><div class="cb-tabs">{tabs}<span class="fn">{first_fn}</span><button class="copy">copy</button></div>{pres}</div>'


def expand(body, codedir):
    def read(fn):
        with open(os.path.join(codedir, fn)) as f:
            return f.read().rstrip()

    def x_code(m):
        return make_code(m.group(1), m.group(2))

    def make_code(attrs, inner):
        blocks = []
        for lang in ("py", "f90"):
            fm = re.search(rf'{lang}="([^"]+)"', attrs)
            if fm:
                blocks.append((lang, read(fm.group(1)), fm.group(1)))
        for lm in re.finditer(r"<(py|f90)>(.*?)</\1>", inner, re.S):
            blocks.append((lm.group(1), dedent(lm.group(2)), None))
        return codebox(blocks)

    def x_f(m):
        attrs, inner = m.group(1), m.group(2)
        name = re.search(r'name="([^"]*)"', attrs).group(1)
        tag = (re.search(r'tag="([^"]*)"', attrs) or [None, "FORMULA"])[1]
        note = re.search(r"<note>(.*?)</note>", inner, re.S)
        blocks = [(lm.group(1), dedent(lm.group(2)), None) for lm in re.finditer(r"<(py|f90)>(.*?)</\1>", inner, re.S)]
        tex = re.sub(r"<(note|py|f90)>.*?</\1>", "", inner, flags=re.S).strip()
        out = (f'<div class="formula" data-name="{html.escape(name)}"><div class="f-head"><span class="t">{tag}</span>{name}'
               f'<span class="hint">hover the symbols ↓</span></div><div class="tex">{tex}</div>')
        if note:
            out += f'<div class="f-note">{note.group(1).strip()}</div>'
        if blocks:
            out += codebox(blocks)
        return out + "</div>"

    def x_hsi(m):
        return (f'<a class="hsi-link" href="../Hyperspectral%20Project/index.html#{m.group(1)}">'
                '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/>'
                '<path d="M3 9h18M3 15h18M9 3v18M15 3v18"/></svg>'
                f'<span><small>RELATED · HYPERSPECTRAL PROJECT</small>{m.group(2).strip()}</span>'
                '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><path d="M5 12h14M13 6l6 6-6 6"/></svg></a>')

    def x_ex(m):
        attrs, inner = m.group(1), m.group(2)
        g = lambda k, d="": (re.search(rf'{k}="([^"]*)"', attrs) or [None, d])[1]
        hint = re.search(r"<hint>(.*?)</hint>", inner, re.S)
        sol = re.search(r"<sol>(.*?)</sol>", inner, re.S)
        q = re.sub(r"<(hint|sol)>.*?</\1>", "", inner, flags=re.S).strip()
        tol = f' data-tol="{g("tol")}"' if g("tol") else ""
        out = (f'<div class="ex" id="{g("id")}" data-ans="{g("ans")}"{tol}><div class="ex-h">EXERCISE <span class="lvl">{g("lvl", "Basic")}</span></div>'
               f'<div class="ex-q">{q}</div><div class="ex-in"><input placeholder="your answer"> <button class="btn go check">Check</button><span class="ex-res"></span></div>')
        if hint:
            out += f"<details><summary>Hint</summary>{hint.group(1).strip()}</details>"
        if sol:
            out += f"<details><summary>Full solution</summary>{sol.group(1).strip()}</details>"
        return out + "</div>"

    body = re.sub(r"<x-code([^>]*?)/>", lambda m: make_code(m.group(1), ""), body)
    body = re.sub(r"<x-code([^>]*)>(.*?)</x-code>", x_code, body, flags=re.S)
    body = re.sub(r"<x-f([^>]*)>(.*?)</x-f>", x_f, body, flags=re.S)
    body = re.sub(r'<x-hsi to="([^"]+)">(.*?)</x-hsi>', x_hsi, body, flags=re.S)
    def x_back(m):
        return (f'<a class="hsi-link back" href="{m.group(1)}">'
                '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><path d="M19 12H5M11 6l-6 6 6 6"/></svg>'
                f'<span><small>THE THEORY · BACK TO THE LESSON</small>{m.group(2).strip()}</span>'
                '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V3H6.5A2.5 2.5 0 0 0 4 5.5z"/></svg></a>')
    body = re.sub(r'<x-back href="([^"]+)">(.*?)</x-back>', x_back, body, flags=re.S)
    body = re.sub(r"<x-ex([^>]*)>(.*?)</x-ex>", x_ex, body, flags=re.S)
    return body


def lesson_page(unit_id, title, body_file, js_file, folder, prefix=None):
    root = "../"
    prefix = prefix if prefix is not None else unit_id + ". "
    with open(os.path.join(SRC, body_file)) as f:
        cd = os.path.join(SITE, folder, "code")
        body = expand(f.read(), cd if os.path.isdir(cd) else os.path.join(SITE, folder))
    page = head(f"{prefix}{title} · Comp Astro", root) + f'''
<body data-unit="{unit_id}" data-root="{root}">
{topbar(root)}
<div class="layout">
  <aside class="sidebar"><nav class="tree"></nav><div class="progress"></div></aside>
  <main class="main">
    <div class="stepper learnonly"><div class="stepper-h">IN THIS LESSON <small id="stepper-n"></small></div><div class="dots"></div></div>
    <div class="card">
      <div class="crumbs"><span>{prefix}{title}</span><span class="sep">/</span><span id="crumb-sub"></span></div>
      <div class="learnonly">
        <div class="titlebar"><h1 id="title"></h1>
          <div class="badges"><span class="badge y" id="b-time">{ICONS["clock"]} </span><span class="badge g" id="b-level">{ICONS["level"]} </span><span class="badge p" id="b-steps">{ICONS["steps"]} </span></div>
        </div>
        <p class="subtitle" id="subtitle"></p>
        {body}
      </div>
      <div class="view v-practice"><h1 style="margin:6px 0 2px">Practice</h1><p class="subtitle">Every exercise of this unit in one place. Answers are checked to about 1%; fractions like 1/3 are accepted.</p></div>
      <div class="view v-code"><h1 style="margin:6px 0 2px">Code Lab</h1><p class="subtitle">Every Python and Fortran snippet of this unit, plus the hyperspectral project tasks. The full files are in the <code>code</code> folder next to this page.</p></div>
      <div class="view v-sheet"><h1 style="margin:6px 0 2px">Cheatsheet</h1><p class="subtitle">All the formulas of this unit. Hover any symbol for its name, meaning and code.</p></div>
    </div>
  </main>
  <aside class="right learnonly">
    <div class="rcard">
      <div class="rtop"><span id="r-step"></span><span class="badge" id="r-badge"></span></div>
      <h2 id="r-title"></h2>
      <div id="r-body"></div>
      <div class="keyidea" id="r-key">{ICONS["spark"]}<b>KEY IDEA</b><span id="r-key-t"></span></div>
      <div class="navbtns"><button class="btn back" id="r-back">{ICONS["arrowl"]} Back</button><button class="btn next" id="r-next"><span id="r-next-t">Next</span>{ICONS["arrowr"]}</button></div>
      <div class="keys"><span class="kbd">←</span><span class="kbd">→</span> step · <span class="kbd">Ctrl K</span> search · hover symbols</div>
    </div>
    <div class="rcard"><div class="rtop">YOU'LL LEARN</div><ul class="learn" id="r-learn"></ul></div>
  </aside>
</div>
<script src="{root}assets/katex/katex.min.js"></script>
<script src="{root}assets/katex/auto-render.min.js"></script>
<script src="{root}assets/js/course.js"></script>
<script src="{root}assets/js/plot.js"></script>
<script src="{root}assets/js/symbols-common.js"></script>
<script src="{js_file}"></script>
<script src="{root}assets/js/app.js"></script>
</body></html>'''
    with open(os.path.join(SITE, folder, "index.html"), "w") as f:
        f.write(page)
    print("wrote", folder, len(page) // 1024, "KB")


def home_page():
    root = "./"
    with open(os.path.join(SRC, "home.html")) as f:
        body = f.read()
    page = head("Comp Astro · Computational Methods in Astrophysics", root) + f'''
<body data-root="{root}">
{topbar(root)}
<div class="layout home">
  <aside class="sidebar"><nav class="tree"></nav><div class="progress"></div></aside>
  <main class="main">{body}</main>
</div>
<script src="{root}assets/katex/katex.min.js"></script>
<script src="{root}assets/katex/auto-render.min.js"></script>
<script src="{root}assets/js/course.js"></script>
<script src="{root}assets/js/plot.js"></script>
<script src="{root}assets/js/symbols-common.js"></script>
<script src="{root}assets/js/home.js"></script>
<script src="{root}assets/js/app.js"></script>
</body></html>'''
    with open(os.path.join(SITE, "index.html"), "w") as f:
        f.write(page)
    print("wrote index.html")


PAGES = {
    "04": ("Direct Linear Solvers", "content04.html", "lesson04.js", "04. Direct Linear Solvers", None),
    "05": ("Conditioning, EVD and SVD", "content05.html", "lesson05.js", "05. Conditioning and SVD", None),
    "06": ("Sparse and Iterative Solvers", "content06.html", "lesson06.js", "06. Sparse and Iterative Solvers", None),
    "07": ("Discrete Signal Analysis", "content07.html", "lesson07.js", "07. Discrete Signal Analysis", None),
    "08": ("The FFT", "content08.html", "lesson08.js", "08. The FFT", None),
    "09": ("Convolution and Correlation", "content09.html", "lesson09.js", "09. Convolution and Correlation", None),
    "10": ("Spectral Estimation", "content10.html", "lesson10.js", "10. Spectral Estimation", None),
    "L1": ("Discrete signal laboratory", "contentL1.html", "lab1.js", "Lab 1", "Lab 1 · "),
    "12": ("Bayesian Inference", "content12.html", "lesson12.js", "12. Bayesian Inference", None),
    "13": ("Monte Carlo Methods", "content13.html", "lesson13.js", "13. Monte Carlo Methods", None),
    "HSI": ("Hyperspectral Project", "contentHSI.html", "hsi.js", "Hyperspectral Project", "Project · "),
}

if __name__ == "__main__":
    which = sys.argv[1:] or ["home"] + list(PAGES)
    if "home" in which:
        home_page()
    for k in which:
        if k in PAGES and os.path.exists(os.path.join(SRC, PAGES[k][1])):
            t, c, j, fo, pre = PAGES[k]
            lesson_page(k, t, c, j, fo, pre)
