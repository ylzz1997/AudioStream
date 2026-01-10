/* global window, document */

(() => {
  const STORE_KEY = "audiostream_docs_state_v1";
  const OVERVIEW_ID = "__overview__";

  const $ = (sel) => document.querySelector(sel);
  const el = (tag, attrs = {}, children = []) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k === "html") node.innerHTML = v;
      else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
      else if (v === true) node.setAttribute(k, "");
      else if (v != null) node.setAttribute(k, String(v));
    }
    for (const c of children) node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    return node;
  };

  const normalize = (s) => String(s || "").trim().toLowerCase();

  function loadState() {
    try {
      const raw = localStorage.getItem(STORE_KEY);
      if (!raw) return null;
      return JSON.parse(raw);
    } catch {
      return null;
    }
  }

  function saveState(state) {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(state));
    } catch {
      // ignore
    }
  }

  function getDocs() {
    const data = window.AUDIOSTREAM_DOCS;
    if (!data || typeof data !== "object") {
      throw new Error("未找到 window.AUDIOSTREAM_DOCS，请检查 document/data.js 是否正确加载。");
    }
    return data;
  }

  function findFirstItem(langData) {
    for (const g of langData.groups || []) {
      const first = (g.items || [])[0];
      if (first) return first;
    }
    return null;
  }

  function findItem(langData, id) {
    for (const g of langData.groups || []) {
      for (const it of g.items || []) {
        if (it.id === id) return it;
      }
    }
    return null;
  }

  function flattenItems(langData) {
    const out = [];
    for (const g of langData.groups || []) {
      for (const it of g.items || []) out.push({ groupTitle: g.title, ...it });
    }
    return out;
  }

  function renderHero(hero, langLabel) {
    const chips = (hero.chips || [])
      .map((c) => `<a class="chip" href="${c.href}" target="_blank" rel="noreferrer">${c.text}</a>`)
      .join("");
    return `
      <div class="hero">
        <div class="hero__kicker">${hero.kicker || langLabel}</div>
        <h1 class="hero__title">${hero.title || `${langLabel} 文档`}</h1>
        <p class="hero__desc">${hero.desc || ""}</p>
        ${chips ? `<div class="chips">${chips}</div>` : ""}
      </div>
    `;
  }

  function trimOuterBlankLines(text) {
    // 只去掉代码块“前后”的空行（通常来自模板字符串缩进/换行），不改动中间内容
    // - 允许空行上带空格缩进：也一并删掉
    let s = String(text || "").replace(/\r\n/g, "\n");
    // 去掉开头连续的“空白行”
    while (/^\s*\n/.test(s)) s = s.replace(/^\s*\n/, "");
    // 去掉结尾连续的“空白行”
    while (/\n\s*$/.test(s)) s = s.replace(/\n\s*$/, "");
    return s;
  }

  function enhanceCodeBlocks(container) {
    const pres = container.querySelectorAll("pre");
    pres.forEach((pre) => {
      if (pre.querySelector(".copybtn")) return;
      const code = pre.querySelector("code");
      if (!code) return;

      // 1) 自动去掉 code 内容前后的多余换行（空白行）
      const before = code.textContent || "";
      const after = trimOuterBlankLines(before);
      if (after !== before) code.textContent = after;

      // 2) 语法高亮（如果 highlight.js 已加载）
      //    注：使用 textContent 设置后再高亮，避免把高亮标签写进复制内容。
      if (window.hljs && typeof window.hljs.highlightElement === "function") {
        try {
          window.hljs.highlightElement(code);
        } catch {
          // ignore highlight errors
        }
      }

      const btn = el("button", { class: "copybtn", type: "button" }, ["复制"]);
      btn.addEventListener("click", async () => {
        const text = code.innerText || "";
        try {
          await navigator.clipboard.writeText(text);
          btn.textContent = "已复制";
          setTimeout(() => (btn.textContent = "复制"), 900);
        } catch {
          btn.textContent = "复制失败";
          setTimeout(() => (btn.textContent = "复制"), 900);
        }
      });
      pre.appendChild(btn);
    });
  }

  function setMainContent(html) {
    const doc = $("#doc");
    doc.innerHTML = html;
    enhanceCodeBlocks(doc);
    $("#main").focus({ preventScroll: true });
  }

  function setNavOpen(open) {
    const nav = $("#nav");
    const btn = $("#btnToggleNav");
    if (!nav || !btn) return;
    nav.classList.toggle("nav--open", open);
    btn.setAttribute("aria-expanded", open ? "true" : "false");
  }

  function isMobileNavMode() {
    return window.matchMedia && window.matchMedia("(max-width: 980px)").matches;
  }

  function buildNavList(langKey, activeId, query) {
    const docs = getDocs();
    const lang = docs[langKey];
    const navList = $("#navList");

    navList.innerHTML = "";

    const q = normalize(query);
    let matchCount = 0;

    // 概览（hero）作为一个条目存在，避免“固定在最上面”的观感
    {
      const overviewTitle = `${lang.label || langKey} 概览`;
      const overviewDesc = "项目简介 / 入口链接";
      const hay = normalize(`${overviewTitle} ${overviewDesc} ${OVERVIEW_ID}`);
      const hit = !q || hay.includes(q);
      if (hit) {
        navList.appendChild(el("div", { class: "navgroup__title" }, ["概览"]));
        const groupWrap = el("div", { class: "navgroup" });
        matchCount += 1;
        groupWrap.appendChild(
          el(
            "button",
            {
              class: `navitem ${OVERVIEW_ID === activeId ? "navitem--active" : ""}`,
              type: "button",
              "data-id": OVERVIEW_ID,
              onclick: () => {
                selectItem(langKey, OVERVIEW_ID, { pushHash: true });
                if (isMobileNavMode()) setNavOpen(false);
              },
            },
            [el("div", { class: "navitem__title" }, [overviewTitle]), el("div", { class: "navitem__desc" }, [overviewDesc])],
          ),
        );
        navList.appendChild(groupWrap);
      }
    }

    for (const g of lang.groups || []) {
      const groupItems = (g.items || []).filter((it) => {
        if (!q) return true;
        const hay = normalize(`${it.title} ${it.desc} ${it.id}`);
        return hay.includes(q);
      });
      if (groupItems.length === 0) continue;

      navList.appendChild(el("div", { class: "navgroup__title" }, [g.title]));

      const groupWrap = el("div", { class: "navgroup" });
      for (const it of groupItems) {
        matchCount += 1;
        const btn = el(
          "button",
          {
            class: `navitem ${it.id === activeId ? "navitem--active" : ""}`,
            type: "button",
            "data-id": it.id,
            onclick: () => {
              selectItem(langKey, it.id, { pushHash: true });
              if (isMobileNavMode()) setNavOpen(false);
            },
          },
          [
            el("div", { class: "navitem__title" }, [it.title]),
            el("div", { class: "navitem__desc" }, [it.desc || ""]),
          ],
        );
        groupWrap.appendChild(btn);
      }
      navList.appendChild(groupWrap);
    }

    if (matchCount === 0) {
      navList.appendChild(el("div", { class: "empty" }, ["没有匹配的条目。"]));
    }
  }

  function setTabs(langKey) {
    const tabPy = $("#tabPython");
    const tabRs = $("#tabRust");
    const isPy = langKey === "python";
    tabPy.setAttribute("aria-selected", isPy ? "true" : "false");
    tabRs.setAttribute("aria-selected", !isPy ? "true" : "false");
  }

  function selectLanguage(langKey, { keepItem = true } = {}) {
    const docs = getDocs();
    if (!docs[langKey]) langKey = "python";

    const state = loadState() || {};
    const prev = state[langKey] || {};
    let nextItemId = keepItem ? prev.activeId : null;

    const lang = docs[langKey];
    if (!nextItemId) nextItemId = OVERVIEW_ID;
    if (nextItemId !== OVERVIEW_ID && !findItem(lang, nextItemId)) nextItemId = OVERVIEW_ID;

    setTabs(langKey);
    $("#search").value = "";
    selectItem(langKey, nextItemId, { pushHash: false });
  }

  function selectItem(langKey, itemId, { pushHash } = {}) {
    const docs = getDocs();
    const lang = docs[langKey];
    const state = loadState() || {};

    if (itemId === OVERVIEW_ID) {
      setMainContent(renderHero(lang.hero || {}, lang.label || langKey));
      buildNavList(langKey, OVERVIEW_ID, $("#search").value || "");
      saveState({ ...state, activeLang: langKey, [langKey]: { ...(state[langKey] || {}), activeId: OVERVIEW_ID } });
    } else {
      const it = findItem(lang, itemId);
      if (!it) return;
      // 点击左侧条目：右侧只渲染条目正文（不再把“使用说明 hero”固定在顶部）
      setMainContent(`${it.body || ""}`);
      buildNavList(langKey, itemId, $("#search").value || "");
      saveState({ ...state, activeLang: langKey, [langKey]: { ...(state[langKey] || {}), activeId: itemId } });
    }

    if (pushHash) {
      const hash = `#${encodeURIComponent(langKey)}/${encodeURIComponent(itemId)}`;
      if (location.hash !== hash) history.pushState(null, "", hash);
    }
  }

  function parseHash() {
    const h = (location.hash || "").replace(/^#/, "");
    if (!h) return null;
    const [lang, id] = h.split("/");
    if (!lang || !id) return null;
    return { lang: decodeURIComponent(lang), id: decodeURIComponent(id) };
  }

  function initBackgroundCanvas() {
    const canvas = document.getElementById("bgCanvas");
    if (!(canvas instanceof HTMLCanvasElement)) return;

    const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const ctx = canvas.getContext("2d", { alpha: false, desynchronized: true });
    if (!ctx) return;

    // 用“低分辨率渲染 + CSS 拉伸”换性能（移动端更明显）
    // scale 越小越省，但细节越糊；0.55~0.75 通常比较平衡
    const computeScale = () => {
      const isSmall = window.matchMedia && window.matchMedia("(max-width: 980px)").matches;
      return isSmall ? 0.55 : 0.7;
    };

    const state = {
      w: 0,
      h: 0,
      scale: computeScale(),
      running: false,
      raf: 0,
      t0: 0,
      last: 0,
    };

    const blobs = [
      { hue: 278, a: 0.22, sx: 0.18, sy: 0.22, r: 0.62, sp: 0.00036, px: 1.3, py: 2.1 },
      { hue: 145, a: 0.18, sx: 0.82, sy: 0.20, r: 0.56, sp: 0.00033, px: 2.2, py: 1.7 },
      { hue: 210, a: 0.17, sx: 0.62, sy: 0.86, r: 0.60, sp: 0.00031, px: 1.6, py: 2.6 },
      { hue: 330, a: 0.14, sx: 0.25, sy: 0.78, r: 0.52, sp: 0.00034, px: 2.9, py: 1.2 },
      { hue: 50, a: 0.10, sx: 0.80, sy: 0.74, r: 0.48, sp: 0.00032, px: 1.1, py: 3.1 },
    ];

    const hsl = (h, s, l, a) => `hsla(${h}, ${s}%, ${l}%, ${a})`;

    function resize() {
      state.scale = computeScale();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.floor(window.innerWidth * dpr * state.scale));
      const h = Math.max(1, Math.floor(window.innerHeight * dpr * state.scale));
      if (w === state.w && h === state.h) return;
      state.w = w;
      state.h = h;
      canvas.width = w;
      canvas.height = h;
      // CSS 尺寸靠样式控制为 100vw/100vh，这里只管内部像素
    }

    function draw(now) {
      if (!state.running) return;

      // 限帧：移动端/低端机器更稳（大概 40fps 上限）
      if (state.last && now - state.last < 25) {
        state.raf = window.requestAnimationFrame(draw);
        return;
      }
      state.last = now;

      const t = now - state.t0;
      const w = state.w;
      const h = state.h;
      const minSide = Math.min(w, h);

      // 背景底色
      ctx.globalCompositeOperation = "source-over";
      ctx.fillStyle = "#0b1220";
      ctx.fillRect(0, 0, w, h);

      // 彩色光斑：用径向渐变（软边）而不是 blur filter，成本更低
      ctx.globalCompositeOperation = "lighter";
      for (const b of blobs) {
        const phase = t * b.sp;
        const x = (b.sx + 0.16 * Math.sin(phase * b.px + b.hue)) * w;
        const y = (b.sy + 0.14 * Math.cos(phase * b.py + b.hue * 0.7)) * h;
        const r = (b.r + 0.06 * Math.sin(phase * 1.3 + b.hue)) * minSide;
        const g = ctx.createRadialGradient(x, y, 0, x, y, r);
        g.addColorStop(0, hsl(b.hue, 92, 62, b.a));
        g.addColorStop(0.55, hsl(b.hue, 92, 52, b.a * 0.45));
        g.addColorStop(1, hsl(b.hue, 92, 45, 0));
        ctx.fillStyle = g;
        // 只绘制光斑范围，减少整屏 overdraw
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
      }

      // 轻微暗角，保证正文对比度
      ctx.globalCompositeOperation = "source-over";
      const vig = ctx.createRadialGradient(w * 0.5, h * 0.45, minSide * 0.25, w * 0.5, h * 0.5, minSide * 0.85);
      vig.addColorStop(0, "rgba(11, 18, 32, 0)");
      vig.addColorStop(1, "rgba(11, 18, 32, 0.55)");
      ctx.fillStyle = vig;
      ctx.fillRect(0, 0, w, h);

      state.raf = window.requestAnimationFrame(draw);
    }

    function start() {
      if (state.running) return;
      state.running = true;
      state.t0 = performance.now();
      state.last = 0;
      resize();
      state.raf = window.requestAnimationFrame(draw);
    }

    function stop() {
      state.running = false;
      if (state.raf) window.cancelAnimationFrame(state.raf);
      state.raf = 0;
    }

    // 初始渲染
    resize();
    if (reduceMotion) {
      state.running = true; // 复用 draw 逻辑画一帧
      state.t0 = performance.now();
      draw(performance.now());
      stop();
    } else {
      start();
    }

    window.addEventListener("resize", () => {
      resize();
      if (reduceMotion) {
        state.running = true;
        state.t0 = performance.now();
        draw(performance.now());
        stop();
      }
    });

    document.addEventListener("visibilitychange", () => {
      if (reduceMotion) return;
      if (document.hidden) stop();
      else start();
    });
  }

  function init() {
    const docs = getDocs();
    const tabPy = $("#tabPython");
    const tabRs = $("#tabRust");
    const search = $("#search");
    const btnToggleNav = $("#btnToggleNav");

    initBackgroundCanvas();

    tabPy.addEventListener("click", () => selectLanguage("python", { keepItem: true }));
    tabRs.addEventListener("click", () => selectLanguage("rust", { keepItem: true }));

    search.addEventListener("input", () => {
      const state = loadState() || {};
      const langKey = state.activeLang || "python";
      const activeId = (state[langKey] || {}).activeId || OVERVIEW_ID;
      buildNavList(langKey, activeId, search.value || "");
    });

    btnToggleNav.addEventListener("click", () => setNavOpen(!$("#nav").classList.contains("nav--open")));

    document.addEventListener("click", (e) => {
      if (!isMobileNavMode()) return;
      const nav = $("#nav");
      if (!nav.classList.contains("nav--open")) return;
      const path = e.composedPath ? e.composedPath() : [];
      const clickedInside = path.includes(nav) || path.includes(btnToggleNav);
      if (!clickedInside) setNavOpen(false);
    });

    window.addEventListener("hashchange", () => {
      const parsed = parseHash();
      if (!parsed) return;
      if (!docs[parsed.lang]) return;
      selectLanguage(parsed.lang, { keepItem: true });
      selectItem(parsed.lang, parsed.id, { pushHash: false });
    });

    // 初始：hash > localStorage > 默认 python
    const parsed = parseHash();
    const state = loadState() || {};
    const initialLang = parsed?.lang && docs[parsed.lang] ? parsed.lang : state.activeLang || "python";
    selectLanguage(initialLang, { keepItem: true });
    if (parsed?.id) {
      selectItem(initialLang, parsed.id, { pushHash: false });
    }
  }

  init();
})();


