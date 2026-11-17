/* BreastScan Twin — front-end (no framework, no build step) */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
  const fmt = (v) => {
    const a = Math.abs(v);
    if (a >= 1000) return v.toFixed(0);
    if (a >= 100) return v.toFixed(1);
    if (a >= 10) return v.toFixed(2);
    if (a >= 1) return v.toFixed(3);
    return v.toFixed(4);
  };
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* ignore */ } },
  };
  const api = async (url, opts, onBusy) => {
    for (let attempt = 0; ; attempt++) {
      let r;
      try { r = await fetch(url, opts); } catch { throw new Error("Can't reach the server. Check your connection and try again."); }
      const j = await r.json().catch(() => ({ error: r.status === 502 || r.status === 504 ? "The server is waking up or restarting. Try again in a minute." : "Server returned an invalid response." }));
      if (r.status === 503 && j.busy && attempt < 40) {  // another scan is running: wait and retry
        if (onBusy) onBusy(attempt);
        await new Promise((res) => setTimeout(res, (j.retry_after || 3) * 1000));
        continue;
      }
      if (!r.ok || j.error) throw new Error(j.error || r.statusText);
      return j;
    }
  };
  const toast = (msg) => {
    const t = $("#toast"); t.textContent = msg; t.classList.add("on");
    clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("on"), 2600);
  };

  const S = { schema: null, metrics: null, values: [], group: "mean", pred: null, population: null, preset: 0 };

  /* ---------------- theme ---------------- */
  const setTheme = (t) => { document.documentElement.dataset.theme = t; store.set("bst-theme", t); redrawMaps(); };
  setTheme(store.get("bst-theme") || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"));
  $("#themeBtn").addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));

  /* ---------------- router ---------------- */
  const pages = ["overview", "twin", "scan", "insights", "learn", "about"];
  function route() {
    const id = (location.hash || "#overview").slice(1);
    const page = pages.includes(id) ? id : "overview";
    pages.forEach((p) => { $("#page-" + p).hidden = p !== page; });
    $$(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.nav === page));
    window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
    if (page === "insights") animateAlgos();
    requestAnimationFrame(redrawMaps);
  }
  addEventListener("hashchange", route);
  document.addEventListener("click", (e) => {
    const t = e.target.closest("[data-kind][data-nav]"); if (t && typeof setKind === "function") setTimeout(() => setKind(t.dataset.kind, false), 0);
  });

  /* ---------------- hero cells ---------------- */
  (function heroCells() {
    const g = $(".orb .cells"); let html = "";
    const rnd = mulberry(11);
    for (let i = 0; i < 26; i++) {
      const a = rnd() * Math.PI * 2, r = 30 + rnd() * 130;
      const x = 200 + Math.cos(a) * r, y = 200 + Math.sin(a) * r, s = 4 + rnd() * 9;
      const m = rnd() < 0.3;
      html += `<circle class="cell" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${s.toFixed(1)}" style="animation-delay:${(-rnd() * 4).toFixed(2)}s;stroke:var(${m ? "--malignant" : "--benign"});fill:color-mix(in srgb,var(${m ? "--malignant" : "--benign"}) 25%,transparent)"/>`;
    }
    g.innerHTML = html;
  })();
  function mulberry(a) { return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

  /* ================= DIGITAL TWIN ================= */
  function buildPresets() {
    const box = $("#presets");
    box.innerHTML = S.schema.presets.map((p, i) =>
      `<button class="preset ${i === S.preset ? "on" : ""}" data-i="${i}"><b>${p.label}</b><span>${p.note}</span></button>`).join("");
    box.onclick = (e) => {
      const b = e.target.closest(".preset"); if (!b) return;
      S.preset = +b.dataset.i;
      $$(".preset", box).forEach((x) => x.classList.toggle("on", x === b));
      setValues(S.schema.presets[S.preset].values);
    };
  }
  function clearPreset() { S.preset = -1; $$(".preset").forEach((x) => x.classList.remove("on")); }

  function buildSliders() {
    const feats = S.schema.features.filter((f) => f.group === S.group);
    $("#sliders").innerHTML = feats.map((f) => {
      const span = f.max - f.min;
      const pb = ((f.benign_mean - f.min) / span) * 100, pm = ((f.malignant_mean - f.min) / span) * 100;
      return `<div class="frow" data-i="${f.index}">
        <div class="fl">${f.label} <span class="info" data-tip="${f.description} Benign avg ${fmt(f.benign_mean)}, malignant avg ${fmt(f.malignant_mean)}.">i</span></div>
        <input type="number" step="${f.step}" min="${f.min}" max="${f.max}" aria-label="${f.label} value">
        <div class="track"><i class="mk b" style="left:calc(${pb}% - 1px)"></i><i class="mk m" style="left:calc(${pm}% - 1px)"></i>
          <input class="fr" type="range" min="${f.min}" max="${f.max}" step="${f.step}" aria-label="${f.label}"></div>
      </div>`;
    }).join("");
    syncInputs();
  }
  function syncInputs() {
    $$("#sliders .frow").forEach((row) => {
      const i = +row.dataset.i, f = S.schema.features[i], v = S.values[i];
      const r = $(".fr", row), n = $("input[type=number]", row);
      r.value = v; if (document.activeElement !== n) n.value = fmt(v);
      r.style.setProperty("--p", (((v - f.min) / (f.max - f.min)) * 100).toFixed(2) + "%");
    });
  }
  $("#sliders").addEventListener("input", (e) => {
    const row = e.target.closest(".frow"); if (!row) return;
    const i = +row.dataset.i, v = parseFloat(e.target.value);
    if (!Number.isFinite(v)) return;
    S.values[i] = v; clearPreset(); syncInputs(); schedulePredict();
  });
  $("#groupTabs").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    $$("#groupTabs button").forEach((x) => x.classList.toggle("on", x === b));
    const paste = b.dataset.group === "paste";
    $("#pastePane").hidden = !paste; $("#sliders").hidden = paste;
    if (!paste) { S.group = b.dataset.group; buildSliders(); }
  });
  $("#pasteApply").addEventListener("click", () => {
    const nums = $("#pasteBox").value.split(/[\s,;]+/).filter(Boolean).map(Number);
    const msg = $("#pasteMsg");
    if (nums.length !== 30 || nums.some((n) => !Number.isFinite(n))) {
      msg.textContent = `Found ${nums.length} numbers; 30 are needed.`; msg.style.color = "var(--malignant)"; return;
    }
    clearPreset(); setValues(nums); msg.textContent = "Applied ✓"; msg.style.color = "var(--benign)";
  });
  $("#btnRandom").addEventListener("click", () => {
    const t = Math.random();
    const v = S.schema.features.map((f) => {
      const base = f.benign_mean + t * (f.malignant_mean - f.benign_mean);
      const noise = (Math.random() - 0.5) * f.std * 0.6;
      return Math.min(f.max, Math.max(f.min, base + noise));
    });
    clearPreset(); setValues(v); toast("Generated a synthetic patient");
  });
  $("#btnTwinReport").addEventListener("click", () => {
    if (!S.pred) return;
    const p = S.pred, m = S.metrics.tabular;
    openReport({ kind: "twin", when: new Date(), case: caseData(), probability: p.probability, label: p.label, classified: true,
      assessment: p.assessment, metrics: m, modelName: `Digital Twin · ${m.model}`, threshold: 0.5 });
  });
  $("#btnCopy").addEventListener("click", async () => {
    const txt = S.schema.features.map((f) => f.key).join(",") + "\n" + S.values.map((v) => +v.toFixed(6)).join(",");
    try { await navigator.clipboard.writeText(txt); toast("Profile copied as CSV"); } catch { toast("Clipboard not available"); }
  });

  function setValues(v) { S.values = v.slice(); syncInputs(); predict(); }
  let predTimer = null, predSeq = 0;
  function schedulePredict() { clearTimeout(predTimer); predTimer = setTimeout(predict, 70); }
  async function predict() {
    const seq = ++predSeq;
    try {
      const r = await api("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ values: S.values }) });
      if (seq !== predSeq) return;
      S.pred = r; renderPrediction(r);
    } catch (e) { toast(e.message); }
  }

  function renderPrediction(r) {
    const p = r.probability;
    $("#gFill").style.strokeDashoffset = (314.16 * (1 - p)).toFixed(2);
    $("#gNeedle").style.transform = `rotate(${(-90 + 180 * p).toFixed(1)}deg)`;
    animateNumber($("#probVal"), p);
    const cls = r.label === "Malignant" ? (r.risk === "High" ? "is-m" : "is-m") : (r.risk === "Low" ? "is-b" : "is-u");
    $("#verdictLabel").textContent = r.label; $("#verdictLabel").className = cls;
    const pill = $("#riskPill"); pill.textContent = r.risk + " risk"; pill.className = "pill " + cls;

    const maxAbs = Math.max(0.03, ...r.contributions.map((c) => Math.abs(c.delta)));
    $("#drivers").innerHTML = r.contributions.slice(0, 7).map((c) => {
      const f = S.schema.features.find((x) => x.key === c.key);
      const w = (Math.abs(c.delta) / maxAbs) * 50, pos = c.delta >= 0;
      const style = pos ? `left:50%;width:${w}%;background:var(--malignant)` : `right:50%;width:${w}%;background:var(--benign)`;
      return `<div class="drv"><span class="n" title="${c.key}">${f.label} <small style="color:var(--faint)">${f.group}</small></span>
        <div class="bar"><i style="${style}"></i></div><span class="v">${pos ? "+" : "−"}${Math.abs(c.delta * 100).toFixed(1)}</span></div>`;
    }).join("") + `<p class="drv-note">Change in malignancy probability (points) if that one measurement were set to the benign average. Orange pushes towards malignant, blue towards benign.</p>`;

    $("#neighbours").innerHTML = r.neighbours.slice(0, 5).map((n) =>
      `<li data-case="${n.case}" title="Load this case into the twin"><span class="c">Case #${n.case}</span>
        <span class="tag ${n.diagnosis === "Malignant" ? "m" : "b"}">${n.diagnosis}</span><span class="c">d = ${n.distance.toFixed(2)}</span></li>`).join("");
    drawRadar(r.z);
    drawMap($("#twinMap"), r.pca);
    if (r.assessment) $("#twinPathway").innerHTML = `<h4 class="mini-title">Clinical pathway</h4>` + assessmentHTML(r.assessment, true);
  }
  $("#neighbours").addEventListener("click", async (e) => {
    const li = e.target.closest("li"); if (!li) return;
    try { const c = await api("/api/case/" + li.dataset.case); clearPreset(); setValues(c.values); toast(`Loaded case #${c.case} (${c.diagnosis})`); } catch (err) { toast(err.message); }
  });

  function animateNumber(el, to) {
    const from = parseFloat(el.dataset.v || "0"), t0 = performance.now();
    el.dataset.v = to;
    cancelAnimationFrame(el._raf);
    const step = (t) => {
      const k = Math.min(1, (t - t0) / 600), e = 1 - Math.pow(1 - k, 3);
      el.textContent = pct(from + (to - from) * e);
      if (k < 1) el._raf = requestAnimationFrame(step);
    };
    el._raf = requestAnimationFrame(step);
  }

  function drawRadar(z) {
    const svg = $("#radar"), feats = S.schema.features.filter((f) => f.group === "mean");
    const cx = 130, cy = 130, R = 92, n = feats.length;
    const pt = (i, zz) => {
      const a = -Math.PI / 2 + (i / n) * Math.PI * 2, r = ((Math.max(-3, Math.min(3, zz)) + 3) / 6) * R;
      return [cx + Math.cos(a) * r, cy + Math.sin(a) * r];
    };
    const poly = (zs) => zs.map((zz, i) => pt(i, zz).map((v) => v.toFixed(1)).join(",")).join(" ");
    const zb = feats.map((f) => (f.benign_mean - f.mean) / f.std), zm = feats.map((f) => (f.malignant_mean - f.mean) / f.std);
    const zp = feats.map((f) => z[f.index]);
    let g = "";
    [1, 2, 3].forEach((k) => { g += `<circle cx="${cx}" cy="${cy}" r="${(R * k) / 3}" stroke="var(--line-2)" stroke-width="1"/>`; });
    feats.forEach((f, i) => {
      const [x, y] = pt(i, 3), [lx, ly] = pt(i, 4.1);
      g += `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="var(--line)" stroke-width="1"/>`;
      g += `<text x="${lx}" y="${ly}" text-anchor="middle" dominant-baseline="middle">${f.label.replace("Fractal Dimension", "Fractal").replace("Concave Points", "Conc. pts")}</text>`;
    });
    g += `<polygon points="${poly(zb)}" stroke="var(--benign)" stroke-width="1.2" stroke-dasharray="3 3" fill="none"/>`;
    g += `<polygon points="${poly(zm)}" stroke="var(--malignant)" stroke-width="1.2" stroke-dasharray="3 3" fill="none"/>`;
    g += `<polygon points="${poly(zp)}" stroke="var(--brand)" stroke-width="2" fill="color-mix(in srgb, var(--brand) 22%, transparent)" style="transition:all .4s"/>`;
    svg.innerHTML = g;
  }

  /* ---------------- PCA maps ---------------- */
  const mapState = new Map();
  function drawMap(canvas, patient) {
    if (!S.population || !canvas || canvas.offsetParent === null) { if (canvas) mapState.set(canvas, patient); return; }
    mapState.set(canvas, patient);
    const dpr = devicePixelRatio || 1, w = canvas.clientWidth, h = +canvas.getAttribute("height");
    canvas.width = w * dpr; canvas.height = h * dpr; canvas.style.height = h + "px";
    const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const pts = S.population.points;
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const x0 = Math.min(...xs) - 1, x1 = Math.max(...xs) + 1, y0 = Math.min(...ys) - 1, y1 = Math.max(...ys) + 1;
    const X = (v) => 14 + ((v - x0) / (x1 - x0)) * (w - 28), Y = (v) => h - 14 - ((v - y0) / (y1 - y0)) * (h - 28);
    const cb = css("--benign"), cm = css("--malignant"), cbr = css("--brand");
    ctx.clearRect(0, 0, w, h);
    ctx.globalAlpha = 0.65;
    for (const [a, b, c] of pts) { ctx.fillStyle = c ? cm : cb; ctx.beginPath(); ctx.arc(X(a), Y(b), 2.6, 0, 7); ctx.fill(); }
    ctx.globalAlpha = 1;
    ctx.fillStyle = css("--faint"); ctx.font = "11px Manrope, sans-serif";
    ctx.fillText(`PC1 ${(S.population.explained[0] * 100).toFixed(0)}% · PC2 ${(S.population.explained[1] * 100).toFixed(0)}% of variance`, 16, h - 6);
    if (patient) {
      const px = X(Math.max(x0, Math.min(x1, patient[0]))), py = Y(Math.max(y0, Math.min(y1, patient[1])));
      ctx.strokeStyle = cbr; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
      ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.moveTo(0, py); ctx.lineTo(w, py); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = cbr; ctx.beginPath(); ctx.arc(px, py, 7, 0, 7); ctx.fill();
      ctx.strokeStyle = css("--surface"); ctx.lineWidth = 3; ctx.stroke();
    }
  }
  function redrawMaps() {
    if (!S.population) return;
    drawMap($("#twinMap"), S.pred ? S.pred.pca : null);
    drawMap($("#bigMap"), null);
  }
  let rT; addEventListener("resize", () => { clearTimeout(rT); rT = setTimeout(redrawMaps, 120); });

  /* ================= SCAN LAB ================= */
  const KIND_INFO = {
    mammogram: { name: "Mammogram", hint: "Full-field mammogram (CC or MLO view). PNG · JPG · TIFF · DICOM" },
    histopathology: { name: "Histology slide", hint: "H&E tissue image: a 50×50 patch or a larger region (tiled automatically)" },
    generic: { name: "MRI / ultrasound / other", hint: "Any breast image: enhancement, segmentation and measurements only" },
  };
  const scan = { kind: "mammogram", file: null, polarity: "auto", res: null, view: null, zoom: 1, panX: 0, panY: 0,
                 heat: 0.7, bright: 100, contrast: 100, invert: false, history: [], models: {} };
  const drop = $("#drop"), fileInput = $("#fileInput");
  const esc = (t) => String(t ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) runScan(f); });
  fileInput.addEventListener("change", () => { if (fileInput.files[0]) runScan(fileInput.files[0]); fileInput.value = ""; });

  function setKind(kind, rerun = true) {
    scan.kind = kind;
    $$("#scanKind button").forEach((b) => { const on = b.dataset.kind === kind; b.classList.toggle("on", on); b.setAttribute("aria-checked", on); });
    $("#polarityRow").hidden = kind !== "generic";
    $("#dropHint").textContent = KIND_INFO[kind].hint;
    renderBanner();
    if (rerun && scan.file) runScan(scan.file);
  }
  $("#scanKind").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b || b.dataset.kind === scan.kind) return;
    setKind(b.dataset.kind, false);
    if (scan.res && scan.res.kind !== b.dataset.kind) toast(`Now reading ${KIND_INFO[b.dataset.kind].name.toLowerCase()} images. Drop a new file.`);
  });
  $("#polarity").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    $$("#polarity button").forEach((x) => x.classList.toggle("on", x === b));
    scan.polarity = b.dataset.v; if (scan.file) runScan(scan.file);
  });
  $("#btnCase").addEventListener("click", () => {
    const f = $("#caseForm"); f.hidden = !f.hidden; $("#btnCase").setAttribute("aria-expanded", !f.hidden);
    if (!f.hidden) f.querySelector("input").focus();
  });
  const caseData = () => Object.fromEntries(new FormData($("#caseForm")).entries());

  /* ---- viewer ---- */
  const VIEWS = {
    mammogram: [["heat", "Heatmap"], ["input", "Input"], ["overlay", "Segment"], ["compare", "Compare"]],
    histopathology: [["heat", "Heatmap"], ["input", "Tissue"], ["compare", "Compare"]],
    generic: [["overlay", "Segmentation"], ["enhanced", "Enhanced"], ["heat", "Irregularity"], ["compare", "Compare"]],
  };
  const CAPTIONS = {
    heat: (r) => r.kind === "mammogram"
      ? "Class-activation map: warmer areas pushed the model towards malignant. The box marks the strongest region."
      : r.kind === "histopathology"
        ? `Each ${r.patches?.patch_px || 50}×${r.patches?.patch_px || 50} px tissue patch coloured by IDC probability (blank background is skipped).`
        : "Texture irregularity map (local variance × edge strength): image analysis, not a model explanation.",
    input: (r) => r.kind === "mammogram" ? "Exactly what the model saw: breast cropped, turned to face left, CLAHE contrast, letterboxed." : "The tissue image as analysed.",
    overlay: () => "Brightest lesion-like region found by classic segmentation (orange outline); measurements are in the details card.",
    enhanced: () => "Contrast-limited adaptive histogram equalisation (CLAHE) and light denoising.",
    compare: () => "Drag across the image to compare the plain scan with the heatmap.",
  };
  function buildViewTabs(r) {
    const views = VIEWS[r.kind].filter(([k]) => k === "compare" || k === "heat" || r.images[k]);
    if (!views.some(([k]) => k === scan.view)) scan.view = views[0][0];
    $("#viewTabs").innerHTML = views.map(([k, l]) => `<button data-view="${k}" class="${k === scan.view ? "on" : ""}">${l}</button>`).join("");
  }
  $("#viewTabs").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    scan.view = b.dataset.view; $$("#viewTabs button").forEach((x) => x.classList.toggle("on", x === b)); showView();
  });
  function baseFor(r, view) {
    if (view === "overlay") return r.images.overlay;
    if (view === "enhanced") return r.images.enhanced;
    return r.images.input || r.images.enhanced;
  }
  function showView() {
    const r = scan.res; if (!r) return;
    const v = scan.view, base = $("#baseImg"), heat = $("#heatImg"), cmp = $("#compare"), layer = $("#layer");
    $("#viewCaption").textContent = CAPTIONS[v](r);
    if (v === "compare") {
      layer.hidden = true; cmp.hidden = false;
      $("#cmpA").src = r.images.heatmap; $("#cmpB").src = baseFor(r, "input");
      $("#cmpB").style.clipPath = "inset(0 50% 0 0)"; $("#cmpRange").value = 50; $("#cmpBar").style.left = "50%";
      applyFilters(); return;
    }
    layer.hidden = false; cmp.hidden = true;
    base.src = baseFor(r, v);
    const showHeat = v === "heat" && (r.images.heat || r.images.heatmap);
    heat.hidden = !showHeat;
    if (showHeat) heat.src = r.images.heat || r.images.heatmap;
    if (showHeat && !r.images.heat) { base.hidden = true; } else { base.hidden = false; }
    base.onload = () => placeRoi();
    if (base.complete) placeRoi();
    applyFilters();
  }
  function placeRoi() {
    const r = scan.res, img = $("#baseImg").hidden ? $("#heatImg") : $("#baseImg"), roi = $("#roi");
    const box = scan.view === "heat" && r.kind === "mammogram" ? r.focus : scan.view === "overlay" ? r.roi || r.classic?.roi : null;
    if (!box || !img.naturalWidth) { roi.classList.remove("show"); return; }
    const k = Math.min(img.clientWidth / img.naturalWidth, img.clientHeight / img.naturalHeight);
    const w = img.naturalWidth * k, h = img.naturalHeight * k;
    const ox = (img.clientWidth - w) / 2, oy = (img.clientHeight - h) / 2, pad = 6;
    Object.assign(roi.style, { left: ox + box.x * w - pad + "px", top: oy + box.y * h - pad + "px",
      width: box.w * w + pad * 2 + "px", height: box.h * h + pad * 2 + "px" });
    roi.dataset.label = scan.view === "heat" ? "FOCUS" : "ROI";
    roi.classList.add("show");
  }
  function applyFilters() {
    const f = `brightness(${scan.bright}%) contrast(${scan.contrast}%)${scan.invert ? " invert(1)" : ""}`;
    $("#baseImg").style.filter = f; $("#cmpB").style.filter = f;
    $("#heatImg").style.opacity = scan.heat;
    $("#layer").style.transform = `translate(${scan.panX}px, ${scan.panY}px) scale(${scan.zoom})`;
    $("#zoomVal").textContent = Math.round(scan.zoom * 100) + "%";
    $("#tInvert").setAttribute("aria-pressed", scan.invert);
    $("#tInvert").classList.toggle("on", scan.invert);
  }
  const bindSlider = (id, key, out, fmtv) => $(id).addEventListener("input", (e) => {
    scan[key] = key === "heat" ? e.target.value / 100 : +e.target.value; $(out).textContent = fmtv(e.target.value); applyFilters();
  });
  bindSlider("#sHeat", "heat", "#vHeat", (v) => v + "%");
  bindSlider("#sBright", "bright", "#vBright", (v) => v + "%");
  bindSlider("#sContrast", "contrast", "#vContrast", (v) => v + "%");
  const zoomBy = (k) => { scan.zoom = Math.min(8, Math.max(1, scan.zoom * k)); if (scan.zoom === 1) { scan.panX = scan.panY = 0; } applyFilters(); };
  const resetView = () => { scan.zoom = 1; scan.panX = scan.panY = 0; applyFilters(); };
  $("#tZoomIn").addEventListener("click", () => zoomBy(1.25));
  $("#tZoomOut").addEventListener("click", () => zoomBy(0.8));
  $("#tFit").addEventListener("click", resetView);
  $("#tInvert").addEventListener("click", () => { scan.invert = !scan.invert; applyFilters(); });
  $("#stage").addEventListener("wheel", (e) => { if (!scan.res || scan.view === "compare") return; e.preventDefault(); zoomBy(e.deltaY < 0 ? 1.12 : 0.89); }, { passive: false });
  (() => {
    let drag = null;
    $("#stage").addEventListener("pointerdown", (e) => {
      if (scan.zoom <= 1 || scan.view === "compare") return;
      drag = { x: e.clientX - scan.panX, y: e.clientY - scan.panY }; $("#stage").setPointerCapture(e.pointerId); $("#stage").classList.add("dragging");
    });
    $("#stage").addEventListener("pointermove", (e) => { if (!drag) return; scan.panX = e.clientX - drag.x; scan.panY = e.clientY - drag.y; applyFilters(); });
    const end = () => { drag = null; $("#stage").classList.remove("dragging"); };
    $("#stage").addEventListener("pointerup", end); $("#stage").addEventListener("pointercancel", end);
  })();
  $("#cmpRange").addEventListener("input", (e) => {
    const v = +e.target.value; $("#cmpB").style.clipPath = `inset(0 ${100 - v}% 0 0)`; $("#cmpBar").style.left = v + "%";
  });
  addEventListener("keydown", (e) => {
    if (location.hash !== "#scan" || !scan.res || /INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) return;
    if (e.key === "+" || e.key === "=") zoomBy(1.25);
    else if (e.key === "-") zoomBy(0.8);
    else if (e.key === "0") resetView();
    else if (e.key.toLowerCase() === "i") { scan.invert = !scan.invert; applyFilters(); }
    else if (e.key.toLowerCase() === "h") {
      const b = $(`#viewTabs [data-view="${scan.view === "heat" ? VIEWS[scan.res.kind][1][0] : "heat"}"]`); if (b) b.click();
    }
  });
  addEventListener("resize", () => scan.res && placeRoi());

  /* ---- run ---- */
  async function runScan(file) {
    scan.file = file;
    $("#viewer").hidden = false; $("#scanning").classList.add("on"); $("#roi").classList.remove("show");
    if (!scan.res) { $("#baseImg").src = URL.createObjectURL(file); $("#heatImg").hidden = true; }
    const fd = new FormData(); fd.append("file", file); fd.append("polarity", scan.polarity); fd.append("kind", scan.kind);
    try {
      const r = await api("/api/scan", { method: "POST", body: fd }, () => {
        $("#scanning span").textContent = "Queued: another scan is being analysed…";
      });
      r.fileName = file.name; r.when = new Date(); r.case = caseData();
      scan.res = r; resetView(); renderScan(r); addHistory(r);
      if (r.kind === "generic" && r.classified && r.id) explainScan(r.id);
    } catch (e) {
      toast(e.message);
      if (!scan.res) $("#viewer").hidden = true;
    } finally { $("#scanning").classList.remove("on"); $("#scanning span").textContent = "Analysing…"; }
  }
  async function explainScan(id) {
    try {
      const r = await api(`/api/scan/${id}/explain`, { method: "POST" });
      if (!scan.res || scan.res.id !== id) return;
      scan.res.images.heatmap = r.heatmap; if (scan.view === "heat") showView();
    } catch { /* keep irregularity map */ }
  }

  /* ---- result rendering ---- */
  const LV = (a) => `lv${a.level ?? 0}`;
  function verdictHTML(r) {
    const a = r.assessment;
    if (!r.classified) {
      return `<div class="vc-top"><span class="pill is-u">Analysis only</span><span class="vc-kind">${KIND_INFO[r.kind].name}</span></div>
        <h3>${r.kind === "generic" ? "No AI verdict for this image type" : esc(r.label)}</h3>
        <p class="muted">${r.kind === "generic" ? "The image was enhanced, segmented and measured. To get verdicts for MRI or ultrasound, train a model with <code>train_scan_model.py</code> on labelled scans."
          : "The model could not analyse this image. Check that the scan type selected above matches the image."}</p>`;
    }
    const p = r.probability, t = r.threshold ?? 0.5, pos = r.kind === "histopathology" ? a.category === "IDC+" : p >= t;
    const cls = !r.quality?.ok && r.kind === "mammogram" && r.quality?.ok === false ? "is-u" : pos ? "is-m" : p >= t * 0.5 ? "is-u" : "is-b";
    const m = r.model?.metrics || {};
    return `<div class="vc-top"><span class="pill ${cls}">${esc(r.risk)} risk</span><span class="vc-kind">${KIND_INFO[r.kind].name} · ${esc(r.fileName || "")}</span></div>
      <h3 class="${cls}">${esc(r.label)}</h3>
      <div class="pbar"><span class="thr" style="left:${(t * 100).toFixed(1)}%" title="Decision threshold ${pct(t)}"></span><i style="left:calc(${(p * 100).toFixed(1)}% - 2px)"></i></div>
      <div class="pbar-l"><span>${r.kind === "histopathology" ? "Non-IDC" : "Benign"}</span><b class="mono">${pct(p)} ${r.kind === "histopathology" ? "slide IDC score" : "malignancy score"}</b><span>${r.kind === "histopathology" ? "IDC" : "Malignant"}</span></div>
      ${r.kind === "histopathology" && r.patches ? `<div class="patch-sum"><div class="patchbar"><i style="width:${(r.patches.fraction * 100).toFixed(1)}%"></i></div>
        <span>${r.patches.positive} of ${r.patches.tissue} tissue patches flagged as IDC (${pct(r.patches.fraction)})${r.patches.sampled ? `. Large slide: an even sample of ${r.patches.tissue.toLocaleString()} of its ${r.patches.tissue_all.toLocaleString()} tissue patches was analysed at full resolution.` : ""}</span></div>` : ""}
      <p class="muted small">Model: ${esc(r.model?.architecture || "")} · test AUC ${m.auc != null ? m.auc.toFixed(3) : "—"} · sensitivity ${m.sensitivity != null ? pct(m.sensitivity, 0) : "—"} · specificity ${m.specificity != null ? pct(m.specificity, 0) : "—"}. Decision threshold ${pct(t)} (dashed mark).</p>`;
  }

  function biradsScale(a) {
    const order = ["0", "1", "2", "3", "4A", "4B", "4C", "5", "6"];
    const lv = { "0": 0, "1": 1, "2": 1, "3": 2, "4A": 3, "4B": 4, "4C": 5, "5": 6, "6": 6 };
    return `<div class="bscale" role="img" aria-label="BI-RADS scale, category ${esc(a.category)} highlighted">${order.map((c) =>
      `<div class="bcell lv${lv[c]} ${c === a.category ? "on" : ""}" title="BI-RADS ${c}"><b>${c}</b></div>`).join("")}</div>`;
  }
  function bandsHTML(a) {
    if (!a.score_bands) return "";
    return `<details class="bands"><summary>How the score maps to a category</summary>
      <div class="band-bar">${a.score_bands.map((b) => `<span class="lv${({ "2": 1, "3": 2, "4A": 3, "4B": 4, "4C": 5, "5": 6 })[b.category]} ${b.category === a.category ? "on" : ""}" style="flex:${Math.max(0.02, b.to - b.from)}" title="${pct(b.from, 0)}–${pct(b.to, 0)}">${b.category}</span>`).join("")}</div>
      <p class="muted small">Cut-offs are anchored on the model's validated decision threshold (${pct(a.threshold || 0.5)}): below it → BI-RADS 2 or 3; above it → 4A, 4B, 4C or 5 depending on how close the score is to 100%. The raw score is not a calibrated cancer likelihood, so the official ACR percentage bands are not applied directly.</p></details>`;
  }
  function assessmentHTML(a, forTwin = false) {
    const isBirads = a.system === "BI-RADS";
    return `<div class="as-head">
        <div class="as-cat ${LV(a)}"><small>${esc(a.system)}</small><b class="${a.category.length > 2 ? "long" : ""}">${esc(a.category)}</b></div>
        <div><h4>${esc(a.name)}</h4><p class="muted small">${isBirads ? `Expected likelihood of malignancy for this category: ${esc(a.likelihood)}` : esc(a.likelihood)}</p></div>
      </div>
      ${isBirads ? biradsScale(a) : ""}
      <div class="referral ur-${esc(a.urgency)}">
        <div class="ref-top"><span class="ubadge">${esc(a.urgency_label)}</span><span class="ref-time">${esc(a.urgency_timeframe)}</span></div>
        <p class="ref-sum">${esc(a.summary)}</p>
        <p class="ref-to"><svg viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6"/></svg>Refer to: <b>${esc(a.refer_to)}</b></p>
        <ol class="ref-actions">${a.actions.map((x) => `<li>${esc(x)}</li>`).join("")}</ol>
      </div>
      ${isBirads ? bandsHTML(a) : ""}
      <p class="as-note">${esc(a.basis)} ${a.note ? esc(a.note) : ""}</p>
      ${forTwin ? "" : `<div class="as-actions"><button class="btn primary sm" data-act="report"><svg viewBox="0 0 24 24"><path d="M6 3h9l4 4v14H6z"/><path d="M14 3v5h5M9 13h7M9 17h5"/></svg>Generate report</button>
        <button class="btn ghost sm" data-act="copy">Copy summary</button></div>`}`;
  }

  function detailHTML(r) {
    const q = r.quality || {};
    const chips = [`<span>Image <b>${q.width}×${q.height}</b></span>`];
    if (q.breast_fraction != null) chips.push(`<span>Breast area <b>${pct(q.breast_fraction, 0)}</b></span>`);
    if (q.contrast != null) chips.push(`<span>Contrast <b>${q.contrast.toFixed(2)}</b></span>`);
    if (r.patches) chips.push(`<span>Patches <b>${r.patches.tissue}/${r.patches.total}</b> tissue</span>`);
    let html = `<h4 class="mini-title first">Image</h4><div class="quality">${chips.join("")}</div>`;
    const s = r.shape || r.classic?.shape;
    if (s) {
      const items = [["Radius", fmt(s.radius) + " px"], ["Area", Math.round(s.area) + " px²"], ["Perimeter", fmt(s.perimeter) + " px"],
        ["Compactness", fmt(s.compactness)], ["Concavity", fmt(s.concavity)], ["Concave points", s.concave_points],
        ["Symmetry", fmt(s.symmetry)], ["Fractal dim.", fmt(s.fractal_dimension)], ["Solidity", fmt(s.solidity)]];
      html += `<details class="morpho-box" ${r.kind === "generic" ? "open" : ""}><summary>Lesion morphometrics <small>(brightest lesion-like region, classic segmentation)</small></summary>
        <div class="morpho">${items.map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join("")}</div></details>`;
    }
    const h = r.kind === "generic" ? r.histogram : null;
    if (h) html += `<h4 class="mini-title">Intensity histogram</h4><div class="hist">${h.map((x) => `<i style="height:${Math.max(1, x * 100)}%"></i>`).join("")}</div>`;
    if (r.model?.caveat) html += `<p class="as-note">${esc(r.model.caveat)}</p>`;
    return html;
  }

  function renderScan(r) {
    buildViewTabs(r); showView();
    $("#scanEmpty").hidden = true; $("#scanOut").hidden = false;
    $("#scanVerdict").innerHTML = verdictHTML(r);
    $("#assessCard").innerHTML = assessmentHTML(r.assessment);
    $("#detailCard").innerHTML = detailHTML(r);
  }
  $("#assessCard").addEventListener("click", (e) => {
    const b = e.target.closest("[data-act]"); if (!b || !scan.res) return;
    if (b.dataset.act === "report") openReport(scan.res);
    else copyText(reportText(scan.res), "Summary copied");
  });

  /* ---- history ---- */
  function addHistory(r) {
    scan.history.unshift(r); scan.history = scan.history.slice(0, 12);
    $("#historyBox").hidden = false;
    $("#historyRow").innerHTML = scan.history.map((h, i) => `<button class="hcard ${h === scan.res ? "on" : ""}" data-i="${i}" title="${esc(h.fileName)}">
      <img src="${h.images.heatmap || h.images.input || h.images.enhanced}" alt="">
      <span class="hcat ${LV(h.assessment)}">${h.assessment.system === "BI-RADS" ? "BI-RADS " : ""}${esc(h.assessment.category)}</span>
      <small>${esc(KIND_INFO[h.kind].name)} · ${h.when.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</small></button>`).join("");
  }
  $("#historyRow").addEventListener("click", (e) => {
    const b = e.target.closest(".hcard"); if (!b) return;
    const r = scan.history[+b.dataset.i]; scan.res = r; scan.file = null;
    setKind(r.kind, false); resetView(); renderScan(r);
    $$(".hcard").forEach((x) => x.classList.toggle("on", x === b));
    $("#viewer").hidden = false;
  });

  /* ---- banner ---- */
  function renderBanner() {
    const b = $("#modelBanner"), kind = scan.kind, info = scan.models[kind];
    const warnIcon = `<svg viewBox="0 0 24 24"><path d="M12 3l9 16H3z"/><path d="M12 10v4M12 17h.01"/></svg>`;
    const okIcon = `<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M8 12l3 3 5-6"/></svg>`;
    if (kind === "generic") {
      const g = scan.models.generic || {};
      b.className = "model-banner " + (g.ready ? "ok" : "info");
      b.innerHTML = g.ready ? `${okIcon}<p><b>Custom scan model loaded</b> (${esc(g.modality)}). Trained with <code>train_scan_model.py</code> on ${g.n_images} images.</p>`
        : `${warnIcon}<p><b>Image analysis mode.</b> No model is trained for MRI or ultrasound, so you get enhancement, segmentation and measurements but no verdict (BI-RADS 0, radiologist review). To add one, put labelled images in <code>data/scans/benign</code> and <code>data/scans/malignant</code> and run <code>python train_scan_model.py</code>.</p>`;
      return;
    }
    if (!info || !info.ready) {
      b.className = "model-banner warn";
      b.innerHTML = `${warnIcon}<p><b>${KIND_INFO[kind].name} model not installed.</b> ${esc(info?.error || "")} Put <code>model.onnx</code> and <code>meta.json</code> from the Kaggle zip into <code>models/${kind}/</code>.</p>`;
      return;
    }
    const m = info.metrics || {};
    b.className = "model-banner ok";
    b.innerHTML = `${okIcon}<p><b>${esc(info.title)}</b> · ${esc(info.architecture)} · test AUC <b>${m.auc.toFixed(3)}</b>, sensitivity ${pct(m.sensitivity, 0)}, specificity ${pct(m.specificity, 0)} on ${m.n_test.toLocaleString()} held-out ${kind === "mammogram" ? "mammograms" : "patches"}.${info.caveat ? ` <span class="muted">${esc(info.caveat)}</span>` : ""}</p>`;
  }
  function renderKindStatus(models) {
    scan.models = models;
    $$("#scanKind button").forEach((b) => {
      const k = b.dataset.kind, info = models[k] || {};
      b.querySelector(".kd").className = "kd " + (info.ready ? "ok" : k === "generic" ? "" : "no");
      b.querySelector(".kmetric").textContent = info.ready && info.metrics?.auc != null ? `AUC ${info.metrics.auc.toFixed(2)}` : k === "generic" ? "" : "not installed";
    });
    const ready = ["mammogram", "histopathology"].filter((k) => models[k]?.ready);
    const st = $("#scanStatus");
    st.className = "status " + (ready.length ? "ok" : "warn");
    st.lastElementChild.textContent = ready.length ? `${ready.length + 1} models ready` : "Image models not installed";
    setKind(scan.kind, false);
  }

  /* ================= REPORTS ================= */
  function copyText(t, msg) {
    (navigator.clipboard ? navigator.clipboard.writeText(t) : Promise.reject()).then(() => toast(msg), () => toast("Clipboard not available"));
  }
  function reportText(r) {
    const a = r.assessment, c = r.case || {};
    return [`BreastScan Twin: AI decision-support report`, `Date: ${r.when.toLocaleString()}`,
      c.pid ? `Patient/study: ${c.pid}` : "", c.age ? `Age: ${c.age}` : "", c.side ? `Side: ${c.side}` : "", c.view ? `View: ${c.view}` : "",
      `Exam: ${KIND_INFO[r.kind]?.name || "Digital Twin (FNA cytology)"}${r.fileName ? " (" + r.fileName + ")" : ""}`,
      r.classified !== false ? `AI result: ${r.label} (score ${pct(r.probability)})` : "AI result: no verdict (analysis only)",
      `${a.system} category: ${a.category}: ${a.name}`, `Urgency: ${a.urgency_label} (${a.urgency_timeframe})`,
      `Recommendation: ${a.summary}`, `Refer to: ${a.refer_to}`, ...a.actions.map((x, i) => `  ${i + 1}. ${x}`),
      `Note: ${a.note || ""}`, `This is AI decision support, not a diagnosis. Final assessment by a qualified clinician.`].filter(Boolean).join("\n");
  }
  let reportSource = null;
  function openReport(r) {
    reportSource = r;
    const a = r.assessment, c = r.case || {}, m = r.model?.metrics || r.metrics || {};
    const img = r.images ? (r.images.heatmap || r.images.input) : null;
    const base = r.images ? (r.images.input || r.images.enhanced) : null;
    $("#report").innerHTML = `
      <header class="rp-head"><div class="rp-brand"><img src="/static/img/logo.svg" alt="" width="34" height="34"><div><b>BreastScan Twin</b><span>AI decision-support report</span></div></div>
        <div class="rp-meta"><span>Report date</span><b>${r.when.toLocaleString()}</b><span>Report ID</span><b>${(r.id || Math.random().toString(16).slice(2, 10)).toUpperCase()}</b></div></header>
      <section class="rp-grid">
        <div><span>Patient / study ID</span><b>${esc(c.pid) || "—"}</b></div><div><span>Age</span><b>${esc(c.age) || "—"}</b></div>
        <div><span>Side</span><b>${esc(c.side) || "—"}</b></div><div><span>View / specimen</span><b>${esc(c.view) || "—"}</b></div>
        <div class="w2"><span>Examination</span><b>${esc(KIND_INFO[r.kind]?.name || "Digital Twin: fine-needle aspirate cytology")}${r.fileName ? ` · ${esc(r.fileName)}` : ""}</b></div>
        <div class="w2"><span>Clinical notes</span><b>${esc(c.notes) || "—"}</b></div>
      </section>
      ${img ? `<section class="rp-imgs"><figure><img src="${base}" alt=""><figcaption>Image analysed</figcaption></figure><figure><img src="${img}" alt=""><figcaption>AI heatmap (warm = more suspicious)</figcaption></figure></section>` : ""}
      <section class="rp-result">
        <div class="rp-cat ${LV(a)}"><small>${esc(a.system)}</small><b>${esc(a.category)}</b></div>
        <div><h2>${esc(a.name)}</h2>
          <p>${r.classified === false ? (r.kind === "generic" ? "No AI verdict for this image type." : esc(r.label) + ".") : `AI result: <b>${esc(r.label)}</b> · score ${pct(r.probability)}${r.threshold != null ? ` (threshold ${pct(r.threshold)})` : ""}`}</p>
          ${r.patches ? `<p>${r.patches.positive} of ${r.patches.tissue} tissue patches flagged as IDC (${pct(r.patches.fraction)}).</p>` : ""}</div>
      </section>
      <section class="rp-rec ur-${esc(a.urgency)}"><h3>Recommendation · <span>${esc(a.urgency_label)}: ${esc(a.urgency_timeframe)}</span></h3>
        <p>${esc(a.summary)}</p><p><b>Refer to:</b> ${esc(a.refer_to)}</p><ol>${a.actions.map((x) => `<li>${esc(x)}</li>`).join("")}</ol></section>
      <section class="rp-model"><h3>Model</h3><p>${esc(r.model?.title || r.modelName || "")}${r.model?.architecture ? " · " + esc(r.model.architecture) : ""}${m.auc != null ? ` · test AUC ${m.auc.toFixed(3)}` : ""}${m.sensitivity != null ? ` · sensitivity ${pct(m.sensitivity, 0)} · specificity ${pct(m.specificity, 0)}` : m.recall != null ? ` · accuracy ${pct(m.accuracy)} · sensitivity ${pct(m.recall, 0)}` : ""}</p>
        <p class="rp-note">${esc(a.basis)} ${esc(a.note || "")}</p></section>
      <section class="rp-sign"><div><span>Reviewed by (radiologist / pathologist)</span><i></i></div><div><span>Signature & date</span><i></i></div><div><span>Final category</span><i></i></div></section>
      <footer class="rp-foot">Generated by BreastScan Twin for research and education. This is AI decision support, not a medical diagnosis. The suggested category and referral must be confirmed by a qualified clinician who reviews the full examination and history.</footer>`;
    $("#reportDlg").showModal();
  }
  $("#btnReportClose").addEventListener("click", () => $("#reportDlg").close());
  $("#btnReportPrint").addEventListener("click", printReport);
  function printReport() {
    const old = document.getElementById("printFrame"); if (old) old.remove();
    const f = document.createElement("iframe");
    f.id = "printFrame"; f.setAttribute("aria-hidden", "true");
    Object.assign(f.style, { position: "fixed", right: "0", bottom: "0", width: "0", height: "0", border: "0" });
    document.body.appendChild(f);
    const css = document.querySelector('link[href*="style.css"]').href;
    const d = f.contentDocument;
    d.open();
    d.write(`<!doctype html><html data-theme="light"><head><meta charset="utf-8"><title>BreastScan Twin report</title>
      <link rel="stylesheet" href="${css}"><style>
      @page { size: A4; margin: 12mm; }
      html, body { background: #fff !important; margin: 0; }
      .report { margin: 0 !important; padding: 0 !important; box-shadow: none !important; font-size: 11.5px; }
      .rp-grid { grid-template-columns: repeat(4, 1fr) !important; }
      .rp-imgs { grid-template-columns: 1fr 1fr !important; }
      .rp-imgs img { height: 175px; }
      .rp-grid, .rp-imgs, .rp-result { padding: 10px 0 !important; }
      .rp-rec { margin: 10px 0 !important; padding: 10px 14px !important; }
      .rp-sign { padding: 16px 0 8px !important; }
      .rp-cat { width: 76px !important; height: 76px !important; }
      .report h2 { font-size: 22px; }
      .rp-sign { grid-template-columns: repeat(3, 1fr) !important; }
      * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
      </style></head><body><article class="report">${$("#report").innerHTML}</article></body></html>`);
    d.close();
    const go = () => { f.contentWindow.focus(); f.contentWindow.print(); };
    const imgs = [...d.images];
    Promise.all(imgs.map((i) => i.complete ? 1 : new Promise((r) => { i.onload = i.onerror = r; })))
      .then(() => setTimeout(go, 250));
  }
  $("#btnReportCopy").addEventListener("click", () => reportSource && copyText(reportText(reportSource), "Report text copied"));
  $("#reportDlg").addEventListener("click", (e) => { if (e.target === $("#reportDlg")) $("#reportDlg").close(); });

  /* ================= INSIGHTS ================= */
  function renderInsights(m) {
    $("#twinModelName").textContent = m.model;
    $("#metricTiles").innerHTML = [
      ["Accuracy", pct(m.accuracy), `${m.n_test} held-out cases`],
      ["Sensitivity", pct(m.recall), "malignant cases detected"],
      ["Precision", pct(m.precision), "of malignant calls correct"],
      ["ROC AUC", m.auc.toFixed(3), `10-fold CV ${pct(m.cv_mean)} ± ${pct(m.cv_std)}`],
    ].map(([k, b, s]) => `<div class="tile"><span class="k">${k}</span><b>${b}</b><span class="s">${s}</span></div>`).join("");
    const [[tn, fp], [fn, tp]] = m.confusion;
    const cell = (n, lbl, good, tone) => `<div class="cell" style="background:color-mix(in srgb,var(${tone}) ${good ? 18 : 6}%,transparent)"><b>${n}</b><span>${lbl}</span></div>`;
    $("#cm").innerHTML = `<div></div><div class="h">Predicted benign</div><div class="h">Predicted malignant</div>
      <div class="h">Actual benign</div>${cell(tn, "true negatives", true, "--benign")}${cell(fp, "false alarms", false, "--malignant")}
      <div class="h">Actual malignant</div>${cell(fn, "missed cancers", false, "--malignant")}${cell(tp, "true positives", true, "--malignant")}`;
    // ROC
    const W = 320, H = 260, L = 40, B = 34, T = 10, R = 12;
    const X = (v) => L + v * (W - L - R), Y = (v) => H - B - v * (H - B - T);
    const path = m.roc.fpr.map((f, i) => `${i ? "L" : "M"}${X(f).toFixed(1)},${Y(m.roc.tpr[i]).toFixed(1)}`).join("");
    let g = "";
    for (let k = 0; k <= 4; k++) {
      const v = k / 4;
      g += `<line x1="${X(0)}" x2="${X(1)}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)" stroke-width="1"/>`;
      g += `<text x="${L - 8}" y="${Y(v) + 3}" text-anchor="end">${v.toFixed(2)}</text><text x="${X(v)}" y="${H - B + 16}" text-anchor="middle">${v.toFixed(2)}</text>`;
    }
    g += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="var(--faint)" stroke-dasharray="4 4" stroke-width="1"/>`;
    g += `<path d="${path} L${X(1)},${Y(0)} L${X(0)},${Y(0)}Z" fill="color-mix(in srgb, var(--brand) 14%, transparent)" stroke="none"/>`;
    g += `<path d="${path}" stroke="var(--brand)" stroke-width="2.5"/>`;
    g += `<text x="${X(0.55)}" y="${Y(0.2)}" style="fill:var(--text);font-size:13px;font-weight:600">AUC = ${m.auc.toFixed(3)}</text>`;
    g += `<text x="${(L + W) / 2}" y="${H - 4}" text-anchor="middle">False positive rate</text>`;
    g += `<text transform="translate(11 ${(H - B) / 2}) rotate(-90)" text-anchor="middle">True positive rate</text>`;
    $("#roc").innerHTML = g;
    // algos
    const lo = 0.85;
    $("#algos").innerHTML = m.notebook_comparison.map((a) => `<div class="algo"><b>${a.name}</b><div class="bars" style="padding-right:56px">
      <div class="b raw" data-w="${((a.raw - lo) / (1 - lo)) * 100}"><em>${pct(a.raw)}</em></div>
      <div class="b scaled" data-w="${((a.scaled - lo) / (1 - lo)) * 100}"><em>${pct(a.scaled)}</em></div></div></div>`).join("") +
      `<p class="drv-note">Axis starts at 85% to make the differences visible.</p>`;
  }
  function animateAlgos() { requestAnimationFrame(() => $$(".algo .b").forEach((b) => { b.style.width = b.dataset.w + "%"; })); }

  function miniRoc(roc, auc) {
    const W = 220, H = 150, L = 26, B = 20, T = 6, R = 6;
    const X = (v) => L + v * (W - L - R), Y = (v) => H - B - v * (H - B - T);
    const d = roc.fpr.map((f, i) => `${i ? "L" : "M"}${X(f).toFixed(1)},${Y(roc.tpr[i]).toFixed(1)}`).join("");
    return `<svg viewBox="0 0 ${W} ${H}" class="roc mini"><line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="var(--faint)" stroke-dasharray="3 3" stroke-width="1"/>
      <path d="${d} L${X(1)},${Y(0)} L${X(0)},${Y(0)}Z" fill="color-mix(in srgb, var(--brand) 14%, transparent)" stroke="none"/><path d="${d}" stroke="var(--brand)" stroke-width="2"/>
      <text x="${X(0.45)}" y="${Y(0.15)}" style="fill:var(--text);font-size:11px;font-weight:600">AUC ${auc.toFixed(3)}</text>
      <text x="${(L + W) / 2}" y="${H - 4}" text-anchor="middle">False positive rate</text></svg>`;
  }
  function renderImageModels(models) {
    const cards = ["mammogram", "histopathology"].map((k) => {
      const i = models[k] || {};
      if (!i.ready) return `<div class="panel"><h3>${k === "mammogram" ? "Mammogram" : "Histopathology"}</h3><p class="missing">Not installed. Copy <code>model.onnx</code> and <code>meta.json</code> into <code>models/${k}/</code>.</p></div>`;
      const m = i.metrics, [[tn, fp], [fn, tp]] = m.confusion;
      return `<div class="panel"><h3>${esc(i.title)}</h3>
        <p class="sub">${esc(i.dataset)} · ${esc(i.architecture)} · trained ${esc(i.trained_at || "")}</p>
        <div class="mini-tiles">
          <div><span>ROC AUC</span><b>${m.auc.toFixed(3)}</b></div><div><span>Accuracy</span><b>${pct(m.accuracy, 1)}</b></div>
          <div><span>Sensitivity</span><b>${pct(m.sensitivity, 1)}</b></div><div><span>Specificity</span><b>${pct(m.specificity, 1)}</b></div>
        </div>
        <div class="im-row">${m.roc ? miniRoc(m.roc, m.auc) : ""}
          <div class="cm small"><div></div><div class="h">Pred. ${esc(i.classes[0])}</div><div class="h">Pred. ${esc(i.classes[1])}</div>
            <div class="h">True ${esc(i.classes[0])}</div><div class="cell">${tn.toLocaleString()}</div><div class="cell warn">${fp.toLocaleString()}</div>
            <div class="h">True ${esc(i.classes[1])}</div><div class="cell warn">${fn.toLocaleString()}</div><div class="cell">${tp.toLocaleString()}</div></div></div>
        <p class="sub">Held-out test set: ${m.n_test.toLocaleString()} ${k === "mammogram" ? "images (official CBIS-DDSM test split)" : "patches from patients never seen in training"} · threshold ${pct(i.threshold)}</p>
        ${i.has_report ? `<details><summary>Training curves</summary><img class="report" src="/models/${k}/training_report.png" alt="Training curves for the ${k} model" loading="lazy"></details>` : ""}
        ${i.caveat ? `<p class="caveat">${esc(i.caveat)}</p>` : ""}</div>`;
    });
    $("#imageModels").innerHTML = cards.join("");
  }
  function renderBiradsTable(ref) {
    const order = ["0", "1", "2", "3", "4A", "4B", "4C", "5", "6"];
    $("#biradsTable").innerHTML = `<div class="bt-row bt-h"><span>Category</span><span>Assessment</span><span>Likelihood of cancer</span><span>Management</span><span>Urgency</span></div>` +
      order.map((c) => { const k = ref.categories[c], m = ref.management[c], u = ref.urgency[m.urgency];
        return `<div class="bt-row"><span><b class="as-cat sm lv${k.level}">${c}</b></span><span>${esc(k.name)}</span><span>${esc(k.likelihood)}</span><span>${esc(m.summary)}</span><span><i class="ubadge ur-${m.urgency}">${esc(u.label)}</i></span></div>`; }).join("");
  }

  function renderLearn() {
    const seen = new Set();
    $("#featExplain").innerHTML = S.schema.features.filter((f) => !seen.has(f.base) && seen.add(f.base))
      .map((f) => `<div><b>${f.label}</b>${f.description}</div>`).join("");
  }

  /* ================= boot ================= */
  async function boot() {
    route();
    try {
      const [schema, metrics] = await Promise.all([api("/api/schema"), api("/api/metrics")]);
      S.schema = schema; S.metrics = metrics; S.population = metrics.population;
      const m = metrics.tabular;
      const dm = metrics.deep || {};
      $("#heroAcc").textContent = pct(m.accuracy);
      $("#heroAuc").textContent = dm.histopathology?.ready ? dm.histopathology.metrics.auc.toFixed(3) : m.auc.toFixed(3);
      $("#tileTwin").textContent = pct(m.accuracy);
      $("#tileTwinSub").textContent = `test accuracy · AUC ${m.auc.toFixed(3)} · ${m.model.replace("StandardScaler → ", "")}`;
      $("#tileMammo").textContent = dm.mammogram?.ready ? dm.mammogram.metrics.auc.toFixed(3) : "—";
      $("#tileMammoSub").textContent = dm.mammogram?.ready ? `ROC AUC · ${dm.mammogram.metrics.n_test} test mammograms` : "not installed";
      $("#tileHisto").textContent = dm.histopathology?.ready ? dm.histopathology.metrics.auc.toFixed(3) : "—";
      $("#tileHistoSub").textContent = dm.histopathology?.ready ? `ROC AUC · ${dm.histopathology.metrics.n_test.toLocaleString()} test patches` : "not installed";
      renderKindStatus({ ...dm, generic: metrics.scan });
      renderImageModels(dm);
      api("/api/birads").then(renderBiradsTable).catch(() => {});
      S.values = schema.presets[0].values.slice();
      buildPresets(); buildSliders(); renderInsights(m); renderLearn();
      setValues(schema.presets[0].values);
      route();
    } catch (e) { console.error(e); toast("Could not start the app: " + e.message); }
  }
  boot();
})();
