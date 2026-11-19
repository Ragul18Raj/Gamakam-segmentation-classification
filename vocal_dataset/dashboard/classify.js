"use strict";
/* "Classify a selection" section: pick a song, drag to select a passage, and name it with the Stage 1 DTW-3NN.
   Classify-only: nothing here writes annotation files. Kept separate from app.js so the detector view is unchanged. */
(() => {
  const qs = new URLSearchParams(location.search);
  const API = qs.get("api") || `http://${location.hostname === "localhost" ? "localhost" : "127.0.0.1"}:8770`;
  const TYPES = ["Kampita", "Nokku", "Etra", "Irakka"];
  const BLIND_CHOICES = ["Kampita", "Nokku", "Etra", "Irakka", "orikai", "not a gamaka", "not sure"];
  const $ = id => document.getElementById(id);
  const canvas = $("cTimeline"), ctx = canvas.getContext("2d"), audio = $("cAudio"), exampleAudio = $("cExampleAudio");
  const S = { aid: null, name: "", known: "", view: null, win: [0, 1], sel: null, result: null, panel: "idle", message: "", guess: "", revealed: true, drag: null, hover: null, stopAt: null, timer: null, layout: null, seq: 0, tracksLoaded: false };
  const store = { get(k) { try { return localStorage.getItem(k); } catch { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch {} } };
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const typeColor = t => TYPES.includes(t) ? css(`--${t.toLowerCase()}`) : css("--faint");
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const r2 = v => Math.round(v * 100) / 100;
  const ms = s => `${Math.round(s * 1000)} ms`;
  function timeText(t, digits = 2) { const m = Math.floor(t / 60), s = (t - m * 60).toFixed(digits).padStart(digits ? digits + 3 : 2, "0"); return `${m}:${s}`; }
  function node(tag, text, cls) { const n = document.createElement(tag); if (text !== undefined && text !== null) n.textContent = text; if (cls) n.className = cls; return n; }
  function swatch(type) { const s = node("span", null, "swatch"); s.style.setProperty("--swatch", typeColor(type)); return s; }
  async function api(path, options) {
    let res;
    try { res = await fetch(API + path, options); } catch { throw Object.assign(new Error("The local analysis service is not running. Open a terminal in vocal_dataset and run dashboard/start_backend.sh."), { offline: true }); }
    let body = null; try { body = await res.json(); } catch {}
    if (!res.ok) throw Object.assign(new Error((body && body.detail) || res.statusText), { status: res.status });
    return body;
  }

  // ---------------------------------------------------------------- tabs
  function setTab(name, save = true) {
    const classify = name === "classify";
    $("detectorView").hidden = classify; $("classifyView").hidden = !classify;
    $("tabDetector").setAttribute("aria-selected", String(!classify)); $("tabClassify").setAttribute("aria-selected", String(classify));
    if (save) store.set("gamaka.tab", name);
    const url = location.pathname + location.search + (classify ? "#classify" : ""); history.replaceState(null, "", url);
    if (classify) { const a = $("audio"); if (a && !a.paused) a.pause(); if (!S.tracksLoaded) loadTracks(); requestAnimationFrame(() => { sizeCanvas(); draw(); renderPanel(); }); }
    else { if (!audio.paused) audio.pause(); exampleAudio.pause(); dispatchEvent(new Event("resize")); }
  }
  const visible = () => !$("classifyView").hidden;
  $("tabDetector").onclick = () => setTab("detector"); $("tabClassify").onclick = () => setTab("classify");
  $("tabDetector").parentElement.addEventListener("keydown", e => { if (e.key === "ArrowRight" || e.key === "ArrowLeft") { const next = visible() ? "detector" : "classify"; setTab(next); $(next === "classify" ? "tabClassify" : "tabDetector").focus(); } });

  // ---------------------------------------------------------------- loading audio
  function progress(title, detail) { $("cProgressTitle").textContent = title; $("cProgressDetail").textContent = detail || ""; $("cProgress").hidden = false; }
  function hideProgress() { $("cProgress").hidden = true; }
  function error(title, text, log) { $("cErrorTitle").textContent = title; $("cErrorText").textContent = text || ""; $("cErrorLog").textContent = log || ""; $("cErrorLog").hidden = !log; $("cError").hidden = false; hideProgress(); }
  function clearError() { $("cError").hidden = true; }
  async function loadTracks() {
    if (S.tracksLoaded || S.tracksLoading) return; S.tracksLoading = true;
    try {
      const tracks = await api("/api/tracks"); S.tracksLoaded = true;
      tracks.forEach(t => { const o = node("option", t.name); o.value = t.id; $("cTrackSelect").append(o); });
      const saved = store.get("gamaka.classify.audio");
      if (saved && !S.aid) { const a = JSON.parse(saved); try { const i = await api(`/api/audio-info/${a.id}`); choose(i.id, i.name, i.known); } catch { store.set("gamaka.classify.audio", ""); } }
    } catch (e) { error("The recording list could not be loaded", e.message); setTimeout(loadTracks, 3000); } finally { S.tracksLoading = false; }
  }
  async function upload(file) {
    clearError(); progress("Adding song", file.name);
    try { const i = await api("/api/upload", { method: "POST", headers: { "Content-Type": "application/octet-stream", "X-File-Name": encodeURIComponent(file.name) }, body: file }); choose(i.id, i.name, i.known); }
    catch (e) { error("This song could not be added", e.message); }
  }
  function choose(id, name, known) {
    clearTimeout(S.timer); S.aid = id; S.name = name; S.known = known || ""; S.view = null; S.sel = null; S.result = null; S.panel = "idle"; S.stopAt = null; S.seq++;
    store.set("gamaka.classify.audio", JSON.stringify({ id, name })); audio.src = `${API}/api/audio/${id}`;
    $("cFileSummary").replaceChildren(node("strong", name), node("span", known ? `Dataset recording, passage group excluded from matching` : "New recording"));
    $("cTrackSelect").value = known ? id : ""; $("cWorkspace").hidden = true; $("cEmpty").hidden = true; clearError(); syncInputs(); loadView();
  }
  async function loadView() {
    const id = S.aid, seq = S.seq; progress("Loading pitch", S.name);
    try {
      const v = await api(`/api/classify/view/${id}`); if (seq !== S.seq) return;
      if (v.state !== "ready") { const secs = v.started ? `${Math.max(0, Math.round(Date.now() / 1000 - v.started))} s` : ""; progress(v.message || "Measuring pitch", secs); S.timer = setTimeout(loadView, 1500); return; }
      S.view = v; S.win = [0, v.duration]; hideProgress(); $("cWorkspace").hidden = false; sizeCanvas(); draw(); renderPanel();
    } catch (e) { if (seq === S.seq) error("The pitch for this recording could not be loaded", e.offline ? e.message : "", e.offline ? "" : e.message); }
  }

  // ---------------------------------------------------------------- selection
  function setSelection(a, b, keepResult) {
    if (!S.view) return; a = r2(clamp(a, 0, S.view.duration)); b = r2(clamp(b, 0, S.view.duration));
    S.sel = b > a ? [a, b] : null; if (!keepResult) { S.result = null; S.panel = "idle"; S.seq++; } syncInputs(); draw(); renderPanel();
  }
  function syncInputs() {
    const has = !!S.sel; if (document.activeElement !== $("cStart")) $("cStart").value = has ? S.sel[0].toFixed(2) : ""; if (document.activeElement !== $("cEnd")) $("cEnd").value = has ? S.sel[1].toFixed(2) : "";
    const len = has ? S.sel[1] - S.sel[0] : 0; $("cLength").textContent = !has ? "No selection" : len < .08 ? `${ms(len)}, too short (80 ms minimum)` : `${len < 1 ? ms(len) : `${len.toFixed(2)} s`} selected`;
    $("cPlaySel").disabled = !has; $("cClassify").disabled = !has || S.panel === "loading"; if (S.view) { $("cStart").max = $("cEnd").max = S.view.duration; }
  }
  function fromInputs() { const a = parseFloat($("cStart").value), b = parseFloat($("cEnd").value); if (Number.isFinite(a) && Number.isFinite(b)) setSelection(a, b); }
  $("cStart").addEventListener("change", fromInputs); $("cEnd").addEventListener("change", fromInputs);
  [$("cStart"), $("cEnd")].forEach(i => i.addEventListener("keydown", e => { if (e.key === "Enter") { fromInputs(); classify(); } }));
  $("cSubseq").addEventListener("change", () => { if (S.result || S.panel === "reject") { S.result = null; S.panel = "idle"; renderPanel(); draw(); } });
  $("cBlind").addEventListener("change", () => { if (!$("cBlind").checked) S.revealed = true; renderPanel(); draw(); });

  // ---------------------------------------------------------------- classify
  async function classify() {
    if (!S.sel || !S.aid || S.panel === "loading") return; const seq = ++S.seq; S.panel = "loading"; S.result = null; renderPanel(); syncInputs();
    try {
      const r = await api("/api/classify_region", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ audio_id: S.aid, start_s: S.sel[0], end_s: S.sel[1], subsequence: $("cSubseq").checked }) });
      if (seq !== S.seq) return; S.result = r; S.panel = "result"; S.guess = ""; S.revealed = !$("cBlind").checked;
    } catch (e) { if (seq !== S.seq) return; S.panel = e.status === 422 || e.status === 409 ? "reject" : "error"; S.message = e.message; }
    renderPanel(); syncInputs(); draw();
  }
  $("cClassify").onclick = classify;
  const hidden = () => S.result && !S.revealed;

  function renderPanel() {
    const box = $("cResult"), out = [], r = S.result, v = S.view;
    if (S.panel === "loading") { out.push(node("h2", "Classifying"), node("p", "Comparing the selection with the marked gamakas", "muted")); }
    else if (S.panel === "reject") { out.push(node("h2", "Cannot classify this selection")); const b = node("div", null, "r-box bad"); b.textContent = S.message; out.push(b); }
    else if (S.panel === "error") { out.push(node("h2", "The classifier failed")); const pre = node("pre", S.message); const b = node("div", null, "r-box bad"); b.append(pre); out.push(b); }
    else if (!r) {
      out.push(node("h2", "Result"), node("p", S.sel ? "Press Classify region to name the selected passage." : "Drag across the timeline to select a passage, or type the start and end times.", "muted"));
      if (v) out.push(libraryNote(v.library_used, v.library_size, v.group, v.group_tracks));
    } else if (hidden()) renderBlind(out, r);
    else renderResult(out, r);
    box.replaceChildren(...out); box.querySelectorAll("canvas[data-nb]").forEach(drawMini);
  }
  function libraryNote(used, total, group, tracks) {
    const p = node("p", null, "muted"); p.style.marginTop = "10px";
    p.textContent = group ? `Matches against ${used} of the ${total} marked gamakas. Group ${group} (${tracks.join(", ")}) is left out so this recording cannot match itself or its other tempo versions.` : `Matches against all ${total} marked gamakas. A re-encoded copy of a dataset recording is not recognised and could match itself.`;
    return p;
  }
  function renderBlind(out, r) {
    out.push(node("h2", "Your answer first"), node("p", "The model's answer is hidden until you choose. Nothing is saved.", "muted"));
    const wrap = node("div", null, "blind"), sel = node("select"); sel.setAttribute("aria-label", "Your type for this passage");
    sel.append(node("option", "Choose a type")); sel.firstChild.value = ""; BLIND_CHOICES.forEach(c => { const o = node("option", c); o.value = c; sel.append(o); }); sel.value = S.guess;
    const btn = node("button", "Reveal model answer", "primary"); btn.type = "button"; btn.disabled = !S.guess;
    sel.onchange = () => { S.guess = sel.value; btn.disabled = !S.guess; }; btn.onclick = () => { S.revealed = true; renderPanel(); draw(); };
    const play = node("button", "Play selection", "ghost"); play.type = "button"; play.onclick = () => playSpan(S.sel[0], S.sel[1]);
    wrap.append(sel, btn, play); out.push(wrap);
  }
  function renderResult(out, r) {
    if (S.guess) {
      const model = r.type || "Unknown", agree = S.guess === r.type || (S.guess === "not a gamaka" && !r.type), v = node("div", null, "verdict");
      v.append(node("span", "You:"), node("b", S.guess), node("span", "Model:"), node("b", model), node("span", agree ? "Same answer" : "Different answers", "muted")); out.push(v);
    }
    out.push(node("h2", "Model answer"));
    const head = node("div", null, "r-head"), t = node("div", null, `r-type${r.type ? "" : " unknown"}`);
    if (r.type) t.append(swatch(r.type), r.type); else t.append("Unknown / doesn't match");
    head.append(t, node("span", r.label, `label-chip ${r.label}`)); out.push(head);
    const why = r.label === "unknown" ? `The nearest marked example is further away than the reject distance. The closest type would be ${r.closest_type} (${r.votes}).` : r.label === "confident" ? `All 3 nearest marked examples are ${r.closest_type}.` : `${r.votes_n} of the 3 nearest marked examples are ${r.closest_type}; a tie goes to the nearest one.`;
    out.push(node("p", why, "muted"));
    const dl = node("dl", null, "r-facts"), add = (k, val) => dl.append(node("dt", k), node("dd", val));
    add("Neighbours agree", r.votes); add("DTW distance", `${r.distance.toFixed(2)} c per step`);
    add("Reject above", `${r.threshold.toFixed(2)} c per step${r.threshold_tuned ? "" : " (untuned)"}`);
    const cs = r.classified_span; add(r.best_match ? "Best match" : "Classified", `${cs.start.toFixed(2)} to ${cs.end.toFixed(2)} s (${ms(cs.end - cs.start)}, ${cs.voiced_frames} voiced frames)`);
    if (r.best_match) add("Whole selection", `${r.whole_selection.type}, ${r.whole_selection.votes}, distance ${r.whole_selection.distance.toFixed(2)}`);
    add("Compared with", r.excluded_group ? `${r.library_used} marks; ${r.excluded_group} left out` : `all ${r.library_used} marks`);
    add("Time", `${Math.round(r.timing_ms.request_ms)} ms`); out.push(dl);
    if (r.best_match) { const b = node("div", null, "r-box"); b.append(node("strong", "Search inside selection is rough"), node("span", "In a check on the marked gamakas with 0.5 s added on each side, it found the marked stretch about half the time and gave the right type about 4 times in 10. It tends to pick short stretches, so short types such as Nokku come up too often. Selecting one gamaka by hand is more reliable.")); out.push(b); }
    if (!r.threshold_tuned) { const b = node("div", null, "r-box"); b.append(node("strong", "Reject distance not tuned yet"), node("span", "It is a placeholder (99th percentile of held-out nearest distances among the marks). Set it in dashboard/classify_config.json.")); out.push(b); }
    const det = r.detector_events, d = node("div", null, `r-box ${r.multiple_gamakas ? "" : "neutral"}`);
    d.append(node("strong", r.multiple_gamakas ? `This selection may hold ${det.length} gamakas` : det.length === 1 ? "Detector check: one gamaka here" : "Detector check: no gamaka found here"));
    if (det.length) { const ul = node("ul"); det.forEach(e => ul.append(node("li", `${e.type} ${e.start.toFixed(2)} to ${e.end.toFixed(2)} s (${ms(e.end - e.start)}, ${e.votes})`))); d.append(ul); }
    if (r.multiple_gamakas) d.append(node("span", "Try a shorter selection or turn on Search inside selection."));
    out.push(d);
    const sec = node("div", null, "r-section"); sec.append(node("h3", "Nearest marked examples"));
    r.nearest.forEach((n, i) => {
      const card = node("div", null, "nb"), h = node("div", null, "nb-head"), tc = node("span", null, "type-cell"); tc.append(swatch(n.type), n.type);
      h.append(node("span", `${i + 1}.`, "nb-meta"), tc, node("span", `${n.track}, ${n.start.toFixed(2)} to ${n.end.toFixed(2)} s`, "nb-meta"), node("span", `${n.distance.toFixed(2)}`, "nb-dist"));
      const c = node("canvas"); c.dataset.nb = i; c.setAttribute("role", "img"); c.setAttribute("aria-label", `Pitch contour of the selection overlaid on example ${i + 1}, a ${n.type} from ${n.track}`);
      const foot = node("div", null, "nb-foot"), lg = node("div", null, "nb-legend"), k1 = node("span"), k2 = node("span"), l1 = node("span", null, "key-line"), l2 = node("span", null, "key-line");
      l1.style.setProperty("--swatch", css("--pitch")); l2.style.setProperty("--swatch", typeColor(n.type)); k1.append(l1, "Selection"); k2.append(l2, "Example"); lg.append(k1, k2);
      const pe = node("button", "Play example", "ghost"); pe.type = "button"; pe.disabled = !n.clip_url; pe.onclick = () => { audio.pause(); exampleAudio.src = API + n.clip_url; exampleAudio.play().catch(() => {}); };
      const ps = node("button", "Play selection", "ghost"); ps.type = "button"; ps.onclick = () => playSpan(cs.start, cs.end);
      foot.append(lg, pe, ps); card.append(h, c, foot); sec.append(card);
    });
    out.push(sec);
  }
  function drawMini(c) {
    const r = S.result, n = r.nearest[+c.dataset.nb], a = r.contour, b = n.contour, dpr = devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
    const all = a.concat(b), lo = Math.min(...all), hi = Math.max(...all), pad = Math.max(20, (hi - lo) * .1), y0 = lo - pad, y1 = hi + pad, len = Math.max(a.length, b.length), L = 40, R = 6, T = 6, B = 16;
    const X = i => L + i / Math.max(1, len - 1) * (w - L - R), Y = v => T + (y1 - v) / (y1 - y0) * (h - T - B);
    g.font = "10px system-ui, sans-serif"; g.fillStyle = css("--muted"); g.textAlign = "right"; g.textBaseline = "middle";
    g.fillStyle = css("--grid"); g.fillRect(L, Math.round(Y(0)), w - L - R, 1); g.fillStyle = css("--muted");
    g.fillText(`${hi > 0 ? "+" : ""}${Math.round(hi)} c`, L - 4, Y(hi)); g.fillText(`${Math.round(lo)} c`, L - 4, Y(lo));
    g.textAlign = "right"; g.textBaseline = "alphabetic"; g.fillText(`${len * 10} ms`, w - R, h - 3); g.textAlign = "left"; g.fillText("0", L, h - 3);
    [[b, typeColor(n.type)], [a, css("--pitch")]].forEach(([arr, col]) => { g.strokeStyle = col; g.lineWidth = 2; g.lineJoin = "round"; g.beginPath(); arr.forEach((v, i) => i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))); g.stroke(); });
  }

  // ---------------------------------------------------------------- timeline
  function sizeCanvas() {
    if (!S.view || !visible()) return; const w = canvas.parentElement.clientWidth; if (!w) return; const dpr = devicePixelRatio || 1, narrow = w < 620, left = narrow ? 52 : 100;
    let y = 8; const rows = {}; rows.wave = [y, 38]; y += 38 + 10; rows.pitch = [y, narrow ? 170 : 230]; y += rows.pitch[1] + 10; rows.det = [y, 20]; y += 20 + 6; rows.axis = [y, 26]; y += 26 + 6; rows.nav = [y, 38]; const h = y + 38 + 8;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); canvas.style.height = `${h}px`; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); S.layout = { w, h, left, right: 12, rows };
    // whole-song loudness for the navigation strip: one column per pixel, loudest frame in each column
    const cols = Math.max(1, Math.floor(w - left - 12)), env = S.view.envelope, nav = new Float32Array(cols);
    for (let i = 0; i < env.length; i++) { const c = Math.min(cols - 1, Math.floor(i / env.length * cols)); if (env[i] > nav[c]) nav[c] = env[i]; }
    S.navEnv = nav;
  }
  const plotW = () => S.layout.w - S.layout.left - S.layout.right;
  const xOf = t => S.layout.left + (t - S.win[0]) / (S.win[1] - S.win[0]) * plotW();
  const tOf = x => S.win[0] + clamp((x - S.layout.left) / plotW(), 0, 1) * (S.win[1] - S.win[0]);
  function niceStep(v) { const p = 10 ** Math.floor(Math.log10(v)), n = v / p; return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p; }
  function niceTime(v) { return [.01, .02, .05, .1, .2, .5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600].find(s => s >= v) || 1200; }
  function range(n, hop, t0) { const a = Math.max(0, Math.floor((S.win[0] - t0) / hop) - 1), b = Math.min(n - 1, Math.ceil((S.win[1] - t0) / hop) + 1); return [a, b, Math.max(1, Math.floor((b - a) / Math.max(1, plotW() * 1.5)))]; }
  function band(a, b, color, alpha, top, bottom) { const x0 = xOf(a), x1 = xOf(b); ctx.globalAlpha = alpha; ctx.fillStyle = color; ctx.fillRect(x0, top, Math.max(1, x1 - x0), bottom - top); ctx.globalAlpha = 1; }
  function draw() {
    const v = S.view, l = S.layout; if (!v || !l || !visible()) return; const muted = css("--muted"), grid = css("--grid"), ink = css("--ink"), r = S.result, pw = plotW();
    ctx.clearRect(0, 0, l.w, l.h); ctx.fillStyle = css("--surface"); ctx.fillRect(0, 0, l.w, l.h); ctx.font = "12px system-ui, sans-serif"; ctx.textBaseline = "middle"; ctx.textAlign = "left";
    [["wave", "Audio", "Audio"], ["pitch", "Pitch", ""], ["det", "Detector", "Det."]].forEach(([k, t, short]) => { const [y, h] = l.rows[k]; ctx.fillStyle = muted; ctx.fillText(l.w < 620 ? short : t, 10, k === "pitch" ? y + 8 : y + h / 2); ctx.fillStyle = grid; ctx.fillRect(l.left, y + h, pw, 1); });
    const top = l.rows.wave[0], bottom = l.rows.det[0] + l.rows.det[1];
    ctx.save(); ctx.beginPath(); ctx.rect(l.left, 0, pw, l.h); ctx.clip();
    if (S.sel) band(S.sel[0], S.sel[1], ink, .08, top, bottom);
    if (r && r.best_match) band(r.best_match.start, r.best_match.end, hidden() ? ink : typeColor(r.closest_type), hidden() ? .14 : .26, top, bottom);
    ctx.restore();
    let [a, b, step] = range(v.envelope.length, v.hop_s, v.t0), [y, h] = l.rows.wave; ctx.fillStyle = css("--faint");
    for (let i = a; i <= b; i += step) { let m = 0; for (let j = i; j < Math.min(i + step, v.envelope.length); j++) m = Math.max(m, v.envelope[j]); const bar = Math.max(1, m * h); ctx.fillRect(xOf(v.t0 + i * v.hop_s), y + (h - bar) / 2, Math.max(1, pw / ((b - a) / step)), bar); }
    [a, b, step] = range(v.pitch.length, v.hop_s, v.t0); [y, h] = l.rows.pitch; const seen = []; for (let i = a; i <= b; i += step) if (v.pitch[i] !== null) seen.push(v.pitch[i]); seen.sort((p, q) => p - q);
    const lo = seen.length ? seen[Math.floor(seen.length * .01)] : -600, hi = seen.length ? seen[Math.floor(seen.length * .99)] : 600, pad = Math.max(60, (hi - lo) * .08), y0 = lo - pad, y1 = hi + pad, py = val => y + h - (val - y0) / (y1 - y0) * h, tick = niceStep((y1 - y0) / 4);
    ctx.textAlign = "right"; for (let val = Math.ceil(y0 / tick) * tick; val <= y1; val += tick) { const yy = py(val); ctx.fillStyle = grid; ctx.fillRect(l.left, Math.round(yy), pw, 1); ctx.fillStyle = muted; ctx.fillText(`${val > 0 ? "+" : ""}${val} c`, l.left - 6, yy); }
    ctx.save(); ctx.beginPath(); ctx.rect(l.left, y, pw, h); ctx.clip(); ctx.strokeStyle = css("--pitch"); ctx.lineWidth = 1.5; ctx.lineJoin = "round"; ctx.beginPath(); let pen = false, last = null;
    for (let i = a; i <= b; i += step) { const val = v.pitch[i]; if (val === null) { pen = false; last = null; continue; } const x = xOf(v.t0 + i * v.hop_s), yy = py(val), jump = last !== null && Math.abs(val - last) > 300; pen && !jump ? ctx.lineTo(x, yy) : ctx.moveTo(x, yy); pen = true; last = val; }
    ctx.stroke(); ctx.restore();
    ctx.save(); ctx.beginPath(); ctx.rect(l.left, 0, pw, l.h); ctx.clip();
    if (S.sel) { ctx.fillStyle = ink; [S.sel[0], S.sel[1]].forEach(t => ctx.fillRect(Math.round(xOf(t)) - 1, top, 2, bottom - top)); }
    if (r) { const [dy, dh] = l.rows.det; r.detector_events.forEach(e => { const x0 = xOf(e.start), w = Math.max(2, xOf(e.end) - x0); ctx.fillStyle = hidden() ? css("--faint") : typeColor(e.type); roundRect(x0, dy + 3, w, dh - 6, 3); ctx.fill(); }); }
    ctx.restore();
    const [ay] = l.rows.axis, span = S.win[1] - S.win[0], tk = niceTime(span / Math.max(3, Math.floor(pw / 92))); ctx.fillStyle = css("--line"); ctx.fillRect(l.left, ay, pw, 1); ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let t = Math.ceil(S.win[0] / tk) * tk; t <= S.win[1] + 1e-6; t += tk) { const x = xOf(t); ctx.fillStyle = css("--line"); ctx.fillRect(x, ay, 1, 4); ctx.fillStyle = muted; ctx.fillText(timeText(t, tk < .1 ? 2 : tk < 1 ? 1 : 0), x, ay + 7); }
    drawNav(ink, muted, grid);
    const t = audio.currentTime || 0; if (t >= S.win[0] && t <= S.win[1]) { ctx.fillStyle = css("--playhead"); ctx.fillRect(xOf(t) - .75, 4, 1.5, ay - 4); }
    if (S.hover !== null) { ctx.fillStyle = muted; ctx.globalAlpha = .5; ctx.fillRect(S.hover, 4, 1, ay - 4); ctx.globalAlpha = 1; }
  }
  // ---------------------------------------------------------------- whole-song strip below the pitch: drag the box to move, its edges to zoom
  const navX = t => S.layout.left + t / S.view.duration * plotW();
  const navT = x => clamp((x - S.layout.left) / plotW(), 0, 1) * S.view.duration;
  const inNav = y => S.layout && y >= S.layout.rows.nav[0] - 4 && y <= S.layout.rows.nav[0] + S.layout.rows.nav[1] + 4;
  function navHit(x) { const a = navX(S.win[0]), b = navX(S.win[1]); return Math.abs(x - a) < 7 ? "start" : Math.abs(x - b) < 7 ? "end" : x > a && x < b ? "inside" : "outside"; }
  function drawNav(ink, muted, grid) {
    const l = S.layout, v = S.view, [ny, nh] = l.rows.nav, pw = plotW(), r = S.result;
    ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillStyle = muted; ctx.fillText(l.w < 620 ? "Song" : "Whole song", 10, ny + nh / 2);
    ctx.fillStyle = grid; roundRect(l.left, ny, pw, nh, 5); ctx.fill();
    ctx.fillStyle = css("--faint"); S.navEnv.forEach((m, i) => { const bar = Math.max(1, m * (nh - 10)); ctx.fillRect(l.left + i, ny + (nh - bar) / 2, 1, bar); });
    if (S.sel) { ctx.globalAlpha = .3; ctx.fillStyle = ink; ctx.fillRect(navX(S.sel[0]), ny, Math.max(2, navX(S.sel[1]) - navX(S.sel[0])), nh); ctx.globalAlpha = 1; }
    if (r && r.best_match) { ctx.fillStyle = hidden() ? ink : typeColor(r.closest_type); ctx.fillRect(navX(r.best_match.start), ny, Math.max(2, navX(r.best_match.end) - navX(r.best_match.start)), nh); }
    const a = navX(S.win[0]), b = Math.max(navX(S.win[1]), a + 4);
    ctx.globalAlpha = .55; ctx.fillStyle = css("--surface"); ctx.fillRect(l.left, ny, a - l.left, nh); ctx.fillRect(b, ny, l.left + pw - b, nh); ctx.globalAlpha = 1;
    ctx.strokeStyle = ink; ctx.lineWidth = 1.5; roundRect(a, ny + .75, b - a, nh - 1.5, 4); ctx.stroke();
    ctx.fillStyle = ink; [a, b].forEach(x => ctx.fillRect(x - 1.5, ny + nh / 2 - 7, 3, 14));
    const t = audio.currentTime || 0; ctx.fillStyle = css("--playhead"); ctx.fillRect(navX(t) - .75, ny, 1.5, nh);
  }
  function roundRect(x, y, w, h, r) { const q = Math.min(r, w / 2, h / 2); ctx.beginPath(); ctx.moveTo(x + q, y); ctx.arcTo(x + w, y, x + w, y + h, q); ctx.arcTo(x + w, y + h, x, y + h, q); ctx.arcTo(x, y + h, x, y, q); ctx.arcTo(x, y, x + w, y, q); ctx.closePath(); }
  function zoom(t, f) { const d = S.view.duration, w = clamp((S.win[1] - S.win[0]) * f, Math.min(.2, d), d), share = (t - S.win[0]) / (S.win[1] - S.win[0]); const a = clamp(t - share * w, 0, d - w); S.win = [a, a + w]; draw(); }
  function show(a, b) { const d = S.view.duration, pad = Math.max(.3, (b - a) * .6), lo = clamp(a - pad, 0, d), hi = clamp(b + pad, 0, d); S.win = [lo, Math.max(hi, Math.min(d, lo + .2))]; draw(); }
  const pos = ev => { const r = canvas.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; };
  canvas.addEventListener("pointerdown", ev => {
    if (!S.view) return; const [x, y] = pos(ev); canvas.setPointerCapture(ev.pointerId); canvas.focus();
    if (inNav(y)) {
      let hit = navHit(x);
      if (hit === "outside") { const w = S.win[1] - S.win[0], a = clamp(navT(x) - w / 2, 0, S.view.duration - w); S.win = [a, a + w]; hit = "inside"; draw(); }
      S.drag = { mode: hit === "inside" ? "navpan" : "navedge", k: hit === "start" ? 0 : 1, x, win: [...S.win], moved: false }; canvas.style.cursor = hit === "inside" ? "grabbing" : "ew-resize"; return;
    }
    const edge = S.sel ? [0, 1].find(k => Math.abs(x - xOf(S.sel[k])) < 6) : undefined;
    S.drag = ev.shiftKey || ev.button === 1 ? { mode: "pan", x, win: [...S.win], moved: false } : edge !== undefined ? { mode: "edge", k: edge, x, moved: false } : { mode: "new", x, t: tOf(x), moved: false };
  });
  canvas.addEventListener("pointermove", ev => {
    if (!S.view || !S.layout) return; const [x, y] = pos(ev), d = S.drag;
    if (d && d.mode.startsWith("nav")) {
      if (Math.abs(x - d.x) > 2) d.moved = true; const dur = S.view.duration, minW = Math.min(.2, dur);
      if (d.mode === "navpan") { const w = d.win[1] - d.win[0], a = clamp(d.win[0] + (x - d.x) / plotW() * dur, 0, dur - w); S.win = [a, a + w]; }
      else if (d.k === 0) S.win = [clamp(navT(x), 0, d.win[1] - minW), d.win[1]];
      else S.win = [d.win[0], clamp(navT(x), d.win[0] + minW, dur)];
      $("cTooltip").hidden = true; S.hover = null; draw(); return;
    }
    if (!d && inNav(y)) {
      const hit = navHit(x), tip = $("cTooltip"); canvas.style.cursor = hit === "inside" ? "grab" : hit === "outside" ? "pointer" : "ew-resize"; S.hover = null;
      tip.replaceChildren(node("b", timeText(navT(x))), node("span", hit === "inside" ? "Drag to move along the song" : hit === "outside" ? "Click to jump here" : "Drag to zoom in or out")); tip.hidden = false;
      const rect = tip.getBoundingClientRect(); tip.style.left = `${clamp(x + 12, 4, canvas.clientWidth - rect.width - 4)}px`; tip.style.top = `${Math.max(4, S.layout.rows.nav[0] - rect.height - 6)}px`; draw(); return;
    }
    if (d) {
      if (Math.abs(x - d.x) > 3) d.moved = true;
      if (d.moved && d.mode === "pan") { const w = d.win[1] - d.win[0], a = clamp(d.win[0] - (x - d.x) / plotW() * w, 0, S.view.duration - w); S.win = [a, a + w]; }
      else if (d.moved && d.mode === "new") { const t = tOf(x); setSelection(Math.min(d.t, t), Math.max(d.t, t)); }
      else if (d.moved && d.mode === "edge") { const s = [...S.sel]; s[d.k] = tOf(x); setSelection(Math.min(...s), Math.max(...s)); if (s[0] > s[1]) d.k = 1 - d.k; }
    }
    canvas.style.cursor = S.sel && [0, 1].some(k => Math.abs(x - xOf(S.sel[k])) < 6) ? "ew-resize" : ev.shiftKey ? "grab" : "crosshair";
    S.hover = x >= S.layout.left ? x : null; const t = tOf(x), i = Math.round((t - S.view.t0) / S.view.hop_s), p = S.view.pitch[i], tip = $("cTooltip");
    tip.replaceChildren(node("b", timeText(t)), node("span", p === null || p === undefined ? "No confident pitch" : `Pitch ${p > 0 ? "+" : ""}${Math.round(p)} c from track median`)); tip.hidden = false;
    const rect = tip.getBoundingClientRect(); tip.style.left = `${clamp(x + 12, 4, canvas.clientWidth - rect.width - 4)}px`; tip.style.top = "8px"; draw();
  });
  canvas.addEventListener("pointerup", ev => { const d = S.drag; S.drag = null; if (!d || !S.view) return; if (d.mode.startsWith("nav")) { canvas.style.cursor = "grab"; return; } if (!d.moved) { audio.currentTime = tOf(pos(ev)[0]); S.stopAt = null; draw(); clock(); } });
  canvas.addEventListener("pointerleave", () => { S.hover = null; $("cTooltip").hidden = true; draw(); });
  canvas.addEventListener("wheel", ev => { if (!S.view) return; ev.preventDefault(); zoom(tOf(pos(ev)[0]), ev.deltaY > 0 ? 1.18 : .85); }, { passive: false });
  $("cZoomIn").onclick = () => S.view && zoom((S.win[0] + S.win[1]) / 2, .65); $("cZoomOut").onclick = () => S.view && zoom((S.win[0] + S.win[1]) / 2, 1.5);
  $("cZoomFit").onclick = () => { if (S.view) { S.win = [0, S.view.duration]; draw(); } }; $("cZoomSel").onclick = () => S.view && S.sel && show(S.sel[0], S.sel[1]);

  // ---------------------------------------------------------------- playback
  function playSpan(a, b) { exampleAudio.pause(); audio.currentTime = Math.max(0, a); S.stopAt = b; audio.play().catch(() => {}); }
  function clock() { $("cClock").textContent = timeText(audio.currentTime || 0); $("cPlay").textContent = audio.paused ? "Play" : "Pause"; }
  $("cPlay").onclick = () => { S.stopAt = null; exampleAudio.pause(); audio.paused ? audio.play().catch(() => {}) : audio.pause(); };
  $("cPlaySel").onclick = () => S.sel && playSpan(S.sel[0], S.sel[1]);
  exampleAudio.addEventListener("play", () => audio.pause());
  ["play", "pause", "seeked", "ended"].forEach(n => audio.addEventListener(n, () => { clock(); draw(); }));
  function tick() {
    if (S.view && !audio.paused && visible()) {
      const t = audio.currentTime; if (S.stopAt !== null && t >= S.stopAt) { audio.pause(); S.stopAt = null; }
      if (t > S.win[1] || t < S.win[0]) { const w = S.win[1] - S.win[0], a = clamp(t - w * .08, 0, S.view.duration - w); S.win = [a, a + w]; }
      clock(); draw();
    }
    requestAnimationFrame(tick);
  }
  // Space plays this section's audio while it is shown (a capture listener runs before the detector view's handler).
  addEventListener("keydown", e => { if (e.code === "Space" && visible() && !["BUTTON", "INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) { e.preventDefault(); e.stopImmediatePropagation(); if (S.view) $("cPlay").click(); } }, true);

  // ---------------------------------------------------------------- inputs and start
  $("cFileInput").onchange = e => e.target.files[0] && upload(e.target.files[0]);
  ["dragenter", "dragover"].forEach(n => $("cDropZone").addEventListener(n, e => { e.preventDefault(); $("cDropZone").classList.add("drag"); }));
  ["dragleave", "drop"].forEach(n => $("cDropZone").addEventListener(n, e => { e.preventDefault(); $("cDropZone").classList.remove("drag"); }));
  $("cDropZone").addEventListener("drop", e => e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]));
  $("cTrackSelect").onchange = e => { const o = e.target.selectedOptions[0]; if (e.target.value) choose(e.target.value, `${o.textContent}.mp3`, o.textContent); };
  addEventListener("resize", () => { if (visible()) { sizeCanvas(); draw(); renderPanel(); } });
  $("themeButton").addEventListener("click", () => setTimeout(() => { if (visible()) { draw(); renderPanel(); } }));
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (visible()) { draw(); renderPanel(); } });
  requestAnimationFrame(tick); renderPanel();
  setTab(location.hash === "#classify" || (!location.hash && store.get("gamaka.tab") === "classify") ? "classify" : "detector", false);
})();
