(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const pct = x => (x * 100).toFixed(1).replace(/\.0$/, "") + "%";

  let META = null;

  /* ---------------- theme ---------------- */
  const root = document.documentElement;
  let saved = null;
  try { saved = localStorage.getItem("heartly-theme"); } catch (e) {}
  const prefersDark = window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = saved || (prefersDark ? "dark" : "light");
  $("#theme").addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    try { localStorage.setItem("heartly-theme", root.dataset.theme); } catch (e) {}
  });

  /* ---------------- routing ---------------- */
  const pages = ["home", "dashboard", "predict"];
  function route() {
    let p = location.hash.replace("#", "");
    if (!pages.includes(p)) p = "home";
    pages.forEach(n => { $("#page-" + n).hidden = n !== p; });
    $$("#nav a").forEach(a => a.classList.toggle("on", a.dataset.page === p));
    window.scrollTo(0, 0);
  }
  window.addEventListener("hashchange", route);

  /* ---------------- chart helpers ---------------- */
  function barChart(el, data) {
    const w = 640, h = 280, pad = { l: 36, b: 34, t: 10, r: 8 };
    const max = Math.ceil(Math.max(...data.map(d => d.count)) / 20) * 20;
    const bw = (w - pad.l - pad.r) / data.length;
    let s = `<svg viewBox="0 0 ${w} ${h}" width="100%" role="img" aria-label="Age distribution bar chart">`;
    for (let i = 0; i <= 4; i++) {
      const v = (max / 4) * i, y = h - pad.b - (v / max) * (h - pad.b - pad.t);
      s += `<line x1="${pad.l}" x2="${w - pad.r}" y1="${y}" y2="${y}" stroke="var(--line)" stroke-dasharray="3 4"/>`;
      s += `<text x="${pad.l - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="var(--muted)" font-family="JetBrains Mono,monospace">${Math.round(v)}</text>`;
    }
    data.forEach((d, i) => {
      const bh = (d.count / max) * (h - pad.b - pad.t), x = pad.l + i * bw + bw * .16;
      s += `<rect x="${x}" y="${h - pad.b - bh}" width="${bw * .68}" height="${bh}" rx="6" fill="var(--teal)"><title>${d.label}: ${d.count} records</title></rect>`;
      s += `<text x="${x + bw * .34}" y="${h - 12}" text-anchor="middle" font-size="11" fill="var(--muted)" font-family="JetBrains Mono,monospace">${d.label}</text>`;
    });
    el.innerHTML = s + "</svg>";
  }
  function donut(el, a, b) {
    const total = a + b, r = 70, c = 2 * Math.PI * r, da = (a / total) * c;
    el.innerHTML = `<svg viewBox="0 0 200 200" width="210" role="img" aria-label="Class balance donut chart">
      <circle cx="100" cy="100" r="${r}" fill="none" stroke="var(--coral)" stroke-width="26"/>
      <circle cx="100" cy="100" r="${r}" fill="none" stroke="var(--teal)" stroke-width="26" stroke-dasharray="${da} ${c}" transform="rotate(-90 100 100)"/>
      <text x="100" y="106" text-anchor="middle" font-size="26" font-weight="600" fill="var(--ink)">${pct(a / total)}</text></svg>
      <div class="legend"><span><i style="background:var(--teal)"></i>Disease present ${a}</span><span><i style="background:var(--coral)"></i>Not present ${b}</span></div>`;
  }
  function rateBars(title, sub, rows) {
    const items = rows.map(r => `<div class="bar"><span>${esc(r.label)} <span class="muted">(n=${r.n})</span></span>
      <div class="track"><div class="fill" style="width:${r.disease_rate * 100}%"></div></div><span>${pct(r.disease_rate)}</span></div>`).join("");
    return `<article class="card"><div class="card-h"><h2>${title}</h2><p>${sub}</p></div><div class="card-b"><div class="bars">${items}</div></div></article>`;
  }

  /* ---------------- dashboard / insights / model ---------------- */
  function renderStats(S) {
    const acc = S.comparison[0].cv_accuracy;
    const kpi = (k, v, p, cls = "") => `<div class="kpi"><div class="k">${k}</div><div class="v ${cls}">${v}</div><p>${p}</p></div>`;
    $("#home-kpis").innerHTML = [
      kpi("Patient records", S.clean_rows, "Unique real patients the model learns from."),
      kpi("Measurements used", S.n_features, "Health readings entered for each prediction.", "t"),
      kpi("Tested accuracy", pct(acc), "Share of unseen patients classified correctly in testing."),
    ].join("");
    $("#kpis").innerHTML = [
      kpi("Patients", S.clean_rows, "Unique records after cleaning."),
      kpi("With disease", S.disease, `${pct(S.disease_share)} of all patients.`, "t"),
      kpi("Without disease", S.no_disease, `${pct(1 - S.disease_share)} of all patients.`),
      kpi("Tested accuracy", pct(acc), "Measured on patients the model had not seen."),
    ].join("");
    barChart($("#chart-age"), S.age_bands);
    donut($("#chart-balance"), S.disease, S.no_disease);
    $("#clean-steps").innerHTML = `
      <li><b>Removed ${S.duplicates_removed} duplicate rows.</b> The file held ${S.raw_rows} rows but only ${S.clean_rows} distinct patients. Keeping copies would let the model memorise patients and overstate its accuracy.</li>
      <li><b>Checked for gaps and impossible values.</b> No missing values, and every reading was within a plausible range.</li>
      <li><b>Corrected the disease label.</b> In the original file a label of 1 marked patients <em>without</em> disease. It is flipped so that a higher risk score means more risk.</li>
      <li><b>Standardised the measurements.</b> Each reading is put on the same scale so that cholesterol, with its large numbers, does not drown out smaller ones like ST depression.</li>`;

    $("#insight-bars").innerHTML = [
      rateBars("Chest pain type", "Share of patients with disease.", S.by_cp.map(r => ({ ...r, label: CP[r.code] || r.label }))),
      rateBars("Exercise-induced angina", "Share of patients with disease.", S.by_exang),
      rateBars("Major vessels coloured", "Share of patients with disease.", S.by_ca.map(r => ({ ...r, label: r.code + " vessels" }))),
      rateBars("Thalassemia scan", "Share of patients with disease.", S.by_thal.map(r => ({ ...r, label: THAL[r.code] || r.label }))),
      rateBars("Sex", "Share of patients with disease.", S.by_sex),
      rateBars("ST segment slope", "Share of patients with disease.", S.by_slope.map(r => ({ ...r, label: SLOPE[r.code] || r.label }))),
    ].join("");
    $("#means").innerHTML = `<thead><tr><th>Measurement</th><th>With disease</th><th>Without disease</th></tr></thead><tbody>` +
      Object.entries(S.feature_means).map(([k, v]) => `<tr><td>${LBL[k]}</td><td class="num">${v.disease}</td><td class="num">${v.no_disease}</td></tr>`).join("") + "</tbody>";
  }

  /* compare a record against the two group averages */
  function renderCompare(vals) {
    const S = META.stats, keys = ["age", "trestbps", "chol", "thalach", "oldpeak"];
    $("#compare").innerHTML = keys.map(k => {
      const [lo, hi] = S.ranges[k], pos = v => Math.min(100, Math.max(0, (v - lo) / (hi - lo) * 100));
      const m = S.feature_means[k];
      return `<div class="cmp"><div class="t"><b>${LBL[k]}</b><span>This record: ${vals[k]}</span></div>
        <div class="rail"><i class="dot n" style="left:${pos(m.no_disease)}%" title="Without disease: ${m.no_disease}"></i>
        <i class="dot d" style="left:${pos(m.disease)}%" title="With disease: ${m.disease}"></i>
        <i class="dot you" style="left:${pos(vals[k])}%"></i></div></div>`;
    }).join("") + `<div class="key"><span><i style="background:var(--ink)"></i>This record</span><span><i style="background:var(--high)"></i>Average with disease</span><span><i style="background:var(--low)"></i>Average without</span></div>`;
    $("#compare-card").hidden = false;
  }

  /* label tables used by insights (kept in sync with api/index.py) */
  const CP = { 0: "Asymptomatic", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Typical angina" };
  const THAL = { 0: "Not recorded", 1: "Fixed defect", 2: "Normal", 3: "Reversible defect" };
  const LBL = { age: "Age (years)", trestbps: "Resting blood pressure (mm Hg)", chol: "Cholesterol (mg/dl)", thalach: "Max heart rate (bpm)", oldpeak: "ST depression (mm)" };
  const SLOPE = { 0: "Downsloping", 1: "Flat", 2: "Upsloping" };

  /* ---------------- assess form ---------------- */
  function buildForm(fields) {
    const groups = {};
    fields.forEach(f => (groups[f.group] = groups[f.group] || []).push(f));
    $("#fields").innerHTML = Object.entries(groups).map(([g, fs]) => `<div class="group"><h3>${g}</h3>` + fs.map(f => {
      if (f.kind === "number") {
        return `<div class="field" data-name="${f.name}"><div class="top"><label for="f-${f.name}">${f.label}</label><span class="tag">${f.name}${f.unit ? " · " + f.unit : ""}</span></div>
          <div class="ctl"><input type="number" id="f-${f.name}" name="${f.name}" min="${f.min}" max="${f.max}" step="${f.step}" placeholder="${f.min}–${f.max}" inputmode="decimal">
          <input type="range" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.min}" aria-label="${f.label} slider" tabindex="-1"></div>
          <div class="hint">Accepted range for this model.</div><div class="err"></div></div>`;
      }
      const opts = Object.entries(f.options).map(([k, v]) => `<option value="${k}">${k} · ${esc(v)}</option>`).join("");
      return `<div class="field" data-name="${f.name}"><div class="top"><label for="f-${f.name}">${f.label}</label><span class="tag">${f.name}</span></div>
        <div class="ctl"><select id="f-${f.name}" name="${f.name}"><option value="">Select a code</option>${opts}</select></div><div class="err"></div></div>`;
    }).join("") + "</div>").join("");

    $$(".field").forEach(el => {
      const num = $("input[type=number]", el), rng = $("input[type=range]", el);
      if (num && rng) {
        rng.addEventListener("input", () => { num.value = rng.value; clearErr(el); });
        num.addEventListener("input", () => { if (num.value !== "") rng.value = num.value; clearErr(el); });
      }
      const sel = $("select", el); if (sel) sel.addEventListener("change", () => clearErr(el));
    });
  }
  const clearErr = el => el.classList.remove("bad");
  function setErr(name, msg) { const el = $(`.field[data-name="${name}"]`); if (!el) return; el.classList.add("bad"); $(".err", el).textContent = msg; }
  function fillForm(vals) {
    Object.entries(vals).forEach(([k, v]) => {
      const el = $(`.field[data-name="${k}"]`); if (!el) return;
      const i = $("input[type=number],select", el); i.value = v;
      const r = $("input[type=range]", el); if (r) r.value = v;
      clearErr(el);
    });
  }

  const emptyAnswer = `<div class="empty"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M5.6 18.4 18.4 5.6"/></svg>
    <strong>No result yet</strong>Fill in the record and select Assess risk. Heartly only shows a risk status once the model has scored it.</div>`;
  $("#answer").innerHTML = emptyAnswer;

  $("#form").addEventListener("submit", async e => {
    e.preventDefault();
    const payload = {}; let missing = false;
    $$(".field").forEach(el => {
      clearErr(el);
      const i = $("input[type=number],select", el);
      if (i.value === "") { setErr(el.dataset.name, "Required."); missing = true; } else payload[el.dataset.name] = Number(i.value);
    });
    if (missing) { $("#answer").innerHTML = `<div class="errbox"><b>Nothing was sent.</b> Fill in the highlighted fields first.</div>`; return; }
    const btn = $("#submit"); btn.disabled = true; btn.textContent = "Scoring…";
    try {
      const r = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const d = await r.json();
      if (!r.ok) {
        Object.entries(d.fields || {}).forEach(([k, m]) => setErr(k, m));
        $("#answer").innerHTML = `<div class="errbox"><b>${esc(d.error || "The model rejected this record.")}</b><ul>${Object.entries(d.fields || {}).map(([k, m]) => `<li><code>${k}</code>: ${esc(m)}</li>`).join("")}</ul></div>`;
      } else {
        $("#answer").innerHTML = `<div class="result">
          <div class="status">Risk status</div>
          <div class="big ${d.risk_status}">${d.risk_status} risk</div>
          <div>Model score ${pct(d.risk_probability)}</div>
          <div class="meter" role="img" aria-label="Risk meter at ${pct(d.risk_probability)}"><b style="left:${Math.min(98, Math.max(2, d.risk_probability * 100))}%"></b></div>
          <div class="scale"><span>0%</span><span>Low | Moderate | High</span><span>100%</span></div>
          <p class="n"><b>${d.neighbors_with_disease} of the ${d.neighbors_used}</b> most similar patients in the dataset had disease present. That share is the score; it is not a calibrated probability.</p>
          <div class="disc">Educational project built on 302 historical records. It is not a diagnosis. Anyone worried about their heart should see a doctor.</div></div>`;
        renderCompare(payload);
      }
    } catch (err) {
      $("#answer").innerHTML = `<div class="errbox"><b>Could not reach the model.</b> Nothing has been scored. Check your connection and try again.</div>`;
    } finally { btn.disabled = false; btn.textContent = "Assess risk"; }
  });
  $("#reset").addEventListener("click", () => {
    $("#form").reset(); $$(".field").forEach(clearErr); $("#compare-card").hidden = true;
    $$("input[type=range]").forEach(r => r.value = r.min); $("#answer").innerHTML = emptyAnswer;
  });
  $$("[data-ex]").forEach(b => b.addEventListener("click", async () => {
    const ex = await (await fetch("/api/examples")).json(); fillForm(ex[b.dataset.ex]);
  }));

  /* ---------------- boot ---------------- */
  fetch("/api/meta").then(r => r.json()).then(m => {
    META = m; renderStats(m.stats); buildForm(m.fields);
  }).catch(() => {
    $("#kpis").innerHTML = `<div class="errbox" style="grid-column:1/-1"><b>Could not load the data.</b> Refresh the page to try again.</div>`;
  });
  route();
})();
