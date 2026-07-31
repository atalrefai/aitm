const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// Keep on window so render paths never hit a TDZ / stale-cache ReferenceError.
if (typeof window._liveSettingsCache === "undefined") window._liveSettingsCache = null;
function getLiveSettingsCache() {
  return window._liveSettingsCache || null;
}
function setLiveSettingsCache(s) {
  window._liveSettingsCache = s || null;
  return window._liveSettingsCache;
}

function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, 3500);
}

function fmt(n, d = 2) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  return Number(n).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: Math.min(d, 2) });
}

function pct(n, d = 1) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  return `${(Number(n) * 100).toFixed(d)}%`;
}

function fmtTs(s) {
  if (!s) return "—";
  return String(s).replace("T", " ").replace("+00:00", "").slice(0, 19);
}

function clsNum(n, goodHigh = true) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "";
  const ok = goodHigh ? Number(n) >= 0 : Number(n) <= 0;
  return ok ? "num-ok" : "num-bad";
}

function kvRows(pairs) {
  return pairs.map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join("");
}

function kpi(label, value, extraClass = "") {
  return `<div class="kpi"><div class="l">${label}</div><div class="v ${extraClass}">${value}</div></div>`;
}

function friendlyFetchError(err) {
  const raw = String(err?.message || err || "");
  if (/failed to fetch|networkerror|load failed|fetch.*aborted/i.test(raw)) {
    return "تعذر الاتصال بالخادم — تأكد أن الواجهة تعمل (python -m atis.web.run) ثم أعد المحاولة";
  }
  // Job errors often include a long traceback after the first line
  const first = raw.split("\n").map((s) => s.trim()).find(Boolean) || raw;
  if (/database is locked/i.test(raw)) {
    return "قاعدة معرفة الأنماط مشغولة بخيط آخر — أعد المحاولة بعد ثوانٍ";
  }
  if (/failed to fetch|networkerror|err_connection_refused|load failed/i.test(raw)) {
    return "السيرفر غير متصل (8787) — تأكد أن python -m atis.web.run يعمل ثم أعد المحاولة";
  }
  return first;
}

async function api(path, opts) {
  let res;
  try {
    res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  } catch (err) {
    throw new Error(friendlyFetchError(err));
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data.detail || data.error || res.statusText;
    throw new Error(friendlyFetchError(typeof detail === "string" ? detail : JSON.stringify(detail)));
  }
  return data;
}

async function apiOptional(path, fallback = null) {
  try {
    return await api(path);
  } catch {
    return fallback;
  }
}

function fmtBytes(n) {
  const v = Number(n) || 0;
  if (v < 1024) return `${v} B`;
  if (v < 1024 * 1024) return `${(v / 1024).toFixed(1)} KB`;
  return `${(v / (1024 * 1024)).toFixed(2)} MB`;
}

function openJsonModal(title, pathLabel, content) {
  const modal = $("#json-modal");
  if (!modal) return;
  $("#json-modal-title").textContent = title || "ملف JSON";
  $("#json-modal-path").textContent = pathLabel || "—";
  $("#json-modal-body").textContent =
    typeof content === "string" ? content : JSON.stringify(content, null, 2);
  modal.hidden = false;
}

function closeJsonModal() {
  const modal = $("#json-modal");
  if (modal) modal.hidden = true;
}

async function viewPatternSectionJson(timeframe, section) {
  try {
    const tf = timeframe || $("#pattern-tf")?.value || "M5";
    const data = await api(`/api/patterns/files/${encodeURIComponent(tf)}/${encodeURIComponent(section)}`);
    const note = data.truncated
      ? `\n\n/* truncated: showing ${ (data.items || []).length } of ${data.total_count || "?"} items */`
      : "";
    openJsonModal(
      `${data.title || section} · ${data.timeframe || tf}`,
      data.relative_path || `data/patterns/${data.symbol || "XAUUSD"}/${data.timeframe || tf}/${section}.json`,
      `${JSON.stringify(data, null, 2)}${note}`,
    );
  } catch (e) {
    toast(e.message || "تعذر فتح ملف الأنماط");
  }
}

function renderPatternJsonFiles(files, currentTf) {
  const body = $("#pattern-json-files");
  if (!body) return;
  const titles = {
    candlesticks: "الشموع",
    structural: "الهيكلية",
    compounds: "المركّبة / المكتشفة",
    knowledge: "قاعدة المعرفة",
    discovery_log: "سجل الاكتشافات",
  };
  const showAll = $("#pattern-files-all-tf")?.checked !== false;
  const filtered = showAll
    ? (files || [])
    : (files || []).filter((f) => f.timeframe === currentTf);
  body.innerHTML = filtered.length
    ? filtered.map((f) => `<tr>
        <td><b>${titles[f.section] || f.section}</b></td>
        <td>${f.timeframe}</td>
        <td>${f.count ?? "—"}</td>
        <td>${fmtTs(f.updated_at)}</td>
        <td style="font-size:0.78rem;direction:ltr;text-align:left">${f.path || "—"}</td>
        <td><button class="btn btn-tiny" type="button" data-view-pattern-json="${f.timeframe}" data-section="${f.section}">استعراض</button></td>
      </tr>`).join("")
    : `<tr><td colspan="6">لا ملفات JSON بعد — اضغط «استكشاف شامل للأنماط» أو «تصدير JSON»</td></tr>`;
}

async function viewRegistryJson(timeframe) {
  try {
    const data = await api(`/api/registry/files/${encodeURIComponent(timeframe)}`);
    openJsonModal(
      `حالة البيانات · ${data.timeframe || timeframe}`,
      data.relative_path || data.path || "—",
      data.content || data,
    );
  } catch (e) {
    toast(e.message || "تعذر فتح الملف");
  }
}

function setNamedProgress(prefix, visible, pct = 0, message = "") {
  const panel = $(`#${prefix}-progress`);
  if (!panel) return;
  panel.hidden = !visible;
  const n = Math.max(0, Math.min(100, Math.round(Number(pct) || 0)));
  const bar = $(`#${prefix}-progress-bar`);
  const pctEl = $(`#${prefix}-progress-pct`);
  const step = $(`#${prefix}-progress-step`);
  const track = panel.querySelector(".progress-track");
  if (bar) bar.style.width = `${n}%`;
  if (pctEl) pctEl.textContent = `${n}%`;
  if (step) step.textContent = message || "—";
  if (track) track.setAttribute("aria-valuenow", String(n));
}

function setPipelineProgress(visible, pct = 0, message = "") {
  setNamedProgress("pipeline", visible, pct, message);
}

function setPatternProgress(visible, pct = 0, message = "") {
  setNamedProgress("pattern", visible, pct, message);
}

function setTrainProgress(visible, pct = 0, message = "") {
  setNamedProgress("train", visible, pct, message);
}

function deriveTrainingPhaseProgress(pct = 0, message = "") {
  const total = Number(pct || 0);
  const msg = String(message || "").toLowerCase();
  const trainEnd = 78;
  const validationEnd = 88;

  let trainPct = Math.max(0, Math.min(100, (total / trainEnd) * 100));
  let validationPct = 0;
  let testPct = 0;

  if (total > trainEnd) {
    trainPct = 100;
    validationPct = Math.max(0, Math.min(100, ((total - trainEnd) / (validationEnd - trainEnd)) * 100));
  }
  if (total > validationEnd) {
    validationPct = 100;
    testPct = Math.max(0, Math.min(100, ((total - validationEnd) / (100 - validationEnd)) * 100));
  }

  if (msg.includes("epoch") || msg.includes("التدريب")) {
    validationPct = Math.min(validationPct, 12);
    testPct = 0;
  }
  if (msg.includes("validation") || msg.includes("التحقق")) {
    trainPct = 100;
    validationPct = Math.max(validationPct, 15);
    testPct = 0;
  }
  if (msg.includes("testing") || msg.includes("اختبار")) {
    trainPct = 100;
    validationPct = 100;
    testPct = Math.max(testPct, 15);
  }
  if (msg.includes("regime") || msg.includes("أنظمة") || msg.includes("dsr") || msg.includes("knowledge")) {
    trainPct = 100;
    validationPct = 100;
    testPct = Math.max(testPct, 70);
  }
  if (msg.includes("final model") || total >= 100) {
    trainPct = 100;
    validationPct = 100;
    testPct = 100;
  }
  return { trainPct, validationPct, testPct };
}

function setTrainingPhaseBars(totalPct = 0, message = "") {
  const phases = deriveTrainingPhaseProgress(totalPct, message);
  const rows = [
    ["train", phases.trainPct],
    ["validation", phases.validationPct],
    ["test", phases.testPct],
  ];
  for (const [name, pct] of rows) {
    const bar = $(`#train-phase-${name}-bar`);
    const label = $(`#train-phase-${name}-pct`);
    const value = Math.max(0, Math.min(100, Math.round(pct)));
    if (bar) bar.style.width = `${value}%`;
    if (label) label.textContent = `${Math.max(0, 100 - value)}% متبق`;
  }
}

async function pollJob(jobId, { onProgress, intervalMs = 800 } = {}) {
  let networkFails = 0;
  for (let i = 0; i < 1800; i++) {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      networkFails = 0;
      // Refresh the jobs table less often — parallel TF polls were flooding /api/jobs
      if (i % 3 === 0) {
        try {
          const jobs = await api("/api/jobs");
          renderJobs(jobs.jobs);
        } catch {
          /* ignore list refresh blips */
        }
      }
      if (onProgress) onProgress(job);
      if (job.status === "success" || job.status === "error" || job.status === "cancelled") return job;
    } catch (err) {
      const msg = String(err?.message || err || "");
      const transient = /غير متصل|failed to fetch|connection_refused|network|404|not found/i.test(msg);
      if (!transient) throw err;
      networkFails += 1;
      if (networkFails > 15) {
        throw new Error(
          "انقطع الاتصال بالسيرفر وفُقدت المهمة — أعد تشغيل الاستكشاف بعد التأكد أن السيرفر يعمل على 8787",
        );
      }
      if (onProgress) {
        onProgress({
          progress: 0,
          message: `انتظار السيرفر… (${networkFails}/15)`,
          status: "running",
        });
      }
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error("انتهت مهلة انتظار المهمة");
}

const _activeJobs = {
  pipeline: null,
  training: null,
  trainingDetails: null,
};
let _refreshPromise = null;
let _dataRefreshPromise = null;
let _latestModels = null;
let _latestTraining = null;

const ACTIVE_TAB_KEY = "atis.activeTab";

function getSavedTab() {
  try {
    return window.localStorage.getItem(ACTIVE_TAB_KEY);
  } catch {
    return null;
  }
}

function saveActiveTab(name) {
  try {
    window.localStorage.setItem(ACTIVE_TAB_KEY, name);
  } catch {}
}

function setPipelineStopEnabled(enabled) {
  const btn = $("#btn-pipeline-stop");
  if (btn) btn.disabled = !enabled;
}

function hasRunningPatternDiscovery() {
  return Object.values(_discoverThreads).some((t) => t?.running);
}

function shouldPauseAutoRefresh() {
  return Boolean(_activeJobs.pipeline) || Boolean(_activeJobs.training) || hasRunningPatternDiscovery();
}

function renderTrainingLogs(logs) {
  const el = $("#train-live-logs");
  if (!el) return;
  el.textContent = (logs && logs.length)
    ? logs.slice(-120).join("\n")
    : "لا توجد سجلات مباشرة بعد";
}

async function exportTrainingPageHtml() {
  const section = $("#tab-train");
  if (!section) throw new Error("تعذر العثور على صفحة التدريب");
  let cssText = "";
  try {
    cssText = await fetch("/static/styles.css").then((r) => r.text());
  } catch {}
  const cloned = section.cloneNode(true);
  cloned.classList.add("active");
  const title = `ATIS Training Report - ${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}`;
  const html = `<!doctype html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>${title}</title>
  <style>${cssText}</style>
</head>
<body>
  <main class="content">
    <div class="banner ok">تصدير صفحة التدريب والاختبار من ATIS Gold Desk</div>
    ${cloned.outerHTML}
  </main>
</body>
</html>`;
  const blob = new Blob([html], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${title}.html`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}

async function openTrainingArtifactPath() {
  const t = _latestTraining;
  if (!t) throw new Error("لا توجد بيانات تدريب معروضة بعد");
  const targetPath =
    t.final_model?.artifact_path
    || t.final_model?.artifact_dir
    || t.llmodel?.artifact_path
    || t.artifact_path
    || t.artifact_dir;
  if (!targetPath) throw new Error("لا يوجد مسار ناتج متاح لهذا الإطار");
  await api("/api/system/open-path", {
    method: "POST",
    body: JSON.stringify({ path: targetPath }),
  });
}

function metricPct(value, { invert = false, max = 1 } = {}) {
  if (value == null || Number.isNaN(Number(value))) return 0;
  const raw = Math.max(0, Math.min(1, Number(value) / max));
  const v = invert ? (1 - raw) : raw;
  return Math.max(0, Math.min(100, Math.round(v * 100)));
}

function renderMetricBars(containerId, rows) {
  const el = $(containerId);
  if (!el) return;
  el.innerHTML = rows.map((row) => {
    const width = Math.max(4, Math.min(100, row.pct || 0));
    const cls = row.tone || "";
    return `<div class="metric-bar-row">
      <div>${row.label}</div>
      <div class="metric-bar-track"><div class="metric-bar-fill ${cls}" style="width:${width}%"></div></div>
      <div><b>${row.value}</b></div>
    </div>`;
  }).join("");
}

function renderStageStrip(training) {
  const el = $("#train-stage-strip");
  if (!el) return;
  const msg = ($("#train-progress-step")?.textContent || "").toLowerCase();
  const hasModel = !(training?.empty) || training?.llmodel?.exists;
  const doneTraining = hasModel;
  const doneValidation = Boolean(training?.metrics?.classification || training?.llmodel?.metrics?.validation);
  const doneTesting = Boolean(training?.metrics?.financial_oos || training?.llmodel?.metrics?.test);
  const doneFinal = Boolean(training?.metadata?.final_model_ready || training?.llmodel?.metadata?.final_model_ready);
  const stages = [
    { key: "data", label: "تحميل JSON", done: hasModel, active: msg.includes("json") || msg.includes("تحميل") },
    { key: "train", label: "التدريب WF", done: doneTraining, active: msg.includes("walk") || msg.includes("epoch") || msg.includes("التدريب") },
    { key: "validate", label: "التحقق", done: doneValidation, active: msg.includes("validation") || msg.includes("التحقق") },
    { key: "test", label: "الاختبار", done: doneTesting, active: msg.includes("testing") || msg.includes("اختبار") || msg.includes("oos") },
    { key: "regime", label: "أنظمة السوق", done: Boolean(training?.metrics?.regime_validation), active: msg.includes("regime") || msg.includes("أنظمة") },
    { key: "final", label: "Final Model", done: doneFinal, active: msg.includes("final model") },
  ];
  el.innerHTML = stages.map((s) => {
    const cls = s.done ? "done" : (s.active ? "active" : "pending");
    const value = s.done ? "مكتمل" : (s.active ? "جارٍ" : "بانتظار");
    return `<div class="stage-card ${cls}">
      <div class="stage-title">${s.label}</div>
      <div class="stage-value">${value}</div>
    </div>`;
  }).join("");
}

function svgLineChart(seriesList, labels, { height = 180 } = {}) {
  if (!seriesList.length || !labels.length) return "لا توجد بيانات كافية للرسم";
  const width = 520;
  const pad = 28;
  const values = seriesList.flatMap((s) => s.values).filter((v) => v != null && !Number.isNaN(Number(v)));
  if (!values.length) return "لا توجد بيانات كافية للرسم";
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const x = (i) => pad + (i * (width - pad * 2)) / Math.max(1, labels.length - 1);
  const y = (v) => height - pad - ((v - min) / (max - min)) * (height - pad * 2);
  const grid = [0, 0.25, 0.5, 0.75, 1].map((g) => {
    const yy = pad + g * (height - pad * 2);
    return `<line x1="${pad}" y1="${yy}" x2="${width - pad}" y2="${yy}" stroke="#e2ddd3" stroke-dasharray="3 3"/>`;
  }).join("");
  const lines = seriesList.map((s) => {
    const points = s.values.map((v, i) => `${x(i)},${y(Number(v ?? min))}`).join(" ");
    return `<polyline fill="none" stroke="${s.color}" stroke-width="3" points="${points}"/>`;
  }).join("");
  const dots = seriesList.map((s) => s.values.map((v, i) => (
    `<circle cx="${x(i)}" cy="${y(Number(v ?? min))}" r="3" fill="${s.color}"/>`
  )).join("")).join("");
  const axisLabels = labels.map((label) => `<span>${label}</span>`).join(" · ");
  return `<div class="chart-wrap">
    <svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}" role="img" aria-label="chart">
      ${grid}
      ${lines}
      ${dots}
    </svg>
    <div class="chart-legend">
      ${seriesList.map((s) => `<span><span class="legend-dot" style="background:${s.color}"></span>${s.label}</span>`).join("")}
    </div>
    <div class="chart-caption">${axisLabels}</div>
  </div>`;
}

function renderTrainingDashboards(training, models) {
  renderStageStrip(training);
  if (training?.llmodel?.exists) {
    const test = training.llmodel.metrics?.test || {};
    const val = training.llmodel.metrics?.validation || {};
    const fin = test.financial || {};
    renderMetricBars("#train-health-bars", [
      { label: "Accuracy", pct: metricPct(test.accuracy, { max: 1 }), value: fmt(test.accuracy, 3), tone: "ok" },
      { label: "Val Acc", pct: metricPct(val.accuracy, { max: 1 }), value: fmt(val.accuracy, 3), tone: "ok" },
      { label: "Sharpe", pct: metricPct(Math.max(0, fin.sharpe || 0), { max: 3 }), value: fmt(fin.sharpe, 2), tone: clsNum(fin.sharpe) },
      { label: "Drawdown", pct: metricPct(Math.abs(fin.max_drawdown || 0), { max: 0.4, invert: true }), value: pct(fin.max_drawdown), tone: "bad" },
    ]);
  } else {
    const cls = training?.metrics?.classification || {};
    const fin = training?.metrics?.financial_oos || {};
    renderMetricBars("#train-health-bars", [
      { label: "Accuracy", pct: metricPct(cls.accuracy, { max: 1 }), value: fmt(cls.accuracy, 3), tone: "ok" },
      { label: "F1", pct: metricPct(cls.f1_macro, { max: 1 }), value: fmt(cls.f1_macro, 3), tone: "ok" },
      { label: "Sharpe", pct: metricPct(Math.max(0, fin.sharpe || 0), { max: 3 }), value: fmt(fin.sharpe, 2), tone: clsNum(fin.sharpe) },
      { label: "Drawdown", pct: metricPct(Math.abs(fin.max_drawdown || 0), { max: 0.4, invert: true }), value: pct(fin.max_drawdown), tone: "bad" },
      { label: "Win Rate", pct: metricPct(fin.win_rate, { max: 1 }), value: pct(fin.win_rate), tone: "ok" },
      { label: "Return", pct: metricPct(Math.max(0, fin.total_return || 0), { max: 1 }), value: pct(fin.total_return), tone: clsNum(fin.total_return) },
    ]);
  }

  const versionTarget = $("#train-version-chart");
  if (versionTarget) {
    const items = [];
    for (const item of (models?.versions || []).slice(0, 6).reverse()) {
      items.push({
        label: (item.meta?.version || "ver").replace("run_", "").slice(-6),
        sharpe: Number(item.metrics?.financial_oos?.sharpe ?? 0),
        ret: Number(item.metrics?.financial_oos?.total_return ?? 0),
      });
    }
    if (training?.llmodel?.exists) {
      items.push({
        label: "LL",
        sharpe: Number(training.llmodel.metrics?.test?.financial?.sharpe ?? 0),
        ret: Number(training.llmodel.metrics?.test?.financial?.total_return ?? 0),
      });
    }
    versionTarget.innerHTML = items.length >= 2
      ? svgLineChart(
          [
            { label: "Sharpe", color: "#8b6914", values: items.map((x) => x.sharpe) },
            { label: "Return", color: "#0f6b45", values: items.map((x) => x.ret) },
          ],
          items.map((x) => x.label),
        )
      : "لا توجد إصدارات كافية للرسم";
  }

  const foldTarget = $("#train-fold-chart");
  if (foldTarget) {
    const folds = training?.metrics?.folds || [];
    if (folds.length) {
      foldTarget.innerHTML = svgLineChart(
        [
          { label: "Accuracy", color: "#171513", values: folds.map((f) => Number(f.accuracy ?? 0)) },
          { label: "F1", color: "#8b6914", values: folds.map((f) => Number(f.f1_macro ?? 0)) },
        ],
        folds.map((f) => `F${f.fold}`),
      );
    } else if (training?.llmodel?.exists) {
      const test = training.llmodel.metrics?.test || {};
      const val = training.llmodel.metrics?.validation || {};
      foldTarget.innerHTML = svgLineChart(
        [
          { label: "Accuracy", color: "#171513", values: [Number(val.accuracy ?? 0), Number(test.accuracy ?? 0)] },
        ],
        ["Validation", "Testing"],
      );
    } else {
      foldTarget.innerHTML = "لا توجد بيانات تحقق كافية للرسم";
    }
  }
}

async function startJob(path, body, label, { showPipelineProgress = false, showPatternProgress = false, showTrainingProgress = false } = {}) {
  toast(`${label}…`);
  const btnPipe = showPipelineProgress ? $("#btn-pipeline") : null;
  const btnPat = showPatternProgress ? $("#btn-pattern-discover") : null;
  const btnTrain = showTrainingProgress ? document.querySelector('[data-run="4"]') : null;
  if (btnPipe) btnPipe.disabled = true;
  if (btnPat) btnPat.disabled = true;
  if (btnTrain) btnTrain.disabled = true;
  if (showPipelineProgress) setPipelineProgress(true, 0, "بدء التحديث…");
  if (showPatternProgress) setPatternProgress(true, 0, "بدء الاستكشاف الشامل…");
  if (showTrainingProgress) {
    setTrainProgress(true, 0, "بدء التدريب والاختبار والتحقق…");
    setTrainingPhaseBars(0, "بدء التدريب");
    renderTrainingLogs([]);
  }
  try {
    const job = await api(path, { method: "POST", body: JSON.stringify(body || {}) });
    if (showPipelineProgress) {
      _activeJobs.pipeline = job.id;
      setPipelineStopEnabled(true);
    }
    if (showTrainingProgress) _activeJobs.training = job.id;
    const done = await pollJob(job.id, {
      intervalMs: (showPipelineProgress || showPatternProgress || showTrainingProgress) ? 600 : 1500,
      onProgress: (j) => {
        if (showPipelineProgress) setPipelineProgress(true, j.progress ?? 0, j.message || label);
        if (showPatternProgress) setPatternProgress(true, j.progress ?? 0, j.message || label);
        if (showTrainingProgress) {
          setTrainProgress(true, j.progress ?? 0, j.message || label);
          setTrainingPhaseBars(j.progress ?? 0, j.message || label);
          renderTrainingLogs(j.logs || []);
          if (j.details) {
            _activeJobs.trainingDetails = j.details;
            renderCurrentRunPanel(_latestTraining || {}, j.details);
          }
        }
      },
    });
    if (done.status === "cancelled") {
      if (showPipelineProgress) setPipelineProgress(true, done.progress ?? 0, done.message || "تم إيقاف التحديث");
      if (showTrainingProgress) setTrainProgress(true, done.progress ?? 0, done.message || "تم إيقاف التدريب");
      throw new Error(done.message || "تم إيقاف المهمة");
    }
    if (done.status === "error") throw new Error(done.error || "فشلت المهمة");
    if (showPipelineProgress) setPipelineProgress(true, 100, "اكتمل تحديث كل الأطر");
    if (showPatternProgress) setPatternProgress(true, 100, "اكتمل الاستكشاف الشامل");
    if (showTrainingProgress) {
      setTrainProgress(true, 100, "اكتمل Training / Testing / Validation وإنشاء Final Model");
      setTrainingPhaseBars(100, "Final Model");
      renderTrainingLogs(done.logs || []);
    }
    toast(`اكتمل: ${label}`);
    await refresh();
    if (showPipelineProgress) setTimeout(() => setPipelineProgress(false), 2500);
    if (showPatternProgress) setTimeout(() => setPatternProgress(false), 3500);
    if (showTrainingProgress) setTimeout(() => setTrainProgress(false), 4000);
    return done;
  } catch (e) {
    if (showPatternProgress) setPatternProgress(true, 0, e.message || "فشل");
    if (showPipelineProgress) setPipelineProgress(true, 0, e.message || "فشل");
    if (showTrainingProgress) setTrainProgress(true, 0, e.message || "فشل");
    if (showTrainingProgress) setTrainingPhaseBars(0, e.message || "فشل");
    throw e;
  } finally {
    if (showPipelineProgress) {
      _activeJobs.pipeline = null;
      setPipelineStopEnabled(false);
    }
    if (showTrainingProgress) _activeJobs.training = null;
    if (showTrainingProgress) _activeJobs.trainingDetails = null;
    if (btnPipe) btnPipe.disabled = false;
    if (btnPat) btnPat.disabled = false;
    if (btnTrain) btnTrain.disabled = false;
  }
}

function applyLiveSettingsForm(s) {
  if (!s) return;
  setLiveSettingsCache(s);
  const toggle = $("#setting-use-spread-filter");
  if (toggle) toggle.checked = !!s.use_live_spread_filter;
  const maxSp = $("#setting-max-spread");
  if (maxSp) maxSp.value = s.max_entry_spread_pips ?? 12;
  const tight = $("#setting-tight-spread");
  if (tight) tight.value = s.tight_spread_pips ?? 12;
  const maxEnt = $("#setting-max-entries");
  if (maxEnt) maxEnt.value = s.max_entries_per_cycle ?? 8;
  const banner = $("#spread-mode-banner");
  if (banner) {
    if (s.use_live_spread_filter) {
      banner.className = "banner ok";
      banner.textContent =
        `الوضع النشط: موديل ثم سبريد · الدخول فقط إذا السبريد ≤ ${s.max_entry_spread_pips} pip · حتى ${s.max_entries_per_cycle} صفقات عند السبريد الضيق`;
    } else {
      banner.className = "banner warn";
      banner.textContent =
        "الوضع النشط: موديل فقط · فلتر السبريد ملغى · صفقة واحدة لكل إشارة صالحة من الموديل";
    }
  }
  const extrasDisabled = !s.use_live_spread_filter;
  [maxSp, tight, maxEnt].forEach((el) => {
    if (el) el.disabled = extrasDisabled;
  });
}

async function loadLiveSettings() {
  const s = await api("/api/settings/live");
  applyLiveSettingsForm(s);
  return s;
}

async function saveLiveSettings() {
  const body = {
    use_live_spread_filter: !!$("#setting-use-spread-filter")?.checked,
    max_entry_spread_pips: Number($("#setting-max-spread")?.value || 12),
    tight_spread_pips: Number($("#setting-tight-spread")?.value || 12),
    max_entries_per_cycle: Number($("#setting-max-entries")?.value || 8),
  };
  const s = await api("/api/settings/live", {
    method: "POST",
    body: JSON.stringify(body),
  });
  applyLiveSettingsForm(s);
  toast(s.use_live_spread_filter ? "تم التفعيل: موديل ثم سبريد" : "تم الإلغاء: موديل فقط");
  return s;
}

function switchTab(name) {
  $$(".nav-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${name}`));
  saveActiveTab(name);
  if (name === "data") {
    refreshDataTab().catch((e) => toast(e.message));
  }
  if (name === "settings") {
    loadLiveSettings().catch((e) => toast(e.message));
  }
}

function stageLabelAr(stage) {
  const map = {
    queued: "قيد الانتظار",
    loading_data: "تحميل بيانات",
    features: "هندسة ميزات",
    labeling: "تسمية",
    feature_selection: "اختيار ميزات",
    walk_forward: "تدريب طيات",
    val_policy: "Validation / سياسة",
    test_oos: "Test OOS",
    regime_validation: "تحقق أنظمة السوق",
    stress_mc: "ضغط / مونت كارلو",
    data_intelligence: "ذكاء البيانات",
    feature_intelligence: "ذكاء الميزات",
    model_zoo: "مقارنة النماذج",
    deploy_holdout: "Deploy holdout",
    gates: "بوابات",
    done: "تم",
    error: "خطأ",
    skipped: "لم يُدرَّب",
    passed: "اجتاز",
    rejected: "رُفض",
    not_trained_this_run: "لم يُدرَّب في هذا التشغيل",
  };
  return map[stage] || stage || "—";
}

const REGIME_LABELS_AR = {
  trending: "ترند",
  ranging: "رينج",
  high_volatility: "تقلب عالٍ",
  low_volatility: "تقلب منخفض",
};

function tiePill(label, value, tone = "") {
  return `<span class="tie-pill ${tone}"><span>${label}</span><b>${value}</b></span>`;
}

function renderTradingIntelligencePanel(t) {
  const chip = $("#tie-pipeline-chip");
  const kpis = $("#tie-kpis");
  const regimeGrid = $("#tie-regime-grid");
  const regimeMeta = $("#tie-regime-meta");
  const advTable = $("#tie-advanced-table");
  const knowBox = $("#tie-knowledge-box");
  const zooTable = $("#tie-zoo-table");
  if (!kpis || !regimeGrid) return;

  const m = {
    ...(t?.metrics || {}),
  };
  // Matrix / compact status may hoist enterprise fields to the top level.
  for (const k of [
    "live_readiness", "model_zoo", "stress_testing", "monte_carlo",
    "intelligent_critique", "self_optimize", "self_optimize_applied",
    "regime_validation", "advanced_eval", "knowledge_loop", "validation_mode",
    "pipeline_version", "label_quality", "feature_explainability",
    "champion_challenger", "smart_recommendations", "nested_hp",
  ]) {
    if ((m[k] == null || (typeof m[k] === "object" && !Object.keys(m[k] || {}).length)) && t?.[k] != null) {
      m[k] = t[k];
    }
  }
  const fin = m.financial_oos || {};
  const regime = m.regime_validation || {};
  const adv = m.advanced_eval || {};
  const dsr = adv.deflated_sharpe || {};
  const pbo = adv.pbo || {};
  const exe = adv.execution || {};
  const know = m.knowledge_loop || {};
  const valMode = m.validation_mode || (m.validation || {}).validation_mode || "—";
  const pipe = m.pipeline_version || t?.pipeline_version || "—";

  const ready = m.live_readiness || {};
  const zoo = m.model_zoo || {};
  const stress = m.stress_testing || {};
  const mc = m.monte_carlo || {};
  const critique = m.intelligent_critique || {};
  const selfOpt = m.self_optimize || {};
  const appliedOpt = m.self_optimize_applied || {};
  const labelQ = m.label_quality || {};
  const featX = m.feature_explainability || {};
  const cc = m.champion_challenger || {};
  const recs = m.smart_recommendations || {};
  const nested = m.nested_hp || {};
  const v15Table = $("#tie-v15-table");
  const recsBox = $("#tie-recs-box");
  const shapTable = $("#tie-shap-table");

  if (chip) {
    const isV16 = String(pipe).includes("v16") || String(pipe).includes("research-factory");
    const isV15 = String(pipe).includes("v15") || String(pipe).includes("intelligent-training");
    const isEnt = String(pipe).includes("v14") || String(pipe).includes("enterprise");
    const isTie = String(pipe).includes("v13") || String(pipe).includes("trading-intelligence");
    chip.textContent = isV16
      ? `Research Factory · ${pipe}`
      : (isV15
        ? `Intelligent · ${pipe}`
        : (isEnt ? `Enterprise · ${pipe}` : (isTie ? `TIE · ${pipe}` : `pipeline ${pipe}`)));
  }

  if (t?.empty && !fin.sharpe && !Object.keys(regime).length) {
    kpis.innerHTML = "";
    regimeGrid.innerHTML = `<div class="card-note">شغّل التدريب لعرض مقاييس محرك التدريب الذكي لهذا الإطار.</div>`;
    if (regimeMeta) regimeMeta.innerHTML = "";
    if (advTable) advTable.innerHTML = "";
    if (knowBox) knowBox.textContent = "لا توجد حلقات معرفة بعد لهذا الإطار.";
    if (zooTable) zooTable.innerHTML = `<tr><td colspan="3" class="muted">—</td></tr>`;
    if (v15Table) v15Table.innerHTML = "";
    if (recsBox) recsBox.textContent = "لا توجد توصيات بعد.";
    if (shapTable) shapTable.innerHTML = `<tr><td colspan="3" class="muted">—</td></tr>`;
    return;
  }

  kpis.innerHTML = [
    kpi("Readiness", `${fmt(ready.score, 0)}/100`, ready.score >= 75 ? "num-ok" : (ready.score < 55 ? "num-bad" : "num-warn")),
    kpi("Labels", `${fmt(labelQ.score, 0)}/100`, Number(labelQ.score) >= 70 ? "num-ok" : (Number(labelQ.score) < 45 ? "num-bad" : "")),
    kpi("Expectancy", fmt(fin.expectancy, 5), clsNum(fin.expectancy)),
    kpi("Challenger", cc.promote ? "ترقية" : (cc.decision || "—").replace(/_/g, " ").slice(0, 18)),
  ].join("");

  const regimes = regime.regimes || {};
  const regimeKeys = ["trending", "ranging", "high_volatility", "low_volatility"];
  regimeGrid.innerHTML = regimeKeys.map((key) => {
    const row = regimes[key] || {};
    if (row.skipped || row.error) {
      return `<div class="regime-card skip">
        <div class="rg-title">${REGIME_LABELS_AR[key] || key}</div>
        <div>${row.reason || row.error || "عينة غير كافية"}</div>
        <div class="muted">bars ${row.n_bars ?? "—"} · trades ${row.n_trades ?? "—"}</div>
      </div>`;
    }
    const sh = Number(row.sharpe || 0);
    const tone = sh >= 0.5 ? "ok" : (sh < 0 ? "bad" : "warn");
    return `<div class="regime-card ${tone}">
      <div class="rg-title">${REGIME_LABELS_AR[key] || key}</div>
      <div>Sharpe <b class="${clsNum(row.sharpe)}">${fmt(row.sharpe, 2)}</b> · Sortino ${fmt(row.sortino, 2)}</div>
      <div>Exp ${fmt(row.expectancy, 4)} · PF ${fmt(row.profit_factor, 2)}</div>
      <div class="muted">trades ${fmt(row.n_trades, 0)} · DD ${pct(row.max_drawdown)}</div>
    </div>`;
  }).join("");

  if (regimeMeta) {
    regimeMeta.innerHTML = kvRows([
      ["ثبات الأنظمة", regime.stable === false ? "غير مستقر" : (regime.stable === true ? "مستقر" : "—")],
      ["فرق Sharpe بين الأنظمة", fmt(regime.sharpe_regime_spread, 2)],
      ["جاهزية حية", `${fmt(ready.score, 0)}/100 · ${ready.verdict_ar || ready.verdict || "—"}`],
      ["فائز Model Zoo", zoo.winner || "—"],
      ["ملاحظات", (regime.notes || []).join(" · ") || "—"],
    ]);
  }

  if (advTable) {
    advTable.innerHTML = kvRows([
      ["وضع التحقق", valMode],
      ["Nested HP", `${nested.mode || (nested.enabled ? "single" : "—")} · ${nested.best_family || "—"}`],
      ["DSR", fmt(dsr.deflated_sharpe, 4)],
      ["PBO", fmt(pbo.pbo, 4)],
      ["Stress robust / worst Sh", `${stress.robust ?? "—"} / ${fmt(stress.worst_sharpe, 2)}`],
      ["MC p_profit / p_dd>25%", `${fmt(mc.p_profit, 3)} / ${fmt(mc.p_dd_gt_25pct, 3)}`],
      ["Zoo tried", fmt(zoo.n_models_tried, 0)],
      ["Root cause", critique.root_cause || "—"],
      ["Strengths", (critique.strengths || []).join(", ") || "—"],
      ["Weaknesses", (critique.weaknesses || []).join(", ") || "—"],
      ["Self-opt notes", (selfOpt.notes || []).slice(0, 2).join(" · ") || "—"],
      ["Applied overrides", Object.keys(appliedOpt).length ? Object.keys(appliedOpt).slice(0, 4).join(", ") : "—"],
      ["Risk-adj return R/|DD|", fmt(fin.risk_adjusted_return, 3)],
      ["Latency / Delay", `${fmt(exe.latency_bars, 0)} / ${fmt(exe.execution_delay_bars, 0)}`],
      ["Dynamic costs", exe.dynamic_costs ? "نعم" : "لا"],
    ]);
  }

  if (zooTable) {
    const ranking = zoo.ranking || [];
    if (!ranking.length) {
      zooTable.innerHTML = `<tr><td colspan="3" class="muted">لا توجد مقارنة نماذج بعد</td></tr>`;
    } else {
      zooTable.innerHTML = ranking.slice(0, 12).map((r, i) => {
        const name = r.family || r.name || r.model || `model_${i + 1}`;
        const acc = r.inner_val_acc ?? r.val_acc ?? r.score ?? r.accuracy;
        const isWin = zoo.winner && String(zoo.winner) === String(name);
        return `<tr>
          <td>${name}${isWin ? " ★" : ""}</td>
          <td>${fmt(acc, 4)}</td>
          <td>${i + 1}</td>
        </tr>`;
      }).join("");
    }
  }

  if (v15Table) {
    const stab = featX.stability || {};
    const noise = (labelQ.noise || {}).noise_rate;
    v15Table.innerHTML = kvRows([
      ["جودة Labels", `${fmt(labelQ.score, 0)}/100`],
      ["ضوضاء Labels", noise != null ? fmt(noise, 3) : "—"],
      ["ملخص Labels", labelQ.summary_ar || "—"],
      ["SHAP", (featX.shap || {}).enabled ? "مفعّل" : "غير متاح"],
      ["استقرار الميزات", stab.summary_ar || fmt(stab.mean_jaccard, 3)],
      ["Champion/Challenger", cc.summary_ar || cc.decision || "—"],
      ["Δ Score / Sharpe", `${fmt(cc.score_delta, 3)} / ${fmt(cc.sharpe_delta, 3)}`],
    ]);
  }

  if (shapTable) {
    const consensus = featX.consensus_top || [];
    const shapTop = ((featX.shap || {}).top) || [];
    const rows = consensus.length
      ? consensus.slice(0, 12).map((r) => ({
          feature: r.feature,
          score: r.score,
          src: "consensus",
        }))
      : shapTop.slice(0, 12).map((r) => ({
          feature: r.feature,
          score: r.shap_share,
          src: "shap",
        }));
    if (!rows.length) {
      shapTable.innerHTML = `<tr><td colspan="3" class="muted">لا توجد تفسيرات ميزات بعد — شغّل تدريب v15</td></tr>`;
    } else {
      shapTable.innerHTML = rows.map((r) => `<tr>
        <td>${r.feature}</td>
        <td>${fmt(r.score, 4)}</td>
        <td>${r.src}</td>
      </tr>`).join("");
    }
  }

  if (recsBox) {
    const items = recs.items || [];
    if (!items.length && !recs.executive_ar) {
      recsBox.textContent = "لا توجد توصيات بعد لهذا الإطار.";
    } else {
      recsBox.innerHTML = `
        <div><b>${recs.executive_ar || "توصيات التحسين"}</b></div>
        <ul style="margin:0.4rem 0 0;padding-inline-start:1.1rem">
          ${items.slice(0, 5).map((it) => `<li><b>P${it.priority}</b> ${it.ar || it.en || it.code}</li>`).join("")}
        </ul>
      `;
    }
  }

  if (knowBox) {
    const advK = know.advisory || {};
    const ema = know.performance_ema || {};
    if (!know.n_episodes && !advK.reason && !ready.score) {
      knowBox.textContent = "لا توجد حلقات معرفة بعد لهذا الإطار — ستُحدَّث بعد أول تدريب v15.";
    } else {
      const retrain = advK.retrain_suggested
        ? `<span class="num-bad">يُنصح بإعادة التدريب</span> (${advK.reason || "—"})`
        : `<span class="num-ok">لا حاجة فورية</span> (${advK.reason || "ok"})`;
      knowBox.innerHTML = `
        <div><b>${ready.verdict_ar || "—"}</b> · درجة ${fmt(ready.score, 0)}/100</div>
        <div>الحلقات: <b>${know.n_episodes ?? "—"}</b> · Zoo: <b>${zoo.winner || "—"}</b></div>
        <div>EMA Sharpe: <b class="${clsNum(ema.sharpe)}">${fmt(ema.sharpe, 3)}</b>
          · Crit: <b>${critique.root_cause || "—"}</b></div>
        <div>توصية: ${retrain}</div>
        ${know.path ? `<div class="muted" style="margin-top:0.35rem;word-break:break-all">${know.path}</div>` : ""}
      `;
    }
  }
}

function gateChipsHtml(details, keys) {
  const list = (details && details.length)
    ? details
    : (keys || []).map((k) => ({ key: k, ar: k }));
  if (!list.length) return "";
  return list.map((g) => `<span class="gate-chip" title="${g.key || ""}">${g.ar || g.key}</span>`).join("");
}

function renderCurrentRunPanel(t, liveDetails) {
  const panelMeta = $("#current-run-meta");
  const pipeEl = $("#current-run-pipeline");
  const cardsEl = $("#current-run-tf-cards");
  const timelineEl = $("#train-timeline");
  if (!cardsEl) return;

  const summary = t?.summary || {};
  const live = liveDetails || {};
  const liveTfs = live.timeframes || {};
  const matrix = (Object.keys(liveTfs).length
    ? Object.values(liveTfs)
    : (t?.matrix_current_run || []));
  const pipeline = live.pipeline_version || summary.pipeline_version || t?.last_run?.pipeline_version || "—";
  if (pipeEl) pipeEl.textContent = `pipeline ${pipeline}`;

  const trained = summary.current_run_trained ?? matrix.filter((m) => m.source === "current_run" && !m.empty && m.status !== "not_trained_this_run").length;
  const passed = summary.current_run_passed_gates ?? matrix.filter((m) => m.passed_gates).length;
  const rejected = summary.current_run_rejected ?? matrix.filter((m) => m.passed_gates === false && !m.error).length;
  const errors = summary.current_run_errors ?? matrix.filter((m) => m.error).length;
  if (panelMeta) {
    const reasons = (summary.current_run_reject_reasons || []).slice(0, 4).join(" · ") || "—";
    panelMeta.innerHTML = `دُرِّب <b>${trained ?? "—"}</b> · اجتاز <b>${passed ?? 0}</b> · رُفض <b>${rejected ?? 0}</b> · أخطاء <b>${errors ?? 0}</b>`
      + (summary.run_id ? ` · run <code>${summary.run_id}</code>` : "")
      + `<br/>أسباب الرفض: ${reasons}`;
  }

  if (timelineEl) {
    const stages = [
      "Data", "DataIntel", "Features", "ModelZoo", "Walk-Forward", "Val Policy", "Test OOS",
      "Regime", "Stress/MC", "Deploy", "Gates", "Readiness", "FinalModel",
    ];
    const activeMsg = ($("#train-progress-step")?.textContent || "").toLowerCase();
    timelineEl.innerHTML = stages.map((s) => {
      const key = s.toLowerCase();
      const on = activeMsg.includes(key)
        || (key === "regime" && (activeMsg.includes("أنظمة") || activeMsg.includes("regime")))
        || (key === "knowledge" && activeMsg.includes("knowledge"))
        || (key === "dsr/pbo" && (activeMsg.includes("dsr") || activeMsg.includes("pbo") || activeMsg.includes("advanced")));
      return `<div class="stage-item ${on ? "active" : ""}"><span>${s}</span></div>`;
    }).join("");
  }

  cardsEl.innerHTML = matrix.length
    ? matrix.map((m) => {
        const mm = m.metrics || m;
        const status = m.stage || m.status || (m.passed_gates ? "passed" : (m.empty ? "not_trained_this_run" : "rejected"));
        const tone = m.error || status === "error" ? "bad"
          : (m.passed_gates === true || status === "passed" || status === "done" && m.passed_gates) ? "ok"
          : (status === "not_trained_this_run" || status === "queued") ? "warn"
          : (["walk_forward", "loading_data", "features", "val_policy", "test_oos", "regime_validation", "gates", "deploy_holdout", "data_intelligence", "feature_intelligence", "model_zoo", "stress_mc"].includes(status) ? "running" : "warn");
        const gates = gateChipsHtml(m.gate_failures_detail, m.gate_failures);
        const sharpe = mm.sharpe ?? mm.test_sharpe;
        const unc = mm.sharpe_uncapped;
        const ci = mm.sharpe_ci_low;
        const folds = m.folds || [];
        const regime = m.regime_validation || {};
        const adv = m.advanced_eval || {};
        const dsr = (adv.deflated_sharpe || {}).deflated_sharpe;
        const pbo = (adv.pbo || {}).pbo;
        const know = m.knowledge_loop || {};
        const bestVal = folds.length
          ? folds.reduce((a, b) => (Number(b.val_sharpe || 0) > Number(a.val_sharpe || 0) ? b : a), folds[0])
          : null;
        const foldsHtml = folds.length
          ? `<details class="folds-toggle"><summary>طيات Walk-Forward (${folds.length})</summary>
              <div class="table-scroll"><table class="data"><thead><tr>
                <th>fold</th><th>acc</th><th>f1</th><th>auc</th><th>rate</th><th>val_sh</th><th>n_val</th>
              </tr></thead><tbody>
              ${folds.map((f) => {
                const mark = bestVal && f.fold === bestVal.fold ? " ★" : "";
                return `<tr>
                  <td>${(f.fold ?? 0) + 1}${mark}</td>
                  <td>${fmt(f.accuracy, 3)}</td>
                  <td>${fmt(f.f1_macro, 3)}</td>
                  <td>${fmt(f.roc_auc_ovr, 3)}</td>
                  <td>${fmt(f.trade_rate, 2)}</td>
                  <td class="${clsNum(f.val_sharpe)}">${fmt(f.val_sharpe, 2)}</td>
                  <td>${fmt(f.n_val_trades, 0)}</td>
                </tr>`;
              }).join("")}
              </tbody></table></div></details>`
          : "";
        const regimeTone = regime.stable === false ? "bad" : (regime.stable === true ? "ok" : "");
        const pboTone = Number(pbo) >= 0.55 ? "bad" : (pbo != null ? "ok" : "");
        const tieStrip = `<div class="tie-live-strip">
          ${tiePill("mode", m.validation_mode || "expanding")}
          ${tiePill("Exp", fmt(mm.expectancy, 4), clsNum(mm.expectancy) === "num-ok" ? "ok" : (clsNum(mm.expectancy) === "num-bad" ? "bad" : ""))}
          ${tiePill("Sortino", fmt(mm.sortino, 2), clsNum(mm.sortino) === "num-ok" ? "ok" : "")}
          ${tiePill("DSR", fmt(dsr, 3))}
          ${tiePill("PBO", fmt(pbo, 3), pboTone)}
          ${tiePill("Regime", regime.stable === false ? "unstable" : (regime.stable === true ? "stable" : "—"), regimeTone)}
          ${tiePill("Ready", `${fmt((m.live_readiness || {}).score, 0)}`, Number((m.live_readiness || {}).score) >= 75 ? "ok" : (Number((m.live_readiness || {}).score) < 55 ? "bad" : "warn"))}
          ${(m.model_zoo || {}).winner ? tiePill("Zoo", String((m.model_zoo || {}).winner)) : ""}
          ${know.advisory?.retrain_suggested ? tiePill("Knowledge", "retrain", "warn") : tiePill("Knowledge", String(know.n_episodes ?? "—"))}
        </div>`;
        return `<div class="tf-status-card ${tone}">
          <h3>${m.timeframe || "—"} · ${stageLabelAr(status)}</h3>
          <div class="meta">${m.pipeline_version || pipeline} · ${m.model_version || m.version || "—"}
            ${m.liquidity_rescue ? " · liquidity rescue" : ""}</div>
          <div class="metric-line"><span>Acc / AUC</span><b>${fmt(mm.acc ?? mm.accuracy, 3)} / ${fmt(mm.auc ?? m.auc, 3)}</b></div>
          <div class="metric-line"><span>Train / Val / Test / Deploy</span>
            <b>${fmt(mm.train_sharpe, 2)} / ${fmt(mm.val_sharpe ?? m.val_sharpe, 2)} / ${fmt(sharpe, 2)} / ${fmt(mm.deploy_sharpe ?? m.deploy_sharpe, 2)}</b>
          </div>
          <div class="metric-line"><span>Sharpe CI low</span><b>${fmt(ci, 2)}</b></div>
          <div class="metric-line"><span class="muted">Uncapped</span><span class="muted">${fmt(unc, 2)}</span></div>
          <div class="metric-line"><span>Trades T/V/Te/D</span>
            <b>${fmt(mm.n_trades_train, 0)}/${fmt(mm.n_trades_val, 0)}/${fmt(mm.n_trades_test ?? m.n_trades_test, 0)}/${fmt(mm.n_trades_deploy ?? m.deploy_trades, 0)}</b>
          </div>
          <div class="metric-line"><span>Fit</span><b>${(m.fit_diagnosis?.status || m.fit_status || "—")}</b></div>
          <div class="metric-line"><span>gap TV / VT</span><b>${fmt(mm.gap_tv ?? m.sharpe_gap_tv, 2)} / ${fmt(mm.gap_vt ?? m.sharpe_gap_vt, 2)}</b></div>
          ${tieStrip}
          <div>${gates || (m.passed_gates ? '<span class="gate-chip" style="background:var(--ok-bg);color:var(--ok);border-color:#c6e6d5">اجتاز</span>' : "")}</div>
          ${foldsHtml}
        </div>`;
      }).join("")
    : `<div class="card-note">لا توجد بطاقات لهذا التشغيل بعد.</div>`;
}

function renderTraining(t) {
  _latestTraining = t;
  const banner = $("#train-banner");
  const matrix = t.matrix_champion || t.matrix || [];
  const currentMatrix = t.matrix_current_run || [];
  const summary = t.summary || {};
  const ll = t.llmodel;

  // Fill TF selector
  const sel = $("#tf-select");
  const tfs = t.all_timeframes || matrix.map((m) => m.timeframe);
  const selected = t.selected_timeframe || t.final_model?.timeframe || t.timeframe || "H1";
  if (sel && sel.options.length !== tfs.length) {
    sel.innerHTML = tfs.map((tf) => `<option value="${tf}">${tf}</option>`).join("");
  }
  if (sel) sel.value = selected;

  renderCurrentRunPanel(t, _activeJobs.trainingDetails || null);
  renderTradingIntelligencePanel(t);

  // Current-run matrix
  const curBody = $("#train-matrix-current");
  if (curBody) {
    curBody.innerHTML = currentMatrix.length
      ? currentMatrix.map((m) => {
          if (m.status === "not_trained_this_run" || m.empty) {
            return `<tr>
              <td><b>${m.timeframe}</b></td>
              <td class="num-warn">لم يُدرَّب في هذا التشغيل</td>
              <td colspan="9">—</td>
              <td></td>
            </tr>`;
          }
          const statusAr = m.error ? "خطأ" : (m.passed_gates ? "اجتاز" : "رُفض");
          const statusCls = m.error ? "num-bad" : (m.passed_gates ? "num-ok" : "num-bad");
          const regime = m.regime_validation || {};
          const pbo = ((m.advanced_eval || {}).pbo || {}).pbo;
          const regimeLabel = regime.stable === false ? "unstable" : (regime.stable === true ? "stable" : "—");
          const regimeCls = regime.stable === false ? "num-bad" : (regime.stable === true ? "num-ok" : "");
          const pboCls = Number(pbo) >= 0.55 ? "num-bad" : "";
          return `<tr>
            <td><b>${m.timeframe}</b></td>
            <td class="${statusCls}">${statusAr}</td>
            <td>${fmt(m.accuracy, 3)}</td>
            <td>${fmt(m.auc, 3)}</td>
            <td class="${clsNum(m.sharpe)}">${fmt(m.sharpe, 2)}</td>
            <td class="${clsNum(m.expectancy)}">${fmt(m.expectancy, 4)}</td>
            <td class="${clsNum(m.sortino)}">${fmt(m.sortino, 2)}</td>
            <td class="${regimeCls}">${regimeLabel}</td>
            <td class="${pboCls}">${fmt(pbo, 3)}</td>
            <td>${fmt(m.sharpe_ci_low, 2)}</td>
            <td>${fmt(m.n_trades_test, 0)}</td>
            <td>${gateChipsHtml(m.gate_failures_detail, m.gate_failures) || (m.passed_gates ? "اجتاز" : "—")}</td>
            <td><button class="btn" data-select-tf="${m.timeframe}" type="button">تفاصيل</button></td>
          </tr>`;
        }).join("")
      : `<tr><td colspan="13">لا نتائج لهذا التشغيل بعد — اضغط تدريب كل الأطر (M1…H4)</td></tr>`;
  }

  // Champion / latest matrix
  $("#train-matrix").innerHTML = matrix.length
    ? matrix.map((m) => {
        const status = m.empty ? "غير مدرّب" : "أثر محفوظ";
        const statusCls = m.empty ? "num-bad" : "num-ok";
        return `<tr>
          <td><b>${m.timeframe}</b></td>
          <td class="${statusCls}">${status}</td>
          <td>${m.rows ?? "—"}</td>
          <td>${fmt(m.accuracy, 3)}</td>
          <td>${fmt(m.f1, 3)}</td>
          <td class="${clsNum(m.val_sharpe)}">${fmt(m.val_sharpe, 2)}</td>
          <td class="${clsNum(m.sharpe)}">${fmt(m.sharpe, 2)}</td>
          <td class="num-bad">${pct(m.max_drawdown)}</td>
          <td class="${clsNum(m.sum_trade_returns)}">${fmt(m.sum_trade_returns, 3)}</td>
          <td>${m.passed_gates ? "اجتاز" : (m.empty ? "—" : "رفض")}</td>
          <td><button class="btn" data-select-tf="${m.timeframe}" type="button">تفاصيل</button></td>
        </tr>`;
      }).join("")
    : `<tr><td colspan="11">لا نتائج محفوظة بعد</td></tr>`;

  $$("[data-select-tf]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const tf = btn.getAttribute("data-select-tf");
      try {
        const details = await api(`/api/training/details?timeframe=${tf}`);
        renderTraining(details);
        renderTrainingVersions(_latestModels, details);
        renderTrainingDashboards(details, _latestModels);
      } catch (e) { toast(e.message); }
    });
  });

  if (sel && !sel._bound) {
    sel._bound = true;
    sel.addEventListener("change", async () => {
      try {
        const details = await api(`/api/training/details?timeframe=${sel.value}`);
        renderTraining(details);
        renderTrainingVersions(_latestModels, details);
        renderTrainingDashboards(details, _latestModels);
      } catch (e) { toast(e.message); }
    });
  }

  const runPassed = summary.current_run_passed_gates ?? t.last_run?.summary?.passed_gates ?? 0;
  const runTrained = summary.current_run_trained ?? t.last_run?.summary?.trained ?? 0;
  const runRejected = summary.current_run_rejected ?? t.last_run?.summary?.rejected ?? 0;
  const runErrors = summary.current_run_errors ?? t.last_run?.summary?.errors ?? 0;
  const champ = t.final_model || {};
  const champStale = !!(champ.kept_existing || champ.skipped_downgrade || champ.champion_from_prior_run)
    || summary.champion_from_this_run === false
    || summary.final_model_from_prior_run;
  const reasons = (summary.current_run_reject_reasons || t.last_run?.summary?.reject_reasons || []).slice(0, 5);
  if (t.last_run?.summary || currentMatrix.length) {
    banner.className = runPassed > 0 ? "banner ok" : "banner warn";
    banner.innerHTML = `هذا التشغيل: دُرِّب <b>${runTrained}</b> · اجتاز <b>${runPassed}</b> · رُفض <b>${runRejected}</b> · أخطاء <b>${runErrors}</b>`
      + `<br/>البطل: <b>${champ.timeframe || summary.final_model_tf || "—"}</b> · <b>${champ.mode || summary.final_model_mode || "—"}</b>`
      + (champStale ? " (محفوظ سابقاً — ليس من هذا التشغيل)" : " (من هذا التشغيل)")
      + (reasons.length ? `<br/>رفض: ${reasons.join(" · ")}` : "");
  } else if (ll?.exists) {
    banner.className = "banner ok";
    banner.innerHTML = `تم إنشاء <b>LLModel</b> متعدد الأطر · الأساسي <b>${ll?.metadata?.base_timeframe || "—"}</b>`;
  } else if (t.final_model?.exists) {
    banner.className = "banner warn";
    banner.innerHTML = `بطل محفوظ فقط · الإطار <b>${t.final_model.timeframe || "—"}</b> — لا تخلطه مع نتيجة التشغيل الحالي`;
  } else if (!t || (t.empty && !(summary.ready > 0))) {
    banner.className = "banner warn";
    banner.textContent = "لا توجد نماذج مدرّبة بعد. اضغط «تدريب كل الأطر الآن».";
  } else {
    banner.className = "banner ok";
    banner.innerHTML = `آثار محفوظة <b>${summary.ready || 0}</b>/<b>${summary.total || 7}</b> · الإطار المعروض: <b>${selected}</b>`;
  }

  if (ll?.exists) {
    const test = ll.metrics?.test || {};
    const val = ll.metrics?.validation || {};
    const fin = test.financial || {};
    const valFin = val.financial || {};
    $("#train-status-table").innerHTML = kvRows([
      ["نوع النموذج", "LLModel"],
      ["الرمز", ll.metadata?.symbol || "—"],
      ["الإطار الأساسي", ll.metadata?.base_timeframe || "—"],
      ["الأطر الزمنية", (ll.metadata?.timeframes || []).join(" · ") || "—"],
      ["اسم Artifact", "LLModel"],
      ["المسار", ll.artifact_path || "—"],
      ["Final Model", ll.metadata?.final_model_ready ? "جاهز" : "جاهز بعد التدريب"],
    ]);
    $("#train-data-table").innerHTML = kvRows([
      ["عدد العينات", ll.metrics?.rows ?? "—"],
      ["طول التسلسل", ll.metrics?.sequence_length ?? "—"],
      ["ميزات السياق", ll.metrics?.context_features ?? "—"],
      ["عدد الأطر", (ll.metadata?.timeframes || []).length],
      ["Validation Accuracy", fmt(val.accuracy, 3)],
      ["Validation Sharpe", fmt(valFin.sharpe, 2)],
    ]);
    $("#cls-kpis").innerHTML = [
      kpi("Accuracy", fmt(test.accuracy, 3)),
      kpi("Base TF", ll.metadata?.base_timeframe || "—"),
      kpi("TF Count", fmt((ll.metadata?.timeframes || []).length, 0)),
      kpi("Context", fmt(ll.metrics?.context_features, 0)),
    ].join("");
    $("#fin-kpis").innerHTML = [
      kpi("Sharpe", fmt(fin.sharpe, 2), clsNum(fin.sharpe)),
      kpi("Max DD", pct(fin.max_drawdown), "num-bad"),
      kpi("Win Rate", pct(fin.win_rate)),
      kpi("Return", pct(fin.total_return), clsNum(fin.total_return)),
    ].join("");
    $("#cls-table").innerHTML = kvRows([["Accuracy", fmt(test.accuracy, 4)]]);
    $("#fin-table").innerHTML = kvRows([
      ["Sharpe", fmt(fin.sharpe, 4)],
      ["Max Drawdown", pct(fin.max_drawdown)],
      ["Win Rate", pct(fin.win_rate)],
      ["Total Return", pct(fin.total_return)],
    ]);
    $("#compare-table").innerHTML = `<tr><td>LLModel</td><td>${fmt(fin.sharpe, 2)}</td><td>—</td><td>${pct(fin.max_drawdown)}</td><td>${pct(fin.win_rate)}</td><td>—</td><td>${pct(fin.total_return)}</td></tr>`;
    $("#folds-table").innerHTML = `<tr><td colspan="6">التدريب العميق الحالي يعتمد على split زمني موحد متعدد الأطر، وليس طيات Walk-Forward التقليدية.</td></tr>`;
    $("#feature-cloud").innerHTML = Object.entries(ll.metrics?.timeframe_feature_counts || {})
      .map(([tf, n]) => `<span class="chip">${tf}: ${n} features</span>`).join("");
    $("#st-train").textContent = "LLModel";
    $("#st-train").className = "ok";
    renderTradingIntelligencePanel({ empty: true, metrics: {} });
    return;
  }

  if (t.empty) {
    $("#train-status-table").innerHTML = kvRows([
      ["الإطار", selected],
      ["الحالة", "لا يوجد نموذج لهذا الإطار"],
    ]);
    $("#train-data-table").innerHTML = kvRows([
      ["بيانات الميزات", (t.dataset && t.dataset.rows) ? `${t.dataset.rows} شمعة` : "—"],
      ["الفترة", t.dataset ? `${fmtTs(t.dataset.first_ts)} → ${fmtTs(t.dataset.last_ts)}` : "—"],
    ]);
    $("#cls-kpis").innerHTML = "";
    $("#fin-kpis").innerHTML = "";
    $("#cls-table").innerHTML = "";
    $("#fin-table").innerHTML = "";
    $("#compare-table").innerHTML = "";
    $("#folds-table").innerHTML = "";
    $("#feature-cloud").innerHTML = "";
    $("#st-train").textContent = `${summary.ready || 0}/${summary.total || 7}`;
    $("#st-train").className = summary.ready ? "warn" : "bad";
    renderTradingIntelligencePanel(t);
    return;
  }

  const passed = !!t.passed_gates;
  $("#st-train").textContent = `${summary.ready || 0}/${summary.total || 7}`;
  $("#st-train").className = summary.ready === summary.total ? "ok" : "warn";

  const meta = t.metadata || {};
  const d = t.dataset || {};
  const m = t.metrics || {};
  const cfg = t.training_config || {};
  const dsMeta = m.data_sources || meta.data_sources || {};
  const valMeta = m.validation || {};

  $("#train-status-table").innerHTML = kvRows([
    ["الرمز", t.symbol],
    ["الإطار الزمني", t.timeframe],
    ["نسخة النموذج", t.version || "—"],
    ["نوع النموذج", d.model_type || m.model || "—"],
    ["تاريخ الإنشاء", fmtTs(meta.created_at)],
    ["حالة البوابات", passed ? "اجتاز (قابل للنشر)" : "رفض النشر — Paper فقط"],
    ["عدد الصفوف المستخدمة", m.n_rows ?? d.n_rows_used ?? "—"],
    ["عدد الميزات", m.n_features ?? d.n_features ?? "—"],
    ["طريقة التسمية", d.labeling || cfg.labeling || "—"],
    ["أفق التنبؤ (شموع)", d.horizon_bars ?? m.horizon_bars ?? cfg.horizon_bars ?? "—"],
    ["Final Model", meta.final_model_ready ? "جاهز" : "—"],
  ]);

  const policy = m.trade_policy || {};
  $("#train-data-table").innerHTML = kvRows([
    ["مصدر البيانات", dsMeta.features_json_path || `data/features/${t.symbol}/${t.timeframe}/features.json`],
    ["ملف Registry", dsMeta.registry_json_path || `data/registry/${t.timeframe}.json`],
    ["ملف Knowledge", dsMeta.pattern_paths?.knowledge || `data/patterns/${t.symbol}/${t.timeframe}/knowledge.json`],
    ["بداية الفترة", fmtTs(d.first_ts)],
    ["نهاية الفترة", fmtTs(d.last_ts)],
    ["مدة العينة (يوم)", d.days != null ? fmt(d.days, 1) : "—"],
    ["شموع الميزات", d.rows ?? "—"],
    ["شموع بعد التسمية", d.n_rows_used ?? m.n_rows ?? "—"],
    ["التقسيم", valMeta.method || valMeta.validation_mode || "—"],
    ["وضع التحقق (v13)", valMeta.validation_mode || m.validation_mode || "expanding"],
    ["عدد الطيات", (m.folds || []).length],
    ["Latency / Delay", `${fmt((m.validation || {}).latency_bars ?? (m.advanced_eval || {}).execution?.latency_bars, 0)} / ${fmt((m.validation || {}).execution_delay_bars ?? (m.advanced_eval || {}).execution?.execution_delay_bars, 0)}`],
    ["Ensemble Deploy", (m.validation || {}).use_ensemble ? "نعم" : "لا"],
    ["عتبة القرار", fmt(policy.decision_threshold ?? m.decision_threshold, 2)],
    ["هامش الاتجاه", fmt(policy.directional_edge, 2)],
    ["كمّ ثقة", fmt(policy.confidence_quantile, 2)],
  ]);

  const cls = m.classification || {};
  const fin = m.financial_oos || {};
  const valFin = m.financial_validation || (valMeta.financial || {});

  $("#cls-kpis").innerHTML = [
    kpi("Accuracy", fmt(cls.accuracy, 3)),
    kpi("Precision", fmt(cls.precision_macro, 3)),
    kpi("Recall", fmt(cls.recall_macro, 3)),
    kpi("F1", fmt(cls.f1_macro, 3)),
  ].join("");

  $("#cls-table").innerHTML = [
    ["Accuracy", fmt(cls.accuracy, 4)],
    ["Precision (macro)", fmt(cls.precision_macro, 4)],
    ["Recall (macro)", fmt(cls.recall_macro, 4)],
    ["F1 (macro)", fmt(cls.f1_macro, 4)],
    ["Trade rate (filtered)", pct(cls.trade_rate_filtered)],
  ].map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");

  $("#fin-kpis").innerHTML = [
    kpi("Test Sharpe", fmt(fin.sharpe, 2), clsNum(fin.sharpe)),
    kpi("Expectancy", fmt(fin.expectancy, 5), clsNum(fin.expectancy)),
    kpi("Sortino", fmt(fin.sortino, 2), clsNum(fin.sortino)),
    kpi("Max DD", pct(fin.max_drawdown), "num-bad"),
  ].join("");

  $("#fin-table").innerHTML = [
    ["Validation Sharpe", fmt(valFin.sharpe, 4), clsNum(valFin.sharpe)],
    ["Sharpe (conservative)", fmt(fin.sharpe, 4), clsNum(fin.sharpe)],
    ["Sharpe uncapped", fmt(fin.sharpe_uncapped, 4), "muted"],
    ["Sharpe CI low / high", `${fmt(fin.sharpe_ci_low, 4)} / ${fmt(fin.sharpe_ci_high, 4)}`, ""],
    ["ann_factor", fmt(fin.ann_factor, 3), ""],
    ["Sortino Ratio", fmt(fin.sortino, 4), clsNum(fin.sortino)],
    ["Expectancy", fmt(fin.expectancy, 6), clsNum(fin.expectancy)],
    ["Avg win / Avg loss", `${fmt(fin.avg_win, 5)} / ${fmt(fin.avg_loss, 5)}`, ""],
    ["Payoff / Kelly approx", `${fmt(fin.payoff_ratio, 3)} / ${fmt(fin.kelly_fraction_approx, 3)}`, ""],
    ["Risk-adjusted return", fmt(fin.risk_adjusted_return, 4), clsNum(fin.risk_adjusted_return)],
    ["Max Drawdown", pct(fin.max_drawdown, 2), "num-bad"],
    ["Win Rate", pct(fin.win_rate, 2), ""],
    ["Profit Factor", fmt(fin.profit_factor, 4), ""],
    ["n_trades", fmt(fin.n_trades, 0), ""],
    ["Mean trade return", fmt(fin.mean_trade_return, 5), clsNum(fin.mean_trade_return)],
    ["Sum trade returns", fmt(fin.sum_trade_returns, 4), clsNum(fin.sum_trade_returns)],
    ["Simple trade equity (1+Σ)", fmt(fin.simple_trade_equity, 4), ""],
    ["Compounded backtest return (not live)", pct(fin.total_return, 2), clsNum(fin.total_return)],
  ].map(([k, v, c]) => `<tr><td>${k}</td><td class="${c}">${v}</td></tr>`).join("");

  renderTradingIntelligencePanel(t);

  const bh = m.buy_hold || {};
  const rnd = m.random_baseline || {};
  const row = (name, o) => `<tr>
    <td><b>${name}</b></td>
    <td class="${clsNum(o.sharpe)}">${fmt(o.sharpe, 3)}</td>
    <td class="${clsNum(o.sortino)}">${fmt(o.sortino, 3)}</td>
    <td class="num-bad">${pct(o.max_drawdown, 2)}</td>
    <td>${pct(o.win_rate, 1)}</td>
    <td>${fmt(o.profit_factor, 3)}</td>
    <td class="${clsNum(o.total_return)}">${pct(o.total_return, 2)}</td>
  </tr>`;
  $("#compare-table").innerHTML = [
    row("النموذج (OOS)", fin),
    row("Buy & Hold", bh),
    row("عشوائي", rnd),
  ].join("");

  const folds = m.folds || [];
  const bestValFold = folds.length
    ? folds.reduce((a, b) => (Number(b.val_sharpe || 0) > Number(a.val_sharpe || 0) ? b : a), folds[0])
    : null;
  $("#folds-table").innerHTML = folds.length
    ? folds.map((f) => {
        const isBest = bestValFold && f.fold === bestValFold.fold;
        const isDeploy = folds.length && f.fold === folds[folds.length - 1].fold;
        const note = [isBest ? "أفضل Val" : "", isDeploy ? "Deploy window" : ""].filter(Boolean).join(" · ");
        return `<tr class="${isBest ? "num-ok" : ""}">
          <td>${(f.fold ?? 0) + 1}${isBest ? " ★" : ""}</td>
          <td>${fmt(f.accuracy, 4)}</td>
          <td>${fmt(f.f1_macro, 4)}</td>
          <td>${fmt(f.roc_auc_ovr, 4)}</td>
          <td>${fmt(f.trade_rate, 3)}</td>
          <td>${fmt(f.val_sharpe, 3)}</td>
          <td>${fmt(f.n_val_trades, 0)}</td>
          <td>${note || "—"}</td>
        </tr>`;
      }).join("")
    : `<tr><td colspan="8">لا توجد طيات اختبار</td></tr>`;

  const feats = d.feature_list || [];
  $("#feature-cloud").innerHTML = feats.length
    ? feats.map((f) => `<span class="tag">${f}</span>`).join("")
    : `<span class="tag">لا قائمة ميزات</span>`;
}

function renderTrainingVersions(models, training) {
  const body = $("#train-version-history");
  if (!body) return;
  const rows = [];
  if (training?.llmodel?.exists) {
    const ll = training.llmodel;
    const fin = ll.metrics?.test?.financial || {};
    rows.push(`<tr>
      <td>LLModel</td>
      <td>Deep Learning</td>
      <td>${fmt(fin.sharpe, 2)}</td>
      <td>${pct(fin.max_drawdown)}</td>
      <td>${pct(fin.total_return)}</td>
      <td>${ll.metadata?.final_model_ready ? "جاهز" : "موجود"}</td>
    </tr>`);
  }
  for (const item of (models?.versions || []).slice(0, 8)) {
    const meta = item.meta || {};
    const fin = item.metrics?.financial_oos || {};
    rows.push(`<tr>
      <td>${meta.version || "—"}</td>
      <td>${item.metrics?.model || meta.model || "Baseline"}</td>
      <td>${fmt(fin.sharpe, 2)}</td>
      <td>${pct(fin.max_drawdown)}</td>
      <td>${pct(fin.total_return)}</td>
      <td>${meta.passed_gates ? "اجتاز" : "رفض"}</td>
    </tr>`);
  }
  body.innerHTML = rows.length
    ? rows.join("")
    : `<tr><td colspan="6">لا توجد إصدارات سابقة بعد</td></tr>`;
}

function renderCoverage(data, overview) {
  const rows = [];
  const seenTf = new Set();
  for (const item of data.coverage || []) {
    for (const layer of ["raw", "clean", "features"]) {
      const L = item.layers[layer] || {};
      const f = L.file || {};
      const r = L.registry || {};
      if (!f.exists && !r) continue;
      const sj = item.state_json || {};
      const openBtn = sj.exists
        ? `<button class="btn btn-tiny" type="button" data-view-state="${item.timeframe}">فتح</button>`
        : "—";
      rows.push(`<tr>
        <td><b>${item.timeframe}</b></td>
        <td>${layer}</td>
        <td>${f.rows ?? r.row_count ?? "—"}</td>
        <td>${fmtTs(f.first_ts || r.first_available_ts)}</td>
        <td>${fmtTs(f.last_ts || r.last_updated_ts)}</td>
        <td>${f.days != null ? fmt(f.days, 1) : "—"}</td>
        <td>${r.last_run_status || (f.exists ? "file" : "—")}</td>
        <td>${seenTf.has(item.timeframe) ? "" : openBtn}</td>
      </tr>`);
      seenTf.add(item.timeframe);
    }
  }
  $("#coverage-body").innerHTML = rows.length ? rows.join("") : `<tr><td colspan="8">لا بيانات</td></tr>`;

  const files = data.state_files || [];
  const rootLabel = $("#registry-root-label");
  if (rootLabel) rootLabel.textContent = data.registry_root || "data/registry/";
  const filesBody = $("#registry-files-body");
  if (filesBody) {
    filesBody.innerHTML = files.length
      ? files.map((f) => `<tr>
          <td><b>${f.timeframe}</b></td>
          <td dir="ltr">${f.relative_path || f.filename}</td>
          <td>${fmtBytes(f.size_bytes)}</td>
          <td>${fmtTs(f.updated_at)}</td>
          <td>${(f.symbols || []).join(", ") || "—"}</td>
          <td><button class="btn btn-tiny" type="button" data-view-state="${f.timeframe}">استعراض</button></td>
        </tr>`).join("")
      : `<tr><td colspan="6">لا ملفات JSON بعد — شغّل «تحديث كل الأطر»</td></tr>`;
  }

  const layers = overview.layers || {};
  $("#layer-summary").innerHTML = ["raw", "clean", "features"].map((layer) => {
    const r = layers[layer] || {};
    return `<article class="card">
      <h2>${layer}</h2>
      <table class="kv">${kvRows([
        ["الشموع", r.row_count ?? "—"],
        ["من", fmtTs(r.first_available_ts)],
        ["إلى", fmtTs(r.last_updated_ts)],
        ["الحالة", r.last_run_status || "—"],
      ])}</table>
    </article>`;
  }).join("");
}

function renderPatterns(p) {
  const s = p.structure || {};
  const bias = p.bias_label || "محايد";
  const banner = $("#pattern-banner");
  const kb = p.knowledge || {};
  const cat = kb.catalog || {};
  if (banner) {
    banner.className = `banner ${bias === "صاعد" ? "ok" : bias === "هابط" ? "bad" : "warn"}`;
    banner.innerHTML = `إطار <b>${p.timeframe}</b> · تحيز: <b>${bias}</b> · أنماط بإصابات: <b>${p.patterns_with_hits ?? 0}</b> / كتالوج <b>${p.catalog_size ?? "—"}</b> · اكتشافات: <b>${p.total_detections ?? 0}</b>`;
  }
  if ($("#bias-label")) {
    $("#bias-label").textContent = bias;
    $("#bias-label").className = `v ${bias === "صاعد" ? "num-ok" : bias === "هابط" ? "num-bad" : ""}`;
  }
  if ($("#bias-strength")) $("#bias-strength").textContent = fmt(s.pat_strength, 2);

  const structLabel = s.structure_hh_hl > 0 ? "HH/HL صاعد" : s.structure_hh_hl < 0 ? "LH/LL هابط" : "عرضي";
  $("#structure-table").innerHTML = kvRows([
    ["الدعم", fmt(s.support_level, 2)],
    ["المقاومة", fmt(s.resist_level, 2)],
    ["المسافة للدعم", pct(s.dist_to_support, 2)],
    ["المسافة للمقاومة", pct(s.dist_to_resist, 2)],
    ["هيكل السوق", structLabel],
    ["درجة النمط الهيكلي", fmt(s.chart_pattern_score, 1)],
    ["ميل خط الاتجاه", fmt(s.trendline_slope, 6)],
    ["تحيز الشموع", s.pat_bias ?? 0],
    ["الجلسة", s.session || "—"],
    ["التذبذب", s.vol_regime || "—"],
    ["RSI(14)", fmt(s.rsi_14, 1)],
    ["ATR", fmt(s.atr, 2)],
  ]);

  const active = p.active_now || [];
  if ($("#active-now")) {
    $("#active-now").innerHTML = active.length
      ? active.map((a) => `<span class="chip ${a.bias === "bullish" ? "gold" : ""}">${a.pattern} · ${a.bias}</span>`).join("")
      : `<span class="chip">لا نمط نشط على الشمعة الحالية</span>`;
  }

  if ($("#pattern-catalog-stats")) {
    $("#pattern-catalog-stats").innerHTML = kvRows([
      ["حجم الكتالوج", cat.catalog_total ?? p.catalog_size ?? "—"],
      ["شموع بإصابات", Object.keys(p.candle_counts || {}).length],
      ["هيكلية بإصابات", Object.keys(p.chart_counts || {}).length],
      ["مركّبة بإصابات", Object.keys(p.compound_counts || {}).length],
      ["مركّبات مكتشفة", cat.discovered_compounds ?? 0],
      ["متوسط نجاح المعرفة", kb.avg_success_rate != null ? pct(kb.avg_success_rate, 1) : "—"],
      ["إجمالي الإصابات", p.total_detections ?? 0],
      ["نوافذ التحليل (شموع)", p.lookback_bars ?? "—"],
    ]);
  }

  const mkChips = (obj) => {
    const entries = Object.entries(obj || {}).sort((a, b) => b[1] - a[1]);
    return entries.length
      ? entries.map(([k, v]) => `<span class="chip ${v ? "gold" : ""}">${k}: ${v}</span>`).join("")
      : `<span class="chip">لا بيانات — شغّل الاستكشاف الشامل</span>`;
  };
  if ($("#candle-counts")) $("#candle-counts").innerHTML = mkChips(p.candle_counts);
  if ($("#chart-counts")) $("#chart-counts").innerHTML = mkChips(p.chart_counts);
  if ($("#compound-counts")) $("#compound-counts").innerHTML = mkChips(p.compound_counts);

  const np = p.new_patterns || {};
  if ($("#new-pattern-counts")) {
    const items = np.items || [];
    $("#new-pattern-counts").innerHTML = items.length
      ? items.slice(0, 12).map((r) => {
          const tag = r.approved ? "gold" : (r.soft_promoted ? "" : "");
          return `<span class="chip ${tag}">${r.name || r.id}${r.approved ? " ✓" : r.soft_promoted ? " ~" : ""}</span>`;
        }).join("")
      : `<span class="chip">لا NewN بعد</span>`;
  }
  if ($("#new-pattern-summary")) {
    $("#new-pattern-summary").textContent =
      `عدد: ${np.count ?? 0} · معتمد: ${np.approved ?? 0} · مرفوض: ${np.rejected ?? 0}`;
  }
  if ($("#rankings-recommended")) {
    const rec = (p.rankings && p.rankings.engine4_recommended) || [];
    $("#rankings-recommended").innerHTML = rec.length
      ? rec.slice(0, 12).map((r) => `<span class="chip gold">${r.name || r.pattern_key}</span>`).join("")
      : `<span class="chip">لا توصيات Engine4 بعد</span>`;
  }
  if ($("#validation-summary")) {
    const v = p.validation_report || {};
    $("#validation-summary").textContent =
      `مقيّم: ${v.count ?? 0} · معتمد: ${v.approved ?? 0} · مرفوض: ${v.rejected ?? 0}`;
  }
  if ($("#relations-summary")) {
    const rel = p.relations || {};
    $("#relations-summary").textContent = rel.summary || "لا علاقات بعد — شغّل الاستكشاف";
  }
  if ($("#relations-body")) {
    const edges = (p.relations && p.relations.edges) || [];
    $("#relations-body").innerHTML = edges.length
      ? edges.slice(0, 20).map((e) => `<tr>
          <td>${e.source}</td>
          <td>${e.relation}</td>
          <td>${e.target}</td>
          <td>${e.count ?? 0}</td>
        </tr>`).join("")
      : `<tr><td colspan="4">—</td></tr>`;
  }

  const top = kb.top || [];
  if ($("#knowledge-body")) {
    $("#knowledge-body").innerHTML = top.length
      ? top.map((r) => `<tr>
          <td><b>${r.name || r.pattern_key}</b></td>
          <td>${r.category || "—"}</td>
          <td>${r.timeframe || "—"}</td>
          <td>${r.occurrences ?? 0}</td>
          <td class="${(r.success_rate || 0) >= 0.55 ? "num-ok" : (r.success_rate || 0) <= 0.45 ? "num-bad" : ""}">${r.success_rate != null ? pct(r.success_rate, 1) : "—"}</td>
          <td>${r.confidence != null ? pct(r.confidence, 0) : "—"}</td>
          <td class="${clsNum(r.avg_forward_return)}">${r.avg_forward_return != null ? pct(r.avg_forward_return, 2) : "—"}</td>
          <td style="max-width:280px;white-space:normal;font-size:0.75rem">${r.conditions || r.catalog_conditions || "—"}</td>
        </tr>`).join("")
      : `<tr><td colspan="8">لا معرفة مخزّنة بعد — اضغط «استكشاف شامل للأنماط»</td></tr>`;
  }

  const files = p.json_files || [];
  window._patternJsonFiles = files;
  if ($("#patterns-root-label") && p.patterns_root) {
    $("#patterns-root-label").textContent =
      `المجلد: ${p.patterns_root} · اضغط «استعراض» لفتح أي قسم بعد الانتهاء`;
  }
  renderPatternJsonFiles(files, p.timeframe);

  const biasAr = { bullish: "صاعد", bearish: "هابط", neutral: "محايد" };
  const catAr = { candle: "شمعة", chart: "هيكلي", compound: "مركّب" };
  const dets = p.detections || [];
  $("#patterns-body").innerHTML = dets.length
    ? dets.map((d) => `<tr>
        <td>${fmtTs(d.timestamp)}</td>
        <td><b>${d.pattern}</b></td>
        <td>${catAr[d.category] || d.category || "—"}</td>
        <td class="${d.bias === "bullish" ? "num-ok" : d.bias === "bearish" ? "num-bad" : ""}">${biasAr[d.bias] || d.bias || "—"}</td>
        <td>${fmt(d.strength, 2)}</td>
        <td>${d.success_rate != null ? pct(d.success_rate, 1) : "—"}</td>
        <td>${d.confidence != null ? pct(d.confidence, 0) : "—"}</td>
        <td>${fmt(d.close, 2)}</td>
      </tr>`).join("")
    : `<tr><td colspan="8">لا اكتشافات — شغّل الاستكشاف الشامل</td></tr>`;

  drawPatternChart(p.ohlc || [], p.markers || []);
  if ($("#price-meta")) {
    $("#price-meta").textContent = `علامات على آخر ${ (p.ohlc || []).length } شمعة · حتى 8 أنماط/شمعة`;
  }
}

function drawPatternChart(ohlc, markers) {
  const canvas = $("#pattern-chart");
  if (!canvas || !ohlc.length) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const highs = ohlc.map((r) => Number(r.high));
  const lows = ohlc.map((r) => Number(r.low));
  const min = Math.min(...lows);
  const max = Math.max(...highs);
  const pad = 16;
  const n = ohlc.length;
  const xAt = (i) => pad + (i / Math.max(n - 1, 1)) * (w - pad * 2);
  const yAt = (p) => h - pad - ((p - min) / (max - min || 1)) * (h - pad * 2);
  ctx.fillStyle = "rgba(139,105,20,0.05)";
  ctx.fillRect(0, 0, w, h);
  const candleW = Math.max(2, (w - pad * 2) / n * 0.55);
  ohlc.forEach((r, i) => {
    const o = Number(r.open), c = Number(r.close), hi = Number(r.high), lo = Number(r.low);
    const x = xAt(i);
    const up = c >= o;
    ctx.strokeStyle = up ? "#0f6b45" : "#a11d1d";
    ctx.fillStyle = up ? "#0f6b45" : "#a11d1d";
    ctx.beginPath();
    ctx.moveTo(x, yAt(hi));
    ctx.lineTo(x, yAt(lo));
    ctx.stroke();
    const y1 = yAt(Math.max(o, c));
    const y2 = yAt(Math.min(o, c));
    ctx.fillRect(x - candleW / 2, y1, candleW, Math.max(1, y2 - y1));
  });
  const byTs = Object.fromEntries((markers || []).map((m) => [m.timestamp, m]));
  ohlc.forEach((r, i) => {
    const m = byTs[r.timestamp];
    if (!m) return;
    const x = xAt(i);
    const y = yAt(Number(r.high)) - 8;
    ctx.beginPath();
    ctx.fillStyle = m.bias > 0 ? "#0f6b45" : m.bias < 0 ? "#a11d1d" : "#8b6914";
    ctx.arc(x, y, 3.5, 0, Math.PI * 2);
    ctx.fill();
  });
}

function getSelectedAutoTimeframes() {
  return $$('#auto-tf-checks input[name="auto-tf"]:checked').map((el) => el.value);
}

function syncAutoTfAllCheckbox() {
  const boxes = $$('#auto-tf-checks input[name="auto-tf"]');
  const all = $("#auto-tf-all");
  if (!boxes.length || !all) return;
  const checked = boxes.filter((el) => el.checked).length;
  all.checked = checked === boxes.length;
  all.indeterminate = checked > 0 && checked < boxes.length;
}

function setAutoTfChecks(timeframes, { locked = false } = {}) {
  const set = new Set((timeframes || []).map((t) => String(t).toUpperCase()));
  const boxes = $$('#auto-tf-checks input[name="auto-tf"]');
  if (set.size) {
    boxes.forEach((el) => {
      el.checked = set.has(el.value);
    });
  }
  boxes.forEach((el) => {
    el.disabled = locked;
  });
  const all = $("#auto-tf-all");
  if (all) all.disabled = locked;
  $("#auto-tf-checks")?.classList.toggle("is-locked", locked);
  all?.closest(".auto-tf-all")?.classList.toggle("is-locked", locked);
  syncAutoTfAllCheckbox();
}

function renderAuto(a, models) {
  a = a || {};
  const tfs = (a.timeframes && a.timeframes.length)
    ? a.timeframes
    : (a.timeframe ? [a.timeframe] : []);
  const tfLabel = tfs.length ? tfs.join(" · ") : (a.timeframe || "H1");

  const stAuto = $("#st-auto");
  if (a.running) {
    stAuto.textContent = `يعمل (${a.mode || "paper"})`;
    stAuto.className = "ok running-pulse";
  } else {
    stAuto.textContent = "متوقف";
    stAuto.className = "";
  }

  const banner = $("#auto-running-banner");
  if (banner) {
    banner.hidden = !a.running;
    if (a.running) {
      $("#auto-running-title").textContent = `التداول الآلي يعمل الآن · ${a.mode || "paper"}`;
      $("#auto-running-meta").textContent =
        `${a.symbol || "XAUUSD"} · الأطر: ${tfLabel} · دورات: ${a.cycles ?? 0} · أوامر: ${a.orders ?? 0}`;
    }
  }

  $("#btn-auto-start") && ($("#btn-auto-start").disabled = !!a.running);
  $("#btn-auto-start-paper") && ($("#btn-auto-start-paper").disabled = !!a.running);
  $("#btn-auto-stop") && ($("#btn-auto-stop").disabled = !a.running);

  if (a.running && tfs.length) {
    setAutoTfChecks(tfs, { locked: true });
  } else {
    $$('#auto-tf-checks input[name="auto-tf"]').forEach((el) => { el.disabled = false; });
    const allTf = $("#auto-tf-all");
    if (allTf) allTf.disabled = false;
    $("#auto-tf-checks")?.classList.remove("is-locked");
    allTf?.closest(".auto-tf-all")?.classList.remove("is-locked");
    syncAutoTfAllCheckbox();
  }

  $("#auto-table").innerHTML = kvRows([
    ["الحالة", a.running ? "يعمل ✓" : "متوقف"],
    ["الوضع", a.mode || "paper"],
    ["الرمز", a.symbol || "XAUUSD"],
    ["الأطر الزمنية", tfLabel],
    ["قرار الأطر", (tfs.length > 1) ? `دمج متعدد (${a.fusion_mode || "weighted_consensus"})` : "إطار واحد · نموذج مدرّب"],
    ["الفاصل", `${a.interval_seconds || 60} ثانية`],
    ["عدد الدورات", a.cycles ?? 0],
    ["الإشارات", a.signals ?? 0],
    ["الأوامر المنفذة", a.orders ?? 0],
    ["بدأ في", fmtTs(a.started_at)],
    ["آخر دورة", fmtTs(a.last_cycle_at)],
    ["آخر خطأ", a.last_error || "لا يوجد"],
    ["وضع الدخول", getLiveSettingsCache()?.mode_label_ar || "—"],
  ]);

  const latest = models?.versions?.[0];
  const finMeta = models?.final_model;
  const ll = models?.llmodel;
  const llFin = ll?.metrics?.test?.financial || {};
  const finMetrics = finMeta?.metrics?.test || latest?.metrics?.financial_oos || {};
  if (ll?.exists) {
    $("#model-table").innerHTML = kvRows([
      ["النوع", "LLModel"],
      ["الرمز", ll?.metadata?.symbol || "—"],
      ["الإطار الأساسي", ll?.metadata?.base_timeframe || "—"],
      ["عدد الأطر", (ll?.metadata?.timeframes || []).join(" · ") || "—"],
      ["Accuracy", fmt(ll?.metrics?.test?.accuracy, 3)],
      ["Sharpe OOS", fmt(llFin.sharpe, 3)],
      ["Max DD", pct(llFin.max_drawdown)],
    ]);
  } else if (finMeta?.exists || finMeta?.artifact_path) {
    const gatesOk = !!finMeta.passed_gates || String(finMeta.mode || "") === "live_ready";
    const modeLabel = gatesOk ? "جاهز للتداول" : (finMeta.mode || "paper");
    $("#model-table").innerHTML = kvRows([
      ["النوع", "FinalModel (المستخدم في التداول)"],
      ["الرمز", finMeta.symbol || "XAUUSD"],
      ["إطار التدريب", finMeta.timeframe || "—"],
      ["النسخة", finMeta.version || "—"],
      ["الوضع", modeLabel],
      ["بوابات", gatesOk ? "اجتاز ✓" : "لم يجتز"],
      ["Sharpe OOS", fmt(finMetrics.sharpe, 3)],
      ["Max DD", pct(finMetrics.max_drawdown)],
      ["الصفقات (Test)", finMetrics.n_trades ?? "—"],
    ]);
  } else {
    const champ = models?.champion;
    const gatesOk = !!(latest?.meta?.passed_gates || champ?.passed_gates);
    $("#model-table").innerHTML = kvRows([
      ["النوع", "Baseline"],
      ["النسخة", latest?.meta?.version || champ?.version || "—"],
      ["Champion", champ ? "نعم" : "لا"],
      ["Sharpe OOS", fmt((latest?.metrics?.financial_oos || {}).sharpe, 3)],
      ["Max DD", pct((latest?.metrics?.financial_oos || {}).max_drawdown)],
      ["بوابات", gatesOk ? "اجتاز ✓" : "لم يجتز"],
    ]);
  }
}

function renderDecision(payload) {
  const d = payload?.decision;
  if (!d) {
    $("#decision-table").innerHTML = kvRows([["الحالة", "لا توجد قرارات بعد"]]);
    return;
  }
  const dbg = d.debug || {};
  const probs = dbg.scenario_probabilities || {};
  const attn = dbg.attention_by_timeframe || {};
  const fusion = dbg.multi_tf_fusion || {};
  const votes = dbg.votes || fusion.votes || [];
  const topTf = Object.entries(attn).sort((a, b) => Number(b[1]) - Number(a[1]))[0]?.[0]
    || fusion.execution_tf
    || "—";
  const side = d.pred > 0 ? "BUY" : d.pred < 0 ? "SELL" : "HOLD";
  const voteTxt = votes.length
    ? votes.map((v) => {
        if (v.error) return `${v.tf}:خطأ`;
        const s = v.pred > 0 ? "شراء" : v.pred < 0 ? "بيع" : "انتظار";
        return `${v.tf}:${s}(${fmt(v.conf, 2)})`;
      }).join(" · ")
    : "—";
  $("#decision-table").innerHTML = kvRows([
    ["الوقت", fmtTs(d.ts)],
    ["نوع النموذج", d.model_type || "—"],
    ["نسخة النموذج", d.model_version || "—"],
    ["إطار النموذج", d.model_timeframe || d.timeframe || "—"],
    ["الأطر المحلَّلة", (d.timeframes || []).join(" · ") || (d.timeframe || "—")],
    ["القرار", side],
    ["الثقة المدمجة", fmt(d.confidence, 3)],
    ["السعر", fmt(d.close, 2)],
    ["سبب القرار", dbg.reason || fusion.reason || "—"],
    ["أصوات الأطر", voteTxt],
    ["شراء/بيع/انتظار", `${fusion.buy_votes ?? "—"} / ${fusion.sell_votes ?? "—"} / ${fusion.flat_votes ?? "—"}`],
    ["أكثر إطار مؤثر", topTf],
    ["السيناريوهات", `بيع ${fmt(probs.sell, 3)} · انتظار ${fmt(probs.hold, 3)} · شراء ${fmt(probs.buy, 3)}`],
    ["العائد المتوقع", fmt(dbg.expected_return, 4)],
    ["مخاطر النموذج", fmt(dbg.risk_score, 3)],
  ]);
}

function renderTrades(trades) {
  $("#trades-body").innerHTML = trades.length
    ? trades.slice(0, 50).map((t) => {
        const row = t.trade || t;
        const mode = row.mode || (row.ticket ? "demo" : "paper");
        const ticket = row.ticket != null ? `#${row.ticket}` : (mode === "paper" ? "ورقي" : "—");
        return `<tr>
          <td>${fmtTs(row.ts)}</td>
          <td>${row.side || "—"}</td>
          <td>${row.volume ?? "—"}</td>
          <td>${fmt(row.entry_price, 2)}</td>
          <td>${fmt(row.sl, 2)}</td>
          <td>${fmt(row.tp, 2)}</td>
          <td>${mode} · ${ticket}</td>
          <td>${fmt(row.confidence, 2)}</td>
        </tr>`;
      }).join("")
    : `<tr><td colspan="8">لا صفقات بعد — Demo يرسل أوامر إلى MT5 · Paper يسجّل هنا فقط</td></tr>`;
}

function renderJobs(list) {
  const ul = $("#jobs-list");
  if (!list?.length) {
    ul.innerHTML = `<li><span>لا مهام</span><strong>—</strong></li>`;
    return;
  }
  ul.innerHTML = list.slice(0, 15).map((j) => `
    <li><span>${j.name} · ${j.id}</span><strong class="status-${j.status}">${j.status}</strong></li>
  `).join("");
}

function setMini(id, lines) {
  const el = $(`#stats-${id}`);
  if (!el) return;
  el.innerHTML = lines.map(([k, v]) => `<div>${k}: <b>${v}</b></div>`).join("");
}

function settledValue(result, fallback = null) {
  return result?.status === "fulfilled" ? result.value : fallback;
}

async function refreshDataTab() {
  if (_dataRefreshPromise) return _dataRefreshPromise;
  _dataRefreshPromise = (async () => {
    const [overview, coverage] = await Promise.all([
      api("/api/overview"),
      api("/api/data/coverage"),
    ]);
    renderCoverage(coverage, overview);
  })();
  try {
    await _dataRefreshPromise;
  } finally {
    _dataRefreshPromise = null;
  }
}

function drawSpark(rows) {
  const canvas = $("#spark");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!rows || rows.length < 2) return;
  const closes = rows.map((r) => Number(r.close));
  const min = Math.min(...closes);
  const max = Math.max(...closes);
  const pad = 12;
  ctx.strokeStyle = "#8b6914";
  ctx.lineWidth = 2;
  ctx.beginPath();
  closes.forEach((c, i) => {
    const x = pad + (i / (closes.length - 1)) * (w - pad * 2);
    const y = h - pad - ((c - min) / (max - min || 1)) * (h - pad * 2);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

async function refresh(patternTf) {
  if (_refreshPromise) return _refreshPromise;
  _refreshPromise = (async () => {
  const tf = patternTf || $("#pattern-tf")?.value || "M5";
  const emptyTraining = {
    empty: true,
    summary: { total: 0, ready: 0, passed_gates: 0 },
    matrix: [],
    all_timeframes: [],
    selected_timeframe: "H1",
  };
  const emptyPatterns = {
    timeframe: tf,
    detections: [],
    candle_counts: {},
    chart_counts: {},
    compound_counts: {},
    active_now: [],
    structure: {},
    bias_label: "محايد",
    knowledge: {},
    json_files: [],
    patterns_root: "",
    ohlc: [],
    markers: [],
    total_detections: 0,
    patterns_with_hits: 0,
    catalog_size: 0,
  };

  // Kick off all requests, but render training/models as soon as they arrive
  // so the banner is not blocked by heavy pattern feature loads.
  const overviewP = api("/api/overview");
  const coverageP = api("/api/data/coverage");
  const patternsP = api(`/api/patterns?limit=500&lookback=5000&timeframe=${tf}`);
  const trainingP = api("/api/training/details");
  const modelsP = api("/api/models");
  const tradesP = api("/api/trades");
  const decisionP = apiOptional("/api/decision/latest", { empty: true });
  const patternFilesP = apiOptional("/api/patterns/files", { files: [] });

  let training = emptyTraining;
  let models = { versions: [], leaderboard: [], llmodel: null };
  let overview = {};
  let coverage = { coverage: [], state_files: [], registry_root: "data/registry/" };
  let patterns = emptyPatterns;
  let trades = { trades: [] };
  let decision = { empty: true };

  const paintTraining = () => {
    renderTraining(training);
    renderTrainingVersions(models, training);
    renderTrainingDashboards(training, models);
    setMini(4, [["صفوف", training.llmodel?.metrics?.rows ?? training.metrics?.n_rows ?? "—"], ["Sharpe", fmt(training.llmodel?.metrics?.test?.financial?.sharpe ?? training.metrics?.financial_oos?.sharpe, 2)]]);
  };

  const trainingPaint = Promise.allSettled([trainingP, modelsP]).then((pair) => {
    training = settledValue(pair[0], emptyTraining);
    models = settledValue(pair[1], { versions: [], leaderboard: [], llmodel: null });
    _latestModels = models;
    paintTraining();
  });

  const overviewPaint = overviewP.then((ov) => {
    overview = ov || {};
    const mt5 = overview.mt5 || {};
    $("#st-mt5").textContent = mt5.ok
      ? `${mt5.reconnected ? "أعيد الاتصال" : "متصل"} · ${fmt(mt5.balance, 2)}`
      : "غير متصل";
    $("#st-mt5").className = mt5.ok ? (mt5.reconnected ? "warn" : "ok") : "bad";
    $("#st-symbol").textContent = overview.symbol || "XAUUSD";
    $("#st-tf").textContent = overview.timeframe || "M5";
    $("#st-price").textContent = overview.price ? fmt(overview.price.close, 2) : "—";
    $("#price-meta").textContent = overview.price
      ? `آخر إغلاق ${fmtTs(overview.price.timestamp)} · ATR ${fmt(overview.price.atr, 2)}`
      : "—";
    if (overview.live_settings) applyLiveSettingsForm(overview.live_settings);
    renderAuto(overview.autotrader, models);
    renderJobs(overview.jobs || []);
    const layers = overview.layers || {};
    setMini(1, [["TF", overview.timeframe || "M5"], ["صفوف", layers.raw?.row_count ?? "—"], ["إلى", fmtTs(layers.raw?.last_updated_ts).slice(0, 10)]]);
    setMini(2, [["صفوف", layers.clean?.row_count ?? "—"], ["حالة", layers.clean?.last_run_status || "—"]]);
    setMini(5, [["آلي", overview.autotrader?.running ? "يعمل ✓" : "متوقف"], ["أوامر", overview.autotrader?.orders ?? 0]]);
    $("#nav-foot").textContent = `آخر تحديث ${new Date().toLocaleTimeString()}\n${overview.symbol || "XAUUSD"}/${overview.timeframe || "M5"}`;
  }).catch((e) => console.warn("overview refresh failed", e));

  const results = await Promise.allSettled([
    overviewP,
    coverageP,
    patternsP,
    trainingP,
    modelsP,
    tradesP,
    decisionP,
    patternFilesP,
    trainingPaint,
    overviewPaint,
  ]);
  const [overviewRes, coverageRes, patternsRes, trainingRes, modelsRes, tradesRes, decisionRes, patternFilesRes] = results;

  overview = settledValue(overviewRes, overview);
  coverage = settledValue(coverageRes, coverage);
  patterns = settledValue(patternsRes, emptyPatterns);
  training = settledValue(trainingRes, training);
  models = settledValue(modelsRes, models);
  _latestModels = models;
  trades = settledValue(tradesRes, { trades: [] });
  decision = settledValue(decisionRes, { empty: true });
  const patternFiles = settledValue(patternFilesRes, { files: [] });

  // Ensure JSON file list is available even when features parquet is missing
  if ((!patterns.json_files || !patterns.json_files.length) && patternFiles.files?.length) {
    patterns.json_files = patternFiles.files;
    patterns.patterns_root = patternFiles.root || patterns.patterns_root;
  }

  paintTraining();
  renderCoverage(coverage, overview);
  renderPatterns(patterns);
  if (overview.live_settings) applyLiveSettingsForm(overview.live_settings);
  renderAuto(overview.autotrader, models);
  renderDecision(decision);
  renderTrades(trades.trades || []);
  renderJobs(overview.jobs || []);

  const layers = overview.layers || {};
  setMini(1, [["TF", overview.timeframe || "M5"], ["صفوف", layers.raw?.row_count ?? "—"], ["إلى", fmtTs(layers.raw?.last_updated_ts).slice(0, 10)]]);
  setMini(2, [["صفوف", layers.clean?.row_count ?? "—"], ["حالة", layers.clean?.last_run_status || "—"]]);
  setMini(3, [["اكتشافات", patterns.total_detections ?? "—"], ["صفوف", layers.features?.row_count ?? "—"]]);
  setMini(5, [["آلي", overview.autotrader?.running ? "يعمل ✓" : "متوقف"], ["أوامر", overview.autotrader?.orders ?? 0]]);

  $("#nav-foot").textContent = `آخر تحديث ${new Date().toLocaleTimeString()}\n${overview.symbol || "XAUUSD"}/${overview.timeframe || "M5"}`;

  const failures = results.slice(0, 8).filter((r) => r.status === "rejected");
  if (failures.length) {
    console.warn("Partial refresh failures", failures);
  }
  })();
  try {
    await _refreshPromise;
  } finally {
    _refreshPromise = null;
  }
}

const _discoverThreads = Object.create(null); // tf -> { jobId, running }

function setTfDiscoverProgress(tf, { pct = 0, message = "جاهز", state = "", details = null } = {}) {
  const card = document.querySelector(`[data-tf-card="${tf}"]`);
  const bar = document.querySelector(`[data-tf-bar="${tf}"]`);
  const pctEl = document.querySelector(`[data-tf-pct="${tf}"]`);
  const msgEl = document.querySelector(`[data-tf-msg="${tf}"]`);
  const btn = document.querySelector(`[data-discover-tf="${tf}"]`);
  const cancelBtn = document.querySelector(`[data-cancel-discover-tf="${tf}"]`);
  const n = Math.max(0, Math.min(100, Math.round(Number(pct) || 0)));
  let rawMsg = String(message || "—").replace(/\s+/g, " ").trim();
  if (details && typeof details === "object") {
    const bits = [];
    if (details.patterns_found != null) bits.push(`${details.patterns_found} نمط`);
    if (details.speed_bars_s != null) bits.push(`${Math.round(details.speed_bars_s)} b/s`);
    if (details.eta_sec != null) bits.push(`ETA ${Math.round(details.eta_sec)}s`);
    if (bits.length) rawMsg = `${rawMsg} · ${bits.join(" · ")}`;
  }
  const shortMsg = rawMsg.length > 64 ? `${rawMsg.slice(0, 63)}…` : rawMsg;
  if (bar) bar.style.width = `${n}%`;
  if (pctEl) pctEl.textContent = `${n}%`;
  if (msgEl) {
    msgEl.textContent = shortMsg;
    msgEl.title = rawMsg;
  }
  if (card) {
    card.classList.remove("running", "success", "error");
    if (state) card.classList.add(state);
  }
  if (btn) btn.disabled = state === "running";
  if (cancelBtn) cancelBtn.hidden = state !== "running";
}

async function cancelPatternDiscoveryThread(tf) {
  const jobId = _discoverThreads[tf]?.jobId;
  if (!jobId) {
    toast(`لا يوجد خيط نشط لـ ${tf}`);
    return;
  }
  try {
    await api(`/api/jobs/${jobId}/cancel`, { method: "POST", body: "{}" });
    setTfDiscoverProgress(tf, { pct: 0, message: "جارٍ الإيقاف…", state: "running" });
    toast(`طلب إيقاف ${tf}`);
  } catch (e) {
    toast(`فشل إيقاف ${tf}: ${friendlyFetchError(e)}`);
  }
}

async function startPatternDiscoveryThread(tf, { resume = true } = {}) {
  if (!tf) return;
  if (_discoverThreads[tf]?.running) {
    toast(`استكشاف ${tf} يعمل بالفعل`);
    return;
  }
  switchTab("patterns");
  if ($("#pattern-tf")) $("#pattern-tf").value = tf;

  _discoverThreads[tf] = { running: true, jobId: null };
  setTfDiscoverProgress(tf, { pct: 1, message: resume ? "استئناف/بدء الخيط…" : "بدء الخيط…", state: "running" });
  toast(`بدء خيط استكشاف ${tf}`);

  try {
    const job = await api("/api/patterns/discover", {
      method: "POST",
      body: JSON.stringify({
        symbols: ["XAUUSD"],
        timeframes: [tf],
        force_rebuild: false,
        resume: !!resume,
      }),
    });
    _discoverThreads[tf].jobId = job.id;

    const done = await pollJob(job.id, {
      intervalMs: 900,
      onProgress: (j) => {
        setTfDiscoverProgress(tf, {
          pct: j.progress ?? 0,
          message: j.message || `خيط ${tf}`,
          state: "running",
          details: j.details?.discovery || null,
        });
      },
    });

    if (done.status === "cancelled") {
      setTfDiscoverProgress(tf, { pct: 0, message: "موقوف — يمكن الاستئناف", state: "error" });
      toast(`أُوقف خيط ${tf} (الاستئناف متاح)`);
      return;
    }
    if (done.status === "error") throw new Error(friendlyFetchError(done.error || "فشلت المهمة"));
    setTfDiscoverProgress(tf, { pct: 100, message: "اكتمل", state: "success" });
    toast(`اكتمل خيط ${tf}`);
    await refresh(tf);
  } catch (e) {
    const msg = friendlyFetchError(e);
    setTfDiscoverProgress(tf, { pct: 0, message: msg || "فشل", state: "error" });
    toast(`فشل ${tf}: ${msg}`);
    throw e;
  } finally {
    _discoverThreads[tf] = { running: false, jobId: null };
  }
}

async function startAllPatternThreads() {
  const tfs = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"];
  switchTab("patterns");
  toast("تشغيل كل الأطر كخيوط متوازية…");
  // Launch without awaiting each other — truly parallel
  const tasks = tfs.map((tf) =>
    startPatternDiscoveryThread(tf).catch(() => null),
  );
  await Promise.all(tasks);
  toast("انتهت كل خيوط الاستكشاف");
  await refresh();
}

async function startPatternDiscovery(timeframes, label) {
  // Backward-compatible: multi-TF list runs as separate threads
  if (!timeframes || !timeframes.length) {
    await startAllPatternThreads();
    return;
  }
  if (timeframes.length === 1) {
    await startPatternDiscoveryThread(timeframes[0]);
    return;
  }
  await Promise.all(timeframes.map((tf) => startPatternDiscoveryThread(tf).catch(() => null)));
  toast(label || "اكتملت الخيوط");
}

function bind() {
  $$(".nav-btn").forEach((btn) => btn.addEventListener("click", () => switchTab(btn.dataset.tab)));

  document.addEventListener("click", (ev) => {
    const sectionBtn = ev.target.closest("[data-view-pattern-section]");
    if (sectionBtn) {
      const tf = $("#pattern-tf")?.value || "M5";
      viewPatternSectionJson(tf, sectionBtn.getAttribute("data-view-pattern-section"));
      return;
    }
    const patBtn = ev.target.closest("[data-view-pattern-json]");
    if (patBtn) {
      viewPatternSectionJson(
        patBtn.getAttribute("data-view-pattern-json"),
        patBtn.getAttribute("data-section"),
      );
      return;
    }
    const discoverTf = ev.target.closest("[data-discover-tf]");
    if (discoverTf) {
      const tf = discoverTf.getAttribute("data-discover-tf");
      // Fire independently — do not block other buttons
      startPatternDiscoveryThread(tf).catch(() => {});
      return;
    }
    const cancelTf = ev.target.closest("[data-cancel-discover-tf]");
    if (cancelTf) {
      cancelPatternDiscoveryThread(cancelTf.getAttribute("data-cancel-discover-tf"));
      return;
    }
    const btn = ev.target.closest("[data-view-state]");
    if (btn) {
      viewRegistryJson(btn.getAttribute("data-view-state"));
      return;
    }
    if (ev.target.closest("[data-close-modal]")) closeJsonModal();
  });
  if ($("#pattern-files-all-tf") && !$("#pattern-files-all-tf")._bound) {
    $("#pattern-files-all-tf")._bound = true;
    $("#pattern-files-all-tf").addEventListener("change", () => {
      renderPatternJsonFiles(window._patternJsonFiles || [], $("#pattern-tf")?.value || "M5");
    });
  }
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") closeJsonModal();
  });

  if ($("#pattern-tf") && !$("#pattern-tf")._bound) {
    $("#pattern-tf")._bound = true;
    $("#pattern-tf").addEventListener("change", () => {
      refresh($("#pattern-tf").value).catch((e) => toast(e.message));
      renderPatternJsonFiles(window._patternJsonFiles || [], $("#pattern-tf").value);
    });
  }

  $("#btn-refresh").addEventListener("click", () => refresh().catch((e) => toast(e.message)));
  const btnExportTraining = $("#btn-export-training-html");
  if (btnExportTraining) {
    btnExportTraining.addEventListener("click", async () => {
      try {
        await exportTrainingPageHtml();
        toast("تم تصدير صفحة التدريب إلى HTML");
      } catch (e) {
        toast(e.message || "تعذر تصدير الصفحة");
      }
    });
  }
  const btnOpenTrainingArtifact = $("#btn-open-training-artifact");
  if (btnOpenTrainingArtifact) {
    btnOpenTrainingArtifact.addEventListener("click", async () => {
      try {
        await openTrainingArtifactPath();
        toast("تم فتح مسار ناتج التدريب");
      } catch (e) {
        toast(e.message || "تعذر فتح المسار");
      }
    });
  }
  $("#btn-pipeline").addEventListener("click", async () => {
    try {
      await startJob(
        "/api/pipeline/1-3",
        { symbols: ["XAUUSD"] },
        "تحديث بيانات الذهب",
        { showPipelineProgress: true },
      );
    } catch (e) {
      toast(e.message);
    }
  });
  const btnPipelineStop = $("#btn-pipeline-stop");
  if (btnPipelineStop) {
    btnPipelineStop.addEventListener("click", async () => {
      const jobId = _activeJobs.pipeline;
      if (!jobId) {
        toast("لا توجد مهمة تحديث جارية");
        setPipelineStopEnabled(false);
        return;
      }
      try {
        btnPipelineStop.disabled = true;
        await api(`/api/jobs/${jobId}/cancel`, { method: "POST", body: "{}" });
        setPipelineProgress(true, 0, "جارٍ إيقاف التحديث…");
        toast("تم إرسال طلب إيقاف التحديث");
      } catch (e) {
        toast(e.message || "تعذر إيقاف المهمة");
        setPipelineStopEnabled(true);
      }
    });
  }
  const btnDiscover = $("#btn-pattern-discover");
  if (btnDiscover) {
    btnDiscover.addEventListener("click", async () => {
      try {
        await startAllPatternThreads();
      } catch (e) {
        toast(e.message);
      }
    });
  }
  const btnDiscoverParallel = $("#btn-pattern-discover-parallel");
  if (btnDiscoverParallel) {
    btnDiscoverParallel.addEventListener("click", async () => {
      try {
        btnDiscoverParallel.disabled = true;
        await startAllPatternThreads();
      } catch (e) {
        toast(e.message);
      } finally {
        btnDiscoverParallel.disabled = false;
      }
    });
  }
  const btnResumeAll = $("#btn-pattern-resume-all");
  if (btnResumeAll) {
    btnResumeAll.addEventListener("click", async () => {
      const tfs = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"];
      switchTab("patterns");
      toast("استئناف خيوط الاستكشاف (resume=true)…");
      await Promise.all(
        tfs.map((tf) => startPatternDiscoveryThread(tf, { resume: true }).catch(() => null)),
      );
      toast("انتهى الاستئناف");
      await refresh();
    });
  }
  const btnExportJson = $("#btn-pattern-export-json");
  if (btnExportJson) {
    btnExportJson.addEventListener("click", async () => {
      try {
        switchTab("patterns");
        await startJob(
          "/api/patterns/export-json",
          { symbols: ["XAUUSD"] },
          "تصدير نتائج الأنماط إلى JSON",
          { showPatternProgress: true },
        );
      } catch (e) {
        toast(e.message);
      }
    });
  }
  $("#btn-kill").addEventListener("click", async () => {
    try {
      await api("/api/kill-switch", { method: "POST", body: JSON.stringify({ active: true, reason: "manual_ui" }) });
      toast("تم تفعيل إيقاف الطوارئ");
      await refresh();
    } catch (e) { toast(e.message); }
  });
  async function startAutoTrading(mode) {
    const btn = mode === "demo" ? $("#btn-auto-start") : $("#btn-auto-start-paper");
    try {
      const timeframes = getSelectedAutoTimeframes();
      if (!timeframes.length) {
        toast("اختر إطاراً زمنياً واحداً على الأقل");
        return;
      }
      if (btn) btn.disabled = true;
      renderAuto({
        running: true,
        mode,
        symbol: "XAUUSD",
        timeframe: timeframes[0],
        timeframes,
        fusion_mode: timeframes.length > 1 ? "weighted_consensus" : undefined,
        interval_seconds: 60,
        cycles: 0,
        signals: 0,
        orders: 0,
        started_at: new Date().toISOString(),
        last_cycle_at: null,
        last_error: null,
      }, _latestModels || {});
      switchTab("trade");
      toast(mode === "demo"
        ? `بدأ التداول الآلي Demo · الأوامر تُرسل إلى MT5 · ${timeframes.join(" · ")}`
        : `بدأ التداول الآلي Paper · تسجيل فقط · ${timeframes.join(" · ")}`);

      const status = await api("/api/autotrade/start", {
        method: "POST",
        body: JSON.stringify({
          mode,
          interval_seconds: 60,
          symbol: "XAUUSD",
          timeframe: timeframes[0],
          timeframes,
        }),
      });
      renderAuto(status, _latestModels || {});
      refresh().catch(() => {});
    } catch (e) {
      toast(e.message);
      try {
        const st = await api("/api/autotrade/status");
        renderAuto(st, _latestModels || {});
      } catch (_) {
        renderAuto({ running: false }, _latestModels || {});
      }
    }
  }

  $("#btn-auto-start").addEventListener("click", () => startAutoTrading("demo"));
  if ($("#btn-auto-start-paper")) {
    $("#btn-auto-start-paper").addEventListener("click", () => startAutoTrading("paper"));
  }
  $("#btn-auto-stop").addEventListener("click", async () => {
    try {
      $("#btn-auto-stop") && ($("#btn-auto-stop").disabled = true);
      renderAuto({ running: false, mode: "paper" }, _latestModels || {});
      const status = await api("/api/autotrade/stop", { method: "POST", body: "{}" });
      toast("تم إيقاف التداول الآلي");
      renderAuto(status, _latestModels || {});
      refresh().catch(() => {});
    } catch (e) { toast(e.message); }
  });

  $("#btn-settings-save")?.addEventListener("click", async () => {
    const btn = $("#btn-settings-save");
    try {
      if (btn) btn.disabled = true;
      await saveLiveSettings();
    } catch (e) {
      toast(e.message);
    } finally {
      if (btn) btn.disabled = false;
    }
  });

  $("#setting-use-spread-filter")?.addEventListener("change", () => {
    const on = !!$("#setting-use-spread-filter").checked;
    applyLiveSettingsForm({
      ...(getLiveSettingsCache() || {}),
      use_live_spread_filter: on,
      max_entry_spread_pips: Number($("#setting-max-spread")?.value || 12),
      tight_spread_pips: Number($("#setting-tight-spread")?.value || 12),
      max_entries_per_cycle: Number($("#setting-max-entries")?.value || 8),
    });
  });

  if ($("#auto-tf-all") && !$("#auto-tf-all")._bound) {
    $("#auto-tf-all")._bound = true;
    $("#auto-tf-all").addEventListener("change", () => {
      const on = $("#auto-tf-all").checked;
      $$('#auto-tf-checks input[name="auto-tf"]').forEach((el) => {
        if (!el.disabled) el.checked = on;
      });
      syncAutoTfAllCheckbox();
    });
    $$('#auto-tf-checks input[name="auto-tf"]').forEach((el) => {
      el.addEventListener("change", syncAutoTfAllCheckbox);
    });
    syncAutoTfAllCheckbox();
  }

  $$("[data-run]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const kind = btn.getAttribute("data-run");
      btn.disabled = true;
      try {
        if (kind === "1") await startJob("/api/engines/1/run", { symbols: ["XAUUSD"], timeframes: ["M5"] }, "جلب البيانات");
        else if (kind === "2") await startJob("/api/engines/2/run", { symbols: ["XAUUSD"], timeframes: ["M5"] }, "تنظيف");
        else if (kind === "3") await startJob("/api/engines/3/run", { symbols: ["XAUUSD"], timeframes: ["M1", "M5", "M15", "M30", "H1", "H4", "W1"] }, "إعادة اكتشاف الأنماط");
        else if (kind === "4") {
          switchTab("train");
          await startJob(
            "/api/engines/4/run",
            {
              symbols: ["XAUUSD"],
              timeframes: ["M1", "M5", "M15", "M30", "H1", "H4"],
            },
            "تدريب الذهب وإنشاء Final Model",
            { showTrainingProgress: true },
          );
        }
        else if (kind === "5-demo") await startJob("/api/engines/5/run", { symbols: ["XAUUSD"], execute_demo: true }, "Demo");
      } catch (e) { toast(e.message); }
      finally { btn.disabled = false; }
    });
  });
}

bind();
switchTab(getSavedTab() || "train");
refresh().catch((e) => toast(e.message));
setInterval(() => {
  if (shouldPauseAutoRefresh()) return;
  refresh().catch(() => {});
}, 15000);
// Faster pulse while autotrader is active so new trades appear quickly.
setInterval(async () => {
  if (shouldPauseAutoRefresh()) return;
  try {
    const st = await api("/api/autotrade/status");
    if (!st?.running) return;
    renderAuto(st, _latestModels || {});
    const trades = await api("/api/trades?limit=50");
    renderTrades(trades.trades || []);
    const decision = await apiOptional("/api/decision/latest", { empty: true });
    if (!decision.empty) renderDecision(decision);
  } catch (_) { /* ignore */ }
}, 5000);
