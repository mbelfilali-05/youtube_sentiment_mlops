// Change this to your Hugging Face Space URL in Phase 6
const API_URL = "http://localhost:8000";
const BATCH_SIZE = 200;      // API accepts up to 500 per request
const MAX_RENDER = 200;      // cap DOM rows to keep the popup fast
const TIMEOUT_MS = 30000;    // a sleeping HF Space can take a while to wake up

const COLORS = { Positive: "#16a34a", Neutral: "#9ca3af", Negative: "#dc2626" };
const $ = (id) => document.getElementById(id);
let results = [];
let filter = "all";
let lastMs = 0;

// ---------- helpers ----------
async function activeTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

function setState(state, message = "") {
  $("loading").classList.toggle("hidden", state !== "loading");
  $("error").classList.toggle("hidden", state !== "error");
  $("results").classList.toggle("hidden", state !== "results");
  $("analyze").disabled = state === "loading";
  if (state === "error") $("error").textContent = message;
}

async function postBatch(comments) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(`${API_URL}/predict_batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments }),
      signal: ctrl.signal,
    });
    if (!res.ok) throw new Error(`API error (HTTP ${res.status})`);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

// ---------- main flow ----------
async function analyze() {
  setState("loading");
  try {
    const tab = await activeTab();
    if (!tab || !tab.url || !tab.url.startsWith("https://www.youtube.com/watch")) {
      throw new Error("Open a YouTube video page first.");
    }

    let res;
    try {
      res = await chrome.tabs.sendMessage(tab.id, {
        type: "EXTRACT_COMMENTS",
        limit: 500,
        includeReplies: $("replies").checked,
      });
    } catch {
      throw new Error("Cannot reach the page. Reload the YouTube tab and try again.");
    }
    if (!res || !res.comments.length) {
      throw new Error("No comments found. Scroll down so comments load, then retry.");
    }

    const preds = [];
    let ms = 0;
    for (let i = 0; i < res.comments.length; i += BATCH_SIZE) {
      const data = await postBatch(res.comments.slice(i, i + BATCH_SIZE));
      preds.push(...data.predictions);
      ms += data.inference_ms;
    }
    results = preds;
    lastMs = ms;
    render();
    setState("results");
  } catch (e) {
    let msg = e.message;
    if (e.name === "AbortError") msg = "The API took too long to respond. Try again.";
    else if (msg.includes("Failed to fetch")) msg = "API unreachable. Is the server running?";
    setState("error", msg);
  }
}

// ---------- rendering ----------
function stats() {
  const counts = { Positive: 0, Neutral: 0, Negative: 0 };
  results.forEach((r) => counts[r.sentiment]++);
  return counts;
}

function render() {
  const counts = stats();
  const total = results.length;
  const pct = (n) => (total ? (100 * n) / total : 0);

  // Donut chart (pure CSS conic-gradient)
  const p = pct(counts.Positive), n = pct(counts.Neutral);
  $("donut").style.background =
    `conic-gradient(${COLORS.Positive} 0 ${p}%, ${COLORS.Neutral} ${p}% ${p + n}%, ${COLORS.Negative} ${p + n}% 100%)`;
  $("total").textContent = total;

  // Bars
  const bars = $("bars");
  bars.replaceChildren();
  for (const name of ["Positive", "Neutral", "Negative"]) {
    const row = document.createElement("div");
    row.className = "bar-row";
    const label = document.createElement("div");
    label.className = "bar-label";
    const a = document.createElement("span");
    a.textContent = name;
    const b = document.createElement("span");
    b.textContent = `${pct(counts[name]).toFixed(1)}% (${counts[name]})`;
    label.append(a, b);
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("div");
    fill.style.width = `${pct(counts[name])}%`;
    fill.style.background = COLORS[name];
    bar.append(fill);
    row.append(label, bar);
    bars.append(row);
  }

  $("meta").textContent = `${total} comments analyzed • model time ${lastMs.toFixed(0)} ms`;
  renderList();
}

function renderList() {
  const list = $("list");
  list.replaceChildren();
  const rows = results.filter((r) => filter === "all" || String(r.label) === filter);
  const frag = document.createDocumentFragment();
  rows.slice(0, MAX_RENDER).forEach((r) => {
    const li = document.createElement("li");
    li.className = r.label === 1 ? "pos" : r.label === -1 ? "neg" : "";
    const text = document.createElement("div");
    text.className = "c-text";
    text.textContent = r.text;               // textContent: never inject comments as HTML
    const meta = document.createElement("div");
    meta.className = "c-meta";
    meta.textContent = `${r.sentiment} • confidence ${(r.confidence * 100).toFixed(0)}%`;
    li.append(text, meta);
    frag.append(li);
  });
  if (rows.length > MAX_RENDER) {
    const li = document.createElement("li");
    li.className = "c-meta";
    li.textContent = `Showing ${MAX_RENDER} of ${rows.length}. Use Export CSV for everything.`;
    frag.append(li);
  }
  if (!rows.length) {
    const li = document.createElement("li");
    li.className = "c-meta";
    li.textContent = "No comments in this category.";
    frag.append(li);
  }
  list.append(frag);
}

// ---------- copy / export ----------
function summaryText() {
  const c = stats();
  const t = results.length;
  const line = (n) => `${n}: ${c[n]} (${((100 * c[n]) / t).toFixed(1)}%)`;
  return [
    `YouTube sentiment analysis (${t} comments)`,
    line("Positive"), line("Neutral"), line("Negative"),
    "",
    ...results.slice(0, 50).map((r) => `[${r.sentiment} ${(r.confidence * 100).toFixed(0)}%] ${r.text}`),
  ].join("\n");
}

async function copyResults() {
  await navigator.clipboard.writeText(summaryText());
  const btn = $("copy");
  btn.textContent = "Copied ✓";
  setTimeout(() => (btn.textContent = "Copy results"), 1500);
}

function exportCsv() {
  const esc = (s) => `"${String(s).replace(/"/g, '""')}"`;
  const lines = ["sentiment,confidence,text",
    ...results.map((r) => `${r.sentiment},${r.confidence},${esc(r.text)}`)];
  const blob = new Blob(["\ufeff" + lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "youtube_sentiment.csv";
  a.click();
  URL.revokeObjectURL(a.href);
}

// ---------- events ----------
$("analyze").addEventListener("click", analyze);
$("copy").addEventListener("click", copyResults);
$("csv").addEventListener("click", exportCsv);
$("scroll").addEventListener("click", async () => {
  const tab = await activeTab();
  try { await chrome.tabs.sendMessage(tab.id, { type: "SCROLL_FOR_COMMENTS" }); } catch {}
});
$("filters").addEventListener("click", (e) => {
  const f = e.target.dataset.f;
  if (!f) return;
  filter = f;
  document.querySelectorAll("#filters button").forEach((b) => b.classList.toggle("active", b === e.target));
  renderList();
});

// Dark mode (persisted)
function applyTheme(dark) { document.body.classList.toggle("dark", dark); }
chrome.storage.local.get("dark").then(({ dark }) => applyTheme(!!dark));
$("theme").addEventListener("click", () => {
  const dark = !document.body.classList.contains("dark");
  applyTheme(dark);
  chrome.storage.local.set({ dark });
});