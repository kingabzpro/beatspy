"use strict";

const COLORS = ["#b4f272", "#79b6ff", "#ffbc79", "#cb9aff", "#ff9292", "#74d8cb", "#f0b9da", "#d1d7de", "#8d9969"];
// Only the states that change how the numbers read earn a pill. "local" (a run
// executed here against the live providers) is every real run's state, so it
// says nothing a reader can use and shows nothing.
const TRUST = {replayed: "Replay checked", synthetic: "Synthetic demo"};
// Every decision model bills input only; output tokens are free and no provider
// offers prompt caching. Rates checked 2026-10-08 against the cited sources.
const PRICES = {
  "gpt-6-luna": {
    label: "OpenAI Decisions (gpt-6-luna)", provider: "openai_decisions", input: 0.10, output: 0,
    source: "https://community.openai.com/t/decisions-api-is-now-available-in-public-beta/1403877",
    note: "Decisions API public beta list price, input only."},
  "jev-latest": {
    label: "TypeSafe Jev", provider: "typesafe", input: 0.042, output: 0,
    source: "https://docs.typesafe.ai/models",
    note: "$42 per 1B input tokens (about $0.042 per 1M); output is free."},
  "clef": {
    label: "Cloudflare Clef (27B)", provider: "cloudflare", input: 0.24, output: 0,
    source: "https://developers.cloudflare.com/workers-ai/models/clef/index.md",
    note: "Workers AI list price, input only."},
  "clef-flash": {
    label: "Cloudflare Clef-flash (9B)", provider: "cloudflare", input: 0.038, output: 0,
    source: "https://developers.cloudflare.com/workers-ai/models/clef-flash/index.md",
    note: "Workers AI list price, input only; cut from $0.09 on 2026-10-09."},
  "fasttrade-subq": {
    label: "FastTrade (SubQ Snap 1.1)", provider: "typesafe", input: 0.05, output: 0,
    source: "https://api.subq.ai/v1/systemone",
    note: "FastTrade's daily feature state asked the benchmark noul question by SubQ Snap 1.1; input only."},
};
const SIGNALS = ["noul", "choice", "rank"];
// A JSON null (or an empty string) is "not measured", not zero: Number(null) is
// 0, so without this guard a missing miscalibration gap rendered as "0.00%",
// which reads as perfect agreement.
const missing = value => value === null || value === undefined || value === "";
const percent = value => !missing(value) && Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(2)}%` : "n/a";
const ratio = value => !missing(value) && Number.isFinite(Number(value)) ? Number(value).toFixed(2) : "n/a";
const score = value => !missing(value) && Number.isFinite(Number(value)) ? Number(value).toFixed(3) : "n/a";
// Decision-model runs cost fractions of a cent, so small values keep more digits.
const dollars = value => {
  if (missing(value)) return "n/a";
  const number = Number(value);
  if (!Number.isFinite(number)) return "n/a";
  return `$${number.toFixed(Math.abs(number) < 0.01 ? 6 : 4)}`;
};
const count = value => !missing(value) && Number.isInteger(Number(value)) ? String(Number(value)) : "n/a";
// Runs record equity in dollars from a $100,000 start, so a $10,000 book is
// the same journey at a tenth. Profit reads as signed whole dollars.
const START_USD = 10_000;
const money = value => !missing(value) && Number.isFinite(Number(value))
  ? `${Number(value) >= 0 ? "+" : "-"}$${Math.abs(Math.round(Number(value))).toLocaleString("en-US")}`
  : "n/a";
const usd = value => Number.isFinite(Number(value)) ? `$${Math.round(Number(value)).toLocaleString("en-US")}` : "n/a";
// Costs in this section span two orders of magnitude, so those columns keep
// three significant digits instead of a fixed number of decimals.
const preciseDollars = value => {
  if (missing(value) || !Number.isFinite(Number(value))) return "n/a";
  const number = Number(value);
  return `$${number === 0 ? "0" : number.toPrecision(3)}`;
};
const plural = (value, word) => `${count(value)} ${word}${Number(value) === 1 ? "" : "s"}`;

function parseCSV(text) {
  const rows = [], row = [];
  let field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      if (quoted && text[i + 1] === '"') {field += '"'; i++;}
      else quoted = !quoted;
    } else if (!quoted && (c === "," || c === "\n")) {
      row.push(field); field = "";
      if (c === "\n") {if (row.some(Boolean)) rows.push([...row]); row.length = 0;}
    } else if (c !== "\r" || quoted) field += c;
  }
  if (quoted) throw new Error("Malformed CSV: unterminated quoted field");
  if (field || row.length) {row.push(field); rows.push([...row]);}
  const headers = rows.shift() || [];
  return rows.map(values => {
    if (values.length !== headers.length) throw new Error("Malformed CSV: wrong column count");
    return Object.fromEntries(headers.map((key, i) => [key, values[i]]));
  });
}

function modelName(model) {
  // Router-served models carry their route as a key prefix ("route-model"); the
  // board names the model, not the transport that happened to serve it.
  return String(model).replace(/^(?:openrouter|requesty|respan)-/, "").split("/").at(-1);
}
// Older runs recorded broad categories such as "ai" and "algo".
const LEGACY_MODEL_CATEGORIES = {
  "fasttrade": "decision-ai", "fasttrade-subq": "decision-ai",
  "llama-3.3-70b": "llm",
  "bettertradingagents": "agents", "bettertradingagents-live": "agents",
  "desk-router": "agents", "trade-memory": "agents", "risk-review": "agents", "bandit-router": "agents",
  "paperrl": "ml", "deeptrade": "hybrids",
  "timestral-209m": "foundation", "chronos-2": "foundation", "timesfm-3.0": "foundation", "t0-gamma": "foundation",
  "tpi-rs": "rule-based", "hrp": "rule-based", "residual-momentum": "rule-based", "residual-reversal": "rule-based",
  "lightgbm": "ml", "logistic-regression": "ml", "logistic-lightgbm": "ml", "stock-jepa": "ml",
  "lambdamart": "ml", "elastic-net": "ml", "shallow-mlp": "ml",
};
const MODEL_CATEGORIES = new Set(["decision-ai", "ml", "foundation", "rule-based", "llm", "agents", "hybrids"]);
const LEGACY_CATEGORIES = {ai: "decision-ai", algo: "rule-based", agent: "agents", hybrid: "hybrids", rl: "ml"};

function modelCategory(run) {
  if (MODEL_CATEGORIES.has(run.category)) return run.category;
  const name = modelName(run.model);
  if (Object.hasOwn(LEGACY_MODEL_CATEGORIES, name)) return LEGACY_MODEL_CATEGORIES[name];
  if (Object.hasOwn(LEGACY_CATEGORIES, run.category)) return LEGACY_CATEGORIES[run.category];
  return run.category == null ? (run.provider === "local" ? "rule-based" : "decision-ai") : null;
}

// Display names and artwork. Models use a project mark or a glyph for their method.
const MODEL_META = {
  "tpi-rs": ["TPI baseline", "tpi.svg"],
  "fasttrade": ["FastTrade", "fasttrade.svg"],
  "fasttrade-subq": ["FastTrade", "fasttrade.svg"],
  "llama-3.3-70b": ["Llama 3.3 70B", "meta.png"],
  "inkling-2-small": ["Inkling 2 Small", "thinking-machines.png"],
  "kimi-k3.1-pro": ["Kimi K3.1 Pro", "kimi.ico"],
  "minimax-m3.1-flash": ["MiniMax M3.1 Flash", "minimax.ico"],
  "glm-5.3-flash": ["GLM-5.3 Flash", "z-ai.svg"],
  "glm-5.3": ["GLM-5.3", "z-ai.svg"],
  "bettertradingagents": ["BetterTradingAgents", "bettertradingagents.png"],
  "bettertradingagents-live": ["BetterTradingAgents Live", "bettertradingagents.png"],
  "desk-router": ["Desk Router", "agent-desk-router.svg"],
  "trade-memory": ["Trade Memory", "agent-trade-memory.svg"],
  "risk-review": ["Risk Review", "agent-risk-review.svg"],
  "bandit-router": ["Bandit Router", "agent-bandit-router.svg"],
  "paperrl": ["PaperRL", "paperrl.svg"],
  "deeptrade": ["DeepTrade", "deeptrade.svg"],
  "gpt-6-luna": ["GPT-6-Luna", "openai.svg"],
  "gpt-6-luna-llm": ["GPT-6 Luna (Responses API)", "openai.svg"],
  "claude-haiku-5-5": ["Claude Haiku 5.5", "claude.png"],
  "lambdamart-hrp": ["LambdaMART + HRP", "hybrid-rank-risk.svg"],
  "chronos-lightgbm": ["Chronos-2 + LightGBM", "hybrid-forecast-trees.svg"],
  "momentum-reversal": ["Momentum + Reversal", "hybrid-signal-switch.svg"],
  "jepa-elastic-net": ["STOCK-JEPA + Elastic Net", "hybrid-jepa-linear.svg"],
  "lightgbm-momentum": ["LightGBM + Momentum", "hybrid-trend-filter.svg"],
  "jev-latest": ["Jev-Latest", "typesafe.png"],
  "jev-1-14": ["Jev 1.14", "typesafe.png"],
  "clef": ["Clef", "cloudflare.svg"],
  "clef-flash": ["Clef-Flash", "cloudflare.svg"],
  "clef-omni": ["Clef-Omni", "cloudflare.svg"],
  "pplx-decider": ["Pplx-Decider", "perplexity.svg"],
  "decision-1": ["Microsoft Decision 1", "microsoft.svg"],
  "microsoft-decision-1": ["Microsoft Decision 1", "microsoft.svg"],
  "kev-4b": ["Kev-4B", "jaredpalmer.ico"],
  "liquid-d1": ["Liquid D1", "liquid.ico"],
  "liquid-d1-ultra": ["Liquid D1 Ultra", "liquid.ico"],
  "liquid-d1-2": ["Liquid D1.2", "liquid.ico"],
  "liquid-d1-2-ultra": ["Liquid D1.2 Ultra", "liquid.ico"],
  "drex": ["Drex v1.5", "nace.svg"],
  "nace-drex-max-1-1": ["Drex Max v1.1", "nace.svg"],
  "span-01-lite": ["Span-01 Lite", "respan.ico"],
  "span-01": ["Span-01", "respan.ico"],
  "span-02": ["Span-02", "respan.ico"],
  "span-01-free": ["Span-01 (free)", "respan.ico"],
  "baseten-mammoth-1": ["Mammoth 1", "baselabs.png"],
  "baseten-elephant-1": ["Elephant 1", "baselabs.png"],
  "baseten-bison-1": ["Bison 1", "baselabs.png"],
  "baseten-groundhog-1": ["Groundhog 1", "baselabs.png"],
  "baseten-joey-1": ["Joey 1", "baselabs.png"],
  "nace-drex-max-finance": ["Drex Max Finance", "nace.svg"],
  "nace-drex-v2-0": ["Drex v2.0", "nace.svg"],
  "google-gemini-decision-1": ["Gemini Decision 1", "google.svg"],
  "standard-one-8b": ["Standard One 8B", "standardthinking.png"],
  "d3-turbo": ["D3 Turbo", "vllmsr.png"],
  "d3-mini": ["D3 Mini", "vllmsr.png"],
  "d3-edge": ["D3 Edge", "vllmsr.png"],
  "subq-snap-1-1": ["SubQ Snap 1.1", "subq.png"],
  "minara-kairos": ["Kairos", "minara-mark.png"],
  "timestral-209m": ["Timestral", "mistral.png"],
  "chronos-2": ["Chronos-2", "chronos.svg"],
  "timesfm-3.0": ["TimesFM 3.0", "google.svg"],
  "t0-gamma": ["T0-Gamma", "forecasting-company.svg"],
  "lightgbm": ["LightGBM", "lightgbm-mark.svg"],
  "logistic-regression": ["Logistic Regression", "logistic.svg"],
  "logistic-lightgbm": ["Logit + LightGBM", "ensemble.svg"],
  "hrp": ["HRP", "hrp.svg"],
  "stock-jepa": ["Stock-JEPA", "stock-jepa.svg"],
  "lambdamart": ["LambdaMART", "lambdamart.svg"],
  "residual-momentum": ["Residual Momentum", "residual-momentum.svg"],
  "elastic-net": ["Elastic Net", "elastic-net.svg"],
  "residual-reversal": ["Residual Reversal", "residual-reversal.svg"],
  "shallow-mlp": ["Shallow MLP", "shallow-mlp.svg"],
};
const displayModel = model => MODEL_META[modelName(model)]?.[0] || modelName(model);
const modelLogo = model => MODEL_META[modelName(model)]?.[1] || null;

// The board shows the current benchmark protocol only. Older-protocol runs stay
// in the payload so their run pages and the archive page keep working.
const isCurrent = run => protocolOf(run) >= 6;

function leaderboardRuns(runs) {
  const ordered = [...runs].sort((a, b) => String(b.created_utc).localeCompare(String(a.created_utc)) || String(b.run_id).localeCompare(String(b.run_id)));
  const models = new Map();
  for (const run of ordered) if (!models.has(run.model)) models.set(run.model, run);
  return [...models.values()].sort((a, b) => (Number(b.metrics?.excess_return_vs_spy) || 0) - (Number(a.metrics?.excess_return_vs_spy) || 0));
}

// The board scores consecutive six-month windows. A run's window is its frozen
// start/end pair, so the board can be read one window at a time or averaged
// across all of them; one window alone keeps the old single-period behavior.
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const monthLabel = iso => {
  const match = /^(\d{4})-(\d{2})/.exec(String(iso || ""));
  return match ? `${MONTHS[Number(match[2]) - 1]} ${match[1].slice(2)}` : "";
};
// Same dates can be scored under two benchmark protocols; they stay separate
// windows so a v2 run never quietly replaces a v5 row it is not comparable with.
const protocolOf = run => Number(run.dcn_protocol) > 0 ? Number(run.dcn_protocol) : 5;
const windowLabel = run => `${monthLabel(run.start)} - ${monthLabel(run.end)}` + (protocolOf(run) >= 6 ? " · v2" : "");
const windowKey = run => `${run.start}..${run.end}#v${protocolOf(run)}`;

function scoredWindows(allRuns) {
  // Newest run per display model inside each window: routes merge (the board
  // names the model, not the transport) and a re-run replaces its own older
  // attempt by execution time, never by score.
  const grouped = new Map();
  for (const run of allRuns) {
    if (!run.start || !run.end) continue;
    const key = windowKey(run);
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(run);
  }
  return [...grouped.entries()].map(([key, rows]) => {
    const ordered = [...rows].sort((a, b) => String(b.created_utc).localeCompare(String(a.created_utc)) || String(b.run_id).localeCompare(String(b.run_id)));
    const models = new Map();
    for (const run of ordered) {
      const name = modelName(run.model);
      if (!models.has(name)) models.set(name, run);
    }
    const spy = rows.find(run => !run.synthetic && !missing(run.metrics?.spy_total_return)
      && Number.isFinite(Number(run.metrics.spy_total_return)))?.metrics.spy_total_return;
    return {
      key, label: windowLabel(rows[0]), start: rows[0].start, end: rows[0].end,
      proto: protocolOf(rows[0]),
      spy: Number(spy), runs: [...models.values()],
    };
  }).sort((a, b) => String(a.end).localeCompare(String(b.end)) || (a.proto - b.proto));
}

function allPeriodRows(windows) {
  // One aggregate row per display model. A model that has not run a window yet
  // shows n-of-N rather than borrowing a number from another window.
  const models = new Map();
  for (const w of windows) for (const run of w.runs) {
    const name = modelName(run.model);
    if (!models.has(name)) models.set(name, {model: run.model, entries: []});
    models.get(name).entries.push({run, window: w.label, gap: Number(run.metrics?.excess_return_vs_spy)});
  }
  return [...models.values()].map(row => {
    row.entries.sort((a, b) => String(a.run.end).localeCompare(String(b.run.end)));
    const gaps = row.entries.map(entry => entry.gap).filter(Number.isFinite);
    const ranked = row.entries.filter(entry => Number.isFinite(entry.gap)).sort((a, b) => b.gap - a.gap);
    row.windows = row.entries.length;
    row.avgGap = gaps.length ? gaps.reduce((a, b) => a + b, 0) / gaps.length : null;
    const returns = row.entries.map(entry => Number(entry.run.metrics?.total_return)).filter(Number.isFinite);
    row.avgReturn = returns.length ? returns.reduce((a, b) => a + b, 0) / returns.length : null;
    row.best = ranked[0] || null;
    row.worst = ranked[ranked.length - 1] || null;
    row.beatIn = row.entries.filter(entry => entry.gap > 0).length;
    row.latest = row.entries[row.entries.length - 1]?.run;
    // A model that has run every window outranks one that has not, whatever
    // the averages say: one lucky window must never top a three-window record.
    row.complete = row.windows === windows.length;
    return row;
    // The default order is completeness first, then the average gap, with
    // windows beaten as the tiebreak: both readings of "who beat SPY" agree on
    // the top, and the Beat SPY in column can flip to wins-first with one click.
  }).sort((a, b) => (b.complete - a.complete) || (b.avgGap ?? -Infinity) - (a.avgGap ?? -Infinity) || b.beatIn - a.beatIn);
}

function inputCost(metrics, rates) {
  const inputTokens = metrics?.input_tokens, outputTokens = metrics?.output_tokens ?? 0;
  // Both rates must be recorded: a partial rate table is not a free rate.
  const input = rates?.input, output = rates?.output;
  if (![inputTokens, outputTokens, input, output].every(value => typeof value === "number" && Number.isFinite(value) && value >= 0)) return null;
  const cost = (inputTokens * input + outputTokens * output) / 1e6;
  return Number.isFinite(cost) ? cost : null;
}

function ratesFor(model, meta) {
  const pricing = meta?.decision?.pricing;
  if (typeof pricing?.cost_per_m_input === "number" && typeof pricing?.cost_per_m_output === "number") {
    return {input: pricing.cost_per_m_input, output: pricing.cost_per_m_output, note: "Rates recorded in run.json"};
  }
  return PRICES[modelName(model)] || null;
}

// Bins with no observations carry no observed rate and must not be plotted.
function reliabilitySeries(calibration) {
  const series = [];
  for (const name of SIGNALS) {
    const stats = calibration?.signals?.[name];
    if (!stats || !Array.isArray(stats.bins)) continue;
    series.push({
      name, n: stats.n, brier: stats.brier, skill: stats.brier_skill_score, ece: stats.expected_calibration_error,
      meanPredicted: stats.mean_predicted, bins: stats.bins,
      points: stats.bins.filter(bin => Number.isFinite(Number(bin.count)) && Number(bin.count) > 0 && Number.isFinite(Number(bin.observed_rate)))
        .map(bin => ({predicted: Number(bin.mean_predicted), observed: Number(bin.observed_rate), count: Number(bin.count)})),
    });
  }
  return series;
}

// An invalid answer set means "hold the previous portfolio", never "liquidate".
function allocationRows(decisions) {
  const rows = [];
  let held = {};
  for (const decision of decisions) {
    const validated = decision?.validated || {};
    if (!validated.invalid) held = {...(validated.weights || {})};
    const invested = Object.values(held).reduce((sum, weight) => sum + (Number(weight) || 0), 0);
    const cash = validated.invalid ? Math.max(0, 1 - invested) : (Number(validated.cash) || 0);
    rows.push({date: decision?.date, ...held, CASH: cash});
  }
  return rows;
}

function catalogProblems(catalog) {
  const problems = [];
  if (!catalog || typeof catalog !== "object") return ["catalog is not an object"];
  if (catalog.schema_version !== 1) problems.push("schema_version must be 1");
  if (catalog.default_trust !== "all") problems.push("default_trust must be all");
  if (!Array.isArray(catalog.runs)) return [...problems, "runs must be an array"];
  for (const run of catalog.runs) {
    if (typeof run?.run_id !== "string" || !run.run_id) problems.push("run without a run_id");
    else if (run.dir !== `runs/${run.run_id}`) problems.push(`bad artifact path for ${run.run_id}`);
    if (typeof run?.model !== "string" || !run.model) problems.push(`run ${run?.run_id} has no model`);
    if (typeof run?.provider !== "string" || !run.provider) problems.push(`run ${run?.run_id} has no provider`);
    if (!Number.isFinite(Number(run?.metrics?.total_return))) problems.push(`run ${run?.run_id} has no total return`);
    if (!Number.isFinite(Number(run?.metrics?.excess_return_vs_spy))) problems.push(`run ${run?.run_id} has no excess return`);
  }
  return problems;
}

if (typeof module !== "undefined") module.exports = {
  parseCSV, leaderboardRuns, modelName, allocationRows, percent, dollars, preciseDollars, count, plural, ratio, score, inputCost, money, usd,
  ratesFor, reliabilitySeries, catalogProblems, PRICES, SIGNALS,
  monthLabel, windowLabel, scoredWindows, allPeriodRows,
  modelCategory, modelLogo,
};

if (typeof document !== "undefined") {
  const $ = id => document.getElementById(id);
  // Each page renders only the widgets it carries, so every renderer checks for
  // its own container first and never assumes the rest of the site is present.
  const setText = (id, text) => {
    const element = $(id);
    if (element && element.textContent !== text) element.textContent = text;
  };
  const setHidden = (id, hidden) => { const element = $(id); if (element) element.hidden = hidden; };
  const node = (tag, text, className) => {
    const result = document.createElement(tag);
    if (text !== undefined) result.textContent = String(text);
    if (className) result.className = className;
    return result;
  };
  const svgNode = (tag, attrs) => {
    const result = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attrs || {})) result.setAttribute(key, String(value));
    return result;
  };
  const COLUMNS = [
    ["model", "Model", "text", "row"],
    ["total_return", "Return", "percent", "metrics"],
    ["excess_return_vs_spy", "vs SPY (pp)", "points", "metrics"],
    ["outperformance_hit_rate", "Weeks beating SPY", "percent", "metrics"],
    ["max_drawdown", "Worst drop", "percent", "metrics"],
    ["hold", "Holds", "percent", "row"],
    ["accuracy", "Right calls", "percent", "row"],
  ];
  const FORMAT = {text: value => value ?? "n/a", percent, signed: percent, ratio, score, dollars, count,
    points: value => missing(value) || !Number.isFinite(Number(value)) ? "n/a" : (Number(value) * 100).toFixed(2),
    // These models answer in tens to hundreds of milliseconds; "0.00 s" hides
    // every difference between them.
    millis: value => {
      const ms = Number(value) * 1000;
      if (!Number.isFinite(ms)) return "n/a";
      return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
    }};
  const rowValue = (run, key, source) => {
    if (key === "model") return modelName(run.model);
    // The hold share lives one level down, in the catalog's action shares.
    if (key === "hold") return run.actions?.hold;
    // Call accuracy is the plain-words stand-in for the calibration scores.
    if (key === "accuracy") return run.metrics?.calibration?.signals?.noul?.accuracy;
    return source === "metrics" ? run.metrics?.[key] : run[key];
  };
  let runs = [], selected = null, loading = 0;
  let sortKey = "excess_return_vs_spy", descending = true;

  async function fetchText(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Could not load ${path} (${response.status})`);
    return response.text();
  }
  function table(target, headers, records) {
    const element = node("table"), head = node("thead"), tr = node("tr"), body = node("tbody");
    for (const title of headers) tr.append(node("th", title));
    head.append(tr); element.append(head, body);
    for (const record of records) {
      const row = node("tr");
      for (const cell of record) row.append(node("td", cell));
      body.append(row);
    }
    $(target).replaceChildren(element);
  }
  const runRates = run => PRICES[modelName(run.model)]
    || (typeof run.pricing?.cost_per_m_input === "number"
      ? {input: run.pricing.cost_per_m_input, output: run.pricing.cost_per_m_output ?? 0} : null);
  function renderPricing() {
    if (!$("pricing-table")) return;
    // One row per model on the board, on units that compare. Rates are per
    // million tokens, so a run's bill is its rate times the tokens the provider
    // billed; the columns standardize that to one decision (the unit the board
    // is scored in) and to the run, and the caption explains why some providers
    // report far more tokens than the rest for the same frozen state.
    // The pricing page prices one window, the latest scored one, so a model
    // is exactly one row even when several windows are published.
    const windows = scoredWindows(runs.filter(isCurrent));
    const onBoard = windows.length ? windows[windows.length - 1].runs : [];
    const records = onBoard.map(run => {
      const rates = runRates(run);
      const decisions = Number(run.decisions ?? run.metrics?.decisions);
      const tokens = Number(run.metrics?.input_tokens);
      const cost = run.estimated_cost_usd ?? inputCost(run.metrics, rates);
      return {
        run, rates,
        tokens: decisions > 0 && Number.isFinite(tokens) ? Math.round(tokens / decisions) : null,
        perDecision: Number.isFinite(cost) && decisions > 0 ? cost / decisions : null,
        cost,
      };
    });
    const element = node("table"), head = node("thead"), header = node("tr"), body = node("tbody");
    const sortable = new Map();
    for (const [title, className, key] of [["Model", ""], ["Input $ / 1M", "num"], ["Tokens / decision", "num"],
      ["Cost / decision", "num", "perDecision"], ["Run cost", "num", "cost"], ["Source", ""]]) {
      const th = node("th", key ? undefined : title, className);
      th.scope = "col";
      if (key) {
        const button = node("button", title);
        button.type = "button";
        const arrow = node("span", undefined, "sort-arrow");
        arrow.setAttribute("aria-hidden", "true");
        button.append(arrow);
        button.onclick = () => {
          const direction = th.getAttribute("aria-sort") === "ascending" ? "descending" : "ascending";
          sortPrices(key, direction);
          setText("pricing-sort-status", `Sorted by ${title}, ${direction}.`);
        };
        th.append(button);
        sortable.set(key, th);
      }
      header.append(th);
    }
    head.append(header); element.append(head, body);
    for (const record of records) {
      const {run, rates, tokens, perDecision, cost} = record;
      const row = node("tr");
      record.row = row;
      const modelCell = node("td");
      const logo = modelLogo(run.model);
      if (logo) {
        const img = node("img");
        img.className = "model-logo"; img.alt = ""; img.loading = "lazy"; img.decoding = "async";
        img.src = `assets/logos/${logo}`;
        modelCell.append(img);
      }
      modelCell.append(displayModel(run.model));
      row.append(modelCell,
        node("td", typeof rates?.input === "number" ? `$${rates.input.toFixed(3)}` : "n/a", "num"),
        node("td", tokens === null ? "n/a" : tokens.toLocaleString(), "num"),
        node("td", preciseDollars(perDecision), "num"),
        node("td", preciseDollars(cost), "num"));
      const cell = node("td");
      if (rates?.source) {
        const link = node("a", "Source ↗");
        link.href = rates.source; link.rel = "noreferrer noopener"; link.target = "_blank";
        cell.append(link);
        row.title = rates.note || "";
      } else {
        cell.append(node("span", "Run record"));
      }
      row.append(cell); body.append(row);
    }
    function sortPrices(key, direction) {
      for (const th of sortable.values()) th.removeAttribute("aria-sort");
      sortable.get(key).setAttribute("aria-sort", direction);
      const factor = direction === "ascending" ? 1 : -1;
      records.sort((a, b) => {
        const av = a[key], bv = b[key];
        if (!Number.isFinite(av)) return Number.isFinite(bv) ? 1 : 0;
        if (!Number.isFinite(bv)) return -1;
        return factor * (av - bv);
      });
      for (const {row} of records) {
        if (typeof body.moveBefore === "function") body.moveBefore(row, null);
        else body.appendChild(row);
      }
    }
    sortPrices("perDecision", "descending");
    $("pricing-table").replaceChildren(element);
    setHidden("cost-overview", !onBoard.length);
    const latest = windows.at(-1);
    setText("cost-window", latest
      ? `${latest.start} to ${latest.end}. Some providers bill the shared market data once per question, so token counts vary by provider.`
      : "");
  }

  function renderReport() {
    if (!$("report-results")) return;
    const candidates = runs.filter(run => protocolOf(run) === 6 && !run.synthetic);
    const windows = scoredWindows(candidates).filter(window => window.runs.some(run =>
      Number.isFinite(run.metrics?.excess_return_vs_spy)));
    if (!windows.length) {
      setText("status", "No published protocol v2 results are available in this catalog.");
      return;
    }
    const categories = {
      "decision-ai": "Decision AI", ml: "Machine learning", foundation: "Foundation models",
      "rule-based": "Rule-based", llm: "LLMs", agents: "Agents", hybrids: "Hybrids",
    };
    const median = values => {
      const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
      if (!sorted.length) return null;
      const middle = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
    };
    const gap = run => run.metrics?.excess_return_vs_spy;
    const summaries = windows.map(window => {
      const scored = window.runs.filter(run => Number.isFinite(gap(run)));
      const baselines = scored[0].metrics.baselines ?? {};
      return {
        key: `${window.start}..${window.end}`,
        label: `${monthLabel(window.start)} to ${monthLabel(window.end)}`,
        scored,
        wins: scored.filter(run => gap(run) > 0).length,
        equalWeight: baselines.equal_weight,
        comparisons: [
          ["SPY", window.spy, "spy"],
          ["Median model", median(scored.map(run => run.metrics.total_return)), "model"],
          ["Equal weight", baselines.equal_weight, "baseline"],
          ["60/40", baselines.sixty_forty, "baseline"],
          ["12-minus-1 momentum", baselines.momentum_12_1, "baseline"],
        ],
      };
    });
    const values = summaries.flatMap(summary => summary.comparisons.map(([, value]) => value)).filter(Number.isFinite);
    const min = Math.floor(Math.min(0, ...values) * 100) / 100;
    const max = Math.ceil(Math.max(.01, ...values) * 100) / 100;
    const position = value => (value - min) / (max - min) * 100;
    setText("report-baseline-scale", `Common scale across windows from ${percent(min)} to ${percent(max)}.`);
    const consistent = allPeriodRows(windows).filter(row => row.complete && row.beatIn === windows.length).length;
    if (windows.length > 1) {
      setText("report-overview", consistent
        ? `${plural(consistent, "model")} beat SPY in every published window.`
        : "No model beat SPY in every published window.");
    }

    function bar(start, end, kind) {
      const track = node("span", undefined, "report-bar-track");
      track.setAttribute("aria-hidden", "true");
      const fill = node("span", undefined, `report-bar-fill report-bar-${kind}`);
      fill.style.left = `${Math.min(start, end)}%`;
      fill.style.width = `${Math.abs(end - start)}%`;
      track.append(fill);
      if (start > 0) {
        const zero = node("span", undefined, "report-bar-zero");
        zero.style.left = `${start}%`;
        track.append(zero);
      }
      return track;
    }

    const chart = $("report-window-chart");
    const buttons = summaries.map((summary, index) => {
      const item = node("li");
      const button = node("button", undefined, "report-window");
      button.type = "button";
      button.setAttribute("aria-controls", "report-baseline-chart report-category-table");
      button.append(node("span", summary.label, "report-bar-label"),
        bar(0, summary.wins / summary.scored.length * 100, "model"),
        node("span", `${summary.wins}/${summary.scored.length}`, "report-bar-value"));
      button.onclick = () => {
        selectWindow(index);
        const params = new URLSearchParams(location.search);
        params.set("window", summary.key);
        window.history.replaceState(window.history.state, "", `?${params}${location.hash}`);
      };
      item.append(button);
      chart.append(item);
      return button;
    });

    function selectWindow(index) {
      const {label, scored, equalWeight, comparisons} = summaries[index];
      buttons.forEach((button, i) => button.setAttribute("aria-pressed", String(i === index)));
      const comparableReturns = scored.map(run => run.metrics.total_return).filter(Number.isFinite);
      const aheadOfEqualWeight = comparableReturns.filter(value => value > equalWeight).length;
      setText("report-selection", Number.isFinite(equalWeight) && comparableReturns.length
        ? `${label}. ${aheadOfEqualWeight} of ${plural(comparableReturns.length, "model")} beat equal weight.`
        : `Showing ${label}.`);
      $("report-baseline-chart").replaceChildren(...comparisons.filter(([, value]) => Number.isFinite(value))
        .map(([name, value, kind]) => {
          const item = node("li", undefined, "report-baseline");
          item.append(node("span", name, "report-bar-label"), bar(position(0), position(value), kind),
            node("span", percent(value), "report-bar-value"));
          return item;
        }));

      const table = node("table");
      const caption = node("caption", `Category summary for ${label}`, "visually-hidden");
      const thead = node("thead"), header = node("tr"), tbody = node("tbody");
      for (const [i, title] of ["Category", "Models", "Beat SPY", "Median gap (pp)"].entries()) {
        const cell = node("th", title, i ? "num" : undefined);
        cell.scope = "col";
        header.append(cell);
      }
      thead.append(header);
      for (const [category, name] of Object.entries(categories)) {
        const group = scored.filter(run => modelCategory(run) === category);
        if (!group.length) continue;
        const middle = median(group.map(gap));
        const points = Math.abs(middle * 100).toFixed(2);
        const row = node("tr"), title = node("th", name);
        title.scope = "row";
        row.append(title, node("td", group.length, "num"),
          node("td", group.filter(run => gap(run) > 0).length, "num"),
          node("td", `${middle < 0 ? "−" : middle > 0 ? "+" : ""}${points}`,
            `num ${middle < 0 ? "negative" : middle > 0 ? "positive" : ""}`));
        tbody.append(row);
      }
      table.append(caption, thead, tbody);
      $("report-category-table").replaceChildren(table);
    }

    const sharedWindow = new URLSearchParams(location.search).get("window");
    const initialIndex = summaries.findIndex(summary => summary.key === sharedWindow);
    selectWindow(initialIndex === -1 ? summaries.length - 1 : initialIndex);
    setHidden("report-results", false);
  }
  // One window at a time, or every window averaged: the toggle picks the lens.
  const ALL_COLUMNS = [
    ["model", "Model", "text"],
    ["windows", "Windows", "count"],
    ["beat_in", "Beat SPY", "text"],
    ["avg_return", "Avg return", "signed"],
    ["avg_gap", "Avg vs SPY", "signed"],
    ["best_gap", "Best window", "signed"],
    ["worst_gap", "Worst window", "signed"],
  ];
  let period = "all";
  // Header tooltips carry what the caption used to: one line per column, read
  // on hover, so the page itself stays short.
  const HINTS = {
    total_return: "Total portfolio return over the window",
    excess_return_vs_spy: "Return minus SPY’s return, in percentage points",
    outperformance_hit_rate: "Share of weeks this book out-returned SPY",
    max_drawdown: "Largest peak-to-trough fall",
    hold: "Share of names left untraded each week",
    windows: "How many of the scored windows the model ran",
    beat_in: "Windows where the model beat SPY",
    avg_return: "Mean total return across the model's windows",
    avg_gap: "Mean gap to SPY across the model's windows",
    best_gap: "Best single window vs SPY",
    worst_gap: "Worst single window vs SPY",
  };
  // The board opens on the newest protocol's window (v2 when present), not the
  // all-windows average, so the current benchmark version is the landing view.
  let periodOpened = false;

  function renderTabs(windows) {
    const bar = $("period-tabs");
    if (!bar) return;
    bar.replaceChildren();
    // One scored window needs no toggle; the board stays exactly as it was.
    bar.hidden = windows.length < 2;
    if (bar.hidden) return;
    const chip = (id, text, title) => {
      const button = node("button", text, `tab${period === id ? " active" : ""}`);
      button.type = "button";
      if (title) button.title = title;
      button.setAttribute("aria-pressed", period === id ? "true" : "false");
      button.onclick = () => {
        period = id;
        sortKey = id === "all" ? "beat_in" : "excess_return_vs_spy";
        descending = true;
        renderBoard();
      };
      return button;
    };
    bar.append(chip("all", `All ${windows.length} windows`, "Every scored window, averaged per model"));
    for (const w of windows) {
      bar.append(chip(w.key, w.label, `${w.start} to ${w.end}`));
    }
  }

  function renderBoard() {
    if (!$("leaderboard-table")) return;
    const windows = scoredWindows(runs.filter(isCurrent));
    const filter = $("model-category");
    if (filter) filter.onchange = renderBoard;
    const search = $("model-search");
    if (search) search.oninput = renderBoard;
    if (!periodOpened) {
      periodOpened = true;
      const newest = windows.filter(w => w.proto >= 6).pop();
      if (newest) period = newest.key;
    }
    if (period !== "all" && !windows.some(w => w.key === period)) period = "all";
    renderTabs(windows);
    const category = filter?.value || "all";
    const allMode = period === "all" && windows.length >= 2;
    const selectedWindows = period === "all" ? windows : windows.filter(w => w.key === period);
    const scopedWindows = selectedWindows.map(w => ({
      ...w, runs: w.runs.filter(run => category === "all" || modelCategory(run) === category),
    }));
    updateResultsSummary(scopedWindows);
    const query = searchWords(search?.value);
    const matches = model => {
      if (!query.length) return true;
      const name = searchWords(`${displayModel(model)} ${model}`).join(" ");
      return query.every(word => name.includes(word));
    };
    const displayed = allMode
      ? renderAllBoard(scopedWindows, matches)
      : renderWindowBoard(scopedWindows[0]?.runs ?? [], matches);
    const hasCategoryResults = scopedWindows.some(w => w.runs.length);
    setText("search-status", query.length && hasCategoryResults
      ? (displayed.length ? plural(displayed.length, "matching model") : "No models match your search.")
      : "");
    setText("status", hasCategoryResults ? "" : category !== "all" && windows.length
      ? "No results for this category in the selected view."
      : runs.length ? "No completed results yet." : "No results yet. Record a run and re-render this page.");
    setHidden("leaderboard-table", displayed.length === 0);
    updateDemoBanner(displayed);
  }

  const searchWords = value => String(value ?? "").toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];

  function updateResultsSummary(windows) {
    if (!windows.length) return;
    const models = new Set(), winners = new Set();
    for (const window of windows) for (const run of window.runs) {
      const gap = run.metrics?.excess_return_vs_spy;
      if (run.synthetic || missing(gap) || !Number.isFinite(Number(gap))) continue;
      const name = modelName(run.model);
      models.add(name);
      if (Number(gap) > 0) winners.add(name);
    }
    const allMode = windows.length > 1;
    const first = windows[0], last = windows.at(-1);
    const reference = first.spy;
    const countSummary = models.size
      ? `${winners.size} of ${plural(models.size, "model")} beat SPY`
        + (allMode ? ` in at least one of the ${windows.length} windows.` : ".")
      : "";
    const benchmark = !allMode && Number.isFinite(reference) ? `SPY returned ${percent(reference)}.` : "";
    setText("results-summary", [countSummary, benchmark].filter(Boolean).join(" "));
    const run = first.runs[0];
    const decisions = run?.decisions ?? run?.metrics?.decisions;
    setText("hero-scope", `${first.start} → ${last.end}` + (!allMode && !missing(decisions)
      ? ` · ${plural(decisions, "decision")}` : ""));
  }

  function updateDemoBanner(source) {
    const syntheticCount = source.filter(run => run.synthetic).length;
    setHidden("demo-banner", syntheticCount === 0);
    if (syntheticCount) {
      setText("demo-banner", syntheticCount === source.length
        ? `Synthetic demo data: all ${syntheticCount} rows are fabricated by \`beatspy demo\`. No real model was run.`
        : `Synthetic demo data: ${syntheticCount} of ${source.length} rows are fabricated by \`beatspy demo\` and are not real results.`);
    }
  }

  function renderWindowBoard(windowRuns, matches) {
    // One sentence of caption; the column meanings live in the header tooltips
    // (HINTS), so the page reads at a glance. The no-skill line comes from the
    // catalog, not a hardcoded number.
    const baseRate = Number(windowRuns[0]?.metrics?.calibration?.signals?.noul?.base_rate);
    setText("board-window", "");
    const ranked = [...windowRuns].sort((a, b) => {
      const column = COLUMNS.find(([key]) => key === sortKey) || COLUMNS[2];
      const av = rowValue(a, column[0], column[3]), bv = rowValue(b, column[0], column[3]);
      if (column[2] === "text") return descending ? String(bv).localeCompare(String(av)) : String(av).localeCompare(String(bv));
      if (!Number.isFinite(Number(av)) || !Number.isFinite(Number(bv))) return Number.isFinite(Number(av)) ? -1 : 1;
      return descending ? Number(bv) - Number(av) : Number(av) - Number(bv);
    });
    const ranks = new Map(ranked.map((run, i) => [run.run_id, i + 1]));
    const visible = ranked.filter(run => matches(run.model));
    const element = node("table"), head = node("thead"), header = node("tr"), body = node("tbody");
    header.append(node("th", "Rank", "col-rank"));
    const hints = {...HINTS,
      accuracy: "How often calls proved right" + (Number.isFinite(baseRate)
        ? `; always calling one way scores ${percent(baseRate)} here, a coin flip is 50%` : "; a coin flip is 50%")};
    for (const [key, title, kind] of COLUMNS) {
      const th = node("th", undefined, `${kind === "text" ? "" : "num"} col-${key}`);
      const button = node("button", title + (sortKey === key ? (descending ? " ↓" : " ↑") : ""));
      button.title = hints[key] || "";
      th.setAttribute("aria-sort", sortKey === key ? (descending ? "descending" : "ascending") : "none");
      button.onclick = () => {descending = sortKey === key ? !descending : true; sortKey = key; renderBoard();};
      th.append(button); header.append(th);
    }
    head.append(header); element.append(head, body);
    for (const run of visible) {
      const row = node("tr");
      if (run.run_id === selected) row.className = "selected";
      if (run.synthetic) row.classList.add("synthetic");
      // Rank 1 earns the eye; a fabricated row keeps its warning tint instead.
      if (ranks.get(run.run_id) === 1 && !run.synthetic) row.classList.add("leader");
      row.append(node("td", ranks.get(run.run_id), "col-rank"));
      // A real link: the run detail is its own page, so a row can be opened in a
      // new tab and shared as a URL.
      const td = node("td", undefined, "col-model"), link = node("a", displayModel(run.model), "run-button");
      link.href = `run.html?run=${encodeURIComponent(run.run_id)}`;
      const logo = modelLogo(run.model);
      if (logo) {
        const img = node("img");
        img.className = "model-logo"; img.alt = ""; img.loading = "lazy"; img.decoding = "async";
        img.src = `assets/logos/${logo}`;
        td.append(img);
      }
      td.append(link);
      if (run.synthetic) {
        // Mark the row itself: the details panel is one click away, and a
        // fabricated number must not read as a real result before that click.
        const chip = node("span", "synthetic", "synthetic-chip");
        chip.title = "Synthetic demo run fabricated by `beatspy demo`.";
        td.append(chip);
      }
      row.append(td);
      // The whole row opens the run; clicks that carry their own intent (the
      // link, new-tab modifiers) fall through to the browser.
      row.onclick = event => {
        if (event.target.closest("a") || event.ctrlKey || event.metaKey || event.shiftKey || event.button !== 0) return;
        location.href = link.href;
      };
      for (const [key, , kind, source_] of COLUMNS.slice(1)) {
        const raw = rowValue(run, key, source_), value = Number(raw);
        // Tone only the columns whose neutral point is zero; Sharpe, drawdown
        // and the calibration scores are all read against their own scales.
        const tone = !missing(raw) && Number.isFinite(value) && (key === "total_return" || key === "excess_return_vs_spy")
          ? (value >= 0 ? "positive" : "negative") : "";
        const cell = node("td", undefined, `num col-${key}`);
        if (tone) cell.append(node("span", `${tone === "positive" ? "▲" : "▼"} ${FORMAT[kind](raw)}`, `chip ${tone}`));
        else cell.textContent = FORMAT[kind](raw);
        row.append(cell);
      }
      body.append(row);
    }
    $("leaderboard-table").replaceChildren(element);
    return visible;
  }

  function renderAllBoard(windows, matches) {
    const rows = allPeriodRows(windows);
    const total = windows.length;
    setText("board-window", "Models with results in every window rank first by default.");
    const value = (row, key) => key === "model" ? displayModel(row.model)
      : key === "windows" ? `${row.windows}`
      : key === "avg_gap" ? row.avgGap
      : key === "avg_return" ? row.avgReturn
      : key === "best_gap" ? row.best?.gap
      : key === "worst_gap" ? row.worst?.gap
      : key === "beat_in" ? row.beatIn
      : null;
    rows.sort((a, b) => {
      const column = ALL_COLUMNS.find(([key]) => key === sortKey) || ALL_COLUMNS[2];
      const av = value(a, column[0]), bv = value(b, column[0]);
      // The headline ranking: complete records first (a model that ran every
      // window outranks one that has not, whatever a lucky average says), then
      // windows beaten, then average return - wins lead, return settles ties.
      // Clicking Avg return or Avg vs SPY simply makes that column the lead.
      if (["beat_in", "avg_return", "avg_gap"].includes(column[0])) {
        const score = key => key === "beat_in" ? b.beatIn - a.beatIn
          : key === "avg_return" ? (b.avgReturn ?? -Infinity) - (a.avgReturn ?? -Infinity)
          : (b.avgGap ?? -Infinity) - (a.avgGap ?? -Infinity);
        const order = column[0] === "avg_return" ? ["avg_return", "beat_in", "avg_gap"]
          : column[0] === "avg_gap" ? ["avg_gap", "beat_in", "avg_return"]
          : ["beat_in", "avg_return", "avg_gap"];
        const ranked = order.reduce((acc, key) => acc || score(key), 0);
        const completeness = (b.complete ? 1 : 0) - (a.complete ? 1 : 0);
        return descending ? (completeness || ranked) : -(completeness || ranked);
      }
      if (column[2] === "text") return descending ? String(bv).localeCompare(String(av)) : String(av).localeCompare(String(bv));
      if (!Number.isFinite(Number(av)) || !Number.isFinite(Number(bv))) return Number.isFinite(Number(av)) ? -1 : 1;
      return descending ? Number(bv) - Number(av) : Number(av) - Number(bv);
    });
    const element = node("table"), head = node("thead"), header = node("tr"), body = node("tbody");
    header.append(node("th", "Rank", "col-rank"));
    for (const [key, title, kind] of ALL_COLUMNS) {
      const th = node("th", undefined, `${kind === "text" ? "" : "num"} col-${key}`);
      const button = node("button", title + (sortKey === key ? (descending ? " ↓" : " ↑") : ""));
      button.title = HINTS[key] || "";
      th.setAttribute("aria-sort", sortKey === key ? (descending ? "descending" : "ascending") : "none");
      button.onclick = () => {descending = sortKey === key ? !descending : true; sortKey = key; renderBoard();};
      th.append(button); header.append(th);
    }
    head.append(header); element.append(head, body);
    const visible = [];
    rows.forEach((row, index) => {
      if (!matches(row.model)) return;
      visible.push(row.latest);
      const tr = node("tr");
      if (index === 0) tr.classList.add("leader");
      tr.append(node("td", index + 1, "col-rank"));
      const td = node("td", undefined, "col-model"), link = node("a", displayModel(row.model), "run-button");
      if (row.latest) {
        link.href = `run.html?run=${encodeURIComponent(row.latest.run_id)}`;
        if (row.latest.run_id === selected) tr.classList.add("selected");
      }
      const logo = modelLogo(row.model);
      if (logo) {
        const img = node("img");
        img.className = "model-logo"; img.alt = ""; img.loading = "lazy"; img.decoding = "async";
        img.src = `assets/logos/${logo}`;
        td.append(img);
      }
      td.append(link);
      tr.append(td);
      tr.onclick = event => {
        if (!row.latest || event.target.closest("a") || event.ctrlKey || event.metaKey || event.shiftKey || event.button !== 0) return;
        location.href = `run.html?run=${encodeURIComponent(row.latest.run_id)}`;
      };
      // The best/worst cells carry their column key so the phone layout can
      // hide the pair with its headers instead of leaving headerless columns.
      const gapCell = (entry, key) => {
        const cell = node("td", undefined, `num col-${key}`);
        if (entry && Number.isFinite(entry.gap)) {
          const tone = entry.gap >= 0 ? "positive" : "negative";
          cell.append(node("span", `${tone === "positive" ? "▲" : "▼"} ${percent(entry.gap)}`, `chip ${tone}`));
          const note = node("span", entry.window, "cell-note");
          cell.append(note);
        } else cell.textContent = "n/a";
        return cell;
      };
      tr.append(node("td", `${row.windows}/${total}`, "num col-windows"));
      tr.append(node("td", `${row.beatIn} of ${row.windows}`, "num col-beat_in"));
      const pctCell = (v, cls) => {
        const cell = node("td", undefined, `num col-${cls}`);
        if (Number.isFinite(v)) {
          const tone = v >= 0 ? "positive" : "negative";
          cell.append(node("span", `${tone === "positive" ? "▲" : "▼"} ${percent(v)}`, `chip ${tone}`));
        } else cell.textContent = "n/a";
        return cell;
      };
      tr.append(pctCell(row.avgReturn, "avg_return"), pctCell(row.avgGap, "avg_gap"), gapCell(row.best, "best_gap"), gapCell(row.worst, "worst_gap"));
      body.append(tr);
    });
    $("leaderboard-table").replaceChildren(element);
    return visible;
  }
  // Equity columns arrive as snake_case keys; the legend should read like prose.
  const SERIES_LABELS = {
    portfolio: "model portfolio",
    spy_buy_hold: "SPY buy-and-hold",
    equal_weight: "whole pool, equal weight",
    selection_equal_weight: "its investable set, equal weight",
    sixty_forty: "60/40, benchmark and Treasuries",
    momentum_12_1: "12-1 momentum, top 3",
  };
  const seriesLabel = key => SERIES_LABELS[key] || String(key).replaceAll("_", " ");
  function chart(target, rows, series, format) {
    const container = $(target); container.replaceChildren();
    if (!rows.length) {container.append(node("p", "No chart data", "caption")); return;}
    const hidden = new Set(), labels = node("div", undefined, "legend"), plot = node("div", undefined, "chart");
    const tooltip = node("div", "Move over the chart for exact values.", "tooltip");
    tooltip.setAttribute("aria-live", "polite");
    const W = 640, H = 250, L = 62, R = 16, T = 16, B = 35;
    function draw() {
      plot.replaceChildren();
      const svg = svgNode("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${target} chart`});
      const keys = series.filter(key => !hidden.has(key));
      const values = rows.flatMap(row => keys.map(key => Number(row[key]))).filter(Number.isFinite);
      let min = Math.min(...values), max = Math.max(...values);
      if (!values.length) {min = 0; max = 1;}
      if (max === min) {max += 1; min -= 1;}
      const x = i => L + i * (W - L - R) / Math.max(rows.length - 1, 1);
      const y = value => T + (max - value) * (H - T - B) / (max - min);
      for (let i = 0; i <= 4; i++) {
        const value = min + (max - min) * i / 4;
        svg.append(svgNode("line", {x1: L, x2: W - R, y1: y(value), y2: y(value), stroke: "#30363d"}));
        const label = svgNode("text", {x: L - 8, y: y(value) + 4, fill: "#a7b0ba", "font-size": 10, "text-anchor": "end"});
        label.textContent = format(value); svg.append(label);
      }
      for (const key of keys) {
        const points = rows.map((row, i) => Number(row[key])).filter(Number.isFinite);
        if (points.length !== rows.length) continue;
        svg.append(svgNode("polyline", {
          points: rows.map((row, i) => `${x(i)},${y(Number(row[key]))}`).join(" "),
          fill: "none", stroke: COLORS[series.indexOf(key) % COLORS.length], "stroke-width": 2,
        }));
      }
      for (const [i, anchor] of [[0, "start"], [rows.length - 1, "end"]]) {
        const label = svgNode("text", {x: x(i), y: H - 9, fill: "#a7b0ba", "font-size": 10, "text-anchor": anchor});
        label.textContent = rows[i].date; svg.append(label);
      }
      const cross = svgNode("line", {x1: L, x2: L, y1: T, y2: H - B, stroke: "#a7b0ba", "stroke-dasharray": "4 4", visibility: "hidden"});
      svg.append(cross);
      svg.onpointermove = event => {
        const box = svg.getBoundingClientRect();
        const px = (event.clientX - box.left) * W / box.width;
        const i = Math.max(0, Math.min(rows.length - 1, Math.round((px - L) * (rows.length - 1) / (W - L - R))));
        cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
        tooltip.textContent = `${rows[i].date} · ${keys.map(key => `${seriesLabel(key)}: ${format(Number(rows[i][key]))}`).join(" · ")}`;
      };
      svg.onpointerleave = () => cross.setAttribute("visibility", "hidden");
      plot.append(svg);
    }
    series.forEach((key, i) => {
      const button = node("button", `● ${seriesLabel(key)}`); button.style.color = COLORS[i % COLORS.length];
      button.setAttribute("aria-pressed", "true");
      button.onclick = () => {if (hidden.has(key)) hidden.delete(key); else hidden.add(key); button.setAttribute("aria-pressed", !hidden.has(key)); draw();};
      labels.append(button);
    });
    container.append(plot, labels, tooltip); draw();
  }
  // Predicted (x) against observed (y) per reliability bin, with the y = x diagonal.
  function reliabilityChart(target, calibration) {
    const container = $(target); container.replaceChildren();
    const series = reliabilitySeries(calibration);
    if (!series.length || !series.some(item => item.points.length)) {
      container.append(node("p", "No scored observations in this run.", "caption")); return;
    }
    const hidden = new Set(), labels = node("div", undefined, "legend"), plot = node("div", undefined, "chart");
    const tooltip = node("div", "Move over a point for its bin and count.", "tooltip");
    tooltip.setAttribute("aria-live", "polite");
    const W = 560, H = 380, L = 56, R = 22, T = 22, B = 46;
    const px = value => L + value * (W - L - R);
    const py = value => T + (1 - value) * (H - T - B);
    function draw() {
      plot.replaceChildren();
      const svg = svgNode("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "reliability curve"});
      const diagonal = svgNode("line", {x1: px(0), y1: py(0), x2: px(1), y2: py(1), stroke: "#a7b0ba",
        "stroke-width": 1, "stroke-dasharray": "5 5"});
      svg.append(diagonal);
      for (let i = 0; i <= 5; i++) {
        const value = i / 5;
        svg.append(svgNode("line", {x1: px(value), y1: T, x2: px(value), y2: py(0), stroke: "#30363d"}));
        svg.append(svgNode("line", {x1: L, y1: py(value), x2: px(1), y2: py(value), stroke: "#30363d"}));
        const left = svgNode("text", {x: L - 8, y: py(value) + 4, fill: "#a7b0ba", "font-size": 10, "text-anchor": "end"});
        left.textContent = percent(value); svg.append(left);
        const bottom = svgNode("text", {x: px(value), y: py(0) + 18, fill: "#a7b0ba", "font-size": 10, "text-anchor": "middle"});
        bottom.textContent = value.toFixed(1); svg.append(bottom);
      }
      const axisX = svgNode("text", {x: px(0.5), y: H - 6, fill: "#a7b0ba", "font-size": 11, "text-anchor": "middle"});
      axisX.textContent = "Predicted probability"; svg.append(axisX);
      for (const item of series) {
        if (hidden.has(item.name) || !item.points.length) continue;
        const color = COLORS[SIGNALS.indexOf(item.name) % COLORS.length];
        svg.append(svgNode("polyline", {
          points: item.points.map(point => `${px(point.predicted)},${py(point.observed)}`).join(" "),
          fill: "none", stroke: color, "stroke-width": 2,
        }));
        for (const point of item.points) {
          const circle = svgNode("circle", {cx: px(point.predicted), cy: py(point.observed),
            r: Math.min(9, 3 + point.count), fill: color, "fill-opacity": 0.85, stroke: "#101214"});
          circle.onpointerenter = () => {
            tooltip.textContent = `${item.name} · predicted ${percent(point.predicted)} → observed ${percent(point.observed)} (n=${point.count})`;
          };
          svg.append(circle);
        }
      }
      plot.append(svg);
    }
    for (const item of series) {
      const color = COLORS[SIGNALS.indexOf(item.name) % COLORS.length];
      const button = node("button", `● ${item.name} · n=${count(item.n)} · Brier ${score(item.brier)}`);
      button.style.color = color;
      button.setAttribute("aria-pressed", "true");
      button.onclick = () => {if (hidden.has(item.name)) hidden.delete(item.name); else hidden.add(item.name); button.setAttribute("aria-pressed", !hidden.has(item.name)); draw();};
      labels.append(button);
    }
    const diagonalLabel = node("span", "┈ y = x (perfect calibration)");
    labels.append(diagonalLabel);
    container.append(plot, labels, tooltip); draw();
    const headers = ["Predicted bin", ...series.map(item => `${item.name} (n)`), ...series.map(item => `${item.name} observed`)];
    const bins = series[0].bins || [];
    const records = bins.map((bin, i) => [
      `${Number(bin.lower).toFixed(1)}–${Number(bin.upper).toFixed(1)}`,
      ...series.map(item => count(item.bins?.[i]?.count)),
      ...series.map(item => item.bins?.[i]?.observed_rate == null ? "n/a" : percent(item.bins[i].observed_rate)),
    ]);
    table("reliability-bins", headers, records);
  }
  async function selectRun(id) {
    if (!$("run-details")) return;
    const run = runs.find(row => row.run_id === id);
    if (!run) {setHidden("run-details", true); setText("status", "That run is not in this catalog."); return;}
    selected = id; const generation = ++loading;
    setHidden("run-details", true);
    try {
      if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$/.test(run.run_id) || run.dir !== `runs/${run.run_id}`) throw new Error("Invalid artifact path");
      const base = `data/${run.dir}/`;
      const texts = await Promise.all(["run.json", "metrics.json", "equity_curve.csv", "decisions.jsonl"].map(name => fetchText(base + name)));
      if (generation !== loading) return;
      const meta = JSON.parse(texts[0]), metrics = JSON.parse(texts[1]);
      const equity = parseCSV(texts[2]), decisions = texts[3].split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line));
      const calibration = metrics.calibration || run.metrics.calibration || {};
      const allocationOnly = run.question_form === "allocation";
      setHidden("run-details", false);
      setText("status", "");
      $("run-title").textContent = `${displayModel(run.model)} · ${allocationOnly ? "allocation" : `${run.question_form} questions`}`;
      // The pill keeps its border when empty, so hide it unless a trust state
      // earns a label.
      $("run-trust").textContent = TRUST[run.trust] || "";
      $("run-trust").hidden = !TRUST[run.trust];
      $("run-meta").textContent = `${run.scenario} · ${run.start} → ${run.end} · Data cutoff ${run.data_cutoff} · Snapshot ${run.snapshot_id || "unknown"}`;
      $("metrics").replaceChildren();
      const noulStats = calibration?.signals?.noul || {};
      const excessCi = run.metrics.excess_return_ci;
      const ciNote = excessCi && !missing(excessCi.lower) && !missing(excessCi.upper)
        ? `95% interval ${percent(excessCi.lower)} to ${percent(excessCi.upper)}`
        : "interval unavailable";
      // [heading, formatted value, signed?, sub-note]. This panel is the review:
      // the board's headline numbers plus everything behind them, each with the
      // line that says how to read it.
      const pool = run.metrics.baselines?.equal_weight;
      const vsPool = Number.isFinite(pool) ? run.metrics.total_return - pool : null;
      // Buy/sell/hold straight from the recorded target books: a name is a hold
      // when its target stayed inside the 2% trading band, so it was never
      // traded. An invalid decision holds the previous book, as the engine did.
      let buys = 0, sells = 0, holds = 0, held = {};
      for (const decision of decisions) {
        const target = decision?.validated && !decision.validated.invalid
          ? Object.fromEntries(Object.entries(decision.validated.weights || {}).map(([ticker, weight]) => [ticker.toUpperCase(), Number(weight)]))
          : held;
        for (const ticker of new Set([...Object.keys(held), ...Object.keys(target)])) {
          const before = held[ticker] || 0, after = target[ticker] || 0;
          if (after > before + 0.02) buys += 1;
          else if (after < before - 0.02) sells += 1;
          else holds += 1;
        }
        held = target;
      }
      const perDecision = value => (value / Math.max(decisions.length, 1)).toFixed(1);
      const fillDetails = `${plural(run.metrics.trades, "fill")} filled`;
      const tiles = [
        ["Portfolio return", percent(run.metrics.total_return), run.metrics.total_return,
          `a $10,000 start ends at ${usd(START_USD * (1 + (Number.isFinite(Number(run.metrics.total_return)) ? run.metrics.total_return : 0)))}`
          + `, SPY's at ${usd(START_USD * (1 + (Number.isFinite(Number(run.metrics.spy_total_return)) ? run.metrics.spy_total_return : 0)))}`],
        ["Excess vs SPY", percent(run.metrics.excess_return_vs_spy), run.metrics.excess_return_vs_spy, ciNote],
        ["Excess vs pool", percent(vsPool), vsPool, "above 0 beats holding the same universe equally"],
        ["Weeks beating SPY", percent(run.metrics.outperformance_hit_rate), null,
          "share of weeks the book out-returned SPY"],
        ["Sharpe", ratio(run.metrics.sharpe), run.metrics.sharpe, "zero risk-free rate"],
        ["Annual volatility", percent(run.metrics.annual_volatility), null, "realized, annualized"],
        ["Maximum drawdown", percent(run.metrics.max_drawdown), run.metrics.max_drawdown, "peak to trough"],
        ["Turnover per decision", percent(run.metrics.avg_turnover), null, "share of the book traded, one-sided"],
        ["Decisions", count(run.metrics.decisions), null,
          allocationOnly ? fillDetails : `${fillDetails} · ${perDecision(buys)} buys, ${perDecision(sells)} sells, ${perDecision(holds)} holds per decision`],
      ];
      if (!allocationOnly) tiles.push(
        ["Answer coverage", percent(run.metrics.answer_coverage), null,
          `${plural(run.metrics.invalid_outputs, "invalid answer")}: missing answers stay unscored and carry no conviction`],
        ["Brier (noul)", score(noulStats.brier), null, `sample rate ${score(noulStats.base_rate)}: what a constant would score`],
        ["Brier skill (noul)", score(noulStats.brier_skill_score), noulStats.brier_skill_score,
          `permutation p = ${score(noulStats.brier_skill_p_value)}: above 0 beats a constant`],
        ["Log loss (noul)", score(noulStats.log_loss), null, "lower is better"],
        ["AUC (noul)", score(noulStats.auc), null, `permutation p = ${score(noulStats.auc_p_value)}: 0.5 is chance`],
        ["ECE (noul)", score(noulStats.expected_calibration_error), null,
          `expected calibration error; worst bin ${score(noulStats.max_calibration_error)}`],
        ["Accuracy (noul)", percent(noulStats.accuracy), null, "share of labels the called side got right"],
      );
      tiles.forEach(([title, formatted, value, note], index) => {
        // The first two tiles are the headline: what it returned and whether
        // that beat SPY. They render larger so the page has a focal point.
        const tile = node("div", undefined, index < 2 ? "metric metric-hero" : "metric");
        tile.append(node("p", title), node("strong", formatted,
          Number.isFinite(Number(value)) ? (Number(value) >= 0 ? "positive" : "negative") : ""));
        if (note) tile.append(node("p", note, "metric-note"));
        $("metrics").append(tile);
      });
      const series = Object.keys(equity[0] || {}).filter(key => key !== "date");
      // Equity is recorded from a $100,000 start; the page reads a $10,000 book.
      chart("performance", equity, series, value => usd(Number(value) * (START_USD / 100_000)));
      const peaks = {}; const drawdown = equity.map(row => {
        const result = {date: row.date};
        for (const key of series) {peaks[key] = Math.max(peaks[key] || -Infinity, Number(row[key])); result[key] = Number(row[key]) / peaks[key] - 1;}
        return result;
      });
      chart("drawdown", drawdown, series, percent);
      setHidden("calibration", allocationOnly);
      if (!allocationOnly) reliabilityChart("reliability", calibration);
      const allocationRowsData = allocationRows(decisions);
      const tickers = [...new Set(allocationRowsData.flatMap(row => Object.keys(row).filter(key => key !== "date")))];
      allocationRowsData.forEach(row => tickers.forEach(key => {row[key] = row[key] || 0;}));
      chart("allocations", allocationRowsData, tickers, percent);
      const rates = ratesFor(run.model, meta), cost = run.metrics.estimated_cost_usd ?? inputCost(run.metrics, rates);
      const policy = run.policy || {};
      const policyText = meta.decision?.portfolio_method || (policy.selection === "core_satellite"
        // Recorded under protocol 3; kept so those runs still render their own rule.
        ? `core ${percent(policy.core_weight)} in the benchmark, ${percent(1 - (policy.core_weight ?? 0))} active sleeve`
        : (policy.selection === "graded_cs" || policy.selection === "graded")
          ? `whole book across the investable set, neutral P = ${policy.min_probability ?? "?"}`
          : `route ${count(policy.top_n)} names at P ≥ ${policy.min_probability ?? "?"}`);
      const selectionText = meta.scenario?.selection_top
        ? ` · trades the top ${count(meta.scenario.selection_top)} trailing performers, ranked point-in-time`
        : "";
      $("run-cost").textContent = allocationOnly
        ? `Direct allocation · ${plural(policy.lookback_sessions, "session")} of trailing history · ${percent(meta.scenario?.max_position_weight)} position cap`
        : `Input-only estimate ${dollars(cost)} · ${count(run.metrics.input_tokens)} input tokens · `
        + `${count(run.metrics.output_tokens)} output tokens · ${plural(run.metrics.requests, "request")} · `
        + `mean latency ${FORMAT.millis(run.metrics.mean_latency_s)} · ${policyText}${selectionText}`;
      $("price-source").replaceChildren();
      if (rates?.source) {
        const link = node("a", "Published pricing ↗"); link.href = rates.source;
        link.rel = "noreferrer noopener"; link.target = "_blank";
        $("price-source").append(link);
      }
      $("settings").textContent = JSON.stringify(meta, null, 2);
      $("coverage").textContent = allocationOnly
        ? "Allocation weights are replayed from the frozen snapshot."
        : `Scored ${plural(calibration.observations, "observation")} over ${plural(calibration.horizons, "horizon")}, `
        + `labeled against realized benchmark-relative outcomes. ${calibration.note || ""}`.trim();
      $("downloads").replaceChildren();
      for (const name of ["run.json", "metrics.json", "equity_curve.csv", "trades.csv", "decisions.jsonl", "events.jsonl"]) {
        const a = node("a", name); a.href = base + name; a.setAttribute("download", name); $("downloads").append(a);
      }
      if (typeof run.snapshot_dir === "string" && /^[a-z0-9/]+$/.test(run.snapshot_dir)) {
        for (const name of ["prices.csv", "MANIFEST.json"]) {
          const a = node("a", `Snapshot ${name}`); a.href = `data/${run.snapshot_dir}/${name}`; $("downloads").append(a);
        }
      }
    } catch (error) {if (generation === loading) {setHidden("run-details", true); setText("status", `Unable to load this run: ${error.message}`);}}
  }
  // The run page carries one run, named in the query string (`run.html?run=<id>`).
  // The `#run=` form still works, and with neither the page opens the top of the
  // board so a bare run.html link is useful on its own.
  function wantedRunId() {
    try {
      const id = new URLSearchParams(location.search || "").get("run");
      if (id) return id;
    } catch (error) { /* no URLSearchParams: fall through to the hash */ }
    if ((location.hash || "").startsWith("#run=")) return decodeURIComponent(location.hash.slice(5));
    return null;
  }
  function renderRun() {
    if (!$("run-details")) return;
    const id = wantedRunId() || (leaderboardRuns(runs.filter(isCurrent))[0] || {}).run_id;
    if (!id) {setText("status", "No recorded runs yet."); return;}
    selectRun(id);
  }
  function hashSelection() {
    const id = wantedRunId();
    if (id) selectRun(id);
  }
  async function init() {
    try {
      const catalog = JSON.parse(await fetchText("data/index.json"));
      const problems = catalogProblems(catalog);
      if (problems.length) throw new Error(`Invalid result catalog (${problems[0]})`);
      runs = catalog.runs;
      window.addEventListener("hashchange", hashSelection);
      renderBoard();
      renderPricing();
      renderReport();
      renderRun();
    } catch (error) {
      setText("status", `Results unavailable: ${error.message}. Try reloading the page.`);
    }
  }
  init();
}
