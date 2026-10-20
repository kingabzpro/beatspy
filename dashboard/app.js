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
  return String(model).replace(/^(?:openrouter|requesty)-/, "").split("/").at(-1);
}

// Presentation only: how each model reads, and the logo of the company that
// built it. Raw ids stay in the artifacts; OpenRouter and Requesty are
// deliberately absent because they are the transport for the routed rows,
// not the creator.
const MODEL_META = {
  "gpt-6-luna": ["GPT-6-Luna", "openai.svg"],
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
  "drex": ["Drex v1.5", "nace.svg"],
  "span-01-lite": ["Span-01 Lite", "respan.ico"],
  "span-01": ["Span-01", "respan.ico"],
  "standard-one-8b": ["Standard One 8B", "standardthinking.png"],
};
const displayModel = model => MODEL_META[modelName(model)]?.[0] || modelName(model);
const companyLogo = model => MODEL_META[modelName(model)]?.[1] || null;

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
const windowLabel = run => `${monthLabel(run.start)} - ${monthLabel(run.end)}`;
const windowKey = run => `${run.start}..${run.end}`;

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
    return {
      key, label: windowLabel(rows[0]), start: rows[0].start, end: rows[0].end,
      spy: Number(rows[0].metrics?.spy_total_return), runs: [...models.values()],
    };
  }).sort((a, b) => String(a.end).localeCompare(String(b.end)));
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
    row.best = ranked[0] || null;
    row.worst = ranked[ranked.length - 1] || null;
    row.beatIn = row.entries.filter(entry => entry.gap > 0).length;
    row.latest = row.entries[row.entries.length - 1]?.run;
    return row;
    // The default order is the average gap, with windows beaten as the
    // tiebreak: both readings of "who beat SPY" agree on the top, and the
    // Beat SPY in column can flip to wins-first with one click.
  }).sort((a, b) => (b.avgGap ?? -Infinity) - (a.avgGap ?? -Infinity) || b.beatIn - a.beatIn);
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
    if (typeof run?.model !== "string" || !run.model) problems.push(`run ${run?.run_id} has no decision model`);
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
};

if (typeof document !== "undefined") {
  const $ = id => document.getElementById(id);
  // Each page renders only the widgets it carries, so every renderer checks for
  // its own container first and never assumes the rest of the site is present.
  const setText = (id, text) => { const element = $(id); if (element) element.textContent = text; };
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
  // The board is its own explanation: what it returned, what that did to
  // $10,000, whether it beat SPY and how often, how bad it got, how much it
  // actually traded, and whether its judgment was any good. The statistical
  // scores (Brier skill, AUC, Sharpe) ask for background, so they live in the
  // run page, which explains them.
  const COLUMNS = [
    ["model", "Model", "text", "row"],
    ["total_return", "Return", "percent", "metrics"],
    ["usd_10k", "on $10K", "money", "metrics"],
    ["excess_return_vs_spy", "vs SPY", "signed", "metrics"],
    ["outperformance_hit_rate", "Weeks beating SPY", "percent", "metrics"],
    ["max_drawdown", "Worst drop", "percent", "metrics"],
    ["hold", "Holds", "percent", "row"],
    ["accuracy", "Right calls", "percent", "row"],
  ];
  const FORMAT = {text: value => value ?? "n/a", percent, signed: percent, ratio, score, dollars, count, money,
    // These models answer in tens to hundreds of milliseconds; "0.00 s" hides
    // every difference between them.
    millis: value => {
      const ms = Number(value) * 1000;
      if (!Number.isFinite(ms)) return "n/a";
      return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
    }};
  const rowValue = (run, key, source) => {
    if (key === "model") return modelName(run.model);
    // Derived, not recorded: what the recorded return does to a $10,000 start.
    if (key === "usd_10k") return Number(run.metrics?.total_return) * START_USD;
    // The hold share lives one level down, in the catalog's action shares.
    if (key === "hold") return run.actions?.hold;
    // Call accuracy is the plain-words stand-in for the calibration scores.
    if (key === "accuracy") return run.metrics?.calibration?.signals?.noul?.accuracy;
    return source === "metrics" ? run.metrics?.[key] : run[key];
  };
  let runs = [], visible = [], selected = null, loading = 0;
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
    // is scored in) and to the run, and the caption explains why two providers
    // report far more tokens than the rest for the same frozen state.
    // The pricing page prices one window, the latest scored one, so a model
    // is exactly one row even when several windows are published.
    const windows = scoredWindows(runs);
    const onBoard = windows.length ? windows[windows.length - 1].runs : [];
    const records = onBoard.map(run => {
      const rates = runRates(run);
      const decisions = Number(run.decisions ?? run.metrics?.decisions);
      const tokens = Number(run.metrics?.input_tokens);
      const cost = run.estimated_cost_usd ?? inputCost(run.metrics, rates);
      return {
        run, rates,
        tokens: decisions > 0 && Number.isFinite(tokens) ? Math.round(tokens / decisions) : null,
        perDecision: Number.isFinite(Number(cost)) && decisions > 0 ? Number(cost) / decisions : null,
        cost,
      };
    }).sort((a, b) => (Number(b.perDecision) || 0) - (Number(a.perDecision) || 0));
    const element = node("table"), head = node("thead"), header = node("tr"), body = node("tbody");
    for (const [title, className] of [["Model", ""], ["Input $ / 1M", "num"], ["Tokens / decision", "num"],
                                      ["Cost / decision", "num"], ["Run cost", "num"], ["Source", ""]]) {
      header.append(node("th", title, className));
    }
    head.append(header); element.append(head, body);
    for (const {run, rates, tokens, perDecision, cost} of records) {
      const row = node("tr");
      const modelCell = node("td");
      const logo = companyLogo(run.model);
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
    $("pricing-table").replaceChildren(element);
    setHidden("cost-overview", !onBoard.length);
      setText("cost-window", onBoard.length
        ? "Published list prices, checked 2026-10-08. Every model bills input only: output is free and no provider offers prompt caching. "
        + "A run's bill is its rate times the tokens the provider counts, and providers count the same frozen state very differently: most "
        + "bill the shared state once per request, while d1 and pplx-decider bill it once per question on OpenRouter, so a decision that asks 25 questions "
        + "is billed as 25 passes over the same state. That is why the published pplx-decider row is served through Requesty, whose route "
        + "bills the state once per request. Compare models on tokens per decision and cost per decision; the run cost is the total "
        + "for one scored window of weekly decisions (the latest)."
        : "");
    runBars("model-costs", onBoard
      .map(run => ({label: displayModel(run.model), cost: run.estimated_cost_usd ?? inputCost(run.metrics, runRates(run))}))
      .sort((a, b) => (Number(b.cost) || 0) - (Number(a.cost) || 0)));
  }
  // One window at a time, or every window averaged: the toggle picks the lens.
  const ALL_COLUMNS = [
    ["model", "Model", "text"],
    ["windows", "Windows", "count"],
    ["avg_gap", "Avg vs SPY", "signed"],
    ["best_gap", "Best window", "signed"],
    ["worst_gap", "Worst window", "signed"],
    ["beat_in", "Beat SPY in", "text"],
  ];
  let period = "all";

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
        sortKey = id === "all" ? "avg_gap" : "excess_return_vs_spy";
        descending = true;
        renderBoard();
      };
      return button;
    };
    bar.append(chip("all", `All ${windows.length} windows`, "Every scored window, averaged per model"));
    for (const w of windows) {
      bar.append(chip(w.key, w.label, `${w.start} to ${w.end}`
        + (Number.isFinite(w.spy) ? `, SPY ${percent(w.spy)}` : "")));
    }
  }

  function renderBoard() {
    if (!$("leaderboard-table")) return;
    const windows = scoredWindows(runs);
    if (period !== "all" && !windows.some(w => w.key === period)) period = "all";
    renderTabs(windows);
    // A single window (or the single-window fallback) renders the plain board.
    if (period !== "all" || windows.length < 2) {
      const w = period === "all" ? windows[windows.length - 1] : windows.find(x => x.key === period);
      visible = w ? w.runs : [];
      renderWindowBoard(w, windows.length >= 2);
    } else {
      visible = [];
      renderAllBoard(windows);
    }
    updateBoardChrome(windows);
  }

  function updateBoardChrome(windows) {
    const source = visible.length ? visible : [];
    const allMode = period === "all" && windows.length >= 2;
    setText("run-count", allMode
      ? `${plural(allPeriodRows(windows).length, "model")} · ${plural(windows.length, "window")}`
      : `${visible.length} result${visible.length === 1 ? "" : "s"}`);
    if (allMode) {
      setText("hero-scope", `${windows[0].start} → ${windows[windows.length - 1].end} · ${plural(windows.length, "window")}`);
    } else if (source.length) {
      const first = source.reduce((a, b) => (String(a.start) <= String(b.start) ? a : b));
      const last = source.reduce((a, b) => (String(a.end) >= String(b.end) ? a : b));
      const decisions = source[0].decisions ?? source[0].metrics?.decisions;
      setText("hero-scope", `${first.start} → ${last.end} · ${plural(decisions, "decision")}`);
    }
    const syntheticCount = source.filter(run => run.synthetic).length;
    setHidden("demo-banner", syntheticCount === 0);
    if (syntheticCount) {
      setText("demo-banner", syntheticCount === source.length
        ? `Synthetic demo data: all ${syntheticCount} rows are fabricated by \`beatspy demo\`. No decision model was called, so every number here is meaningless as a result.`
        : `Synthetic demo data: ${syntheticCount} of ${source.length} rows are fabricated by \`beatspy demo\` and are not real results.`);
    }
    const runCount = $("run-count");
    if (runCount) runCount.className = syntheticCount === source.length && source.length ? "pill warn" : "pill";
    setText("status", source.length || !allMode ? (source.length ? "" : (runs.length
      ? "No completed results yet."
      : "No results yet. Record a decision-model run and re-render this page.")) : "");
  }

  function renderWindowBoard(w, multiWindow) {
    // The caption teaches the two columns whose meaning a first read can't
    // guess; the no-skill line comes from the catalog, not a hardcoded number.
    const baseRate = Number(visible[0]?.metrics?.calibration?.signals?.noul?.base_rate);
    setText("board-window", (multiWindow && w
      ? `${w.label} window${Number.isFinite(w.spy) ? `, SPY ${percent(w.spy)}` : ""}. `
      : "One row per model, latest completed result. ")
      + "Ranked by vs SPY, the gap to the index in this window; click any column to re-sort. "
      + "Holds is the share of names left untraded each week; Right calls is how often its calls proved right"
      + (Number.isFinite(baseRate)
        ? ` (always calling the same way scores ${percent(baseRate)} here, 50% is a coin flip)`
        : " (50% is a coin flip)")
      + ". Open a model for its full run.");
    visible.sort((a, b) => {
      const column = COLUMNS.find(([key]) => key === sortKey) || COLUMNS[3];
      const av = rowValue(a, column[0], column[3]), bv = rowValue(b, column[0], column[3]);
      if (column[2] === "text") return descending ? String(bv).localeCompare(String(av)) : String(av).localeCompare(String(bv));
      if (!Number.isFinite(Number(av)) || !Number.isFinite(Number(bv))) return Number.isFinite(Number(av)) ? -1 : 1;
      return descending ? Number(bv) - Number(av) : Number(av) - Number(bv);
    });
    // Rank follows the sorted order the reader sees, so the numbers can never
    // disagree with the rows under the current sort.
    const ranks = new Map(visible.map((run, i) => [run.run_id, i + 1]));
    const element = node("table"), head = node("thead"), header = node("tr"), body = node("tbody");
    header.append(node("th", "Rank", "col-rank"));
    for (const [key, title, kind] of COLUMNS) {
      const th = node("th", undefined, `${kind === "text" ? "" : "num"} col-${key}`);
      const button = node("button", title + (sortKey === key ? (descending ? " ↓" : " ↑") : ""));
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
      const logo = companyLogo(run.model);
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
        chip.title = "Synthetic demo run: fabricated by `beatspy demo`, not produced by a decision model.";
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
        const tone = Number.isFinite(value) && (key === "total_return" || key === "excess_return_vs_spy" || key === "usd_10k")
          ? (value >= 0 ? "positive" : "negative") : "";
        const cell = node("td", undefined, `num col-${key}`);
        if (tone) cell.append(node("span", `${tone === "positive" ? "▲" : "▼"} ${FORMAT[kind](raw)}`, `chip ${tone}`));
        else cell.textContent = FORMAT[kind](raw);
        row.append(cell);
      }
      body.append(row);
    }
    $("leaderboard-table").replaceChildren(element);
  }

  function renderAllBoard(windows) {
    const rows = allPeriodRows(windows);
    const total = windows.length;
    setText("board-window", `Every model on every scored window, oldest to newest. `
      + `Ranked by Avg vs SPY, the mean gap across its windows, with windows beaten as the tiebreak; click any column to re-sort (Beat SPY in ranks by wins first). `
      + `Windows shows how many of the ${total} windows a model has run in; `
      + `Best and Worst name the extremes. Open a model for its most recent full run.`);
    const value = (row, key) => key === "model" ? displayModel(row.model)
      : key === "windows" ? `${row.windows}`
      : key === "avg_gap" ? row.avgGap
      : key === "best_gap" ? row.best?.gap
      : key === "worst_gap" ? row.worst?.gap
      : key === "beat_in" ? row.beatIn
      : null;
    rows.sort((a, b) => {
      const column = ALL_COLUMNS.find(([key]) => key === sortKey) || ALL_COLUMNS[2];
      const av = value(a, column[0]), bv = value(b, column[0]);
      // The headline ranking: average gap first, windows beaten as the tiebreak.
      // The Beat SPY in column flips that: wins first, average as the tiebreak.
      if (column[0] === "avg_gap" || column[0] === "beat_in") {
        const avgFirst = column[0] === "avg_gap";
        const primary = avgFirst ? (b.avgGap ?? -Infinity) - (a.avgGap ?? -Infinity) : b.beatIn - a.beatIn;
        const secondary = avgFirst ? b.beatIn - a.beatIn : (b.avgGap ?? -Infinity) - (a.avgGap ?? -Infinity);
        return descending ? primary || secondary : -(primary || secondary);
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
      th.setAttribute("aria-sort", sortKey === key ? (descending ? "descending" : "ascending") : "none");
      button.onclick = () => {descending = sortKey === key ? !descending : true; sortKey = key; renderBoard();};
      th.append(button); header.append(th);
    }
    head.append(header); element.append(head, body);
    rows.forEach((row, index) => {
      const tr = node("tr");
      if (index === 0) tr.classList.add("leader");
      tr.append(node("td", index + 1, "col-rank"));
      const td = node("td", undefined, "col-model"), link = node("a", displayModel(row.model), "run-button");
      if (row.latest) {
        link.href = `run.html?run=${encodeURIComponent(row.latest.run_id)}`;
        if (row.latest.run_id === selected) tr.classList.add("selected");
      }
      const logo = companyLogo(row.model);
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
      const gapCell = (entry) => {
        const cell = node("td", undefined, "num col-gap");
        if (entry && Number.isFinite(entry.gap)) {
          const tone = entry.gap >= 0 ? "positive" : "negative";
          cell.append(node("span", `${tone === "positive" ? "▲" : "▼"} ${percent(entry.gap)}`, `chip ${tone}`));
          const note = node("span", entry.window, "cell-note");
          cell.append(note);
        } else cell.textContent = "n/a";
        return cell;
      };
      tr.append(node("td", `${row.windows}/${total}`, "num col-windows"));
      const avgCell = node("td", undefined, "num col-avg_gap");
      if (Number.isFinite(row.avgGap)) {
        const tone = row.avgGap >= 0 ? "positive" : "negative";
        avgCell.append(node("span", `${tone === "positive" ? "▲" : "▼"} ${percent(row.avgGap)}`, `chip ${tone}`));
      } else avgCell.textContent = "n/a";
      tr.append(avgCell, gapCell(row.best), gapCell(row.worst));
      tr.append(node("td", `${row.beatIn} of ${row.windows}`, "num col-beat_in"));
      body.append(tr);
    });
    $("leaderboard-table").replaceChildren(element);
  }
  function runBars(target, rows) {
    const container = $(target); container.replaceChildren();
    const known = rows.filter(row => typeof row.cost === "number" && Number.isFinite(row.cost));
    if (!rows.length || !known.length) {container.append(node("p", "No recorded input cost.", "caption")); return;}
    const max = Math.max(1e-12, ...known.map(row => row.cost));
    for (const row of rows) {
      const line = node("div", undefined, "bar-row"), track = node("div", undefined, "bar-track");
      line.append(node("span", row.label, "bar-label"));
      if (typeof row.cost === "number" && Number.isFinite(row.cost)) {
        const bar = node("span", undefined, "bar-fill");
        bar.style.width = `${row.cost / max * 100}%`; bar.style.background = COLORS[0]; track.append(bar);
      }
      line.append(track, node("span", typeof row.cost === "number" ? preciseDollars(row.cost) : "n/a", "bar-value"));
      container.append(line);
    }
  }
  // Equity columns arrive as snake_case keys; the legend should read like prose.
  const SERIES_LABELS = {
    portfolio: "decision model",
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
      setHidden("run-details", false);
      setText("status", "");
      $("run-title").textContent = `${displayModel(run.model)} · ${run.question_form} questions`;
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
          `${plural(run.metrics.trades, "fill")} filled · ${perDecision(buys)} buys, ${perDecision(sells)} sells, ${perDecision(holds)} holds per decision`],
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
      ];
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
      reliabilityChart("reliability", calibration);
      const allocationRowsData = allocationRows(decisions);
      const tickers = [...new Set(allocationRowsData.flatMap(row => Object.keys(row).filter(key => key !== "date")))];
      allocationRowsData.forEach(row => tickers.forEach(key => {row[key] = row[key] || 0;}));
      chart("allocations", allocationRowsData, tickers, percent);
      const rates = ratesFor(run.model, meta), cost = run.metrics.estimated_cost_usd ?? inputCost(run.metrics, rates);
      const policy = run.policy || {};
      const policyText = policy.selection === "core_satellite"
        // Recorded under protocol 3; kept so those runs still render their own rule.
        ? `core ${percent(policy.core_weight)} in the benchmark, ${percent(1 - (policy.core_weight ?? 0))} active sleeve`
        : (policy.selection === "graded_cs" || policy.selection === "graded")
          ? `whole book across the investable set, neutral P = ${policy.min_probability ?? "?"}`
          : `route ${count(policy.top_n)} names at P ≥ ${policy.min_probability ?? "?"}`;
      const selectionText = meta.scenario?.selection_top
        ? ` · trades the top ${count(meta.scenario.selection_top)} trailing performers, ranked point-in-time`
        : "";
      $("run-cost").textContent = `Input-only estimate ${dollars(cost)} · ${count(run.metrics.input_tokens)} input tokens · `
        + `${count(run.metrics.output_tokens)} output tokens · ${plural(run.metrics.requests, "request")} · `
        + `mean latency ${FORMAT.millis(run.metrics.mean_latency_s)} · ${policyText}${selectionText}`;
      $("price-source").replaceChildren();
      if (rates?.source) {
        const link = node("a", "Published pricing ↗"); link.href = rates.source;
        link.rel = "noreferrer noopener"; link.target = "_blank";
        $("price-source").append(link);
      }
      $("settings").textContent = JSON.stringify(meta, null, 2);
      $("coverage").textContent = `Scored ${plural(calibration.observations, "observation")} over ${plural(calibration.horizons, "horizon")}, `
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
    const id = wantedRunId() || (leaderboardRuns(runs)[0] || {}).run_id;
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
      renderRun();
    } catch (error) {
      setText("run-count", "Unavailable");
      setText("status", `Results unavailable: ${error.message}. Try reloading the page.`);
    }
  }
  init();
}