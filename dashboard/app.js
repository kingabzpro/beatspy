"use strict";

const COLORS = ["#b4f272", "#79b6ff", "#ffbc79", "#cb9aff", "#ff9292", "#74d8cb", "#f0b9da", "#d1d7de", "#8d9969"];
const TRUST = {replayed: "Replay checked", local: "Local", synthetic: "Synthetic demo"};
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
    label: "Cloudflare Clef-flash (9B)", provider: "cloudflare", input: 0.09, output: 0,
    source: "https://developers.cloudflare.com/workers-ai/models/clef/index.md",
    note: "Workers AI list price, input only."},
};
const SIGNALS = ["noul", "choice", "rank"];
const percent = value => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(2)}%` : "—";
const ratio = value => Number.isFinite(Number(value)) ? Number(value).toFixed(2) : "—";
const score = value => Number.isFinite(Number(value)) ? Number(value).toFixed(3) : "—";
// Decision-model runs cost fractions of a cent, so small values keep more digits.
const dollars = value => {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return `$${number.toFixed(Math.abs(number) < 0.01 ? 6 : 4)}`;
};
const count = value => Number.isInteger(Number(value)) ? String(Number(value)) : "—";
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
  return String(model).split("/").at(-1);
}

function leaderboardRuns(runs) {
  const ordered = [...runs].sort((a, b) => String(b.created_utc).localeCompare(String(a.created_utc)) || String(b.run_id).localeCompare(String(a.run_id)));
  const models = new Map();
  for (const run of ordered) if (!models.has(run.model)) models.set(run.model, run);
  return [...models.values()].sort((a, b) => (Number(b.metrics?.excess_return_vs_spy) || 0) - (Number(a.metrics?.excess_return_vs_spy) || 0));
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
  parseCSV, leaderboardRuns, modelName, allocationRows, percent, dollars, count, plural, ratio, score, inputCost,
  ratesFor, reliabilitySeries, catalogProblems, PRICES, SIGNALS,
};

if (typeof document !== "undefined") {
  const $ = id => document.getElementById(id);
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
  // [row key, heading, formatter, source of the value]
  const COLUMNS = [
    ["model", "Decision model", "text", "row"],
    ["provider", "Provider", "text", "row"],
    ["total_return", "Return", "percent", "metrics"],
    ["excess_return_vs_spy", "Excess vs SPY", "signed", "metrics"],
    ["sharpe", "Sharpe", "ratio", "metrics"],
    ["max_drawdown", "Max DD", "percent", "metrics"],
    ["brier", "Brier", "score", "row"],
    ["brier_skill_score", "Brier skill", "score", "row"],
    ["auc", "AUC", "score", "row"],
    ["ece", "ECE", "score", "row"],
    ["estimated_cost_usd", "Cost", "money", "row"],
    ["mean_latency_s", "Mean latency", "millis", "row"],
    ["decisions", "Decisions", "count", "row"],
    ["invalid_outputs", "Invalid", "count", "row"],
  ];
  const FORMAT = {text: value => value ?? "—", percent, signed: percent, ratio, score, dollars, count,
    // Decision-model costs land in fractions of a cent, so four decimals would
    // render several providers as the same $0.0000. Show micro-dollars instead.
    money: value => {
      const amount = Number(value);
      if (!Number.isFinite(amount)) return "—";
      if (amount === 0) return "$0";
      return amount >= 0.01 ? `$${amount.toFixed(4)}` : `$${amount.toFixed(6)}`;
    },
    // These models answer in tens to hundreds of milliseconds; "0.00 s" hides
    // every difference between them.
    millis: value => {
      const ms = Number(value) * 1000;
      if (!Number.isFinite(ms)) return "—";
      return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
    }};
  const rowValue = (run, key, source) => key === "model" ? modelName(run.model) : source === "metrics" ? run.metrics?.[key] : run[key];
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
  function renderPricing() {
    const records = Object.entries(PRICES).map(([model, rates]) => {
      const row = node("tr");
      row.append(node("td", model), node("td", rates.provider), node("td", `$${rates.input.toFixed(3)}`, "num"),
        node("td", `$${rates.output.toFixed(2)}`, "num"), node("td", "None"));
      const cell = node("td"), link = node("a", "Source ↗");
      link.href = rates.source; link.rel = "noreferrer noopener"; link.target = "_blank";
      cell.append(link); row.append(cell);
      row.title = rates.note;
      return row;
    });
    const element = node("table"), head = node("thead"), header = node("tr"), body = node("tbody");
    for (const title of ["Decision model", "Provider", "Input USD / M", "Output USD / M", "Prompt caching", "Pricing source"]) {
      header.append(node("th", title));
    }
    head.append(header); element.append(head, body);
    for (const row of records) body.append(row);
    $("pricing-table").replaceChildren(element);
  }
  function renderBoard() {
    visible = leaderboardRuns(runs);
    $("board-window").textContent = "Latest completed result per decision model; every row shows its own evaluation window.";
    const ranks = new Map(visible.map((run, i) => [run.run_id, i + 1]));
    visible.sort((a, b) => {
      const column = COLUMNS.find(([key]) => key === sortKey) || COLUMNS[3];
      const av = rowValue(a, column[0], column[3]), bv = rowValue(b, column[0], column[3]);
      if (column[2] === "text") return descending ? String(bv).localeCompare(String(av)) : String(av).localeCompare(String(bv));
      if (!Number.isFinite(Number(av)) || !Number.isFinite(Number(bv))) return Number.isFinite(Number(av)) ? -1 : 1;
      return descending ? Number(bv) - Number(av) : Number(av) - Number(bv);
    });
    $("run-count").textContent = `${visible.length} result${visible.length === 1 ? "" : "s"}`;
    const syntheticCount = visible.filter(run => run.synthetic).length;
    $("demo-banner").hidden = syntheticCount === 0;
    if (syntheticCount) {
      $("demo-banner").textContent = syntheticCount === visible.length
        ? `Synthetic demo data — all ${syntheticCount} rows are fabricated by \`beatspy demo\`. No decision model was called, so every number here is meaningless as a result.`
        : `Synthetic demo data — ${syntheticCount} of ${visible.length} rows are fabricated by \`beatspy demo\` and are not real results.`;
    }
    $("run-count").className = syntheticCount === visible.length && visible.length ? "pill warn" : "pill";
    $("status").textContent = visible.length ? "" : (runs.length
      ? "No completed results yet."
      : "No results yet. Record a decision-model run and re-render this page.");
    const element = node("table"), head = node("thead"), header = node("tr"), body = node("tbody");
    header.append(node("th", "Rank"));
    for (const [key, title, kind] of COLUMNS) {
      const th = node("th", undefined, kind === "text" ? "" : "num");
      const button = node("button", title + (sortKey === key ? (descending ? " ↓" : " ↑") : ""));
      th.setAttribute("aria-sort", sortKey === key ? (descending ? "descending" : "ascending") : "none");
      button.onclick = () => {descending = sortKey === key ? !descending : true; sortKey = key; renderBoard();};
      th.append(button); header.append(th);
    }
    header.append(node("th", "Period")); head.append(header); element.append(head, body);
    for (const run of visible) {
      const row = node("tr");
      if (run.run_id === selected) row.className = "selected";
      if (run.synthetic) row.classList.add("synthetic");
      row.append(node("td", ranks.get(run.run_id)));
      const td = node("td"), button = node("button", modelName(run.model), "run-button");
      button.onclick = () => {
        const hash = `#run=${encodeURIComponent(run.run_id)}`;
        if (location.hash === hash) selectRun(run.run_id); else location.hash = hash;
      };
      td.append(button);
      if (run.synthetic) {
        // Mark the row itself: the details panel is one click away, and a
        // fabricated number must not read as a real result before that click.
        const chip = node("span", "synthetic", "synthetic-chip");
        chip.title = "Synthetic demo run: fabricated by `beatspy demo`, not produced by a decision model.";
        td.append(chip);
      }
      row.append(td);
      for (const [key, , kind, source] of COLUMNS.slice(1)) {
        const raw = rowValue(run, key, source), value = Number(raw);
        const tone = Number.isFinite(value) ? (value >= 0 ? "positive" : "negative") : "";
        row.append(node("td", FORMAT[kind](raw), `num ${key === "excess_return_vs_spy" ? tone : ""}`));
      }
      row.append(node("td", `${run.start} → ${run.end}`));
      body.append(row);
    }
    $("leaderboard-table").replaceChildren(element);
    $("cost-overview").hidden = !visible.length;
    $("cost-window").textContent = visible.length
      ? "Published list prices on 2026-10-08. Decision models bill input only: every provider charges $0 for output tokens and none offers prompt caching."
      : "";
    runBars("model-costs", visible.map(run => ({label: modelName(run.model), cost: run.estimated_cost_usd ?? inputCost(run.metrics, PRICES[run.model])})));
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
      line.append(track, node("span", typeof row.cost === "number" ? dollars(row.cost) : "—", "bar-value"));
      container.append(line);
    }
  }
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
        tooltip.textContent = `${rows[i].date} · ${keys.map(key => `${key}: ${format(Number(rows[i][key]))}`).join(" · ")}`;
      };
      svg.onpointerleave = () => cross.setAttribute("visibility", "hidden");
      plot.append(svg);
    }
    series.forEach((key, i) => {
      const button = node("button", `● ${key}`); button.style.color = COLORS[i % COLORS.length];
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
      ...series.map(item => item.bins?.[i]?.observed_rate == null ? "—" : percent(item.bins[i].observed_rate)),
    ]);
    table("reliability-bins", headers, records);
  }
  function renderGap(calibration) {
    const callout = $("gap-callout"); callout.replaceChildren();
    const gap = calibration?.miscalibration_gap;
    const noul = calibration?.signals?.noul, choice = calibration?.signals?.choice;
    callout.append(node("strong", Number.isFinite(Number(gap)) ? percent(gap) : "—"));
    // The reported failure mode is a choice form that answers further from 0.5 than
    // the noul form. State it only when this run's numbers actually show it.
    const noulMean = Number(noul?.mean_predicted), choiceMean = Number(choice?.mean_predicted);
    const reach = value => Number.isFinite(value) ? Math.abs(value - 0.5) : NaN;
    let verdict = "compare the two curves above";
    if (Number.isFinite(reach(noulMean)) && Number.isFinite(reach(choiceMean))) {
      if (reach(choiceMean) > reach(noulMean)) verdict = "the choice form answers further from 0.5, so it is the less calibrated of the two";
      else if (reach(choiceMean) < reach(noulMean)) verdict = "the noul form answers further from 0.5 on this run, so the usual ordering is reversed";
      else verdict = "both forms answer equally far from 0.5 on this run";
    }
    callout.append(node("p", Number.isFinite(Number(gap))
      ? `Paired miscalibration gap: the mean absolute difference between the noul and choice probabilities asked about the same ticker and horizon, over ${plural(calibration.observations, "scored observation")}. `
        + `Mean predicted noul ${percent(noul?.mean_predicted)} against choice ${percent(choice?.mean_predicted)}: ${verdict}.`
      : "No paired noul/choice observations were scored in this run."));
    if (calibration?.note) callout.append(node("p", calibration.note, "caption"));
  }
  async function selectRun(id) {
    const run = runs.find(row => row.run_id === id);
    if (!run) {$("run-details").hidden = true; $("status").textContent = "That run is not in this catalog."; return;}
    selected = id; const generation = ++loading;
    $("run-details").hidden = true; renderBoard();
    try {
      if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$/.test(run.run_id) || run.dir !== `runs/${run.run_id}`) throw new Error("Invalid artifact path");
      const base = `data/${run.dir}/`;
      const texts = await Promise.all(["run.json", "metrics.json", "equity_curve.csv", "decisions.jsonl"].map(name => fetchText(base + name)));
      if (generation !== loading) return;
      const meta = JSON.parse(texts[0]), metrics = JSON.parse(texts[1]);
      const equity = parseCSV(texts[2]), decisions = texts[3].split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line));
      const calibration = metrics.calibration || run.metrics.calibration || {};
      $("run-details").hidden = false;
      $("run-title").textContent = `${modelName(run.model)} · ${run.question_form} questions`;
      $("run-trust").textContent = TRUST[run.trust] || "Unrecorded";
      $("run-meta").textContent = `${run.scenario} · ${run.start} → ${run.end} · Data cutoff ${run.data_cutoff} · Snapshot ${run.snapshot_id || "unknown"} · Provider ${run.provider}`;
      $("metrics").replaceChildren();
      for (const [key, title] of [["total_return", "Portfolio return"], ["excess_return_vs_spy", "Excess vs SPY"],
        ["sharpe", "Sharpe"], ["max_drawdown", "Maximum drawdown"], ["brier", "Brier (noul)"], ["calibration_gap", "Miscalibration gap"]]) {
        const raw = run[key] ?? run.metrics[key], value = Number(raw);
        const formatted = key === "sharpe" ? ratio(raw) : key === "brier" ? score(raw) : percent(raw);
        const tile = node("div", undefined, "metric");
        tile.append(node("p", title), node("strong", formatted, Number.isFinite(value) ? (value >= 0 ? "positive" : "negative") : ""));
        $("metrics").append(tile);
      }
      const series = Object.keys(equity[0] || {}).filter(key => key !== "date");
      chart("performance", equity, series, value => `$${value.toLocaleString(undefined, {maximumFractionDigits: 0})}`);
      const peaks = {}; const drawdown = equity.map(row => {
        const result = {date: row.date};
        for (const key of series) {peaks[key] = Math.max(peaks[key] || -Infinity, Number(row[key])); result[key] = Number(row[key]) / peaks[key] - 1;}
        return result;
      });
      chart("drawdown", drawdown, series, percent);
      reliabilityChart("reliability", calibration);
      renderGap(calibration);
      const allocationRowsData = allocationRows(decisions);
      const tickers = [...new Set(allocationRowsData.flatMap(row => Object.keys(row).filter(key => key !== "date")))];
      allocationRowsData.forEach(row => tickers.forEach(key => {row[key] = row[key] || 0;}));
      chart("allocations", allocationRowsData, tickers, percent);
      const rates = ratesFor(run.model, meta), cost = run.metrics.estimated_cost_usd ?? inputCost(run.metrics, rates);
      $("run-cost").textContent = `Input-only estimate ${dollars(cost)} · ${count(run.metrics.input_tokens)} input tokens · `
        + `${count(run.metrics.output_tokens)} output tokens · ${plural(run.metrics.requests, "request")} · `
        + `mean latency ${FORMAT.millis(run.metrics.mean_latency_s)} · route ${count(run.policy?.top_n)} names at `
        + `P ≥ ${run.policy?.min_probability ?? "?"}`;
      $("price-source").replaceChildren();
      if (rates?.source) {
        const link = node("a", "Published pricing ↗"); link.href = rates.source;
        link.rel = "noreferrer noopener"; link.target = "_blank";
        $("price-source").append(link);
      }
      $("settings").textContent = JSON.stringify(meta, null, 2);
      $("coverage").textContent = `Scored ${plural(calibration.observations, "observation")} over ${plural(calibration.horizons, "horizon")} · `
        + `${plural(calibration.observations, "observation")} over ${plural(calibration.horizons, "horizon")}, `
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
    } catch (error) {if (generation === loading) {$("run-details").hidden = true; $("status").textContent = `Unable to load this run: ${error.message}`;}}
  }
  function hashSelection() {
    if (!location.hash.startsWith("#run=")) return;
    try {
      const id = decodeURIComponent(location.hash.slice(5));
      selectRun(id);
    } catch (error) {$("status").textContent = `Invalid run link: ${error.message}`;}
  }
  async function init() {
    try {
      renderPricing();
      const catalog = JSON.parse(await fetchText("data/index.json"));
      const problems = catalogProblems(catalog);
      if (problems.length) throw new Error(`Invalid result catalog (${problems[0]})`);
      runs = catalog.runs;
      window.addEventListener("hashchange", hashSelection); renderBoard();
      if (location.hash.startsWith("#run=")) hashSelection(); else if (catalog.selected) selectRun(catalog.selected);
    } catch (error) {$("run-count").textContent = "Unavailable"; $("status").textContent = `Results unavailable: ${error.message}. Try reloading the page.`;}
  }
  init();
}