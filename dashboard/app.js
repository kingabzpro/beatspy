"use strict";

const COLORS = ["#b4f272", "#79b6ff", "#ffbc79", "#cb9aff", "#ff9292", "#74d8cb", "#f0b9da", "#d1d7de", "#8d9969"];
const TRUST = {maintainer: "Maintainer run", community: "Community submitted", local: "Local · unverified", synthetic: "SYNTHETIC", mixed: "Mixed evidence"};
const percent = value => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(2)}%` : "—";
const dollars = value => typeof value === "number" && Number.isFinite(value) ? `$${value.toFixed(4)}` : "—";
// Reference API rates checked 2026-10-04. Hosted/subscription runs are API-equivalent estimates, not invoices.
const PRICES = {
  "GLM-5.3": {input: 1.4, output: 4.4, source: "https://docs.z.ai/guides/overview/pricing", note: "Z.AI list price"},
  "GLM-5.3-Flash": {input: .15, output: .5, source: "https://docs.z.ai/guides/overview/pricing", note: "Z.AI list price"},
  "DeepSeek-V4.1-Flash": {input: .3, output: 1.2, source: "https://api-docs.deepseek.com/quick_start/pricing/", note: "DeepSeek peak list price; off-peak is half"},
  "gpt-6-luna": {input: .1, output: .5, source: "https://developers.openai.com/api/docs/pricing", note: "OpenAI standard short-context list price"},
  "mimo-v2.6-pro": {input: .435, output: .87, source: "https://mimo.mi.com/docs/en-US/quick-start/usage-guide/text-generation/batch-api", note: "MiMo overseas real-time API price; Token Plan billing differs"},
};

function tokenCost(usage, rates) {
  if (![usage?.input_tokens, usage?.output_tokens, rates?.input, rates?.output].every(value => typeof value === "number" && Number.isFinite(value) && value >= 0)) return null;
  const cost = (usage.input_tokens * rates.input + usage.output_tokens * rates.output) / 1e6;
  return Number.isFinite(cost) ? cost : null;
}

function usageBreakdown(decisions, rates) {
  const agents = Object.create(null), steps = [];
  let cumulative = 0;
  for (const decision of decisions) {
    const total = {input_tokens: 0, output_tokens: 0, requests: 0, tool_calls: 0};
    const usages = Object.entries(decision.usage || {});
    for (const [role, usage] of usages) {
      agents[role] ||= {label: role, input_tokens: 0, output_tokens: 0, requests: 0, tool_calls: 0};
      for (const key of Object.keys(total)) {
        const raw = usage[key] ?? (key.endsWith("tokens") ? NaN : 0);
        const value = typeof raw === "number" && Number.isFinite(raw) && raw >= 0 ? raw : NaN;
        agents[role][key] += value; total[key] += value;
      }
    }
    if (!usages.length) {total.input_tokens = NaN; total.output_tokens = NaN;}
    const cost = usages.length ? tokenCost(total, rates) : null;
    cumulative = cost === null || cumulative === null ? null : cumulative + cost;
    steps.push({date: decision.date, ...total, cost, cumulative});
  }
  return {agents: Object.values(agents).map(usage => ({...usage, cost: tokenCost(usage, rates)})), steps};
}

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

function allocationRow(row) {
  const held = row.validated?.invalid ? row.market_brief : null;
  return {date: row.date, ...(held?.current_portfolio_weights || row.validated?.weights || row.decision?.allocations?.reduce((obj, a) => ({...obj, [a.ticker]: a.weight}), {}) || {}),
    CASH: held?.current_cash_weight ?? row.validated?.cash ?? row.decision?.cash_weight ?? 0};
}

function leaderboardRuns(runs, allYears = false) {
  if (allYears) {
    const years = [...new Set(runs.map(run => run.year || String(run.end).slice(0, 4)))].sort();
    const models = new Map();
    for (const year of years) {
      for (const run of leaderboardRuns(runs.filter(row => (row.year || String(row.end).slice(0, 4)) === year))) {
        if (!models.has(run.model)) models.set(run.model, []);
        models.get(run.model).push(run);
      }
    }
    return [...models].map(([model, members]) => {
      const metrics = {};
      for (const key of new Set(members.flatMap(run => Object.keys(run.metrics)))) {
        const values = members.map(run => run.metrics[key]).filter(value => typeof value === "number" && Number.isFinite(value));
        metrics[key] = values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
      }
      return {model, members, metrics, run_id: `average:${model}`, trust: members.every(run => run.trust === members[0].trust) ? members[0].trust : "mixed"};
    }).sort((a, b) => b.metrics.excess_return_vs_spy - a.metrics.excess_return_vs_spy);
  }
  const ordered = [...runs].sort((a, b) => String(b.end).localeCompare(String(a.end)) || String(b.created_utc).localeCompare(String(a.created_utc)));
  if (!ordered.length) return [];
  const latest = ordered[0], models = new Map();
  for (const run of ordered) {
    if (run.start !== latest.start || run.end !== latest.end || run.scenario !== latest.scenario) continue;
    if (!models.has(run.model)) models.set(run.model, run);
  }
  return [...models.values()].sort((a, b) => b.metrics.excess_return_vs_spy - a.metrics.excess_return_vs_spy);
}

if (typeof module !== "undefined") module.exports = {parseCSV, leaderboardRuns, modelName, allocationRow, percent, tokenCost, usageBreakdown, PRICES};

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
  let runs = [], visible = [], selected = null, loading = 0;
  let sortKey = "excess_return_vs_spy", descending = true;
  const priceOverrides = new Map();
  let costContext = null;
  function ratesFor(model, meta) {
    return priceOverrides.get(model) || (meta?.model?.cost_per_m_input != null && meta?.model?.cost_per_m_output != null
      ? {input: meta.model.cost_per_m_input, output: meta.model.cost_per_m_output, note: "Recorded run rates"} : PRICES[modelName(model)]);
  }
  function costForRun(run) {
    if (run.members) {
      const values = run.members.map(costForRun);
      return values.every(value => value !== null) ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
    }
    if (!priceOverrides.has(run.model) && Number.isFinite(run.metrics.estimated_cost_usd)) return run.metrics.estimated_cost_usd;
    return tokenCost(run.metrics, ratesFor(run.model));
  }

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
  function options(id, values, initial) {
    $(id).replaceChildren(new Option(initial, ""));
    for (const value of [...new Set(values)].filter(Boolean).sort()) $(id).add(new Option(value, value));
  }
  function filter() {
    renderBoard();
    if (selected && !visible.some(run => run.run_id === selected)) {
      selected = null; loading++; $("run-details").hidden = true; $("average-details").hidden = true;
    } else if (selected?.startsWith("average:")) {
      showAverage(visible.find(run => run.run_id === selected));
    }
  }
  function changeFilters() {
    filter();
    if (!selected && /^#(run|model)=/.test(location.hash)) history.replaceState(null, "", location.pathname + location.search);
  }
  function renderBoard() {
    const allYears = !$("year").value;
    visible = leaderboardRuns(runs.filter(run => ($("trust").value === "all" || run.trust === $("trust").value)
      && (!$("year").value || run.year === $("year").value)
      && (!$("scenario").value || run.scenario === $("scenario").value)), allYears);
    $("board-window").textContent = !visible.length ? "" : allYears
      ? "All years · Equal-weight averages of the latest yearly windows · Select a model for dates and individual runs"
      : `${visible[0].scenario} · ${visible[0].start} → ${visible[0].end} · Latest run per model`;
    const ranks = new Map(visible.map((run, i) => [run.run_id, i + 1]));
    visible.sort((a, b) => {
      const av = sortKey === "model" ? modelName(a.model) : sortKey === "estimated_cost_usd" ? costForRun(a) : a.metrics[sortKey];
      const bv = sortKey === "model" ? modelName(b.model) : sortKey === "estimated_cost_usd" ? costForRun(b) : b.metrics[sortKey];
      if (sortKey === "estimated_cost_usd" && (av === null || bv === null)) return av === null ? (bv === null ? 0 : 1) : -1;
      const result = typeof av === "string" ? av.localeCompare(bv) : Number(av) - Number(bv);
      return descending ? -result : result;
    });
    $("run-count").textContent = `${visible.length} result${visible.length === 1 ? "" : "s"}`;
    $("status").textContent = visible.length ? "" : (runs.length
      ? "No results match these filters. Try Community submitted, All results, or another year."
      : "No published runs yet. Run a benchmark and open a pull request to add the first result.");
    const element = node("table"), head = node("thead"), header = node("tr"), body = node("tbody");
    header.append(node("th", "Rank"));
    const columns = [["model", "Model"], ["total_return", "Return"], ["spy_total_return", "SPY"],
      ["excess_return_vs_spy", "Excess"], ["sharpe", "Sharpe"], ["max_drawdown", "Max DD"],
      ["directional_accuracy", "Direction"], ["input_tokens", "Tokens in"], ["estimated_cost_usd", "Est. cost"]];
    for (const [key, title] of columns) {
      const th = node("th", undefined, key === "model" ? "" : "num");
      const button = node("button", (allYears && key !== "model" ? "Avg " : "") + title + (sortKey === key ? (descending ? " ↓" : " ↑") : ""));
      th.setAttribute("aria-sort", sortKey === key ? (descending ? "descending" : "ascending") : "none");
      button.onclick = () => {descending = sortKey === key ? !descending : key !== "model"; sortKey = key; renderBoard();};
      th.append(button); header.append(th);
    }
    if (allYears) header.append(node("th", "Years"));
    header.append(node("th", "Evidence")); head.append(header); element.append(head, body);
    for (const run of visible) {
      const row = node("tr");
      if (run.run_id === selected) row.className = "selected";
      row.append(node("td", ranks.get(run.run_id)));
      const td = node("td"), button = node("button", modelName(run.model), "run-button");
      button.onclick = () => {
        const hash = run.members ? `#model=${encodeURIComponent(run.model)}` : `#run=${encodeURIComponent(run.run_id)}`;
        if (location.hash === hash) {if (run.members) showAverage(run); else selectRun(run.run_id);} else location.hash = hash;
      };
      td.append(button); row.append(td);
      for (const [key] of columns.slice(1)) {
        const raw = key === "estimated_cost_usd" ? costForRun(run) : run.metrics[key], value = Number(raw);
        const formatted = raw == null ? "—" : key === "estimated_cost_usd" ? dollars(raw) : key === "sharpe" ? value.toFixed(2) : key === "input_tokens" ? Math.round(value).toLocaleString() : percent(value);
        row.append(node("td", formatted, `num ${key === "excess_return_vs_spy" ? (value >= 0 ? "positive" : "negative") : ""}`));
      }
      if (allYears) row.append(node("td", run.members.map(member => member.year || member.end.slice(0, 4)).join(", ")));
      row.append(node("td", TRUST[run.trust] || "Unverified")); body.append(row);
    }
    $("leaderboard-table").replaceChildren(element);
    $("cost-overview").hidden = !visible.length;
    $("cost-window").textContent = allYears ? "Average per run across the available yearly windows." : "Per run in the selected benchmark window.";
    bars("model-costs", visible.map(run => ({label: modelName(run.model), cost: costForRun(run)})), ["cost"], dollars);
    bars("model-tokens", visible.map(run => ({label: modelName(run.model), input: run.metrics.input_tokens, output: run.metrics.output_tokens})), ["input", "output"], value => Math.round(value).toLocaleString());
  }
  function showAverage(run) {
    selected = run.run_id; loading++;
    $("run-details").hidden = true; $("average-details").hidden = false;
    $("average-title").textContent = modelName(run.model);
    table("average-runs", ["Year", "Window", "Return", "SPY", "Excess", "Est. cost"], run.members.map(member => [
      member.year || member.end.slice(0, 4), `${member.start} → ${member.end}`,
      ...["total_return", "spy_total_return", "excess_return_vs_spy"].map(key => percent(member.metrics[key])),
      dollars(costForRun(member)),
    ]));
    $("average-runs").querySelectorAll("tbody tr").forEach((row, i) => {
      const member = run.members[i], link = node("a", member.year || member.end.slice(0, 4));
      link.href = `#run=${encodeURIComponent(member.run_id)}`; row.firstChild.replaceChildren(link);
    });
    renderBoard();
  }
  function bars(target, rows, series, format) {
    const container = $(target); container.replaceChildren();
    const max = Math.max(1e-12, ...rows.map(row => series.reduce((sum, key) => sum + (Number.isFinite(row[key]) ? row[key] : 0), 0)));
    if (!rows.length) {container.append(node("p", "No recorded usage.", "caption")); return;}
    for (const row of rows) {
      const line = node("div", undefined, "bar-row"), track = node("div", undefined, "bar-track");
      const known = series.every(key => typeof row[key] === "number" && Number.isFinite(row[key]) && row[key] >= 0);
      const total = known ? series.reduce((sum, key) => sum + row[key], 0) : null;
      line.append(node("span", row.label, "bar-label"));
      if (known) series.forEach((key, i) => {
        const bar = node("span", undefined, "bar-fill"); bar.style.width = `${row[key] / max * 100}%`; bar.style.background = COLORS[i]; track.append(bar);
      });
      line.title = known ? series.map(key => `${key}: ${format(row[key])}`).join(" · ") : "No price or usage recorded";
      line.append(track, node("span", total === null ? "—" : format(total), "bar-value")); container.append(line);
    }
    if (series.length > 1) {
      const legend = node("div", undefined, "legend");
      series.forEach((key, i) => {const label = node("span", `● ${key}`); label.style.color = COLORS[i]; legend.append(label);}); container.append(legend);
    }
  }
  function renderCosts() {
    if (!costContext) return;
    const {run, meta, decisions} = costContext, rates = ratesFor(run.model, meta);
    const {agents, steps} = usageBreakdown(decisions, rates), cost = tokenCost(run.metrics, rates);
    $("cost-summary").textContent = `Estimated model cost ${dollars(cost)} · Per decision ${dollars(decisions.length && cost !== null ? cost / decisions.length : null)} · ${run.metrics.requests ?? "—"} model requests`;
    $("price-note").textContent = `${rates?.note || "Enter both rates to calculate cost"}. Input $${rates?.input ?? "—"} / Output $${rates?.output ?? "—"} per million tokens. Estimates assume uncached input; cache tiers, long-context pricing, hosting, subscriptions, and paid tools are not included. Reference prices checked 2026-10-04.`;
    $("price-source").replaceChildren();
    if (rates?.source) {const link = node("a", "Official pricing ↗"); link.href = rates.source; $("price-source").append(link);}
    chart("running-cost", steps.length && steps.every(step => step.cumulative !== null) ? steps : [], ["cumulative"], dollars);
    bars("agent-costs", agents, ["cost"], dollars);
    bars("decision-tokens", steps.map(step => ({label: step.date, input: step.input_tokens, output: step.output_tokens})), ["input", "output"], value => Math.round(value).toLocaleString());
    table("cost-decisions", ["Decision", "Input tokens", "Output tokens", "Requests", "Est. USD", "Running USD"], steps.map(step => [step.date, ...[step.input_tokens, step.output_tokens, step.requests].map(value => Number.isFinite(value) ? value.toLocaleString() : "—"), dollars(step.cost), dollars(step.cumulative)]));
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
      if (target === "running-cost") {min = 0; max = Math.max(max, 1e-6);}
      else if (max === min) {max += 1; min -= 1;}
      const x = i => L + i * (W - L - R) / Math.max(rows.length - 1, 1);
      const y = value => T + (max - value) * (H - T - B) / (max - min);
      for (let i = 0; i <= 4; i++) {
        const value = min + (max - min) * i / 4;
        svg.append(svgNode("line", {x1: L, x2: W - R, y1: y(value), y2: y(value), stroke: "#30363d"}));
        const label = svgNode("text", {x: L - 8, y: y(value) + 4, fill: "#a7b0ba", "font-size": 10, "text-anchor": "end"});
        label.textContent = format(value); svg.append(label);
      }
      for (const key of keys) svg.append(svgNode("polyline", {
        points: rows.map((row, i) => `${x(i)},${y(Number(row[key]))}`).join(" "),
        fill: "none", stroke: COLORS[series.indexOf(key) % COLORS.length], "stroke-width": 2,
      }));
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
  async function selectRun(id) {
    costContext = null;
    $("average-details").hidden = true;
    const run = runs.find(row => row.run_id === id);
    if (!run) {$("run-details").hidden = true; $("status").textContent = "That run is not in this catalog."; return;}
    selected = id; const generation = ++loading;
    $("run-details").hidden = true; renderBoard();
    try {
      if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$/.test(run.run_id) || run.dir !== `runs/${run.run_id}`) throw new Error("Invalid artifact path");
      const base = `data/${run.dir}/`;
      const texts = await Promise.all(["run.json", "equity_curve.csv", "decisions.jsonl", "trades.csv"].map(name => fetchText(base + name)));
      if (generation !== loading) return;
      const meta = JSON.parse(texts[0]), equity = parseCSV(texts[1]), decisions = texts[2].split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line)), trades = parseCSV(texts[3]);
      $("run-details").hidden = false;
      $("run-title").textContent = modelName(run.model);
      $("run-trust").textContent = TRUST[run.trust] || "Unverified";
      $("run-meta").textContent = `${run.scenario} · ${run.start} → ${run.end} · Data cutoff ${run.data_cutoff} · Snapshot ${run.snapshot_id || "unknown"}`;
      $("metrics").replaceChildren();
      for (const [key, title] of [["total_return", "Portfolio return"], ["spy_total_return", "SPY buy & hold"], ["excess_return_vs_spy", "Excess vs SPY"], ["max_drawdown", "Maximum drawdown"]]) {
        const tile = node("div", undefined, "metric"); tile.append(node("p", title), node("strong", percent(run.metrics[key]), Number(run.metrics[key]) >= 0 ? "positive" : "negative")); $("metrics").append(tile);
      }
      const series = Object.keys(equity[0] || {}).filter(key => key !== "date");
      chart("performance", equity, series, value => `$${value.toLocaleString(undefined, {maximumFractionDigits: 0})}`);
      const peaks = {}; const drawdown = equity.map(row => {
        const result = {date: row.date};
        for (const key of series) {peaks[key] = Math.max(peaks[key] || -Infinity, Number(row[key])); result[key] = Number(row[key]) / peaks[key] - 1;}
        return result;
      });
      chart("drawdown", drawdown, series, percent);
      const allocationRows = decisions.map(allocationRow);
      const tickers = [...new Set(allocationRows.flatMap(row => Object.keys(row).filter(key => key !== "date")))];
      allocationRows.forEach(row => tickers.forEach(key => {row[key] = row[key] || 0;}));
      chart("allocations", allocationRows, tickers, percent);
      const totals = Object.create(null);
      decisions.forEach(row => Object.entries(row.usage || {}).forEach(([role, usage]) => {
        totals[role] ||= {input_tokens: 0, output_tokens: 0, requests: 0, tool_calls: 0};
        for (const key of Object.keys(totals[role])) totals[role][key] += Number(usage[key] || 0);
      }));
      table("telemetry", ["Agent", "Input tokens", "Output tokens", "Requests", "Tools"], Object.entries(totals).map(([role, usage]) => [role, ...Object.values(usage).map(value => value.toLocaleString())]));
      costContext = {run, meta, decisions};
      const rates = ratesFor(run.model, meta);
      $("input-price").value = rates?.input ?? ""; $("output-price").value = rates?.output ?? "";
      const updatePrices = () => {
        const values = [$("input-price"), $("output-price")].map(input => input.value === "" ? null : input.valueAsNumber);
        if (values.some(value => value !== null && (!Number.isFinite(value) || value < 0))) {$("price-note").textContent = "Enter nonnegative, finite rates."; return;}
        priceOverrides.set(run.model, {input: values[0], output: values[1], note: "Custom rates for this visit"});
        renderCosts(); renderBoard();
      };
      $("input-price").oninput = updatePrices; $("output-price").oninput = updatePrices;
      $("reset-prices").onclick = () => {priceOverrides.delete(run.model); const original = ratesFor(run.model, meta); $("input-price").value = original?.input ?? ""; $("output-price").value = original?.output ?? ""; renderCosts(); renderBoard();};
      renderCosts();
      $("decisions").replaceChildren();
      for (const row of decisions) {
        const detail = node("details", undefined, "decision");
        detail.append(node("summary", `${row.date} · ${row.predicted_direction || "flat"} · ${row.parse_errors?.length ? "Output issues recorded" : "Parsed"}`), node("p", row.rationale || "No rationale recorded."), node("pre", JSON.stringify(row, null, 2)));
        $("decisions").append(detail);
      }
      if (!decisions.length) $("decisions").append(node("p", "No decisions recorded.", "caption"));
      table("trades", Object.keys(trades[0] || {date: "", note: ""}), trades.length ? trades.map(row => Object.values(row)) : [["—", "No trades"]]);
      $("settings").textContent = JSON.stringify(meta, null, 2);
      $("coverage").textContent = `${meta.event_feed?.note || "Curated events are not a complete news history."} Feed through: ${meta.event_feed?.through || "not recorded"}. Provider/model identity is a claim unless this is a maintainer-controlled run.`;
      $("downloads").replaceChildren();
      for (const name of ["run.json", "metrics.json", "equity_curve.csv", "trades.csv", "decisions.jsonl", "events.jsonl"]) {
        const a = node("a", name); a.href = base + name; a.setAttribute("download", name); $("downloads").append(a);
      }
      const snapshot = meta.data?.snapshot_id;
      if (/^[a-f0-9]{64}$/.test(snapshot || "") && run.trust !== "local") {
        for (const name of ["prices.csv", "MANIFEST.json"]) {const a = node("a", `Snapshot ${name}`); a.href = `data/snapshots/${snapshot}/${name}`; $("downloads").append(a);}
      }
    } catch (error) {if (generation === loading) {$("run-details").hidden = true; $("status").textContent = `Unable to load this run: ${error.message}`;}}
  }
  function hashSelection() {
    if (location.hash.startsWith("#model=")) {
      try {
        const model = decodeURIComponent(location.hash.slice(7));
        $("year").value = ""; $("scenario").value = ""; filter();
        const run = visible.find(row => row.model === model);
        if (run) showAverage(run);
        else {selected = null; loading++; $("run-details").hidden = true; $("average-details").hidden = true; $("status").textContent = "That model has no results matching these filters.";}
      } catch (error) {$("status").textContent = `Invalid model link: ${error.message}`;}
      return;
    }
    if (!location.hash.startsWith("#run=")) return;
    try {
      const id = decodeURIComponent(location.hash.slice(5));
      const run = runs.find(row => row.run_id === id);
      if (run) {$("trust").value = "all"; $("year").value = run.year; $("scenario").value = run.scenario; filter();}
      selectRun(id);
    } catch (error) {$("status").textContent = `Invalid run link: ${error.message}`;}
  }
  async function init() {
    try {
      const catalog = JSON.parse(await fetchText("data/index.json"));
      if (catalog.schema_version !== 1 || !Array.isArray(catalog.runs)) throw new Error("Invalid result catalog");
      runs = catalog.runs;
      $("trust").value = catalog.default_trust || "maintainer";
      options("year", runs.map(run => run.year), "All years"); options("scenario", runs.map(run => run.scenario), "All scenarios");
      const defaultRuns = runs.filter(run => catalog.default_trust === "all" || run.trust === $("trust").value);
      $("year").value = defaultRuns.map(run => run.year).filter(Boolean).sort().at(-1) || "";
      ["trust", "scenario"].forEach(id => {$(id).onchange = changeFilters;});
      $("year").onchange = () => {$("scenario").value = ""; changeFilters();};
      window.addEventListener("hashchange", hashSelection); filter();
      if (location.hash.startsWith("#run=") || location.hash.startsWith("#model=")) hashSelection(); else if (catalog.selected) selectRun(catalog.selected);
    } catch (error) {$("run-count").textContent = "Unavailable"; $("status").textContent = `Results unavailable: ${error.message}. Try reloading the page.`;}
  }
  init();
}
