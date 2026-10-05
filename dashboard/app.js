"use strict";

const COLORS = ["#b4f272", "#79b6ff", "#ffbc79", "#cb9aff", "#ff9292", "#74d8cb", "#f0b9da", "#d1d7de", "#8d9969"];
const TRUST = {verified: "Verified", replayed: "Replay checked · unsigned", local: "Local · unsigned", synthetic: "Synthetic demo"};
const percent = value => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(2)}%` : "—";
const dollars = value => typeof value === "number" && Number.isFinite(value) ? `$${value.toFixed(4)}` : "—";
// OpenRouter standard uncached token rates checked 2026-10-05; estimates are not invoices.
const PRICES = {
  "Kimi-K3": {"input": 0.67, "output": 14.0, "source": "https://openrouter.ai/moonshotai/kimi-k3", "note": "OpenRouter list price; subscriptions and provider discounts may differ"},
  "gpt-6-luna": {"input": 0.1, "output": 0.5, "source": "https://openrouter.ai/openai/gpt-6-luna", "note": "OpenRouter list price; subscriptions and provider discounts may differ"},
  "mimo-v2.6-pro": {"input": 0.435, "output": 0.87, "source": "https://openrouter.ai/xiaomi/mimo-v2.6-pro", "note": "OpenRouter list price; subscriptions and provider discounts may differ"},
  "GLM-5.3": {"input": 0.05, "output": 7.0, "source": "https://openrouter.ai/z-ai/glm-5.3", "note": "OpenRouter list price; subscriptions and provider discounts may differ"},
  "GLM-5.3-Flash": {"input": 0.15, "output": 0.5, "source": "https://openrouter.ai/z-ai/glm-5.3-flash", "note": "OpenRouter list price; subscriptions and provider discounts may differ"},
  "DeepSeek-V4.1-Flash": {"input": 0.3, "output": 1.2, "source": "https://openrouter.ai/deepseek/deepseek-v4.1-flash", "note": "OpenRouter list price; subscriptions and provider discounts may differ"},
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

function leaderboardRuns(runs) {
  const ordered = [...runs].sort((a, b) => String(b.created_utc).localeCompare(String(a.created_utc)) || String(b.run_id).localeCompare(String(a.run_id)));
  const models = new Map();
  for (const run of ordered) if (!models.has(run.model)) models.set(run.model, run);
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
  function renderBoard() {
    visible = leaderboardRuns(runs);
    $("board-window").textContent = "Latest completed result per model. Each row shows its evaluation period.";
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
      ? "No completed results yet."
      : "No new results yet. Completed benchmark runs will appear here after verification.");
    const element = node("table"), head = node("thead"), header = node("tr"), body = node("tbody");
    header.append(node("th", "Rank"));
    const columns = [["model", "Model"], ["total_return", "Return"], ["spy_total_return", "SPY"],
      ["excess_return_vs_spy", "Excess"], ["sharpe", "Sharpe"], ["max_drawdown", "Max DD"],
      ["directional_accuracy", "Direction"], ["input_tokens", "Tokens in"], ["estimated_cost_usd", "Est. cost"]];
    for (const [key, title] of columns) {
      const th = node("th", undefined, key === "model" ? "" : "num");
      const button = node("button", title + (sortKey === key ? (descending ? " ↓" : " ↑") : ""));
      th.setAttribute("aria-sort", sortKey === key ? (descending ? "descending" : "ascending") : "none");
      button.onclick = () => {descending = sortKey === key ? !descending : key !== "model"; sortKey = key; renderBoard();};
      th.append(button); header.append(th);
    }
    header.append(node("th", "Period"));
    header.append(node("th", "Evidence")); head.append(header); element.append(head, body);
    for (const run of visible) {
      const row = node("tr");
      if (run.run_id === selected) row.className = "selected";
      row.append(node("td", ranks.get(run.run_id)));
      const td = node("td"), button = node("button", modelName(run.model), "run-button");
      button.onclick = () => {
        const hash = `#run=${encodeURIComponent(run.run_id)}`;
        if (location.hash === hash) selectRun(run.run_id); else location.hash = hash;
      };
      td.append(button); row.append(td);
      for (const [key] of columns.slice(1)) {
        const raw = key === "estimated_cost_usd" ? costForRun(run) : run.metrics[key], value = Number(raw);
        const formatted = raw == null ? "—" : key === "estimated_cost_usd" ? dollars(raw) : key === "sharpe" ? value.toFixed(2) : key === "input_tokens" ? Math.round(value).toLocaleString() : percent(value);
        row.append(node("td", formatted, `num ${key === "excess_return_vs_spy" ? (value >= 0 ? "positive" : "negative") : ""}`));
      }
      row.append(node("td", `${run.start} → ${run.end}`));
      row.append(node("td", TRUST[run.trust] || "Unverified")); body.append(row);
    }
    $("leaderboard-table").replaceChildren(element);
    $("cost-overview").hidden = !visible.length;
    $("cost-window").textContent = "API-equivalent estimates for the latest result of each model.";
    bars("model-costs", visible.map(run => ({label: modelName(run.model), cost: costForRun(run)})), ["cost"], dollars);
    bars("model-tokens", visible.map(run => ({label: modelName(run.model), input: run.metrics.input_tokens, output: run.metrics.output_tokens})), ["input", "output"], value => Math.round(value).toLocaleString());
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
    $("price-note").textContent = `${rates?.note || "Enter both rates to calculate cost"}. Input $${rates?.input ?? "—"} / Output $${rates?.output ?? "—"} per million tokens. Estimates assume uncached input; cache tiers, long-context pricing, hosting, subscriptions, and paid tools are not included. Reference prices checked 2026-10-05.`;
    $("price-source").replaceChildren();
    if (rates?.source) {const link = node("a", "OpenRouter pricing ↗"); link.href = rates.source; $("price-source").append(link);}
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
      $("coverage").textContent = `${meta.event_feed?.note || "Curated events are not a complete news history."} Feed through: ${meta.event_feed?.through || "not recorded"}. Signed results come from the trusted runner; older replay-checked results remain unsigned.`;
      $("verification").textContent = run.verification_id ? `Verification ID: ${run.verification_id}` : "Unsigned result. Request a trusted rerun for signed leaderboard verification.";
      $("submit-result").href = "https://github.com/kingabzpro/beatspy/issues/new?title=" + encodeURIComponent(`Benchmark verification: ${run.model}`)
        + "&body=" + encodeURIComponent("Please verify this model on the trusted runner.\n\n```json\n" + JSON.stringify({model: run.model, benchmark: "2026-ytd"}, null, 2) + "\n```\n");
      $("downloads").replaceChildren();
      for (const name of ["run.json", "metrics.json", "equity_curve.csv", "trades.csv", "decisions.jsonl", "events.jsonl"]) {
        const a = node("a", name); a.href = base + name; a.setAttribute("download", name); $("downloads").append(a);
      }
      if (run.verification_id) {const a = node("a", "Signed verification"); a.href = base + "verification.json"; $("downloads").append(a);}
      const snapshot = meta.data?.snapshot_id;
      if (/^[a-f0-9]{64}$/.test(snapshot || "") && run.trust !== "local") {
        for (const name of ["prices.csv", "MANIFEST.json"]) {const a = node("a", `Snapshot ${name}`); a.href = `data/snapshots/${snapshot}/${name}`; $("downloads").append(a);}
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
      const catalog = JSON.parse(await fetchText("data/index.json"));
      if (catalog.schema_version !== 1 || !Array.isArray(catalog.runs)) throw new Error("Invalid result catalog");
      runs = catalog.runs;
      window.addEventListener("hashchange", hashSelection); renderBoard();
      if (location.hash.startsWith("#run=")) hashSelection(); else if (catalog.selected) selectRun(catalog.selected);
    } catch (error) {$("run-count").textContent = "Unavailable"; $("status").textContent = `Results unavailable: ${error.message}. Try reloading the page.`;}
  }
  init();
}
