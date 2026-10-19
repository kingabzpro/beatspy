const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {
  parseCSV, leaderboardRuns, modelName, allocationRows, percent, dollars, preciseDollars, count, plural, ratio, score, inputCost, money, usd, monthLabel, scoredWindows, allPeriodRows,
  ratesFor, reliabilitySeries, catalogProblems, PRICES, SIGNALS
} = require('../dashboard/app.js');

// A minimal catalog row in the DCN shape the Python renderer writes.
function catalogRun(overrides = {}) {
  const runId = overrides.run_id || '20261008-101010_2026-ytd_clef-flash_abcdef1234';
  return {
    run_id: runId, dir: `runs/${runId}`, model: 'clef-flash', model_label: 'Cloudflare Clef-flash (9B)',
    provider: 'cloudflare', question_form: 'twin', policy: {min_probability: 0.5, top_n: 3, min_names: 2},
    pricing: {cost_per_m_input: 0.09, cost_per_m_output: 0.0}, scenario: '2026-ytd',
    start: '2026-01-02', end: '2026-10-07', year: '2026', data_cutoff: '2026-10-07',
    snapshot_id: 'a'.repeat(64), created_utc: '2026-10-08T10:10:10+00:00', synthetic: false, trust: 'replayed',
    label: null, group: 'abc123', decisions: 9, invalid_outputs: 0, answer_coverage: 1.0, mean_latency_s: 0.4,
    estimated_cost_usd: 0.0009, brier: 0.2, brier_skill_score: -0.17, auc: 0.71, ece: 0.2,
    max_calibration_error: 0.36, calibration_gap: 0.355,
    actions: {buy: 0.06, sell: 0.03, hold: 0.91, avg_buys: 1.4, avg_sells: 0.8, avg_holds: 22.8},
    metrics: {total_return: 0.05, spy_total_return: 0.01, excess_return_vs_spy: 0.04, sharpe: -0.98,
      max_drawdown: -0.07, annual_volatility: 0.11, avg_turnover: 0.153, trades: 40,
      input_tokens: 1000, output_tokens: 0, requests: 9, answer_coverage: 1.0,
      mean_latency_s: 0.4, decisions: 9, invalid_outputs: 0, estimated_cost_usd: 0.0009,
      outperformance_hit_rate: 0.44,
      baselines: {equal_weight: 0.02, sixty_forty: 0.01, momentum_12_1: -0.03},
      return_ci: {lower: -0.01, upper: 0.11, level: 0.95, draws: 1000, method: 'decision-window block bootstrap'},
      excess_return_ci: {lower: -0.02, upper: 0.09, level: 0.95, draws: 1000, method: 'decision-window block bootstrap'},
      calibration: {observations: 9, horizons: 3, coverage: 1.0, miscalibration_gap: 0.355, signals: {
        noul: {n: 9, brier: 0.2, brier_skill_score: -0.17, auc: 0.71, expected_calibration_error: 0.2,
          brier_skill_p_value: 0.55, auc_p_value: 0.42, log_loss: 0.61, accuracy: 0.55, base_rate: 0.48,
          mean_predicted: 0.58, bins: [
            {lower: 0, upper: 0.5, count: 0, mean_predicted: 0, observed_rate: null},
            {lower: 0.5, upper: 0.6, count: 6, mean_predicted: 0.55, observed_rate: 0.67},
            {lower: 0.6, upper: 0.7, count: 3, mean_predicted: 0.64, observed_rate: 1.0}]},
        choice: {n: 9, brier: 0.28, brier_skill_score: -0.6, auc: 0.71, expected_calibration_error: 0.26,
          mean_predicted: 0.63, bins: [
            {lower: 0, upper: 0.5, count: 3, mean_predicted: 0.07, observed_rate: 0.67},
            {lower: 0.5, upper: 0.6, count: 0, mean_predicted: 0, observed_rate: null},
            {lower: 0.6, upper: 0.7, count: 3, mean_predicted: 0.84, observed_rate: 0.67}]}}}},
    ...overrides,
  };
}

function catalog(runs) {
  return {schema_version: 1, default_trust: 'all', runs};
}

test('CSV supports CRLF, quotes, escaped quotes, and embedded newlines', () => {
  assert.deepEqual(parseCSV('date,note\r\n2026-10-02,"a, b"\r\n2026-10-01,"a ""quote""\nand newline"\r\n'), [
    {date: '2026-10-02', note: 'a, b'}, {date: '2026-10-01', note: 'a "quote"\nand newline'}
  ]);
  assert.deepEqual(parseCSV('date,price\n'), []);
  assert.throws(() => parseCSV('date,price\n"broken'), /Malformed CSV/);
  assert.throws(() => parseCSV('date,price\n1,2,3'), /column count/);
});

test('one leaderboard row per decision model, chosen by execution time, never best score', () => {
  const rows = [
    catalogRun({run_id: 'old', model: 'clef', created_utc: '2026-10-01', metrics: {...catalogRun().metrics, excess_return_vs_spy: 0.9}}),
    catalogRun({run_id: 'new', model: 'clef', created_utc: '2026-10-04', metrics: {...catalogRun().metrics, excess_return_vs_spy: 0.1}}),
    catalogRun({run_id: 'jev', model: 'jev-latest', created_utc: '2026-10-02', metrics: {...catalogRun().metrics, excess_return_vs_spy: 0.2}})
  ];
  assert.deepEqual(leaderboardRuns(rows).map(row => row.run_id), ['jev', 'new']);
  assert.deepEqual(leaderboardRuns([]), []);
  assert.equal(modelName('zai-org/GLM-5.3'), 'GLM-5.3');
  assert.equal(modelName('clef-flash'), 'clef-flash');
  // Router-served models are presented by model name, not by their route.
  assert.equal(modelName('openrouter-kev-4b'), 'kev-4b');
  assert.equal(percent(0.1234), '12.34%');
  assert.equal(percent(undefined), 'n/a');
  // An input-only decision-model run costs fractions of a cent, so tiny values keep more digits.
  assert.equal(dollars(0.000135), '$0.000135');
  assert.equal(dollars(0.5), '$0.5000');
  assert.equal(dollars(undefined), 'n/a');
  // The pricing column spans two orders of magnitude, so it holds three
  // significant digits instead of a fixed number of decimals.
  assert.equal(preciseDollars(0.07924), '$0.0792');
  assert.equal(preciseDollars(0.004209), '$0.00421');
  assert.equal(preciseDollars(0), '$0');
  assert.equal(preciseDollars(undefined), 'n/a');
  assert.equal(ratio(-0.9822), '-0.98');
  assert.equal(score(0.2), '0.200');
  // The $10,000 view: signed whole dollars for a column, plain for a sentence.
  assert.equal(money(500), '+$500');
  assert.equal(money(-1234.6), '-$1,235');
  assert.equal(money(null), 'n/a');
  assert.equal(usd(10500), '$10,500');
  assert.equal(count(9), '9');
  assert.equal(count(undefined), 'n/a');
  assert.equal(plural(1, 'request'), '1 request');
  assert.equal(plural(3, 'observation'), '3 observations');
  assert.equal(plural(1, 'horizon'), '1 horizon');
});

test('signal helpers read the DCN calibration block', () => {
  const [noul, choice] = reliabilitySeries(catalogRun().metrics.calibration);
  assert.deepEqual(reliabilitySeries(catalogRun().metrics.calibration).map(item => item.name), ['noul', 'choice']);
  assert.equal(noul.n, 9);
  assert.deepEqual(noul.points, [{predicted: 0.55, observed: 0.67, count: 6}, {predicted: 0.64, observed: 1.0, count: 3}]);
  assert.equal(noul.ece, 0.2);
  assert.equal(choice.points.length, 2);
  // Empty bins are never plotted, and a missing block returns no series at all.
  assert.equal(noul.points.some(point => point.count === 0), false);
  assert.deepEqual(reliabilitySeries(undefined), []);
  assert.deepEqual(reliabilitySeries({signals: {}}), []);
  assert.deepEqual(SIGNALS, ['noul', 'choice', 'rank']);
});

test('invalid decisions hold the previous portfolio instead of liquidating', () => {
  const rows = allocationRows([
    {date: '2026-01-02', validated: {invalid: false, weights: {AAPL: 0.5}, cash: 0.5}},
    {date: '2026-02-02', validated: {invalid: true, weights: {}, cash: 1}},
    {date: '2026-03-02', validated: {invalid: false, weights: {MSFT: 0.25}, cash: 0.75}},
    {date: '2026-04-02', validated: {}}
  ]);
  assert.deepEqual(rows[0], {date: '2026-01-02', AAPL: 0.5, CASH: 0.5});
  assert.deepEqual(rows[1], {date: '2026-02-02', AAPL: 0.5, CASH: 0.5});
  assert.deepEqual(rows[2], {date: '2026-03-02', MSFT: 0.25, CASH: 0.75});
  // A run that only ever failed to answer holds cash, and stays investable at 100%.
  assert.deepEqual(allocationRows([{date: '2026-01-02', validated: {invalid: true}}]), [{date: '2026-01-02', CASH: 1}]);
});

test('cost is input-only: output tokens are free and missing usage is never priced as free', () => {
  const rates = {input: 2, output: 0};
  assert.equal(inputCost({input_tokens: 1e6, output_tokens: 2e6}, rates), 2);
  assert.equal(inputCost({input_tokens: 1e6, output_tokens: 0}, {input: 0, output: 0}), 0);
  for (const invalid of [undefined, {input: 2}, {input: -1, output: 0}, {input: Infinity, output: 0}]) {
    assert.equal(inputCost({input_tokens: 1e6, output_tokens: 0}, invalid), null);
  }
  assert.equal(inputCost({output_tokens: 0}, rates), null);
  assert.equal(inputCost({input_tokens: 1e308, output_tokens: 1e308}, {input: 1e308, output: 1e308}), null);
  assert.equal(inputCost({input_tokens: 1e6, output_tokens: 1e6}, PRICES['gpt-6-luna']), 0.1);
  assert.equal(inputCost({input_tokens: 1e6, output_tokens: 1e6}, PRICES['clef']), 0.24);
  // A recorded run rate wins over the published list price; a junk row falls back to it.
  const meta = {decision: {pricing: {cost_per_m_input: 1, cost_per_m_output: 0}}};
  assert.equal(ratesFor('clef-flash', meta).input, 1);
  assert.equal(ratesFor('clef-flash', {decision: {pricing: {}}}).input, 0.038);
  assert.equal(ratesFor('clef-flash', undefined).input, 0.038);
});

test('the pricing table covers the three providers and four decision models, input only', () => {
  assert.deepEqual(Object.keys(PRICES).sort(), ['clef', 'clef-flash', 'gpt-6-luna', 'jev-latest']);
  const expected = {
    'gpt-6-luna': {provider: 'openai_decisions', input: 0.10, source: 'https://community.openai.com/t/decisions-api-is-now-available-in-public-beta/1403877'},
    'jev-latest': {provider: 'typesafe', input: 0.042, source: 'https://docs.typesafe.ai/models'},
    'clef': {provider: 'cloudflare', input: 0.24, source: 'https://developers.cloudflare.com/workers-ai/models/clef/index.md'},
    'clef-flash': {provider: 'cloudflare', input: 0.038, source: 'https://developers.cloudflare.com/workers-ai/models/clef/index.md'}
  };
  for (const [model, spec] of Object.entries(expected)) {
    assert.equal(PRICES[model].provider, spec.provider);
    assert.equal(PRICES[model].input, spec.input);
    assert.equal(PRICES[model].output, 0, `${model} must bill no output tokens`);
    assert.match(PRICES[model].source, /^https:\/\//);
  }
  assert.match(PRICES['gpt-6-luna'].source, /community\.openai\.com/);
  assert.match(PRICES['jev-latest'].source, /docs\.typesafe\.ai\/models/);
  for (const model of ['clef', 'clef-flash']) assert.match(PRICES[model].source, /developers\.cloudflare\.com/);
  assert.deepEqual(new Set(Object.values(PRICES).map(rate => rate.provider)),
    new Set(['openai_decisions', 'typesafe', 'cloudflare']));
});

test('catalog validation accepts the DCN payload and reports real problems', () => {
  assert.deepEqual(catalogProblems(catalog([catalogRun()])), []);
  assert.deepEqual(catalogProblems(undefined), ['catalog is not an object']);
  assert.deepEqual(catalogProblems({schema_version: 2, default_trust: 'all', runs: []}),
    ['schema_version must be 1']);
  assert.deepEqual(catalogProblems({schema_version: 1, runs: []}), ['default_trust must be all']);
  assert.deepEqual(catalogProblems({schema_version: 1, default_trust: 'all'}), ['runs must be an array']);
  const broken = catalog([catalogRun({run_id: 'x', dir: '../secret', model: '', provider: '', metrics: {}})]);
  assert.equal(broken.runs[0].dir, '../secret');
  const problems = catalogProblems(broken);
  assert.ok(problems.some(problem => problem.includes('bad artifact path')));
  assert.ok(problems.some(problem => problem.includes('no decision model')));
  assert.ok(problems.some(problem => problem.includes('no provider')));
  assert.ok(problems.some(problem => problem.includes('no total return')));
  assert.deepEqual(catalogProblems(catalog([])), []);
});

test('the frontend uses safe text rendering, keeps the design, and never mentions agents', () => {
  const js = readFileSync('dashboard/app.js', 'utf8');
  const css = readFileSync('dashboard/styles.css', 'utf8');
  const pages = Object.fromEntries(['index.html', 'run.html', 'pricing.html', 'method.html', 'contribute.html']
    .map(name => [name, readFileSync(`dashboard/${name}`, 'utf8')]));
  const html = Object.values(pages).join('\n');
  assert.doesNotMatch(js, /innerHTML|outerHTML|document\.write|eval\(/);
  assert.match(js, /textContent/);
  assert.match(pages['index.html'], /aria-live="polite"/);
  assert.match(pages['index.html'], /styles\.css/);
  assert.match(pages['index.html'], /app\.js/);
  // Design language kept: the same dark palette and panel classes.
  assert.match(css, /--accent:#b4f272/);
  assert.match(css, /\.panel\{/);
  assert.match(html, /class="eyebrow"/);
  assert.match(html, /class="brandmark"/);
  for (const [name, text] of [...Object.entries(pages), ['app.js', js], ['styles.css', css]]) {
    for (const word of ['agent', 'pipeline', 'critic', 'reasoning effort', 'telemetry', 'tool call', 'tool_call']) {
      assert.doesNotMatch(text.toLowerCase(), new RegExp(word), `${name} mentions ${word}`);
    }
  }
  // Every widget the script binds to exists somewhere in the site, and the
  // reliability chart and per-run statistics live in the script.
  for (const anchor of ['leaderboard', 'calibration', 'reliability', 'reliability-bins',
    'pricing-table', 'model-costs', 'run-cost', 'performance', 'drawdown', 'allocations']) {
    assert.match(html, new RegExp(`id="${anchor}"`), anchor);
  }
  assert.match(html, /Reliability/);
  assert.match(html, /Brier/);
  assert.match(js, /reliabilityChart/);
  assert.match(js, /excess_return_ci/);
  assert.match(js, /auc_p_value/);
});

test('the committed public catalog is a valid DCN catalog', () => {
  const catalog = JSON.parse(readFileSync('dashboard/data/index.json', 'utf8'));
  assert.deepEqual(catalogProblems(catalog), []);
  for (const run of catalog.runs) assert.equal(run.dir, `runs/${run.run_id}`);
});

// A tiny stand-in for the DOM so the whole render path can run under `node --test`
// without a browser. Fetch is served from memory: no network, no files.
function fakeElement(tag) {
  const element = {
    tagName: tag, attrs: {}, children: [], style: {}, className: "", _text: "", href: "", title: "",
    hidden: false, id: "",
    append(...nodes) {this.children.push(...nodes);},
    replaceChildren(...nodes) {this.children = [...nodes];},
    setAttribute(key, value) {this.attrs[key] = String(value);},
    getAttribute(key) {return this.attrs[key] ?? null;},
    set textContent(value) {this._text = String(value);},
    get textContent() {return this._text;},
    descendants() {
      const out = [];
      const walk = node => {out.push(node); for (const child of node.children || []) walk(child);};
      walk(this);
      return out;
    },
  };
  // The page tags rows with classList.add, so the stand-in needs one. It stays
  // consistent with className because tests read both.
  const classes = new Set();
  const sync = () => {element.className = [...classes].join(" ");};
  element.classList = {
    add(...names) {for (const name of names) if (name) classes.add(String(name)); sync();},
    remove(...names) {for (const name of names) classes.delete(String(name)); sync();},
    contains(name) {return classes.has(String(name));},
    toggle(name) {classes.has(String(name)) ? classes.delete(String(name)) : classes.add(String(name)); sync(); return classes.has(String(name));},
  };
  return element;
}

// The site is several pages sharing one script, so each page's element ids are
// listed separately: a page only gets the nodes its own HTML declares, which is
// what the renderers guard on. Kept in sync with the dashboard/*.html files.
const PAGE_IDS = {
  'index.html': ['leaderboard-table', 'board-window', 'run-count', 'demo-banner', 'status', 'hero-scope', 'period-tabs'],
  'run.html': ['run-details', 'run-trust', 'run-title', 'run-meta', 'status', 'metrics', 'calibration', 'reliability',
    'reliability-bins', 'reliability-bins-caption', 'performance', 'drawdown', 'allocations',
    'run-cost', 'price-source', 'price-note', 'coverage', 'settings', 'downloads'],
  'pricing.html': ['cost-overview', 'cost-window', 'status', 'pricing-table', 'model-costs'],
};

// Stand up one page against an in-memory site: no network, no files, no browser.
function mountSite(site, page = 'index.html', search = '') {
  const elements = new Map();
  for (const id of PAGE_IDS[page] || []) {const element = fakeElement('div'); element.id = id; elements.set(id, element);}
  global.document = {
    // Absent ids are genuinely absent: that is how the script knows which page
    // it is on and which widgets to render.
    getElementById(id) {return elements.get(id) || null;},
    createElement: fakeElement,
    createElementNS: (_namespace, tag) => fakeElement(tag),
  };
  global.window = {addEventListener() {}};
  global.location = {hash: '', search};
  global.fetch = async path => ({
    ok: path in site, status: path in site ? 200 : 404,
    text: async () => {if (!(path in site)) throw new Error(`missing ${path}`); return site[path];},
  });
  delete require.cache[require.resolve('../dashboard/app.js')];
  require('../dashboard/app.js');
  return elements;
}

const settle = async () => {for (let i = 0; i < 50; i++) await new Promise(resolve => setTimeout(resolve, 0));};

// Two runs, artifacts present for the first: enough to exercise the board, the
// run page and the pricing page against one payload.
function siteFixture() {
  const first = catalogRun();
  const second = catalogRun({run_id: 'run-b', dir: 'runs/run-b', model: 'clef', provider: 'cloudflare',
    question_form: 'noul', created_utc: '2026-10-08T09:00:00+00:00',
    actions: {buy: 0.25, sell: 0.23, hold: 0.52, avg_buys: 6.2, avg_sells: 5.8, avg_holds: 13.0}});
  const meta = {
    run_id: first.run_id, scenario: {name: '2026-ytd'},
    decision: {pricing: {cost_per_m_input: 0.09, cost_per_m_output: 0}},
    data: {snapshot_id: first.snapshot_id},
  };
  const site = {
    'data/index.json': JSON.stringify({schema_version: 1, default_trust: 'all', selected: null, runs: [first, second]}),
    [`data/${first.dir}/run.json`]: JSON.stringify(meta),
    [`data/${first.dir}/metrics.json`]: JSON.stringify(first.metrics),
    [`data/${first.dir}/equity_curve.csv`]:
      'date,portfolio,spy_buy_hold,equal_weight,sixty_forty,momentum_12_1\n'
      + '2026-01-02,100,100,100,100,100\n2026-02-02,105,101,102,101,99\n',
    [`data/${first.dir}/decisions.jsonl`]: JSON.stringify(
      {date: '2026-01-02', validated: {weights: {AAPL: 0.5}, cash: 0.5, invalid: false}}) + '\n',
  };
  return {site, first, second};
}

test('the leaderboard page renders the board and links each row to its run page', async () => {
  const {site, first} = siteFixture();
  const elements = mountSite(site, 'index.html');
  await settle();
  const $ = id => elements.get(id);
  assert.equal($('run-count').textContent, '2 results');
  assert.equal($('status').textContent, '');
  assert.equal($('demo-banner').hidden, true, 'real results must not show the synthetic warning');
  // One scored window: the toggle stays hidden and the board is unchanged.
  assert.equal($('period-tabs').hidden, true);
  const table = $('leaderboard-table').children[0];
  assert.equal(table.descendants().filter(node => node.tagName === 'tr').length, 3);
  // The board carries the headline metrics only, named in plain words: the
  // statistical scores need a statistics background, so they stay off it.
  const headers = table.children[0].descendants()
    .filter(node => node.tagName === 'th')
    .map(node => String(node.children[0]?.textContent ?? node.textContent).replace(/ [↓↑]$/, ''));
  assert.deepEqual(headers, ['Rank', 'Model', 'Return', 'on $10K', 'vs SPY', 'Weeks beating SPY', 'Worst drop', 'Holds', 'Right calls']);
  // The plain-word columns render the recorded values for the leading row.
  const firstRowCells = table.descendants().filter(node => node.tagName === 'tr')[1]
    .children.map(node => node.textContent);
  assert.equal(firstRowCells[5], '44.00%', 'the share of weeks the book beat SPY');
  assert.equal(firstRowCells[6], '-7.00%', 'the worst peak-to-trough drop');
  assert.equal(firstRowCells[7], '91.00%', 'the hold share of the book');
  assert.equal(firstRowCells[8], '55.00%', 'the share of calls that proved right');
  // The caption teaches the two columns a first read cannot guess, with the
  // no-skill line taken from the catalog.
  assert.match($('board-window').textContent, /Holds is the share of names left untraded each week/);
  assert.match($('board-window').textContent, /always calling the same way scores 48\.00% here/);
  // A row is a real link to the run page, so it opens in a new tab and shares well.
  const runLink = table.descendants().find(node => node.className === 'run-button');
  assert.equal(runLink.tagName, 'a');
  assert.equal(runLink.href, `run.html?run=${first.run_id}`);
  // Widgets that belong to other pages are absent here, and the script must cope.
  assert.equal($('run-details'), undefined);
  assert.equal($('pricing-table'), undefined);
});

test('scored windows group runs by frozen window and average gaps per model', () => {
  assert.equal(monthLabel('2026-04-09'), 'Apr 26');
  const gap = value => ({metrics: {...catalogRun().metrics, excess_return_vs_spy: value}});
  const rows = [
    {...catalogRun({run_id: 'w1-flash'}), start: '2026-01-02', end: '2026-06-30', ...gap(0.001)},
    {...catalogRun({run_id: 'w1-clef', model: 'clef'}), start: '2026-01-02', end: '2026-06-30', ...gap(-0.01)},
    // Same display model through a second route: the window keeps one row.
    {...catalogRun({run_id: 'w1-clef-older', model: 'openrouter-clef', created_utc: '2026-10-01T00:00:00+00:00'}),
      start: '2026-01-02', end: '2026-06-30', ...gap(0.9)},
    {...catalogRun({run_id: 'w2-flash'}), start: '2026-07-01', end: '2026-10-07', ...gap(0.002)},
    {...catalogRun({run_id: 'w2-clef', model: 'clef'}), start: '2026-07-01', end: '2026-10-07', ...gap(0.05)},
    {...catalogRun({run_id: 'w2-jev', model: 'jev-latest'}), start: '2026-07-01', end: '2026-10-07', ...gap(0.01)},
  ];
  const windows = scoredWindows(rows);
  assert.deepEqual(windows.map(w => [w.label, w.runs.length]), [['Jan 26 - Jun 26', 2], ['Jul 26 - Oct 26', 3]]);
  // Inside a window the newest execution of a display model wins, never the
  // better score: openrouter-clef ran first, so clef's row is w1-clef (-0.01).
  assert.deepEqual(windows[0].runs.map(run => run.run_id).sort(), ['w1-clef', 'w1-flash']);
  const aggregate = allPeriodRows(windows);
  // The default order is the average gap with windows beaten as the tiebreak:
  // clef's fatter average leads even though flash won both its windows.
  assert.deepEqual(aggregate.map(row => [modelName(row.model), row.windows, row.beatIn, Math.round(row.avgGap * 1e6) / 1e6]),
    [['clef', 2, 1, 0.02], ['jev-latest', 1, 1, 0.01], ['clef-flash', 2, 2, 0.0015]]);
  const clef = aggregate.find(row => modelName(row.model) === 'clef');
  assert.equal(clef.best.gap, 0.05);
  assert.equal(clef.best.window, 'Jul 26 - Oct 26');
  assert.equal(clef.worst.gap, -0.01);
  assert.equal(clef.beatIn, 1);
  assert.equal(clef.latest.run_id, 'w2-clef');
});

test('the leaderboard toggles between all-window averages and one window at a time', async () => {
  const first = catalogRun({run_id: 'w1-flash'});
  const clef = catalogRun({run_id: 'w1-clef', model: 'clef'});
  const flash2 = catalogRun({run_id: 'w2-flash', created_utc: '2026-10-09T09:00:00+00:00'});
  const clef2 = catalogRun({run_id: 'w2-clef', model: 'clef', created_utc: '2026-10-09T09:00:00+00:00'});
  for (const [run, start, end, gap] of [[first, '2026-01-02', '2026-06-30', 0.02], [clef, '2026-01-02', '2026-06-30', -0.01],
                                        [flash2, '2026-07-01', '2026-10-07', -0.03], [clef2, '2026-07-01', '2026-10-07', 0.05]]) {
    run.start = start; run.end = end;
    run.metrics = {...run.metrics, excess_return_vs_spy: gap, spy_total_return: 0.10};
  }
  const site = {'data/index.json': JSON.stringify({schema_version: 1, default_trust: 'all', selected: null,
    runs: [first, clef, flash2, clef2]})};
  const elements = mountSite(site, 'index.html');
  await settle();
  const $ = id => elements.get(id);
  const headers = () => $('leaderboard-table').children[0].children[0].descendants()
    .filter(node => node.tagName === 'th')
    .map(node => String(node.children[0]?.textContent ?? node.textContent).replace(/ [↓↑]$/, ''));
  // Two windows: the toggle appears and the default view averages them.
  assert.equal($('period-tabs').hidden, false);
  assert.deepEqual(headers(), ['Rank', 'Model', 'Windows', 'Avg vs SPY', 'Best window', 'Worst window', 'Beat SPY in']);
  assert.equal($('run-count').textContent, '2 models · 2 windows');
  const rows = $('leaderboard-table').children[0].children[1].children;
  assert.equal(rows[0].descendants().find(node => node.className === 'run-button').textContent, 'Clef', 'averages, not best runs, order the all-windows view');
  assert.equal(rows[0].children[2].textContent, '2/2');
  assert.equal(rows[0].children[3].children[0].textContent, '▲ 2.00%');
  assert.equal(rows[0].children[6].textContent, '1 of 2');
  // A window chip narrows the board to that window, plain columns and all.
  const chips = $('period-tabs').children;
  assert.equal(chips.length, 3);
  chips[2].onclick();
  assert.deepEqual(headers(), ['Rank', 'Model', 'Return', 'on $10K', 'vs SPY', 'Weeks beating SPY', 'Worst drop', 'Holds', 'Right calls']);
  assert.equal($('run-count').textContent, '2 results');
  assert.match($('board-window').textContent, /Jul 26 - Oct 26 window, SPY 10\.00%\./);
});

test('per-window rank numbers follow the displayed vs-SPY order', async () => {
  const make = (id, model, created, gap) => {
    const run = catalogRun({run_id: id, model, created_utc: created});
    run.metrics = {...run.metrics, excess_return_vs_spy: gap, spy_total_return: 0.10};
    run.start = '2026-01-02'; run.end = '2026-06-30';
    return run;
  };
  // Created newest-first so the old execution-time ranks would disagree with
  // the vs-SPY display order.
  const site = {'data/index.json': JSON.stringify({schema_version: 1, default_trust: 'all', selected: null,
    runs: [make('late', 'clef', '2026-10-09T09:00:00+00:00', -0.02), make('early', 'clef-flash', '2026-10-01T09:00:00+00:00', 0.03)]})};
  const elements = mountSite(site, 'index.html');
  await settle();
  const rows = elements.get('leaderboard-table').children[0].children[1].children;
  assert.equal(rows[0].children[0].textContent, '1');
  assert.equal(rows[0].descendants().find(node => node.className === 'run-button').textContent, 'Clef-Flash');
  assert.equal(rows[1].children[0].textContent, '2');
});

test('the run page loads one run from the query string', async () => {
  const {site, first} = siteFixture();
  const elements = mountSite(site, 'run.html', `?run=${encodeURIComponent(first.run_id)}`);
  await settle();
  const $ = id => elements.get(id);
  assert.equal($('run-details').hidden, false, `run details shown (status: ${$('status').textContent})`);
  assert.match($('run-title').textContent, /Clef-Flash/);
  // A replay still earns its pill; a locally executed run is every real run,
  // so its pill stays empty and hidden.
  assert.equal($('run-trust').textContent, 'Replay checked');
  assert.equal($('run-trust').hidden, false);
  const local = siteFixture();
  local.site['data/index.json'] = JSON.stringify({schema_version: 1, default_trust: 'all', selected: null,
    runs: [local.first, local.second].map(run => ({...run, trust: 'local'}))});
  const localElements = mountSite(local.site, 'run.html', `?run=${encodeURIComponent(local.first.run_id)}`);
  await settle();
  assert.equal(localElements.get('run-trust').textContent, '');
  assert.equal(localElements.get('run-trust').hidden, true);
  // The run summary: sixteen tiles, each with the line that reads it.
  assert.equal($('metrics').children.length, 16);
  const metricText = $('metrics').descendants().map(node => node.textContent).join(' ');
  for (const label of ['Excess vs SPY', 'Excess vs pool', 'Turnover per decision', 'Weeks beating SPY',
                       'Log loss', 'ECE (noul)', 'Answer coverage', '95% interval', 'permutation p =']) {
    assert.match(metricText, new RegExp(label.replace(/[()]/g, '\\$&')));
  }
  assert.equal($('reliability').children[0].descendants().filter(node => node.tagName === 'svg').length, 1);
  const circles = $('reliability').children[0].descendants().filter(node => node.tagName === 'circle');
  assert.equal(circles.length, 4);
  // Geometry: bins land inside the plot box, x rises with the prediction, y rises
  // with the observation, and the y = x diagonal is drawn corner to corner.
  for (const circle of circles) {
    assert.ok(Number(circle.attrs.cx) >= 56 && Number(circle.attrs.cx) <= 538, circle.attrs.cx);
    assert.ok(Number(circle.attrs.cy) >= 22 && Number(circle.attrs.cy) <= 334, circle.attrs.cy);
  }
  const [firstNoul] = circles;  // noul bin: predicted 0.55, observed 0.67
  const diagonalAtCx = 22 + (1 - (Number(firstNoul.attrs.cx) - 56) / 482) * 312;
  assert.ok(Number(firstNoul.attrs.cy) < diagonalAtCx, 'an under-confident bin sits above the diagonal');
  const diagonal = $('reliability').children[0].descendants().find(node => node.attrs["stroke-dasharray"] === '5 5');
  assert.deepEqual([diagonal.attrs.x1, diagonal.attrs.y1, diagonal.attrs.x2, diagonal.attrs.y2], ['56', '334', '538', '22']);
  assert.equal($('performance').children[0].descendants().filter(node => node.tagName === 'polyline').length, 5);
  assert.match($('run-cost').textContent, /Input-only estimate \$0\.000900 · 1000 input tokens · 0 output tokens · 9 requests/);
  assert.match($('coverage').textContent, /9 observations over 3 horizons, labeled against realized benchmark-relative outcomes/);
  assert.equal($('reliability-bins').children[0].descendants().filter(node => node.tagName === 'tr').length, 4);
  assert.match($('settings').textContent, /"decision"/);
  assert.equal($('downloads').children.length, 6);
  // No board on this page: the script must not require it.
  assert.equal($('leaderboard-table'), undefined);
});

test('the pricing page renders the standardized table', async () => {
  const {site} = siteFixture();
  const elements = mountSite(site, 'pricing.html');
  await settle();
  const $ = id => elements.get(id);
  assert.equal($('cost-overview').hidden, false);
  const pricing = $('pricing-table').children[0].descendants();
  const pricingRows = pricing.filter(node => node.tagName === 'tr');
  assert.equal(pricingRows.length, 3, 'header + one row per model on the board');
  assert.deepEqual(pricingRows[0].children.map(cell => cell.textContent),
    ['Model', 'Input $ / 1M', 'Tokens / decision', 'Cost / decision', 'Run cost', 'Source']);
  // The model cell carries the creator's logo plus the display name.
  const cellText = row => row.children[0].descendants()
    .map(child => (typeof child === "string" ? child : child.textContent)).join("");
  assert.deepEqual(pricingRows.slice(1).map(row => cellText(row)).sort(), ['Clef', 'Clef-Flash']);
  for (const row of pricingRows.slice(1)) {
    assert.match(row.children[1].textContent, /^\$\d+\.\d{3}$/);   // list rate per 1M input tokens
    assert.equal(row.children[2].textContent, '111');              // 1000 billed tokens over 9 decisions
    assert.equal(row.children[3].textContent, '$0.000100');        // $0.0009 spread over 9 decisions
    assert.equal(row.children[4].textContent, '$0.000900');        // the run's own recorded cost
  }
  assert.ok(pricing.filter(node => node.tagName === 'a').every(link => link.href.startsWith('https://')));
  assert.match($('cost-window').textContent, /bill it once per question/);
  assert.equal($('model-costs').children.length, 2, 'one bar per model');
  assert.equal($('leaderboard-table'), undefined);
});

// A DOM `hidden` property is invisible to a fake DOM, so this checks the
// stylesheet. `.pill.warn` sets `display:block`, which overrides the browser's
// own `[hidden]` rule; without an explicit `[hidden]{display:none}` the warning
// banner rendered even when hidden and labelled real results as fabricated.
test('the stylesheet makes hidden elements actually hidden', () => {
  const css = readFileSync(require.resolve('../dashboard/styles.css'), 'utf8');
  const override = css.match(/\[hidden\]\s*\{[^}]*\}/);
  assert.ok(override, 'styles.css must define a [hidden] rule');
  assert.match(override[0], /display\s*:\s*none/, '[hidden] must set display:none');
  assert.ok(
    override[0].includes('!important'),
    '[hidden] needs !important to beat a class that sets its own display'
  );
  // The specific collision that caused the bug must still be covered.
  assert.match(css, /\.pill\.warn\s*\{[^}]*display\s*:\s*block/);
});

// A fabricated run must be labelled before it is clicked. This regression exists
// because the trust badge originally lived only inside the click-through details
// panel, so the leaderboard showed demo numbers with no warning at all.
test('a synthetic run is disclosed on the leaderboard itself, before any click', async () => {
  // One real run plus two demo runs. Every model must be distinct: the board
  // keeps one result per decision model, so duplicate models collapse into one row.
  const real = catalogRun();
  const fakeJev = catalogRun({
    run_id: '20261008-163612_2022-bear_jev-latest_demo1', dir: 'runs/20261008-163612_2022-bear_jev-latest_demo1',
    model: 'jev-latest', provider: 'typesafe', synthetic: true, trust: 'synthetic', label: 'demo',
    created_utc: '2026-10-08T16:36:12+00:00',
  });
  const fakeLuna = catalogRun({
    run_id: '20261008-163612_2022-bear_gpt-6-luna_demo0', dir: 'runs/20261008-163612_2022-bear_gpt-6-luna_demo0',
    model: 'gpt-6-luna', provider: 'openai_decisions', synthetic: true, trust: 'synthetic', label: 'demo',
    created_utc: '2026-10-08T16:36:12+00:00',
  });
  const elements = mountSite({
    'data/index.json': JSON.stringify({
      schema_version: 1, default_trust: 'all', selected: null, runs: [real, fakeJev, fakeLuna],
    }),
  });
  await settle();
  const $ = id => elements.get(id);
  assert.equal($('run-count').textContent, '3 results');
  const banner = $('demo-banner');
  assert.equal(banner.hidden, false, 'the warning must be visible on load');
  assert.match(banner.textContent, /Synthetic demo data/);
  assert.match(banner.textContent, /2 of 3 rows/);
  assert.match(banner.textContent, /not real results/);

  const rows = $('leaderboard-table').children[0].descendants().filter(node => node.tagName === 'tr');
  assert.equal(rows.length, 4);  // header plus one row per model
  assert.equal(rows.filter(row => row.className.includes('synthetic')).length, 2);
  const chips = rows.flatMap(row => row.descendants()).filter(node => node.className === 'synthetic-chip');
  assert.equal(chips.length, 2);
  assert.equal(chips[0].textContent, 'synthetic');
  assert.match(chips[0].title, /fabricated by/);
});

test('an all-synthetic leaderboard says so unambiguously', async () => {
  const elements = mountSite({
    'data/index.json': JSON.stringify({
      schema_version: 1, default_trust: 'all', selected: null,
      runs: [
        catalogRun({synthetic: true, trust: 'synthetic'}),
        catalogRun({run_id: 'b', dir: 'runs/b', model: 'clef', synthetic: true, trust: 'synthetic'}),
      ],
    }),
  });
  await settle();
  const $ = id => elements.get(id);
  assert.equal($('run-count').textContent, '2 results');
  assert.match($('demo-banner').textContent, /all 2 rows are fabricated/);
  assert.match($('demo-banner').textContent, /No decision model was called/);
  // The result count is styled as a warning too, not a neutral pill.
  assert.equal($('run-count').className, 'pill warn');
});