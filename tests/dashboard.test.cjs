const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {
  parseCSV, leaderboardRuns, modelName, allocationRows, percent, dollars, count, plural, ratio, score, inputCost,
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
    metrics: {total_return: 0.05, spy_total_return: 0.01, excess_return_vs_spy: 0.04, sharpe: -0.98,
      max_drawdown: -0.07, input_tokens: 1000, output_tokens: 0, requests: 9, answer_coverage: 1.0,
      mean_latency_s: 0.4, decisions: 9, invalid_outputs: 0, estimated_cost_usd: 0.0009,
      baselines: {equal_weight: 0.02, sixty_forty: 0.01, momentum_12_1: -0.03},
      calibration: {observations: 9, horizons: 3, coverage: 1.0, miscalibration_gap: 0.355, signals: {
        noul: {n: 9, brier: 0.2, brier_skill_score: -0.17, auc: 0.71, expected_calibration_error: 0.2,
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
  assert.equal(percent(0.1234), '12.34%');
  assert.equal(percent(undefined), '—');
  // An input-only decision-model run costs fractions of a cent, so tiny values keep more digits.
  assert.equal(dollars(0.000135), '$0.000135');
  assert.equal(dollars(0.5), '$0.5000');
  assert.equal(dollars(undefined), '—');
  assert.equal(ratio(-0.9822), '-0.98');
  assert.equal(score(0.2), '0.200');
  assert.equal(count(9), '9');
  assert.equal(count(undefined), '—');
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
  assert.equal(ratesFor('clef-flash', {decision: {pricing: {}}}).input, 0.09);
  assert.equal(ratesFor('clef-flash', undefined).input, 0.09);
});

test('the pricing table covers the three providers and four decision models, input only', () => {
  assert.deepEqual(Object.keys(PRICES).sort(), ['clef', 'clef-flash', 'gpt-6-luna', 'jev-latest']);
  const expected = {
    'gpt-6-luna': {provider: 'openai_decisions', input: 0.10, source: 'https://community.openai.com/t/decisions-api-is-now-available-in-public-beta/1403877'},
    'jev-latest': {provider: 'typesafe', input: 0.042, source: 'https://docs.typesafe.ai/models'},
    'clef': {provider: 'cloudflare', input: 0.24, source: 'https://developers.cloudflare.com/workers-ai/models/clef/index.md'},
    'clef-flash': {provider: 'cloudflare', input: 0.09, source: 'https://developers.cloudflare.com/workers-ai/models/clef/index.md'}
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
  const html = readFileSync('dashboard/index.html', 'utf8');
  const css = readFileSync('dashboard/styles.css', 'utf8');
  assert.doesNotMatch(js, /innerHTML|outerHTML|document\.write|eval\(/);
  assert.match(js, /textContent/);
  assert.match(html, /aria-live="polite"/);
  assert.match(html, /styles\.css/);
  assert.match(html, /app\.js/);
  // Design language kept: the same dark palette and panel classes.
  assert.match(css, /--accent:#b4f272/);
  assert.match(css, /\.panel\{/);
  assert.match(html, /class="eyebrow"/);
  assert.match(html, /class="brandmark"/);
  for (const [name, text] of [['index.html', html], ['app.js', js], ['styles.css', css]]) {
    for (const word of ['agent', 'pipeline', 'critic', 'reasoning effort', 'telemetry', 'tool call', 'tool_call']) {
      assert.doesNotMatch(text.toLowerCase(), new RegExp(word), `${name} mentions ${word}`);
    }
  }
  // The reliability section and its callout are part of the site.
  for (const anchor of ['leaderboard', 'calibration', 'reliability', 'reliability-bins', 'gap-callout',
    'pricing-table', 'model-costs', 'run-cost', 'performance', 'drawdown', 'allocations']) {
    assert.match(html, new RegExp(`id="${anchor}"`), anchor);
  }
  assert.match(html, /Reliability/);
  assert.match(html, /Brier/);
  // The miscalibration callout and the reliability chart live in the script.
  assert.match(js, /miscalibration_gap/);
  assert.match(js, /reliabilityChart/);
  assert.match(js, /gap-callout/);
});

test('the committed public catalog is a valid DCN catalog', () => {
  const catalog = JSON.parse(readFileSync('dashboard/data/index.json', 'utf8'));
  assert.deepEqual(catalogProblems(catalog), []);
  for (const run of catalog.runs) assert.equal(run.dir, `runs/${run.run_id}`);
});

// A tiny stand-in for the DOM so the whole render path can run under `node --test`
// without a browser. Fetch is served from memory: no network, no files.
function fakeElement(tag) {
  return {
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
}

test('the payload actually drives the page: leaderboard, equity, reliability and the gap callout', async () => {
  const first = catalogRun();
  const second = catalogRun({run_id: 'run-b', dir: 'runs/run-b', model: 'clef', provider: 'cloudflare',
    question_form: 'noul', created_utc: '2026-10-08T09:00:00+00:00'});
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
  const elements = new Map();
  global.document = {
    getElementById(id) {
      if (!elements.has(id)) {const element = fakeElement('div'); element.id = id; elements.set(id, element);}
      return elements.get(id);
    },
    createElement: fakeElement,
    createElementNS: (_namespace, tag) => fakeElement(tag),
  };
  global.window = {addEventListener() {}};
  global.location = {hash: ''};
  global.fetch = async path => ({
    ok: path in site, status: path in site ? 200 : 404,
    text: async () => {if (!(path in site)) throw new Error(`missing ${path}`); return site[path];},
  });
  delete require.cache[require.resolve('../dashboard/app.js')];
  require('../dashboard/app.js');
  const settle = async () => {for (let i = 0; i < 50; i++) await new Promise(resolve => setTimeout(resolve, 0));};
  await settle();
  const $ = id => elements.get(id);
  assert.equal($('run-count').textContent, '2 results');
  assert.equal($('status').textContent, '');
  const table = $('leaderboard-table').children[0];
  assert.equal(table.descendants().filter(node => node.tagName === 'tr').length, 3);
  assert.equal($('cost-overview').hidden, false);
  // The pricing panel is static: three providers, input-only, with cited sources.
  const pricing = $('pricing-table').children[0].descendants();
  assert.deepEqual(pricing.filter(node => node.tagName === 'tr' && node.children.some(cell => cell.textContent === 'gpt-6-luna')).length, 1);
  assert.ok(pricing.filter(node => node.tagName === 'a').every(link => link.href.startsWith('https://')));
  const runButton = table.descendants().find(node => node.className === 'run-button');
  runButton.onclick();  // sets the hash
  runButton.onclick();  // selects the run
  await settle();
  assert.equal($('run-details').hidden, false, `run details shown (status: ${$('status').textContent})`);
  assert.equal($('metrics').children.length, 6);
  assert.match($('gap-callout').descendants().map(node => node.textContent).join(' '), /35\.50% Paired miscalibration gap:/);
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
  assert.match($('coverage').textContent, /Scored 9 observations over 3 horizons · label coverage 100\.00%/);
  assert.equal($('reliability-bins').children[0].descendants().filter(node => node.tagName === 'tr').length, 4);
  assert.match($('settings').textContent, /"decision"/);
  assert.equal($('downloads').children.length, 6);
});