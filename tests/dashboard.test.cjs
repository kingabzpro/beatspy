const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {parseCSV, leaderboardRuns, modelName, allocationRow, percent, tokenCost, usageBreakdown, PRICES} = require('../dashboard/app.js');

test('CSV supports CRLF, quotes, escaped quotes, and embedded newlines', () => {
  assert.deepEqual(parseCSV('date,note\r\n2026-10-02,"a, b"\r\n2026-10-01,"a ""quote""\nand newline"\r\n'), [
    {date: '2026-10-02', note: 'a, b'}, {date: '2026-10-01', note: 'a "quote"\nand newline'}
  ]);
  assert.deepEqual(parseCSV('date,price\n'), []);
  assert.throws(() => parseCSV('date,price\n"broken'), /Malformed CSV/);
  assert.throws(() => parseCSV('date,price\n1,2,3'), /column count/);
});
test('one leaderboard merges reasoning modes, selects latest results independently for each model', () => {
  const rows = [
    {model:'a', start:'2026-07-05',end:'2026-10-02',scenario:'2026-recent',created_utc:'1',group:'a',metrics:{excess_return_vs_spy:.09}},
    {model:'b', start:'2025-10-03',end:'2025-12-31',scenario:'2025-recent',created_utc:'3',metrics:{excess_return_vs_spy:.9}},
    {model:'a', start:'2026-07-05',end:'2026-10-02',scenario:'2026-recent',created_utc:'2',group:'b',metrics:{excess_return_vs_spy:.12}},
    {model:'c', start:'2026-07-05',end:'2026-10-02',scenario:'2026-recent',created_utc:'2',group:'c',metrics:{excess_return_vs_spy:.08}}
  ];
  assert.deepEqual(leaderboardRuns(rows).map(x => x.metrics.excess_return_vs_spy), [.9,.12,.08]);
  assert.equal(modelName('zai-org/GLM-5.3'), 'GLM-5.3');
  assert.equal(modelName('gpt-6-luna'), 'gpt-6-luna');
  assert.equal(percent(.1234), '12.34%');
});
test('invalid decisions display retained holdings instead of a false cash liquidation', () => {
  assert.deepEqual(allocationRow({date:'2025-11-28',validated:{invalid:true,weights:{},cash:1},
    market_brief:{current_portfolio_weights:{SPY:.3,TLT:.2},current_cash_weight:.5}}),
    {date:'2025-11-28',SPY:.3,TLT:.2,CASH:.5});
  assert.deepEqual(allocationRow({date:'2025-10-03',validated:{invalid:false,weights:{SPY:.35},cash:.65}}),
    {date:'2025-10-03',SPY:.35,CASH:.65});
});

test('latest results are chosen by execution time, never best score or cutoff', () => {
  const rows = [
    {model:'a', run_id:'old', created_utc:'2026-10-01', end:'2026-10-02', metrics:{excess_return_vs_spy:.9}},
    {model:'a', run_id:'new', created_utc:'2026-10-04', end:'2025-12-31', metrics:{excess_return_vs_spy:.1}},
    {model:'b', run_id:'b', created_utc:'2026-10-02', end:'2026-10-02', metrics:{excess_return_vs_spy:.2}}
  ];
  assert.deepEqual(leaderboardRuns(rows).map(row => row.run_id), ['b','new']);
  assert.deepEqual(leaderboardRuns([]), []);
  const html = readFileSync('dashboard/index.html', 'utf8');
  assert.doesNotMatch(html, /id="(trust|year|scenario)"|class="filters"|Maintainer/);
});
test('frontend uses safe text rendering and independent static files', () => {
  const js = readFileSync('dashboard/app.js', 'utf8');
  const html = readFileSync('dashboard/index.html', 'utf8');
  assert.doesNotMatch(js, /innerHTML|outerHTML|document\.write|eval\(/);
  assert.match(js, /textContent/);
  assert.match(html, /aria-live="polite"/);
  assert.match(html, /styles\.css/);
  const catalog = JSON.parse(readFileSync('dashboard/data/index.json', 'utf8'));
  assert.equal(catalog.schema_version, 1);
  assert.equal(catalog.default_trust, 'all');
  assert.ok(Array.isArray(catalog.runs));
  for (const run of catalog.runs) {
    assert.equal(run.synthetic, false);
    assert.ok(['verified', 'replayed'].includes(run.trust));
    assert.equal(run.dir, `runs/${run.run_id}`);
  }
});

test('cost uses input and output rates, reconciles agents and decisions, and never prices missing usage as free', () => {
  const rates = {input: 2, output: 8};
  assert.equal(tokenCost({input_tokens:1e6, output_tokens:2e6}, rates), 18);
  assert.equal(tokenCost({input_tokens:1e6, output_tokens:2e6}, {input:0,output:0}), 0);
  for (const invalid of [undefined, {input:2}, {input:-1,output:2}, {input:Infinity,output:2}]) {
    assert.equal(tokenCost({input_tokens:1e6,output_tokens:2e6}, invalid), null);
  }
  assert.equal(tokenCost({input_tokens:100}, rates), null);
  assert.equal(tokenCost({input_tokens:1e308,output_tokens:1e308}, {input:1e308,output:1e308}), null);
  const rows = [{date:'2026-07-31',usage:{analyst:{input_tokens:1e6,output_tokens:0,requests:1},critic:{input_tokens:0,output_tokens:1e6,requests:2}}},
    {date:'2026-08-31',usage:{analyst:{input_tokens:2e6,output_tokens:1e6,requests:3}}}];
  const result = usageBreakdown(rows, rates);
  assert.deepEqual(result.steps.map(step => [step.cost,step.cumulative]), [[10,10],[12,22]]);
  assert.equal(result.agents.reduce((sum,agent) => sum+agent.cost,0), 22);
  assert.deepEqual(result.agents.map(agent => agent.requests), [4,2]);
  assert.equal(usageBreakdown([...rows,{date:'2026-09-30'}],rates).steps.at(-1).cumulative, null);
  assert.equal(usageBreakdown([{date:'2026-09-30',usage:{critic:{input_tokens:20}}}],rates).agents[0].cost, null);
  assert.equal(usageBreakdown(rows,undefined).steps[0].cost, null);
  assert.equal(tokenCost({input_tokens:1e6,output_tokens:1e6},PRICES['gpt-6-luna']), .6);
  assert.equal(tokenCost({input_tokens:1e6,output_tokens:1e6},PRICES['GLM-5.3']), 7.06);
  assert.match(PRICES['GLM-5.3'].source, /^https:\/\/openrouter\.ai\//);
});
