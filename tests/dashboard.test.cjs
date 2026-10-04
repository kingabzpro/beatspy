const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {parseCSV, leaderboardRuns, modelName, allocationRow, percent} = require('../dashboard/app.js');

test('CSV supports CRLF, quotes, escaped quotes, and embedded newlines', () => {
  assert.deepEqual(parseCSV('date,note\r\n2026-10-02,"a, b"\r\n2026-10-01,"a ""quote""\nand newline"\r\n'), [
    {date: '2026-10-02', note: 'a, b'}, {date: '2026-10-01', note: 'a "quote"\nand newline'}
  ]);
  assert.deepEqual(parseCSV('date,price\n'), []);
  assert.throws(() => parseCSV('date,price\n"broken'), /Malformed CSV/);
  assert.throws(() => parseCSV('date,price\n1,2,3'), /column count/);
});
test('one leaderboard merges reasoning modes, selects latest models, and excludes other windows', () => {
  const rows = [
    {model:'a', start:'2026-07-05',end:'2026-10-02',scenario:'2026-recent',created_utc:'1',group:'a',metrics:{excess_return_vs_spy:.09}},
    {model:'b', start:'2025-10-03',end:'2025-12-31',scenario:'2025-recent',created_utc:'3',metrics:{excess_return_vs_spy:.9}},
    {model:'a', start:'2026-07-05',end:'2026-10-02',scenario:'2026-recent',created_utc:'2',group:'b',metrics:{excess_return_vs_spy:.12}},
    {model:'c', start:'2026-07-05',end:'2026-10-02',scenario:'2026-recent',created_utc:'2',group:'c',metrics:{excess_return_vs_spy:.08}}
  ];
  assert.deepEqual(leaderboardRuns(rows).map(x => x.metrics.excess_return_vs_spy), [.12,.08]);
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

test('All years averages latest yearly windows once per model and exposes coverage', () => {
  const run = (model, year, created, value, extra = {}) => ({model, year, run_id:`${model}-${year}-${created}`,
    start:`${year}-10-03`,end:`${year}-12-31`,scenario:`${year}-recent`,created_utc:created,trust:'maintainer',
    metrics:{total_return:value,spy_total_return:.02,excess_return_vs_spy:value-.02,cost:null,...extra}});
  const rows = [run('a','2025','1',.9),run('a','2025','2',.1),run('a','2026','1',.3,{cost:4}),
    run('b','2026','1',.15)];
  const averages = leaderboardRuns(rows, true);
  assert.deepEqual(averages.map(row => row.model), ['a','b']);
  assert.equal(averages[0].metrics.total_return, .2);
  assert.ok(Math.abs(averages[0].metrics.excess_return_vs_spy - .18) < 1e-12);
  assert.equal(averages[0].metrics.spy_total_return, .02);
  assert.equal(averages[0].metrics.cost, 4);
  assert.deepEqual(averages[0].members.map(row => row.run_id), ['a-2025-2','a-2026-1']);
  assert.deepEqual(averages[1].members.map(row => row.year), ['2026']);
  assert.equal(averages[1].metrics.cost, null);
  assert.deepEqual(leaderboardRuns([], true), []);
  rows[2].trust = 'community';
  assert.equal(leaderboardRuns(rows, true)[0].trust, 'mixed');
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
  assert.equal(catalog.default_trust, 'maintainer');
  assert.ok(Array.isArray(catalog.runs));
  for (const run of catalog.runs) {
    assert.equal(run.synthetic, false);
    assert.ok(['maintainer', 'community'].includes(run.trust));
    assert.equal(run.dir, `runs/${run.run_id}`);
  }
});
