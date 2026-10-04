const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {parseCSV, comparableRuns, percent} = require('../dashboard/app.js');

test('CSV supports CRLF, quotes, escaped quotes, and embedded newlines', () => {
  assert.deepEqual(parseCSV('date,note\r\n2026-10-02,"a, b"\r\n2026-10-01,"a ""quote""\nand newline"\r\n'), [
    {date: '2026-10-02', note: 'a, b'}, {date: '2026-10-01', note: 'a "quote"\nand newline'}
  ]);
  assert.deepEqual(parseCSV('date,price\n'), []);
  assert.throws(() => parseCSV('date,price\n"broken'), /Malformed CSV/);
  assert.throws(() => parseCSV('date,price\n1,2,3'), /column count/);
});
test('ranking excludes incomparable windows and sorts numerically', () => {
  const rows = [
    {group:'a', metrics:{excess_return_vs_spy:.09}},
    {group:'b', metrics:{excess_return_vs_spy:.9}},
    {group:'a', metrics:{excess_return_vs_spy:.12}}
  ];
  assert.deepEqual(comparableRuns(rows, 'a').map(x => x.metrics.excess_return_vs_spy), [.12,.09]);
  assert.equal(percent(.1234), '12.34%');
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
