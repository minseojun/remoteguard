/*
 * RemoteGuard — feature extraction driver
 * ---------------------------------------
 * data/raw/*.json  ->  data/out/windows.csv
 *
 * Deliberately calls shared/features.js, the same module the browser SDK
 * loads. Training and inference therefore cannot drift apart; if this file
 * ever grows its own feature logic, the guarantee is gone.
 *
 * usage: node pipeline/extract.mjs [--in data/raw] [--out data/out/windows.csv]
 */
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const RG = require('../shared/features.js');

const args = process.argv.slice(2);
const get = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const inDir = get('in', 'data/raw');
const outFile = get('out', 'data/out/windows.csv');

const files = fs.readdirSync(inDir).filter(f => f.endsWith('.json'));
if (!files.length) {
  console.error(`no sessions in ${inDir}. run: node pipeline/synth.mjs  (or collect real ones)`);
  process.exit(1);
}

const META = ['session_id', 'subject_id', 'condition', 'label', 'synthetic',
              'device_hz', 'clock_skew_ms', 'win_start_ms', 'n_input', 'n_motion'];
const rows = [];
const perSession = [];
let skipped = 0;

for (const f of files) {
  let sess;
  try { sess = JSON.parse(fs.readFileSync(path.join(inDir, f), 'utf8')); }
  catch (e) { console.warn(`  ! unreadable, skipped: ${f}`); skipped++; continue; }

  if (!Array.isArray(sess.events) || !sess.events.length) { skipped++; continue; }
  if (sess.label !== 0 && sess.label !== 1) {
    console.warn(`  ! missing label, skipped: ${f}`); skipped++; continue;
  }

  const wins = RG.extractWindows(sess.events);
  perSession.push({ id: sess.session_id, cond: sess.condition, n: wins.length });
  if (!wins.length) continue;

  for (const w of wins) {
    const row = {
      session_id: sess.session_id,
      subject_id: sess.subject_id || 'unknown',
      condition: sess.condition,
      label: sess.label,
      synthetic: sess.synthetic ? 1 : 0,
      device_hz: (sess.diagnostics && sess.diagnostics.motion_hz) || 0,
      clock_skew_ms: (sess.diagnostics && sess.diagnostics.clock_skew_mean_ms) || 0,
      win_start_ms: w.__t_start,
      n_input: w.__n_input,
      n_motion: w.__n_motion
    };
    for (const name of RG.FEATURE_NAMES) row[name] = w[name];
    rows.push(row);
  }
}

const cols = [...META, ...RG.FEATURE_NAMES];
const csv = [cols.join(',')];
for (const r of rows) {
  csv.push(cols.map(c => {
    const v = r[c];
    return typeof v === 'number' ? (Number.isFinite(v) ? +v.toPrecision(8) : 0) : v;
  }).join(','));
}
fs.mkdirSync(path.dirname(outFile), { recursive: true });
fs.writeFileSync(outFile, csv.join('\n'));

const byCond = {};
for (const r of rows) byCond[r.condition] = (byCond[r.condition] || 0) + 1;
const thin = perSession.filter(s => s.n < 3);

console.log(`sessions read : ${files.length}  (skipped ${skipped})`);
console.log(`windows       : ${rows.length}`);
console.log(`features      : ${RG.FEATURE_NAMES.length}`);
for (const [k, v] of Object.entries(byCond)) console.log(`  ${k.padEnd(16)} ${v}`);
if (thin.length) {
  console.log(`\n${thin.length} session(s) produced <3 windows — too short to be useful:`);
  thin.slice(0, 8).forEach(s => console.log(`  ${s.id} (${s.n})`));
}
console.log(`\nwrote ${outFile}`);
