/*
 * RemoteGuard — synthetic session generator
 * =========================================
 * PURPOSE: validate that the pipeline runs end to end (collector schema ->
 * features -> training -> evaluation) BEFORE any real phone data exists.
 *
 * ⚠ THIS IS NOT EVIDENCE. Numbers trained on synthetic sessions say nothing
 * about whether the method works; they only say the plumbing is connected.
 * Every reported metric must come from real collected sessions. The generator
 * exists so that a broken CSV column or a leaking split is caught on day one
 * instead of the night before submission.
 *
 * The four conditions encode the actual hypotheses under test:
 *   handheld    (0) real finger, phone in hand   -> sway + tremor + coupling
 *   desk        (0) real finger, phone on table  -> real touch properties,
 *                                                   flat IMU  ← the false-positive case
 *   remote_scrcpy (1) injected touches, phone still -> degenerate pressure,
 *                                                   straight paths, frame-quantised timing
 *   remote_a11y   (1) injected touches, phone HELD  -> sway present but NO
 *                                                   touch/IMU coupling ← the hard case
 *
 * usage: node pipeline/synth.mjs --out data/raw --subjects 12 --per 6
 */
import fs from 'node:fs';
import path from 'node:path';

// ---- deterministic RNG so runs are reproducible ----
function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
let rng = mulberry32(20260816);
const U = (a, b) => a + rng() * (b - a);
const N = (mu, sd) => {
  const u = Math.max(1e-9, rng()), v = rng();
  return mu + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

const G = 9.80665;

// ---------------------------------------------------------------- strokes
/* A human reach: bell-shaped primary movement along a curved path plus one or
   two corrective sub-movements near the target (Fitts / Meyer). An injected
   gesture: linear interpolation between two points at a fixed step count. */
function humanStroke(t0, from, to, prof) {
  const evs = [];
  const dist = Math.hypot(to.x - from.x, to.y - from.y);
  const dur = clamp(180 + dist * N(1.5, 0.3), 160, 900);
  const nSteps = Math.max(6, Math.round(dur / N(prof.frameMs, 2)));
  // curved via a perpendicular offset control point
  const mx = (from.x + to.x) / 2, my = (from.y + to.y) / 2;
  const nx = -(to.y - from.y), ny = (to.x - from.x);
  const nl = Math.hypot(nx, ny) || 1;
  const bow = N(0, dist * 0.12);
  const cx = mx + nx / nl * bow, cy = my + ny / nl * bow;

  let t = t0;
  for (let i = 0; i <= nSteps; i++) {
    let u = i / nSteps;
    // minimum-jerk velocity profile
    u = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5;
    const x = (1 - u) ** 2 * from.x + 2 * (1 - u) * u * cx + u * u * to.x;
    const y = (1 - u) ** 2 * from.y + 2 * (1 - u) * u * cy + u * u * to.y;
    evs.push({
      t: +t.toFixed(2), type: i === 0 ? 'touchstart' : (i === nSteps ? 'touchend' : 'touchmove'),
      x: +(x + N(0, 0.6)).toFixed(3), y: +(y + N(0, 0.6)).toFixed(3),
      force: +clamp(N(prof.force, prof.forceSd), 0.05, 0.95).toFixed(4),
      rx: +Math.max(2, N(prof.radius, prof.radiusSd)).toFixed(2),
      ry: +Math.max(2, N(prof.radius, prof.radiusSd)).toFixed(2),
      ptype: 'touch', id: 0
    });
    t += dur / nSteps + N(0, prof.jitterMs);
  }
  // corrective sub-movements around the target
  const nCorr = rng() < 0.75 ? 1 + Math.floor(rng() * 2) : 0;
  for (let c = 0; c < nCorr; c++) {
    for (let i = 0; i < 3; i++) {
      t += U(12, 26);
      evs.push({
        t: +t.toFixed(2), type: 'touchmove',
        x: +(to.x + N(0, 3.5)).toFixed(3), y: +(to.y + N(0, 3.5)).toFixed(3),
        force: +clamp(N(prof.force, prof.forceSd), 0.05, 0.95).toFixed(4),
        rx: +Math.max(2, N(prof.radius, prof.radiusSd)).toFixed(2),
        ry: +Math.max(2, N(prof.radius, prof.radiusSd)).toFixed(2),
        ptype: 'touch', id: 0
      });
    }
  }
  evs[evs.length - 1].type = 'touchend';
  return { events: evs, tEnd: t };
}

function injectedStroke(t0, from, to, prof) {
  const evs = [];
  const nSteps = prof.fixedSteps;             // dispatchGesture / scrcpy: fixed interpolation
  const step = prof.frameMs;                  // quantised to the encoder frame rate
  let t = t0;
  for (let i = 0; i <= nSteps; i++) {
    const u = i / nSteps;                     // strictly linear
    const x = Math.round(from.x + (to.x - from.x) * u);   // integer device pixels
    const y = Math.round(from.y + (to.y - from.y) * u);
    evs.push({
      t: +t.toFixed(2), type: i === 0 ? 'touchstart' : (i === nSteps ? 'touchend' : 'touchmove'),
      x, y,
      force: prof.force,                      // constant, usually exactly 1
      rx: prof.radius, ry: prof.radius,       // constant, usually 0
      ptype: 'touch', id: 0
    });
    // frame-rate quantisation + occasional RTT stall
    t += step + (rng() < 0.06 ? U(60, 220) : N(0, 0.8));
  }
  return { events: evs, tEnd: t };
}

/* ADAPTIVE ATTACKER.
   Assume the attacker has read our paper. Everything a remote tool controls in
   software, it can randomise: pressure, touch radius, sub-pixel coordinates,
   path curvature, inter-event timing, even corrective sub-movements. So the
   evasive profile reuses the human stroke generator outright.
   What it cannot do is move the handset. The coupling between touch instants
   and the accelerometer is generated by physics at the far end of the wire, and
   there is no API on the attacker's side that produces it. This condition is
   the whole justification for keeping the IMU layer: under evasion the P/G/T
   layers are supposed to collapse, and if M does not hold the line here, the
   layer is decoration and should be cut. */
function evasiveStroke(t0, from, to, prof) {
  return humanStroke(t0, from, to, prof);
}

// ---------------------------------------------------------------- motion
/* Handheld: postural sway (~1 Hz) + physiological tremor (~9-11 Hz) + a short
   impulse at each touch-down. Resting on a surface: no sway, no tremor, but a
   touch on a table-mounted phone still transmits an impulse. */
function motionStream(t0, t1, fs, touchTimes, cfg) {
  const out = [];
  const dt = 1000 / fs;
  const swayF = U(0.7, 1.6), tremF = U(8.5, 11.5);
  const swayPh = U(0, 6.28), tremPh = U(0, 6.28);
  for (let t = t0; t <= t1; t += dt + N(0, cfg.sampleJitter)) {
    let a = 0;
    a += cfg.sway * Math.sin(2 * Math.PI * swayF * t / 1000 + swayPh);
    a += cfg.tremor * Math.sin(2 * Math.PI * tremF * t / 1000 + tremPh);
    a += N(0, cfg.noise);
    if (cfg.impulse > 0) {
      for (const tt of touchTimes) {
        const d = t - tt;
        if (d >= -10 && d < 90) a += cfg.impulse * Math.exp(-d / 22) * Math.sin(2 * Math.PI * 28 * d / 1000);
      }
    }
    // gravity split across axes with a slowly drifting tilt
    const tilt = 0.25 + 0.05 * Math.sin(2 * Math.PI * 0.3 * t / 1000);
    out.push({
      t: +t.toFixed(2), type: 'motion',
      ax: +(a * 0.5 + G * Math.sin(tilt)).toFixed(4),
      ay: +(a * 0.4 + N(0, cfg.noise)).toFixed(4),
      az: +(a * 0.6 + G * Math.cos(tilt)).toFixed(4),
      gx: +(cfg.gyro * Math.sin(2 * Math.PI * swayF * t / 1000) + N(0, cfg.gyroNoise)).toFixed(3),
      gy: +(cfg.gyro * 0.7 * Math.cos(2 * Math.PI * swayF * t / 1000) + N(0, cfg.gyroNoise)).toFixed(3),
      gz: +N(0, cfg.gyroNoise).toFixed(3),
      interval: +dt.toFixed(2)
    });
  }
  return out;
}

// ---------------------------------------------------------------- profiles
const CONDITIONS = {
  handheld: {
    label: 0, injected: false,
    touch: { force: 0.42, forceSd: 0.13, radius: 14, radiusSd: 4.5, frameMs: 16, jitterMs: 3.2 },
    motion: { sway: 0.16, tremor: 0.045, noise: 0.02, impulse: 0.35, gyro: 3.2, gyroNoise: 0.5, sampleJitter: 1.2 }
  },
  desk: {   // real finger, phone lying on a surface -> the false-positive case
    label: 0, injected: false,
    touch: { force: 0.44, forceSd: 0.12, radius: 15, radiusSd: 4.2, frameMs: 16, jitterMs: 3.0 },
    motion: { sway: 0.004, tremor: 0.002, noise: 0.012, impulse: 0.22, gyro: 0.15, gyroNoise: 0.12, sampleJitter: 0.9 }
  },
  remote_scrcpy: {
    label: 1, injected: true,
    touch: { force: 1, radius: 0, frameMs: 16.67, fixedSteps: 12 },
    motion: { sway: 0.003, tremor: 0.001, noise: 0.011, impulse: 0, gyro: 0.1, gyroNoise: 0.1, sampleJitter: 0.9 }
  },
  remote_a11y: {   // victim holds the phone, operator drives it -> sway but no coupling
    label: 1, injected: true,
    touch: { force: 1, radius: 0, frameMs: 20, fixedSteps: 8 },
    motion: { sway: 0.15, tremor: 0.042, noise: 0.02, impulse: 0, gyro: 3.0, gyroNoise: 0.5, sampleJitter: 1.2 }
  },
  remote_evasive: { // adaptive attacker: forges every software-side signal
    label: 1, injected: true, evasive: true,
    touch: { force: 0.43, forceSd: 0.13, radius: 14, radiusSd: 4.4, frameMs: 16, jitterMs: 3.1 },
    motion: { sway: 0.15, tremor: 0.043, noise: 0.02, impulse: 0, gyro: 3.1, gyroNoise: 0.5, sampleJitter: 1.2 }
  }
};

// per-subject device variation: sample rate and sensor gain differ across phones
function subjectProfile(sid) {
  const r = mulberry32(1000 + sid * 7717);
  return {
    fs: [60, 60, 50, 30, 60, 25][Math.floor(r() * 6)],
    gain: 0.7 + r() * 0.8,      // sensor scale factor
    speed: 0.8 + r() * 0.5      // how fast this person moves
  };
}

function makeSession(sid, cond, trial) {
  const C = CONDITIONS[cond];
  const sp = subjectProfile(sid);
  const W = 390, H = 780;
  const targets = [
    { x: 195, y: 240 }, { x: 195, y: 330 }, { x: 100, y: 520 }, { x: 195, y: 520 },
    { x: 290, y: 520 }, { x: 100, y: 590 }, { x: 195, y: 660 }, { x: 195, y: 720 }
  ];

  let t = 0, from = { x: U(50, 340), y: U(600, 760) };
  const inputs = [];
  const nActions = 14 + Math.floor(rng() * 8);
  for (let i = 0; i < nActions; i++) {
    const to = targets[Math.floor(rng() * targets.length)];
    const prof = Object.assign({}, C.touch);
    if (!C.injected || C.evasive) { prof.frameMs = C.touch.frameMs / sp.speed; }
    const s = C.evasive ? evasiveStroke(t, from, to, prof)
            : C.injected ? injectedStroke(t, from, to, prof)
            : humanStroke(t, from, to, prof);
    inputs.push(...s.events);
    // an evasive operator also randomises dwell time between actions
    t = s.tEnd + ((C.injected && !C.evasive) ? U(150, 600) : U(280, 1400));
    from = to;
  }

  const touchDowns = inputs.filter(e => e.type === 'touchstart').map(e => e.t);
  const mcfg = Object.assign({}, C.motion);
  for (const k of ['sway', 'tremor', 'noise', 'impulse', 'gyro']) mcfg[k] *= sp.gain;
  const motion = motionStream(0, t + 500, sp.fs, touchDowns, mcfg);

  const events = [...inputs, ...motion].sort((a, b) => a.t - b.t);
  return {
    schema: 'remoteguard.session.v1',
    session_id: `synth_s${String(sid).padStart(2, '0')}_${cond}_${trial}`,
    subject_id: `synth_s${String(sid).padStart(2, '0')}`,
    trial, condition: cond, label: C.label,
    synthetic: true,
    collected_at: new Date().toISOString(),
    duration_ms: t + 500,
    device: { ua: 'synthetic', dpr: 3, screen: { w: W, h: H }, touch_points: 5 },
    diagnostics: { motion_hz: sp.fs, clock_skew_mean_ms: 0, clock_skew_sd_ms: 0,
                   motion_samples: motion.length, input_events: inputs.length },
    events
  };
}

// ---------------------------------------------------------------- main
const args = process.argv.slice(2);
const get = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const outDir = get('out', 'data/raw');
const nSubjects = +get('subjects', 12);
const perCond = +get('per', 6);

fs.mkdirSync(outDir, { recursive: true });
let n = 0;
for (let s = 1; s <= nSubjects; s++) {
  for (const cond of Object.keys(CONDITIONS)) {
    for (let k = 1; k <= perCond; k++) {
      const sess = makeSession(s, cond, k);
      fs.writeFileSync(path.join(outDir, sess.session_id + '.json'), JSON.stringify(sess));
      n++;
    }
  }
}
console.log(`wrote ${n} synthetic sessions to ${outDir}`);
console.log('REMINDER: synthetic data validates the pipeline only. Never report these numbers.');
