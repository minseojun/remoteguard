/*
 * RemoteGuard — physical presence challenge
 * -----------------------------------------
 * L4. When the session looks risky we do not block; we ask the person to pick
 * the phone up and shake it. A remote operator cannot move the handset.
 *
 * Threat model, stated honestly:
 *   - defeats  : fully remote / unattended operation (the "victim asleep at
 *                3am" pattern, which is where the large losses sit)
 *   - degrades : coached fraud, where the caller tells the victim to shake.
 *                The challenge only adds friction there; the cooling delay and
 *                situational prompt carry that case, not this.
 *   - assumes  : the page itself is not attacker-controlled. A client that can
 *                rewrite our JS can forge any trace. That is a native-SDK
 *                problem, not a web one, and we say so rather than pretending.
 *
 * Verification is a signal test, not a boolean. A constant/replayed trace, a
 * table tap, or a single jolt all fail.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.RGChallenge = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var G = 9.80665;

  var THRESHOLDS = {
    minDurationMs: 600,
    maxDurationMs: 6000,
    minSamples: 20,
    minPeakDev: 3.0,      // m/s^2 above gravity — a deliberate shake, not a tap
    minReversals: 4,      // direction changes: shaking is oscillatory
    minDomFreq: 1.5,      // Hz
    maxDomFreq: 8.0,
    minGyroRms: 20.0,     // deg/s — the wrist rotates; a desk-tap does not
    minAccelStd: 1.2
  };

  function mean(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return a.length ? s / a.length : 0; }
  function std(a) {
    if (a.length < 2) return 0;
    var m = mean(a), s = 0;
    for (var i = 0; i < a.length; i++) s += (a[i] - m) * (a[i] - m);
    return Math.sqrt(s / (a.length - 1));
  }

  /* Summarise a raw motion trace into 8 numbers. Runs on-device: the raw trace
     never leaves the phone, only this summary does. */
  function summarize(samples) {
    if (!samples || samples.length < 2) {
      return { n: 0, duration_ms: 0, peak_dev: 0, accel_std: 0, gyro_rms: 0, reversals: 0, dom_freq: 0, flat_ratio: 1 };
    }
    var t0 = samples[0].t, t1 = samples[samples.length - 1].t;
    var dur = t1 - t0;
    var fs = dur > 0 ? (samples.length - 1) / (dur / 1000) : 0;

    var dev = [], gyro = [];
    for (var i = 0; i < samples.length; i++) {
      var s = samples[i];
      dev.push(Math.hypot(s.ax || 0, s.ay || 0, s.az || 0) - G);
      gyro.push(Math.hypot(s.gx || 0, s.gy || 0, s.gz || 0));
    }

    var peak = 0;
    for (var j = 0; j < dev.length; j++) peak = Math.max(peak, Math.abs(dev[j]));

    // zero crossings of the mean-removed signal -> oscillation count
    var m = mean(dev), rev = 0;
    for (var k = 1; k < dev.length; k++) {
      if ((dev[k] - m) * (dev[k - 1] - m) < 0) rev++;
    }
    var domFreq = dur > 0 ? (rev / 2) / (dur / 1000) : 0;

    // a replayed constant / synthetic trace has near-zero local variation
    var diffs = [];
    for (var q = 1; q < dev.length; q++) diffs.push(Math.abs(dev[q] - dev[q - 1]));
    var flat = diffs.filter(function (v) { return v < 1e-4; }).length / (diffs.length || 1);

    return {
      n: samples.length,
      duration_ms: dur,
      sample_rate: fs,
      peak_dev: peak,
      accel_std: std(dev),
      gyro_rms: Math.sqrt(mean(gyro.map(function (v) { return v * v; }))),
      reversals: rev,
      dom_freq: domFreq,
      flat_ratio: flat
    };
  }

  function verify(summary, th) {
    th = Object.assign({}, THRESHOLDS, th || {});
    var s = summary || {};
    var checks = [
      ['samples', (s.n || 0) >= th.minSamples, 'not enough motion samples'],
      ['duration', (s.duration_ms || 0) >= th.minDurationMs && (s.duration_ms || 0) <= th.maxDurationMs, 'shake too short or too long'],
      ['amplitude', (s.peak_dev || 0) >= th.minPeakDev, 'movement too weak'],
      ['variation', (s.accel_std || 0) >= th.minAccelStd, 'movement not sustained'],
      ['oscillation', (s.reversals || 0) >= th.minReversals, 'a single jolt, not a shake'],
      ['frequency', (s.dom_freq || 0) >= th.minDomFreq && (s.dom_freq || 0) <= th.maxDomFreq, 'shake rhythm out of range'],
      ['rotation', (s.gyro_rms || 0) >= th.minGyroRms, 'device did not rotate — was it lifted?'],
      ['not_replayed', (s.flat_ratio == null ? 0 : s.flat_ratio) < 0.5, 'trace looks synthetic']
    ];
    var failed = checks.filter(function (c) { return !c[1]; });
    return {
      passed: failed.length === 0,
      failed: failed.map(function (c) { return { check: c[0], reason: c[2] }; }),
      checks: checks.map(function (c) { return { check: c[0], ok: c[1] }; })
    };
  }

  return { THRESHOLDS: THRESHOLDS, summarize: summarize, verify: verify };
});
