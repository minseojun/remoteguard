/*
 * RemoteGuard — shared feature extractor
 * ---------------------------------------
 * THE single implementation of feature extraction. Runs in two places:
 *   1. the browser SDK  (live scoring; only the feature vector leaves the device)
 *   2. the Node pipeline (offline training on collected raw sessions)
 *
 * Both paths call extractWindows() on the same event schema, so there is no
 * train/serve skew. Do not fork this file.
 *
 * Event schema (one array, time-ordered, t = performance.now() ms):
 *   input : {t, type:'touchstart'|'touchmove'|'touchend'|'mousedown'|'mousemove'|'mouseup'|'pointerdown'|...,
 *            x, y, force, rx, ry, ptype:'touch'|'mouse'|'pen', id}
 *   motion: {t, type:'motion', ax, ay, az, gx, gy, gz, interval}
 *            ax/ay/az = accelerationIncludingGravity (m/s^2), gx/gy/gz = rotationRate (deg/s)
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.RGFeatures = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var WINDOW_MS = 3000;
  var HOP_MS = 1500;
  var G = 9.80665;

  // ---------- small numeric helpers ----------
  function mean(a) { if (!a.length) return 0; var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s / a.length; }
  function variance(a) { if (a.length < 2) return 0; var m = mean(a), s = 0; for (var i = 0; i < a.length; i++) s += (a[i] - m) * (a[i] - m); return s / (a.length - 1); }
  function std(a) { return Math.sqrt(variance(a)); }
  function median(a) {
    if (!a.length) return 0;
    var b = a.slice().sort(function (x, y) { return x - y; });
    var h = b.length >> 1;
    return b.length % 2 ? b[h] : (b[h - 1] + b[h]) / 2;
  }
  function finite(v, d) { return (typeof v === 'number' && isFinite(v)) ? v : d; }

  /* Goertzel band energy: sum of |X(f)|^2 over a frequency band, normalised by
     sample count. Cheaper than a full FFT and works on the short, irregularly
     sampled motion streams browsers actually deliver. */
  function bandEnergy(sig, fs, f0, f1, nBins) {
    if (sig.length < 8 || !(fs > 0)) return 0;
    nBins = nBins || 8;
    var m = mean(sig), N = sig.length, total = 0;
    for (var b = 0; b < nBins; b++) {
      var f = f0 + (f1 - f0) * (b + 0.5) / nBins;
      if (f >= fs / 2) break;
      var w = 2 * Math.PI * f / fs, c = 2 * Math.cos(w);
      var s0 = 0, s1 = 0, s2 = 0;
      for (var n = 0; n < N; n++) {
        s0 = (sig[n] - m) + c * s1 - s2;
        s2 = s1; s1 = s0;
      }
      total += (s1 * s1 + s2 * s2 - c * s1 * s2);
    }
    return total / (N * N);
  }

  function shannonEntropy(vals, nBins) {
    if (vals.length < 4) return 0;
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    if (hi - lo < 1e-9) return 0;
    var counts = new Array(nBins).fill(0);
    for (var i = 0; i < vals.length; i++) {
      var k = Math.min(nBins - 1, Math.floor((vals[i] - lo) / (hi - lo) * nBins));
      counts[k]++;
    }
    var H = 0;
    for (var j = 0; j < nBins; j++) {
      if (!counts[j]) continue;
      var p = counts[j] / vals.length;
      H -= p * Math.log2(p);
    }
    return H / Math.log2(nBins); // normalised 0..1
  }

  function modalFraction(vals, nBins) {
    if (vals.length < 4) return 0;
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    if (hi - lo < 1e-9) return 1;
    var counts = new Array(nBins).fill(0);
    for (var i = 0; i < vals.length; i++) {
      var k = Math.min(nBins - 1, Math.floor((vals[i] - lo) / (hi - lo) * nBins));
      counts[k]++;
    }
    return Math.max.apply(null, counts) / vals.length;
  }

  function isIntegerish(v) { return Math.abs(v - Math.round(v)) < 1e-6; }

  // ---------- the 24 features ----------
  // Grouped by layer so ablation studies can drop a whole group.
  var FEATURE_GROUPS = {
    P: ['p_force_var', 'p_force_uniq_ratio', 'p_force_degenerate', 'p_radius_var',
      'p_radius_missing', 'p_int_snap_frac', 'p_nontouch_ratio'],
    G: ['g_path_efficiency', 'g_curvature_mean', 'g_speed_peaks_per_stroke',
      'g_jerk_norm', 'g_endpoint_correction'],
    T: ['t_iei_cv', 't_iei_entropy', 't_iei_modal_frac', 't_burst_ratio'],
    M: ['m_sample_rate', 'm_sway_energy', 'm_tremor_energy', 'm_still_frac',
      'm_accel_std', 'm_gyro_std', 'm_touch_coupling', 'm_coupling_z']
  };
  var FEATURE_NAMES = [].concat(FEATURE_GROUPS.P, FEATURE_GROUPS.G, FEATURE_GROUPS.T, FEATURE_GROUPS.M);

  function emptyFeatures() {
    var o = {};
    for (var i = 0; i < FEATURE_NAMES.length; i++) o[FEATURE_NAMES[i]] = 0;
    return o;
  }

  // --- P: pointer authenticity -------------------------------------------
  // Synthesised input (scrcpy, AccessibilityService#dispatchGesture, WebDriver)
  // carries default pressure/size and integer coordinates. Real fingers do not.
  function pointerFeatures(inputs, f) {
    var downs = inputs.filter(function (e) { return /down|start/.test(e.type); });
    var touchLike = inputs.filter(function (e) { return e.ptype === 'touch'; });

    f.p_nontouch_ratio = inputs.length ? 1 - touchLike.length / inputs.length : 0;

    var forces = touchLike.map(function (e) { return finite(e.force, -1); })
      .filter(function (v) { return v >= 0; });
    if (forces.length >= 3) {
      f.p_force_var = variance(forces);
      var uniq = {}; var n = 0;
      for (var i = 0; i < forces.length; i++) {
        var k = forces[i].toFixed(4);
        if (!uniq[k]) { uniq[k] = 1; n++; }
      }
      f.p_force_uniq_ratio = n / forces.length;
      // exactly 0 or exactly 1 for (almost) every sample => injected
      var deg = forces.filter(function (v) { return v === 0 || v === 1; }).length;
      f.p_force_degenerate = deg / forces.length;
    } else {
      f.p_force_uniq_ratio = 0;
      f.p_force_degenerate = touchLike.length ? 1 : 0; // no pressure reported at all
    }

    var radii = touchLike.map(function (e) { return finite(e.rx, -1); })
      .filter(function (v) { return v >= 0; });
    f.p_radius_var = radii.length >= 3 ? variance(radii) : 0;
    f.p_radius_missing = touchLike.length
      ? touchLike.filter(function (e) { return !finite(e.rx, 0); }).length / touchLike.length : 0;

    if (inputs.length) {
      var snapped = 0;
      for (var j = 0; j < inputs.length; j++) {
        if (isIntegerish(inputs[j].x) && isIntegerish(inputs[j].y)) snapped++;
      }
      f.p_int_snap_frac = snapped / inputs.length;
    }
    return downs;
  }

  // --- G: trajectory geometry --------------------------------------------
  // Fitts-style human movement: curved approach + corrective sub-movements
  // near the target. Injected gestures interpolate on a straight line.
  function strokes(inputs) {
    var out = [], cur = null;
    for (var i = 0; i < inputs.length; i++) {
      var e = inputs[i];
      if (/down|start/.test(e.type)) { cur = [e]; }
      else if (/move/.test(e.type)) { if (cur) cur.push(e); }
      else if (/up|end/.test(e.type)) { if (cur) { cur.push(e); if (cur.length >= 4) out.push(cur); cur = null; } }
    }
    if (cur && cur.length >= 4) out.push(cur);
    return out;
  }

  function geometryFeatures(inputs, f) {
    var ss = strokes(inputs);
    if (!ss.length) { f.g_path_efficiency = 1; return; }
    var effs = [], curvs = [], peaks = [], jerks = [], corrs = [];

    for (var s = 0; s < ss.length; s++) {
      var p = ss[s], pathLen = 0, speeds = [], angles = [];
      for (var i = 1; i < p.length; i++) {
        var dx = p[i].x - p[i - 1].x, dy = p[i].y - p[i - 1].y;
        var d = Math.hypot(dx, dy), dt = Math.max(1, p[i].t - p[i - 1].t);
        pathLen += d;
        speeds.push(d / dt);
        if (d > 0.5) angles.push(Math.atan2(dy, dx));
      }
      var straight = Math.hypot(p[p.length - 1].x - p[0].x, p[p.length - 1].y - p[0].y);
      if (pathLen > 1) effs.push(Math.min(1, straight / pathLen));

      // mean absolute heading change per step = curvature proxy
      var turn = 0, tn = 0;
      for (var a = 1; a < angles.length; a++) {
        var d2 = angles[a] - angles[a - 1];
        while (d2 > Math.PI) d2 -= 2 * Math.PI;
        while (d2 < -Math.PI) d2 += 2 * Math.PI;
        turn += Math.abs(d2); tn++;
      }
      if (tn) curvs.push(turn / tn);

      // speed-profile local maxima => corrective sub-movements
      var pk = 0;
      for (var k = 1; k < speeds.length - 1; k++) {
        if (speeds[k] > speeds[k - 1] && speeds[k] >= speeds[k + 1] && speeds[k] > 0.05) pk++;
      }
      peaks.push(pk);

      if (speeds.length >= 3) {
        var jr = 0;
        for (var m = 1; m < speeds.length; m++) jr += Math.abs(speeds[m] - speeds[m - 1]);
        jerks.push(jr / (speeds.length - 1) / (mean(speeds) + 1e-6));
      }

      // path length spent inside the final 10% of the straight-line distance
      if (straight > 5) {
        var tail = 0;
        for (var q = 1; q < p.length; q++) {
          var distToEnd = Math.hypot(p[p.length - 1].x - p[q].x, p[p.length - 1].y - p[q].y);
          if (distToEnd < 0.1 * straight) tail += Math.hypot(p[q].x - p[q - 1].x, p[q].y - p[q - 1].y);
        }
        corrs.push(tail / pathLen);
      }
    }
    f.g_path_efficiency = effs.length ? mean(effs) : 1;
    f.g_curvature_mean = curvs.length ? mean(curvs) : 0;
    f.g_speed_peaks_per_stroke = peaks.length ? mean(peaks) : 0;
    f.g_jerk_norm = jerks.length ? mean(jerks) : 0;
    f.g_endpoint_correction = corrs.length ? mean(corrs) : 0;
  }

  // --- T: timing structure ------------------------------------------------
  // Remote streaming quantises events to the encoder frame rate and stalls on
  // network RTT: low interval entropy + heavy tail.
  function timingFeatures(inputs, f) {
    var moves = inputs.filter(function (e) { return /move/.test(e.type); });
    if (moves.length < 6) return;
    var iei = [];
    for (var i = 1; i < moves.length; i++) {
      var d = moves[i].t - moves[i - 1].t;
      if (d > 0 && d < 500) iei.push(d);
    }
    if (iei.length < 5) return;
    var mu = mean(iei), md = median(iei);
    f.t_iei_cv = mu > 0 ? std(iei) / mu : 0;
    f.t_iei_entropy = shannonEntropy(iei, 16);
    f.t_iei_modal_frac = modalFraction(iei, 16);
    f.t_burst_ratio = iei.filter(function (v) { return v > 3 * md; }).length / iei.length;
  }

  // --- M: device motion ---------------------------------------------------
  // A phone held in a hand is never still: postural sway (0.5-2 Hz) and
  // physiological tremor (8-12 Hz) are always present. m_touch_coupling asks a
  // sharper question: does the IMU react at the moments the screen is touched?
  // m_coupling_z compares that against a permutation null, so it is comparable
  // across devices with different sensor noise floors.
  function motionFeatures(motion, downs, winMs, f) {
    if (motion.length < 8) { f.m_sample_rate = 0; f.m_still_frac = 1; return; }
    var span = motion[motion.length - 1].t - motion[0].t;
    var fs = span > 0 ? (motion.length - 1) / (span / 1000) : 0;
    f.m_sample_rate = fs;

    var mag = [], gyro = [];
    for (var i = 0; i < motion.length; i++) {
      var m = motion[i];
      mag.push(Math.hypot(finite(m.ax, 0), finite(m.ay, 0), finite(m.az, G)) - G);
      gyro.push(Math.hypot(finite(m.gx, 0), finite(m.gy, 0), finite(m.gz, 0)));
    }
    f.m_accel_std = std(mag);
    f.m_gyro_std = std(gyro);
    f.m_sway_energy = bandEnergy(mag, fs, 0.5, 2.0, 6);
    f.m_tremor_energy = bandEnergy(mag, fs, 8.0, 12.0, 6);

    var mm = median(mag.map(Math.abs));
    f.m_still_frac = mag.filter(function (v) { return Math.abs(v) < Math.max(0.02, 0.5 * mm); }).length / mag.length;

    // --- touch/IMU coupling -------------------------------------------------
    if (!downs.length) { f.m_touch_coupling = 0; f.m_coupling_z = 0; return; }
    var HALF = 150; // ms around each touch-down
    var absMag = mag.map(Math.abs);
    var baseline = mean(absMag) + 1e-9;

    function couplingAt(times) {
      var hits = [];
      for (var k = 0; k < times.length; k++) {
        var t0 = times[k] - HALF, t1 = times[k] + HALF, acc = [];
        for (var j = 0; j < motion.length; j++) {
          if (motion[j].t >= t0 && motion[j].t <= t1) acc.push(absMag[j]);
        }
        if (acc.length) hits.push(mean(acc));
      }
      return hits.length ? mean(hits) / baseline : 0;
    }

    var realTimes = downs.map(function (e) { return e.t; });
    var observed = couplingAt(realTimes);
    f.m_touch_coupling = observed;

    // permutation null: same number of events, random times inside the window
    var t0w = motion[0].t, t1w = motion[motion.length - 1].t;
    var nulls = [];
    var rng = mulberry32(Math.floor(t0w) >>> 0);
    for (var p = 0; p < 40; p++) {
      var fake = [];
      for (var q = 0; q < realTimes.length; q++) fake.push(t0w + rng() * (t1w - t0w));
      nulls.push(couplingAt(fake));
    }
    var ns = std(nulls);
    f.m_coupling_z = ns > 1e-9 ? (observed - mean(nulls)) / ns : 0;
  }

  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  // ---------- window driver ----------
  function featuresFor(events) {
    var f = emptyFeatures();
    var inputs = [], motion = [];
    for (var i = 0; i < events.length; i++) {
      if (events[i].type === 'motion') motion.push(events[i]); else inputs.push(events[i]);
    }
    var downs = pointerFeatures(inputs, f);
    geometryFeatures(inputs, f);
    timingFeatures(inputs, f);
    motionFeatures(motion, downs, WINDOW_MS, f);
    for (var k in f) if (!isFinite(f[k])) f[k] = 0;
    return f;
  }

  /* Slice a session into overlapping windows. Windows with no user input at
     all are dropped — scoring an idle screen tells us nothing and would flood
     the training set with trivial negatives. */
  function extractWindows(events, opts) {
    opts = opts || {};
    var winMs = opts.windowMs || WINDOW_MS, hopMs = opts.hopMs || HOP_MS;
    var minInputs = opts.minInputs == null ? 3 : opts.minInputs;
    if (!events.length) return [];
    var sorted = events.slice().sort(function (a, b) { return a.t - b.t; });
    var t0 = sorted[0].t, tEnd = sorted[sorted.length - 1].t;
    var out = [];
    for (var start = t0; start + winMs <= tEnd + hopMs; start += hopMs) {
      var end = start + winMs;
      var slice = sorted.filter(function (e) { return e.t >= start && e.t < end; });
      var nIn = slice.filter(function (e) { return e.type !== 'motion'; }).length;
      if (nIn < minInputs) continue;
      var f = featuresFor(slice);
      f.__t_start = start - t0;
      f.__n_input = nIn;
      f.__n_motion = slice.length - nIn;
      out.push(f);
    }
    return out;
  }

  function toVector(f) {
    var v = [];
    for (var i = 0; i < FEATURE_NAMES.length; i++) v.push(finite(f[FEATURE_NAMES[i]], 0));
    return v;
  }

  return {
    FEATURE_NAMES: FEATURE_NAMES,
    FEATURE_GROUPS: FEATURE_GROUPS,
    WINDOW_MS: WINDOW_MS,
    HOP_MS: HOP_MS,
    featuresFor: featuresFor,
    extractWindows: extractWindows,
    toVector: toVector,
    _internal: { bandEnergy: bandEnergy, shannonEntropy: shannonEntropy, strokes: strokes }
  };
});
