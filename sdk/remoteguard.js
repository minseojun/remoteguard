/*
 * RemoteGuard SDK
 * ===============
 * One <script> in a bank's transfer page. Watches the session, scores it every
 * 1.5 s, and hands the host page a decision.
 *
 *   RemoteGuard.init({
 *     endpoint: 'https://api.example.com',
 *     sessionId: 'txn-8842',
 *     onDecision: function (d) { ... }      // {risk, decision, reasons}
 *   });
 *   RemoteGuard.setStage('amount_entry');   // optional flow context
 *   RemoteGuard.runChallenge().then(ok => ...);
 *
 * What leaves the device: 24 floats per window. Not coordinates, not
 * timestamps, not accelerometer samples. Feature extraction happens here, in
 * shared/features.js — the same module the training pipeline runs under Node,
 * so the browser and the model can never disagree about what a feature means.
 *
 * Requires shared/features.js and shared/challenge.js to be loaded first.
 */
(function (root) {
  'use strict';

  var RG = {
    _buf: [], _cfg: null, _timer: null, _stage: 'unknown', _windowIndex: 0,
    _motionOK: false, _lastDecision: null, VERSION: '1.0.0'
  };

  var WINDOW_MS = 3000, HOP_MS = 1500, MAX_BUF = 4000;

  function now() { return performance.now(); }

  function record(e) {
    RG._buf.push(e);
    if (RG._buf.length > MAX_BUF) RG._buf.splice(0, RG._buf.length - MAX_BUF);
  }

  function onInput(e, type) {
    var t = (typeof e.timeStamp === 'number' && e.timeStamp > 0) ? e.timeStamp : now();
    var pts = e.changedTouches ? e.changedTouches : [e];
    for (var i = 0; i < pts.length; i++) {
      var p = pts[i];
      record({
        t: t, type: type, x: p.clientX, y: p.clientY,
        force: (p.force != null ? p.force : (p.pressure != null ? p.pressure : null)),
        rx: (p.radiusX != null ? p.radiusX : (p.width != null ? p.width : null)),
        ry: (p.radiusY != null ? p.radiusY : (p.height != null ? p.height : null)),
        ptype: e.changedTouches ? 'touch' : (p.pointerType || 'mouse'),
        id: (p.identifier != null ? p.identifier : (p.pointerId != null ? p.pointerId : 0))
      });
    }
  }

  function onMotion(e) {
    var a = e.accelerationIncludingGravity || {}, r = e.rotationRate || {};
    if (a.x == null) return;
    RG._motionOK = true;
    record({ t: now(), type: 'motion', ax: a.x, ay: a.y, az: a.z,
             gx: r.alpha, gy: r.beta, gz: r.gamma, interval: e.interval });
  }

  function attach() {
    ['touchstart', 'touchmove', 'touchend'].forEach(function (k) {
      window.addEventListener(k, function (e) { onInput(e, k); }, { passive: true, capture: true });
    });
    ['mousedown', 'mousemove', 'mouseup'].forEach(function (k) {
      window.addEventListener(k, function (e) {
        if (e.sourceCapabilities && e.sourceCapabilities.firesTouchEvents) return;
        onInput(e, k);
      }, { passive: true, capture: true });
    });
    window.addEventListener('devicemotion', onMotion);
  }

  /* iOS requires a user gesture before motion is available at all. Call this
     from a real tap early in the flow, or the M layer is simply absent — in
     which case we say so rather than silently scoring on three layers. */
  RG.requestMotion = function () {
    var need = typeof DeviceMotionEvent !== 'undefined' &&
               typeof DeviceMotionEvent.requestPermission === 'function';
    if (!need) return Promise.resolve(true);
    return DeviceMotionEvent.requestPermission()
      .then(function (r) { return r === 'granted'; })
      .catch(function () { return false; });
  };

  function tick() {
    var t1 = now(), t0 = t1 - WINDOW_MS;
    var win = RG._buf.filter(function (e) { return e.t >= t0; });
    var nIn = win.filter(function (e) { return e.type !== 'motion'; }).length;
    if (nIn < 3) return;                       // idle screen: nothing to judge

    var f;
    try { f = root.RGFeatures.featuresFor(win); }
    catch (err) { return; }

    var payload = {
      session_id: RG._cfg.sessionId,
      features: f,
      window_index: RG._windowIndex++,
      stage: RG._stage,
      client_version: RG.VERSION
    };

    if (RG._cfg.onFeatures) RG._cfg.onFeatures(f);

    fetch(RG._cfg.endpoint.replace(/\/$/, '') + '/api/v1/score', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload), keepalive: true
    }).then(function (r) { return r.json(); })
      .then(function (d) {
        d.motion_available = RG._motionOK;
        RG._lastDecision = d;
        if (RG._cfg.onDecision) RG._cfg.onDecision(d);
      })
      .catch(function (e) {
        // fail open: a scoring outage must never block a legitimate transfer
        if (RG._cfg.onError) RG._cfg.onError(e);
      });
  }

  RG.init = function (cfg) {
    if (!cfg || !cfg.endpoint || !cfg.sessionId) throw new Error('endpoint and sessionId required');
    if (!root.RGFeatures) throw new Error('shared/features.js must be loaded first');
    RG._cfg = cfg;
    attach();
    RG._timer = setInterval(tick, HOP_MS);
    return RG;
  };

  RG.setStage = function (s) { RG._stage = s; };
  RG.lastDecision = function () { return RG._lastDecision; };
  RG.stop = function () { clearInterval(RG._timer); RG._timer = null; };

  /* L4 — physical presence.
     Collects motion for up to `ms`, summarises on-device, and asks the server
     to verify. The raw trace never leaves the phone; the summary is 8 numbers. */
  RG.runChallenge = function (ms) {
    ms = ms || 4000;
    if (!root.RGChallenge) return Promise.reject(new Error('shared/challenge.js not loaded'));
    return RG.requestMotion().then(function (ok) {
      if (!ok) return { passed: false, failed: [{ check: 'permission', reason: 'motion access denied' }] };
      return new Promise(function (resolve) {
        var samples = [], start = now();
        function grab(e) {
          var a = e.accelerationIncludingGravity || {}, r = e.rotationRate || {};
          if (a.x == null) return;
          samples.push({ t: now(), ax: a.x, ay: a.y, az: a.z, gx: r.alpha, gy: r.beta, gz: r.gamma });
          if (RG._cfg && RG._cfg.onChallengeSample) RG._cfg.onChallengeSample(samples);
          // finish as soon as the shake is unambiguous — no need to wait it out
          if (now() - start > 900) {
            var s = root.RGChallenge.summarize(samples);
            if (root.RGChallenge.verify(s).passed) finish();
          }
        }
        var done = false;
        function finish() {
          if (done) return; done = true;
          window.removeEventListener('devicemotion', grab);
          var summary = root.RGChallenge.summarize(samples);
          fetch(RG._cfg.endpoint.replace(/\/$/, '') + '/api/v1/challenge', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: RG._cfg.sessionId, summary: summary })
          }).then(function (r) { return r.json(); })
            .then(function (v) { v.summary = summary; resolve(v); })
            .catch(function () {
              // verifier unreachable: fall back to the local check, and say so
              var v = root.RGChallenge.verify(summary);
              v.summary = summary; v.local_only = true; resolve(v);
            });
        }
        window.addEventListener('devicemotion', grab);
        setTimeout(finish, ms);
      });
    });
  };

  root.RemoteGuard = RG;
})(typeof self !== 'undefined' ? self : this);
