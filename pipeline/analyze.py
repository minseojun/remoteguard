#!/usr/bin/env python3
"""
RemoteGuard — Phase 0 signal check
==================================
Run this on your FIRST real sessions, before touching the model. It answers the
three questions that decide whether the IMU layer exists at all on the hardware
you actually have:

  1. Are the touch clock and the motion clock the same clock?
     If event.timeStamp and performance.now() disagree, every coupling number
     downstream is noise. Check this first; nothing else matters if it fails.

  2. What sample rate does the browser really deliver?
     At 60 Hz a finger-tap impulse spans 2-3 samples and impulse detection is
     marginal. At 25-30 Hz it is gone, and the layer has to rest on band energy
     (held vs. not held) instead. Measure, don't assume.

  3. Do the conditions actually separate?
     Band-energy spectra and the touch-triggered average. If handheld and
     remote overlap here, the idea does not work and you want to know in week
     one, not week six.

usage: python3 pipeline/analyze.py [--in data/raw] [--out data/out]
"""
import argparse, glob, json, os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = 9.80665
COND_COLOR = {"handheld": "#0E7C7B", "desk": "#A15C00",
              "remote_scrcpy": "#B3261E", "remote_a11y": "#7A1FA2",
              "remote_evasive": "#1D3557"}


def load(indir):
    out = []
    for f in sorted(glob.glob(os.path.join(indir, "*.json"))):
        try:
            s = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if s.get("events"):
            out.append(s)
    return out


def series(sess):
    mot = [e for e in sess["events"] if e.get("type") == "motion"]
    if len(mot) < 8:
        return None
    t = np.array([e["t"] for e in mot], float)
    a = np.array([np.hypot(np.hypot(e.get("ax") or 0, e.get("ay") or 0),
                           e.get("az") or G) for e in mot]) - G
    g = np.array([np.hypot(np.hypot(e.get("gx") or 0, e.get("gy") or 0),
                           e.get("gz") or 0) for e in mot])
    downs = np.array([e["t"] for e in sess["events"]
                      if e.get("type") in ("touchstart", "mousedown", "pointerdown")], float)
    fs = (len(t) - 1) / ((t[-1] - t[0]) / 1000) if t[-1] > t[0] else 0
    return t, a, g, downs, fs


def resample(t, v, fs):
    """Browsers deliver motion on an irregular clock. Put it on a uniform grid
    before any spectral estimate, or the spectrum is an artefact of the jitter."""
    if fs <= 0 or len(t) < 8:
        return None, None
    grid = np.arange(t[0], t[-1], 1000.0 / fs)
    return grid, np.interp(grid, t, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="data/raw")
    ap.add_argument("--out", dest="outdir", default="data/out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    sessions = load(args.indir)
    if not sessions:
        raise SystemExit(f"no sessions in {args.indir}")

    synth = any(s.get("synthetic") for s in sessions)
    print("=" * 68)
    if synth:
        print("!! synthetic sessions present — these plots check the code path,")
        print("!! not the physics. Re-run on real phone sessions.")
    print(f"sessions: {len(sessions)}")

    # --- 1. clock alignment ------------------------------------------------
    print("\n1) CLOCK ALIGNMENT   performance.now() - event.timeStamp")
    skews = [(s["condition"], s.get("diagnostics", {}).get("clock_skew_mean_ms"),
              s.get("diagnostics", {}).get("clock_skew_sd_ms")) for s in sessions]
    real = [x for x in skews if x[1] is not None]
    if real:
        m = np.mean([x[1] for x in real]); sd = np.mean([x[2] or 0 for x in real])
        print(f"   mean {m:+.2f} ms   (within-session sd {sd:.2f} ms)")
        if abs(m) > 20 or sd > 20:
            print("   ✗ the two clocks are NOT the same base. Fix alignment before")
            print("     trusting any coupling feature — subtract the measured offset.")
        else:
            print("   ✓ same clock base. Coupling features are meaningful.")

    # --- 2. sample rate ----------------------------------------------------
    print("\n2) MOTION SAMPLE RATE")
    rates = defaultdict(list)
    for s in sessions:
        r = series(s)
        if r:
            rates[s["condition"]].append(r[4])
    for c, v in sorted(rates.items()):
        print(f"   {c:<16} {np.mean(v):6.1f} Hz   (min {np.min(v):.0f}, max {np.max(v):.0f})")
    allr = [x for v in rates.values() for x in v]
    if allr:
        mr = np.mean(allr)
        print(f"   -> {'impulse detection viable' if mr >= 50 else 'impulse detection marginal; rely on band energy'}"
              f" at {mr:.0f} Hz")

    # --- 3. spectra + touch-triggered average ------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    for cond in sorted(rates):
        acc = []
        for s in sessions:
            if s["condition"] != cond:
                continue
            r = series(s)
            if not r:
                continue
            t, a, g, downs, fs = r
            grid, av = resample(t, a, fs)
            if grid is None or len(av) < 64:
                continue
            n = 1 << int(np.floor(np.log2(len(av))))
            sp = np.abs(np.fft.rfft((av[:n] - av[:n].mean()) * np.hanning(n))) ** 2
            fr = np.fft.rfftfreq(n, 1 / fs)
            acc.append(np.interp(np.linspace(0.3, 15, 120), fr, sp))
        if acc:
            ax.semilogy(np.linspace(0.3, 15, 120), np.mean(acc, 0),
                        color=COND_COLOR.get(cond, "#555"), label=cond, lw=1.6)
    ax.axvspan(0.5, 2, color="#0E7C7B", alpha=.08)
    ax.axvspan(8, 12, color="#B3261E", alpha=.08)
    ax.set_title("accel spectrum\nshaded: sway 0.5–2 Hz, tremor 8–12 Hz", fontsize=10)
    ax.set_xlabel("Hz"); ax.set_ylabel("power"); ax.legend(fontsize=7)

    ax = axes[1]
    for cond in sorted(rates):
        stack = []
        for s in sessions:
            if s["condition"] != cond:
                continue
            r = series(s)
            if not r:
                continue
            t, a, g, downs, fs = r
            for d in downs:
                sel = (t >= d - 150) & (t <= d + 250)
                if sel.sum() < 6:
                    continue
                stack.append(np.interp(np.linspace(-150, 250, 40), t[sel] - d, np.abs(a[sel])))
        if stack:
            ax.plot(np.linspace(-150, 250, 40), np.mean(stack, 0),
                    color=COND_COLOR.get(cond, "#555"), label=f"{cond} (n={len(stack)})", lw=1.6)
    ax.axvline(0, color="#333", lw=.8, ls="--")
    ax.set_title("touch-triggered average |a|\nthe core claim: real touches move the phone", fontsize=10)
    ax.set_xlabel("ms from touch-down"); ax.legend(fontsize=7)

    ax = axes[2]
    for cond in sorted(rates):
        vals = []
        for s in sessions:
            if s["condition"] != cond:
                continue
            r = series(s)
            if not r:
                continue
            t, a, g, downs, fs = r
            if len(downs) < 3:
                continue
            base = np.mean(np.abs(a)) + 1e-9
            hit = [np.mean(np.abs(a[(t >= d - 150) & (t <= d + 150)]))
                   for d in downs if ((t >= d - 150) & (t <= d + 150)).sum() > 2]
            if hit:
                vals.append(np.mean(hit) / base)
        if vals:
            ax.scatter([cond] * len(vals), vals, s=14, alpha=.55,
                       color=COND_COLOR.get(cond, "#555"))
    ax.axhline(1.0, color="#333", lw=.8, ls="--")
    ax.set_title("touch/IMU coupling ratio per session\n1.0 = touches carry no motion", fontsize=10)
    ax.tick_params(axis="x", rotation=20, labelsize=8)

    plt.tight_layout()
    p = os.path.join(args.outdir, "phase0_signals.png")
    plt.savefig(p, dpi=130)
    print(f"\nwrote {p}")
    print("\nread the middle panel first: if handheld shows a bump at t=0 and the")
    print("remote conditions stay flat, the layer is real. If not, stop and rethink.")


if __name__ == "__main__":
    main()
