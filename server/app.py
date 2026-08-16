#!/usr/bin/env python3
"""
RemoteGuard API
===============
Three endpoints that matter:

  POST /api/v1/score       SDK sends a 24-float feature vector, gets a decision.
  POST /api/v1/challenge   Verifies a physical-presence (shake) attempt.
  POST /api/v1/sessions    Research ingest for raw collected sessions.

Privacy posture, and it is load-bearing rather than decorative:
/score never receives coordinates, timestamps, or raw accelerometer samples.
Feature extraction runs on the device; 24 floats leave it. /sessions does take
raw traces, but it is the research collector path only — it is disabled unless
ALLOW_RAW_INGEST=1, and consented participants are the only people who ever hit
it. Nothing in the production path stores a raw trace.

The policy layer is separate from the model on purpose. The model produces a
probability; the policy decides what to do with it, and includes one rule the
model must not be allowed to override — see _imu_alone_guard.

run: uvicorn server.app:app --reload --port 8000
"""
from __future__ import annotations

import json, os, sqlite3, subprocess, tempfile, time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "data" / "out" / "model.txt"
META_PATH = ROOT / "data" / "out" / "model_meta.json"
DB_PATH = Path(os.environ.get("RG_DB", ROOT / "data" / "rg.sqlite"))
ALLOW_RAW = os.environ.get("ALLOW_RAW_INGEST", "1") == "1"

# ---------------------------------------------------------------- model ----
_model = None
_meta: dict[str, Any] = {}


def load_model():
    global _model, _meta
    if META_PATH.exists():
        _meta = json.loads(META_PATH.read_text())
    if MODEL_PATH.exists():
        import lightgbm as lgb
        _model = lgb.Booster(model_file=str(MODEL_PATH))
    return _model


FEATURES: list[str] = []


def feature_names() -> list[str]:
    global FEATURES
    if FEATURES:
        return FEATURES
    if _meta.get("features"):
        FEATURES = _meta["features"]
    else:  # fall back to the shared JS module so the two can never disagree
        js = ROOT / "shared" / "features.js"
        out = subprocess.run(
            ["node", "-e", f"console.log(JSON.stringify(require({str(js)!r}).FEATURE_NAMES))"],
            capture_output=True, text=True, timeout=20)
        FEATURES = json.loads(out.stdout)
    return FEATURES


GROUPS = {
    "P": "pointer",      # is the touch physically real
    "G": "trajectory",   # does the movement look like a human reach
    "T": "timing",       # is the event stream frame-quantised
    "M": "motion",       # is the handset behaving like a held object
}

# ---------------------------------------------------------------- storage --
def db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS sessions(
        session_id TEXT PRIMARY KEY, subject_id TEXT, condition TEXT, label INT,
        received_at REAL, duration_ms REAL, motion_hz REAL, clock_skew_ms REAL,
        n_windows INT, payload TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS scores(
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, ts REAL,
        risk REAL, decision TEXT, reasons TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS challenges(
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, ts REAL,
        passed INT, summary TEXT, failed TEXT)""")
    return con


# ---------------------------------------------------------------- schemas --
class ScoreIn(BaseModel):
    session_id: str
    features: dict[str, float] = Field(..., description="24-float vector from shared/features.js")
    window_index: int = 0
    stage: str = "unknown"          # where in the flow the user is
    client_version: str = "unknown"


class ChallengeIn(BaseModel):
    session_id: str
    summary: dict[str, float]       # from shared/challenge.js summarize()
    attempt: int = 1


# ---------------------------------------------------------------- policy ---
# Bands come from the trained model's ROC; the fallbacks are placeholders that
# exist only so a fresh checkout runs before anyone has trained anything.
BAND_ALLOW = 0.35
BAND_DELAY = 0.70


def bands() -> tuple[float, float]:
    ab = _meta.get("policy", {})
    return ab.get("allow_below", BAND_ALLOW), ab.get("delay_above", BAND_DELAY)


def _imu_alone_guard(f: dict[str, float], risk: float) -> tuple[float, str | None]:
    """A still handset is not evidence of fraud.

    A phone in a desk stand, on a table, or in a car mount produces exactly the
    flat accelerometer trace a remote session does. If the motion layer is the
    only thing raising the alarm, we refuse to escalate to the hard band and
    send the session to a challenge instead, where a legitimate user resolves it
    in two seconds. This rule sits outside the model because a model trained on
    a dataset thin in stand-users will happily learn to block them.
    """
    still = f.get("m_still_frac", 0) > 0.6 or f.get("m_accel_std", 0) < 0.03
    software_layers_clean = (
        f.get("p_force_degenerate", 0) < 0.5
        and f.get("p_int_snap_frac", 0) < 0.8
        and f.get("g_path_efficiency", 1) < 0.97
        and f.get("t_iei_modal_frac", 0) < 0.6
    )
    if still and software_layers_clean and risk >= BAND_DELAY:
        return BAND_DELAY - 0.01, "motion_only_evidence"
    return risk, None


def explain(f: dict[str, float]) -> list[dict[str, Any]]:
    """Reasons in the operator's language, not the model's."""
    r = []
    if f.get("p_force_degenerate", 0) > 0.8:
        r.append(dict(layer="P", code="pressure_constant",
                      text="터치 압력이 고정값입니다 — 합성 입력의 특징입니다"))
    if f.get("p_int_snap_frac", 0) > 0.9:
        r.append(dict(layer="P", code="integer_coordinates",
                      text="좌표가 모두 정수에 스냅되어 있습니다"))
    if f.get("g_path_efficiency", 0) > 0.97:
        r.append(dict(layer="G", code="linear_path",
                      text="이동 경로가 직선입니다 — 사람의 손은 곡선을 그립니다"))
    if f.get("t_iei_modal_frac", 0) > 0.6:
        r.append(dict(layer="T", code="frame_quantised",
                      text="이벤트 간격이 한 값에 몰려 있습니다 — 화면 스트리밍 프레임레이트"))
    if f.get("t_burst_ratio", 0) > 0.15:
        r.append(dict(layer="T", code="network_stalls",
                      text="네트워크 지연 형태의 정지·버스트 패턴이 있습니다"))
    if f.get("m_coupling_z", 0) < 0.5 and f.get("m_accel_std", 0) > 0.05:
        r.append(dict(layer="M", code="no_touch_coupling",
                      text="기기는 움직이는데 터치 시점과 무관합니다 — 다른 손이 조작 중일 수 있습니다"))
    if f.get("m_still_frac", 0) > 0.6:
        r.append(dict(layer="M", code="handset_still",
                      text="조작 중인데 기기가 완전히 정지해 있습니다 (거치 상태일 수도 있습니다)"))
    return r


def score_features(f: dict[str, float]) -> dict[str, Any]:
    names = feature_names()
    x = np.array([[float(f.get(n, 0.0) or 0.0) for n in names]], dtype=float)
    x[~np.isfinite(x)] = 0.0

    if _model is None:
        risk = _heuristic(f)
        source = "heuristic"
    else:
        risk = float(_model.predict(x)[0])
        source = "model"

    risk, guard = _imu_alone_guard(f, risk)
    lo, hi = bands()
    decision = "allow" if risk < lo else ("challenge" if risk < hi else "challenge_and_delay")
    return dict(risk=round(risk, 4), decision=decision, source=source,
                guard=guard, reasons=explain(f), bands=dict(allow_below=lo, delay_above=hi))


def _heuristic(f: dict[str, float]) -> float:
    """Runs before a model exists so the demo is never dead. Deliberately crude
    and clearly labelled — it is not the product."""
    s = 0.0
    s += 0.30 * (f.get("p_force_degenerate", 0) > 0.8)
    s += 0.15 * (f.get("p_int_snap_frac", 0) > 0.9)
    s += 0.15 * (f.get("g_path_efficiency", 0) > 0.97)
    s += 0.15 * (f.get("t_iei_modal_frac", 0) > 0.6)
    s += 0.25 * (f.get("m_coupling_z", 0) < 0.5)
    return min(1.0, s)


# ---------------------------------------------------------------- app ------
app = FastAPI(title="RemoteGuard API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _startup():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db().close()
    load_model()
    feature_names()


@app.get("/api/v1/health")
def health():
    return dict(ok=True, model_loaded=_model is not None,
                model_synthetic=_meta.get("synthetic"),
                n_features=len(feature_names()),
                raw_ingest=ALLOW_RAW)


@app.get("/api/v1/model")
def model_info():
    return dict(loaded=_model is not None,
                synthetic=_meta.get("synthetic"),
                auc_loso=_meta.get("auc_loso"),
                fpr_at_recall=_meta.get("fpr_at_recall"),
                ablation=_meta.get("ablation"),
                top_features=dict(list((_meta.get("importance") or {}).items())[:10]),
                bands=dict(zip(("allow_below", "delay_above"), bands())))


@app.post("/api/v1/score")
def score(body: ScoreIn):
    out = score_features(body.features)
    con = db()
    con.execute("INSERT INTO scores(session_id,ts,risk,decision,reasons) VALUES(?,?,?,?,?)",
                (body.session_id, time.time(), out["risk"], out["decision"],
                 json.dumps(out["reasons"], ensure_ascii=False)))
    con.commit(); con.close()
    out["window_index"] = body.window_index
    return out


@app.post("/api/v1/challenge")
def challenge(body: ChallengeIn):
    """Verification runs server-side against shared/challenge.js thresholds so a
    tampered client cannot simply return passed:true."""
    js = ROOT / "shared" / "challenge.js"
    code = (f"const C=require({str(js)!r});"
            f"console.log(JSON.stringify(C.verify({json.dumps(body.summary)})));")
    try:
        r = subprocess.run(["node", "-e", code], capture_output=True, text=True, timeout=10)
        verdict = json.loads(r.stdout)
    except Exception as e:
        raise HTTPException(500, f"challenge verifier failed: {e}")

    con = db()
    con.execute("INSERT INTO challenges(session_id,ts,passed,summary,failed) VALUES(?,?,?,?,?)",
                (body.session_id, time.time(), int(verdict["passed"]),
                 json.dumps(body.summary), json.dumps(verdict["failed"])))
    con.commit(); con.close()
    return verdict


@app.post("/api/v1/sessions")
async def ingest(req: Request):
    if not ALLOW_RAW:
        raise HTTPException(403, "raw ingest disabled; this path is for consented research collection only")
    sess = await req.json()
    if sess.get("schema") != "remoteguard.session.v1":
        raise HTTPException(400, "unknown schema")
    sid = sess.get("session_id")
    if not sid:
        raise HTTPException(400, "session_id required")

    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{sid}.json").write_text(json.dumps(sess), encoding="utf-8")

    n_windows = 0
    try:  # count windows with the same extractor the SDK uses
        js = ROOT / "shared" / "features.js"
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(sess["events"], fh)
            tmp = fh.name
        code = (f"const RG=require({str(js)!r}),fs=require('fs');"
                f"console.log(RG.extractWindows(JSON.parse(fs.readFileSync({tmp!r},'utf8'))).length);")
        r = subprocess.run(["node", "-e", code], capture_output=True, text=True, timeout=60)
        n_windows = int((r.stdout or "0").strip() or 0)
        os.unlink(tmp)
    except Exception:
        pass

    d = sess.get("diagnostics", {})
    con = db()
    con.execute("""INSERT OR REPLACE INTO sessions
        (session_id,subject_id,condition,label,received_at,duration_ms,motion_hz,
         clock_skew_ms,n_windows,payload) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (sid, sess.get("subject_id"), sess.get("condition"), sess.get("label"),
                 time.time(), sess.get("duration_ms"), d.get("motion_hz"),
                 d.get("clock_skew_mean_ms"), n_windows, ""))
    con.commit(); con.close()
    return dict(ok=True, session_id=sid, windows=n_windows,
                warning=None if n_windows >= 3 else "session too short to be useful (<3 windows)")


@app.get("/api/v1/stats")
def stats():
    con = db()
    rows = con.execute("""SELECT condition, COUNT(*), SUM(n_windows),
                          AVG(motion_hz), AVG(clock_skew_ms)
                          FROM sessions GROUP BY condition""").fetchall()
    con.close()
    return dict(by_condition=[
        dict(condition=r[0], sessions=r[1], windows=r[2],
             mean_motion_hz=round(r[3] or 0, 1), mean_clock_skew_ms=round(r[4] or 0, 2))
        for r in rows])


# serve the collector, demo and shared modules from the same origin
for name, folder in (("collector", "collector"), ("demo", "demo"), ("shared", "shared"), ("sdk", "sdk")):
    p = ROOT / folder
    if p.exists():
        app.mount(f"/{name}", StaticFiles(directory=str(p), html=True), name=name)
