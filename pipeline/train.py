#!/usr/bin/env python3
"""
RemoteGuard — training & evaluation
===================================
Reads data/out/windows.csv, trains a LightGBM window classifier, and reports
the numbers that actually decide whether this service could ship.

Design decisions that matter more than the model:

1. LEAVE-ONE-SUBJECT-OUT. A random split puts windows from the same person in
   both train and test. Windows overlap by 50% and a person's motor signature is
   stable, so a random split inflates AUC by a large margin and the number is
   meaningless. Grouping by subject is the only honest split here.

2. FPR AT FIXED RECALL IS THE HEADLINE, NOT AUC. Blocking a legitimate transfer
   is the failure that kills the product. We fix recall at 95% and report the
   false-positive rate there.

3. PER-CONDITION BREAKDOWN. Aggregate FPR hides the case we care about: a phone
   sitting in a desk stand, operated by its real owner. That condition is scored
   separately and is the number to defend in review.

4. ABLATION BY LAYER. If pointer features alone do the work, the IMU layer is
   decoration and should be cut. The ablation says which layers earn their place.

usage: python3 pipeline/train.py [--csv data/out/windows.csv] [--out data/out]
"""
import argparse, json, os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except ImportError:
    sys.exit("lightgbm not installed:  pip install lightgbm")

from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

GROUPS = {
    "P": ["p_force_var", "p_force_uniq_ratio", "p_force_degenerate", "p_radius_var",
          "p_radius_missing", "p_int_snap_frac", "p_nontouch_ratio"],
    "G": ["g_path_efficiency", "g_curvature_mean", "g_speed_peaks_per_stroke",
          "g_jerk_norm", "g_endpoint_correction"],
    "T": ["t_iei_cv", "t_iei_entropy", "t_iei_modal_frac", "t_burst_ratio"],
    "M": ["m_sample_rate", "m_sway_energy", "m_tremor_energy", "m_still_frac",
          "m_accel_std", "m_gyro_std", "m_touch_coupling", "m_coupling_z"],
}
ALL_FEATS = [f for g in GROUPS.values() for f in g]

PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.05, num_leaves=15,
    min_data_in_leaf=25, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, verbose=-1, num_threads=4,
)
N_ROUNDS = 400


def fpr_at_recall(y, p, target=0.95):
    """Smallest achievable FPR at or above the target recall."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = tpr >= target
    if not ok.any():
        return 1.0, 0.0
    i = int(np.argmax(ok))
    return float(fpr[i]), float(thr[i])


def loso_predict(df, feats, seed=0):
    """Out-of-fold predictions, one fold per subject."""
    oof = np.full(len(df), np.nan)
    subjects = sorted(df.subject_id.unique())
    if len(subjects) < 3:
        sys.exit(f"only {len(subjects)} subject(s) in the data — LOSO needs at least 3. "
                 "Collect from more participants before trusting any metric.")
    for s in subjects:
        te = df.subject_id == s
        tr = ~te
        if df.loc[tr, "label"].nunique() < 2:
            continue
        ds = lgb.Dataset(df.loc[tr, feats], label=df.loc[tr, "label"])
        p = dict(PARAMS); p["seed"] = seed
        m = lgb.train(p, ds, num_boost_round=N_ROUNDS)
        oof[te.values] = m.predict(df.loc[te, feats])
    return oof


def session_level(df, oof, thr):
    """Aggregate windows to a decision per session.

    'events_to_alarm' answers the question a reviewer will ask: was the session
    flagged before the confirm button, or only after the money left?
    """
    out = []
    for sid, g in df.assign(p=oof).groupby("session_id"):
        g = g.sort_values("win_start_ms")
        hits = g.p >= thr
        # two consecutive flagged windows = alarm (debounce against single spikes)
        alarm_at, run = None, 0
        for _, r in g.iterrows():
            run = run + 1 if r.p >= thr else 0
            if run >= 2:
                alarm_at = r.win_start_ms
                break
        out.append(dict(
            session_id=sid, subject_id=g.subject_id.iloc[0], condition=g.condition.iloc[0],
            label=int(g.label.iloc[0]), n_windows=len(g), max_p=float(g.p.max()),
            mean_p=float(g.p.mean()), alarmed=alarm_at is not None,
            alarm_ms=alarm_at if alarm_at is not None else np.nan,
            frac_flagged=float(hits.mean()),
        ))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/out/windows.csv")
    ap.add_argument("--out", default="data/out")
    ap.add_argument("--recall", type=float, default=0.95)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.out, exist_ok=True)

    missing = [f for f in ALL_FEATS if f not in df.columns]
    if missing:
        sys.exit(f"csv is missing feature columns: {missing[:5]} — re-run extract.mjs")
    df[ALL_FEATS] = df[ALL_FEATS].replace([np.inf, -np.inf], 0).fillna(0)

    synthetic = df.get("synthetic", pd.Series([0] * len(df))).max() == 1
    print("=" * 72)
    if synthetic:
        print("!! SYNTHETIC DATA IN THIS RUN — pipeline check only.")
        print("!! These numbers are not evidence and must not appear in the report.")
    print(f"windows {len(df)} | subjects {df.subject_id.nunique()} | "
          f"sessions {df.session_id.nunique()} | positives {df.label.mean():.1%}")
    print(df.groupby("condition").agg(windows=("label", "size"),
                                      sessions=("session_id", "nunique")).to_string())
    print("=" * 72)

    # ---------------- ablation ----------------
    sets = {
        "P only": GROUPS["P"],
        "G only": GROUPS["G"],
        "T only": GROUPS["T"],
        "M only": GROUPS["M"],
        "P+G+T (no IMU)": GROUPS["P"] + GROUPS["G"] + GROUPS["T"],
        "ALL": ALL_FEATS,
    }
    print("\nABLATION  (leave-one-subject-out)")
    print(f"{'feature set':<18}{'AUC':>8}{'AP':>8}{f'FPR@{args.recall:.0%}':>12}")
    print("-" * 46)
    results, best = {}, None
    for name, feats in sets.items():
        oof = loso_predict(df, feats)
        ok = ~np.isnan(oof)
        auc = roc_auc_score(df.label[ok], oof[ok])
        ap_ = average_precision_score(df.label[ok], oof[ok])
        f, thr = fpr_at_recall(df.label[ok], oof[ok], args.recall)
        results[name] = dict(auc=auc, ap=ap_, fpr=f, thr=thr)
        print(f"{name:<18}{auc:>8.3f}{ap_:>8.3f}{f:>12.3f}")
        if name == "ALL":
            best = (oof, thr, f, auc)

    oof, thr, fpr, auc = best

    # ---------------- operating points ----------------
    # "95% recall" is the research framing. The product framing is the reverse:
    # pick the false-positive budget the bank will tolerate, then see what
    # recall that buys. If recall at 1% FPR is poor, a hard block is off the
    # table and the mid band has to route to the L4 challenge instead.
    ok = ~np.isnan(oof)
    yv, pv = df.label[ok].values, oof[ok]
    fpr_c, tpr_c, thr_c = roc_curve(yv, pv)
    print("\nOPERATING POINTS  (what a fixed false-positive budget buys)")
    print(f"{'FPR budget':<14}{'recall':>10}{'threshold':>12}")
    print("-" * 36)
    op = {}
    for budget in (0.01, 0.02, 0.05, 0.10):
        i = int(np.searchsorted(fpr_c, budget, side="right")) - 1
        i = max(0, min(i, len(tpr_c) - 1))
        op[budget] = dict(recall=float(tpr_c[i]), threshold=float(thr_c[i]))
        print(f"{budget:<14.0%}{tpr_c[i]:>10.3f}{thr_c[i]:>12.4f}")

    # three-tier policy sizing: how much traffic lands in each band
    t_hi = op[0.01]["threshold"]
    i_lo = int(np.searchsorted(fpr_c, 0.20, side="right")) - 1
    t_lo = float(thr_c[max(0, min(i_lo, len(thr_c) - 1))])
    neg = pv[yv == 0]
    print(f"\nPOLICY BANDS   allow < {t_lo:.3f} <= challenge < {t_hi:.3f} <= challenge+delay")
    print(f"  legitimate sessions sent to a challenge : {float(((neg >= t_lo) & (neg < t_hi)).mean()):.1%}")
    print(f"  legitimate sessions hitting the delay   : {float((neg >= t_hi).mean()):.1%}")
    print(f"  attack recall inside the two risk bands : {float((pv[yv == 1] >= t_lo).mean()):.1%}")

    # ---------------- per-condition ----------------
    print(f"\nPER-CONDITION  (threshold {thr:.3f}, set for {args.recall:.0%} recall)")
    print(f"{'condition':<18}{'n':>7}{'flag rate':>12}{'meaning':>10}")
    print("-" * 47)
    ok = ~np.isnan(oof)
    d = df[ok].assign(p=oof[ok])
    for cond, g in d.groupby("condition"):
        rate = float((g.p >= thr).mean())
        kind = "recall" if g.label.iloc[0] == 1 else "FPR"
        print(f"{cond:<18}{len(g):>7}{rate:>12.3f}{kind:>10}")
    desk = d[d.condition == "desk"]
    if len(desk):
        print(f"\n  desk FPR = {float((desk.p >= thr).mean()):.3f}  <- the number that decides "
              "whether this can be deployed")

    # ---------------- session level ----------------
    sess = session_level(d, d.p.values, thr)
    print("\nSESSION LEVEL  (alarm = 2 consecutive flagged windows)")
    print(f"{'condition':<18}{'sessions':>10}{'alarm rate':>12}{'median t':>12}")
    print("-" * 52)
    for cond, g in sess.groupby("condition"):
        med = g.alarm_ms.median()
        print(f"{cond:<18}{len(g):>10}{g.alarmed.mean():>12.3f}"
              f"{('%.1fs' % (med / 1000)) if not np.isnan(med) else '—':>12}")

    # ---------------- importance ----------------
    ds = lgb.Dataset(df[ALL_FEATS], label=df.label)
    full = lgb.train(dict(PARAMS, seed=0), ds, num_boost_round=N_ROUNDS)
    imp = pd.Series(full.feature_importance("gain"), index=ALL_FEATS).sort_values(ascending=False)
    print("\nTOP FEATURES BY GAIN")
    for k, v in imp.head(12).items():
        grp = next(g for g, fs in GROUPS.items() if k in fs)
        print(f"  [{grp}] {k:<28}{v:>10.1f}")

    full.save_model(os.path.join(args.out, "model.txt"))
    sess.to_csv(os.path.join(args.out, "sessions.csv"), index=False)
    meta = dict(
        threshold=float(thr), target_recall=args.recall, auc_loso=float(auc),
        fpr_at_recall=float(fpr), features=ALL_FEATS, groups=GROUPS,
        n_windows=int(len(df)), n_subjects=int(df.subject_id.nunique()),
        synthetic=bool(synthetic), ablation={k: {m: float(x) for m, x in v.items()}
                                            for k, v in results.items()},
        importance={k: float(v) for k, v in imp.items()},
        policy=dict(allow_below=float(t_lo), delay_above=float(t_hi)),
        operating_points={str(k): v for k, v in op.items()},
    )
    with open(os.path.join(args.out, "model_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nsaved model.txt, model_meta.json, sessions.csv -> {args.out}")
    if synthetic:
        print("\n!! reminder: synthetic run. replace data/raw with real sessions before reporting.")


if __name__ == "__main__":
    main()
