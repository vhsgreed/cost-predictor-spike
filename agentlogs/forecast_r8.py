#!/usr/bin/env python3
"""Round 8: learned during-run model. Plan: PLAN.md 'Round 8 pre-registration'. One look on test.

  python forecast_r8.py --smoke   # fit-internal check only: fit instances split 80/20, test never read
  python forecast_r8.py           # the registered run (one look)

Part A: LightGBM quantile regression (alpha .1/.5/.9 on log final size) vs the fixed k-only table.
Part B: success-from-here classifier -> stop policy, threshold chosen on fit (out-of-fold),
        backtested once on test against no-stop and the group-p90 alarm.
Size target = R7's quantity (context size at the last assistant turn, chars/4); the summed
quantity (sum of context sizes over turns = estimated tokens processed) is used for cost in Part B
and as a descriptive secondary in Part A. Own code; no TokenCast code used."""
import glob, json, os, pickle, random, sys
from collections import defaultdict
import numpy as np
import lightgbm as lgb
import pyarrow.parquet as pq
from traj_common import DATASETS, HERE, EXT, kb, tq, q, steps, load, split

SMOKE = "--smoke" in sys.argv
SEED = 20261016
PARAMS = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, seed=SEED, verbose=-1, num_threads=14)
NROUND = 500
FEATS = ["grp", "k", "log_c", "x", "d1", "d3", "dmax", "log_first", "log_med", "log_p90", "log_sum"]
TAUS = [i / 100 for i in range(0, 51)]
MAX_LOST = 0.05


def load_labeled(ds):
    """Same parse as traj_common._load plus the outcome label; asserted identical to the R6/R7 cache."""
    p = os.path.join(HERE, f"traj_cache_r8_{ds}.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    rows, seen = [], set()
    if ds == "SWE-smith":
        for f in sorted(glob.glob(f"{EXT}/SWE-bench__SWE-smith-trajectories/**/train-*.parquet", recursive=True)):
            for b in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["messages", "instance_id", "resolved", "model", "traj_id"]):
                for r in b.to_pylist():
                    if r["traj_id"] in seen:
                        continue
                    seen.add(r["traj_id"])
                    s = steps(r["messages"], "role", "content", "assistant")
                    if s:
                        rows.append((r["model"], r["instance_id"], s, bool(r["resolved"])))
    elif ds == "SWE-rebench-OH":
        for b in pq.ParquetFile(f"{EXT}/nebius__SWE-rebench-openhands-trajectories/trajectories.parquet").iter_batches(
                batch_size=128, columns=["trajectory", "instance_id", "resolved", "repo"]):
            for r in b.to_pylist():
                s = steps(r["trajectory"], "role", "content", "assistant")
                if s:
                    rows.append((r["repo"], r["instance_id"], s, bool(r["resolved"])))
    else:
        for f in sorted(glob.glob(f"{EXT}/nebius__SWE-agent-trajectories/**/*.parquet", recursive=True)):
            for b in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["trajectory", "instance_id", "target", "model_name"]):
                for r in b.to_pylist():
                    s = steps(r["trajectory"], "role", "text", "ai")
                    if s:
                        rows.append((r["model_name"], r["instance_id"], s, bool(r["target"])))
    cache = load(ds)
    assert len(cache) == len(rows) and all(a == b[:3] for a, b in zip(cache, rows)), "labeled parse != R6/R7 cache"
    pickle.dump(rows, open(p, "wb"))
    return rows


def gstats(runs):
    d = defaultdict(list)
    for g, _, s, _ in runs:
        d[g].append(s[-1])
    return {g: (q(v, .5), q(v, .9)) for g, v in d.items()}


def featurise(runs, gst_for, gcode, kmin=1):
    """One row per update point k >= kmin. gst_for(run_index) -> (median, p90) of fit-group final size."""
    X, meta = [], []
    for ri, (g, inst, s, ok) in enumerate(runs):
        med, p90 = gst_for(ri)
        cum, dmax = 0.0, 0.0
        for k in range(1, len(s) + 1):
            c = s[k - 1]; cum += c
            if k > 1:
                dmax = max(dmax, c - s[k - 2])
            if k < kmin:
                continue
            dl = [s[j] - s[j - 1] for j in range(max(1, k - 3), k)]
            X.append((gcode[g], k, np.log1p(c), c / med, dl[-1] if dl else 0.0, sum(dl) / len(dl) if dl else 0.0,
                      dmax, np.log1p(s[0]),
                      np.log1p(med), np.log1p(p90), np.log1p(cum)))
            meta.append((ri, k, c, cum))
    return np.asarray(X, dtype=np.float64), meta


def train(X, y, obj, alpha=None):
    p = dict(PARAMS, objective=obj)
    if alpha is not None:
        p["alpha"] = alpha
    ds = lgb.Dataset(X, y, feature_name=FEATS, categorical_feature=["grp"], free_raw_data=False)
    return lgb.train(p, ds, num_boost_round=NROUND)


def boot(per, ncol, stat, seed=20261015, B=200):
    rng = random.Random(seed); ids = list(per); out = []
    for _ in range(B):
        a = [0.0] * ncol
        for _ in ids:
            v = per[ids[rng.randrange(len(ids))]]
            for i in range(ncol):
                a[i] += v[i]
        out.append(stat(a))
    out.sort()
    return out[10], out[189]


def analyse(ds, r7):
    rows = load_labeled(ds)
    fit3, test3, gfin, med, skipped = split([r[:3] for r in rows])
    fit_i = {r[1] for r in fit3}
    fit = [r for r in rows if r[1] in fit_i]                 # same order and rows as split()
    test = [r for r in rows if r[1] not in fit_i and len(gfin.get(r[0], ())) >= 50]
    assert [r[:3] for r in fit] == fit3 and [r[:3] for r in test] == test3
    fit = [r for r in fit if len(gfin[r[0]]) >= 50]          # same support rule as the test skip
    if SMOKE:                                                # fit-internal: never reads test
        inst = sorted({r[1] for r in fit}); random.Random(SEED + 99).shuffle(inst)
        hold = set(inst[int(.8 * len(inst)):])
        test = [r for r in fit if r[1] in hold]; fit = [r for r in fit if r[1] not in hold]
        gfin = defaultdict(list)
        for g, _, s, _ in fit:
            gfin[g].append(s[-1])
        test = [r for r in test if len(gfin.get(r[0], ())) >= 50]
        fit = [r for r in fit if len(gfin[r[0]]) >= 50]
    groups = sorted({r[0] for r in fit}); gcode = {g: i for i, g in enumerate(groups)}
    gst = gstats(fit)

    # k-only reference table (group, turn bucket) on final size, and on summed size (secondary)
    kc, kcs, gfs, gss = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for g, _, s, _ in fit:
        tot = sum(s); gfs[g].append(s[-1]); gss[g].append(tot)
        for k in range(1, len(s) + 1):
            kc[(g, kb(k))].append(s[-1]); kcs[(g, kb(k))].append(tot)
    KC = {key: q(v, .5) for key, v in kc.items() if len(v) >= 50}
    KCS = {key: q(v, .5) for key, v in kcs.items() if len(v) >= 50}
    GM = {g: q(v, .5) for g, v in gfs.items()}; GMS = {g: q(v, .5) for g, v in gss.items()}
    ref = lambda g, k: KC.get((g, kb(k)), GM[g])
    refs = lambda g, k: KCS.get((g, kb(k)), GMS[g])

    # out-of-fold group stats for fit rows (5 folds by instance), full-fit stats for test rows
    inst = sorted({r[1] for r in fit}); random.Random(SEED).shuffle(inst)
    fold = {i: n % 5 for n, i in enumerate(inst)}
    oof = []
    for f in range(5):
        oof.append(gstats([r for r in fit if fold[r[1]] != f]))
    allst = gst
    fit_gst = lambda ri: oof[fold[fit[ri][1]]].get(fit[ri][0], allst[fit[ri][0]])
    Xf, mf = featurise(fit, fit_gst, gcode)
    Xt, mt = featurise(test, lambda ri: allst[test[ri][0]], gcode)
    yf = np.log1p([fit[ri][2][-1] for ri, _, _, _ in mf])
    ysf = np.log1p([sum(fit[ri][2]) for ri, _, _, _ in mf])
    print(f"{ds}{' [SMOKE fit-internal]' if SMOKE else ''}: fit {len(fit):,} runs / {len(Xf):,} points; "
          f"test {len(test):,} runs / {len(Xt):,} points (skipped {skipped:,})", flush=True)

    # ---------- Part A: quantile forecast of final size ----------
    pred = {a: np.expm1(train(Xf, yf, "quantile", a).predict(Xt)) for a in (.1, .5, .9)}
    preds = {a: np.expm1(train(Xf, ysf, "quantile", a).predict(Xt)) for a in (.1, .5, .9)}
    per = defaultdict(lambda: [0.0] * 6)     # e8, eref, hit8, n, e8s, erefs
    turn = defaultdict(lambda: [0.0, 0.0, 0])
    for j, (ri, k, c, cum) in enumerate(mt):
        if k < 3:
            continue
        g, inst_, s, ok = test[ri]; fin, tot = s[-1], sum(s)
        lo, p, hi = (max(c, pred[a][j]) for a in (.1, .5, .9))          # final >= current (known)
        ps = max(cum, preds[.5][j])
        e8, er = abs(p - fin), abs(ref(g, k) - fin)
        a = per[inst_]
        a[0] += e8; a[1] += er; a[2] += lo <= fin <= hi; a[3] += 1
        a[4] += abs(ps - tot); a[5] += abs(refs(g, k) - tot)
        b = turn[min(k, 21)]; b[0] += e8; b[1] += er; b[2] += 1
    tot_ = [sum(v[i] for v in per.values()) for i in range(6)]
    n = tot_[3]; ratio = tot_[0] / tot_[1]; cov = tot_[2] / n
    rci = boot(per, 6, lambda a: a[0] / a[1]); cci = boot(per, 6, lambda a: a[2] / a[3])
    sci = boot(per, 6, lambda a: a[4] / a[5])
    okA = ratio <= .8 and rci[1] < .8 and cci[0] <= .8 <= cci[1]
    if not SMOKE:   # pairing check: reference reproduces R7 exactly
        assert len(test) == r7["test"] and n == r7["update_points"], (len(test), n)
        assert abs(tot_[1] / n - r7["mae_ref"]) < 1e-6 * r7["mae_ref"], (tot_[1] / n, r7["mae_ref"])
    print(f"  A update points k>=3: {n:,} | MAE R8 {tot_[0]/n:,.0f} vs k-only {tot_[1]/n:,.0f} | ratio {ratio:.3f} "
          f"[{rci[0]:.3f},{rci[1]:.3f}] | coverage {cov:.1%} [{cci[0]:.1%},{cci[1]:.1%}] | {'PASS' if okA else 'fail'}")
    if not SMOKE:
        print(f"    vs R7 table (descriptive): R8 MAE / R7 MAE = {tot_[0]/n/r7['mae_r7']:.3f}")
    print(f"  A secondary, summed size (descriptive): ratio {tot_[4]/tot_[5]:.3f} [{sci[0]:.3f},{sci[1]:.3f}]")
    turns = {}
    for k in sorted(turn):
        b = turn[k]; lab = "21+" if k == 21 else str(k)
        turns[lab] = {"n": b[2], "mae_r8": b[0] / b[2], "mae_ref": b[1] / b[2]}
        print(f"    turn {lab:>3}: n={b[2]:8,} | R8 {b[0]/b[2]:8,.0f} | k-only {b[1]/b[2]:8,.0f} | ratio {b[0]/b[1]:.2f}", flush=True)

    # ---------- Part B: success-from-here classifier -> stop policy ----------
    yc = np.array([fit[ri][3] for ri, _, _, _ in mf], dtype=float)
    fold_pt = np.array([fold[fit[ri][1]] for ri, _, _, _ in mf])
    oofp = np.zeros(len(Xf))
    for f in range(5):
        m = fold_pt != f
        oofp[~m] = train(Xf[m], yc[m], "binary").predict(Xf[~m])
    pt = train(Xf, yc, "binary").predict(Xt)

    def policy(runs, meta, p, tau):
        """Stop at first update point k >= 3 with p < tau. Returns per-run (tokens, resolved)."""
        out = {}
        for j, (ri, k, c, cum) in enumerate(meta):
            if ri in out and out[ri][2]:
                continue
            s, ok = runs[ri][2], runs[ri][3]
            if k >= 3 and p[j] < tau:
                out[ri] = (cum, False, True)
            elif k == len(s):
                out[ri] = (cum, ok, True)
        return out

    def score(runs, res):
        tok = sum(v[0] for v in res.values()); rs = sum(v[1] for v in res.values())
        return tok, rs

    base_tok = sum(sum(r[2]) for r in fit); base_res = sum(r[3] for r in fit)
    best = (base_tok / base_res, 0.0, 0.0)
    for tau in TAUS[1:]:
        tok, rs = score(fit, policy(fit, mf, oofp, tau))
        lost = 1 - rs / base_res
        if lost <= MAX_LOST and rs and tok / rs < best[0]:
            best = (tok / rs, tau, lost)
    tau = best[1]
    print(f"  B tau chosen on fit (out-of-fold): {tau:.2f} (fit tokens/resolved {best[0]/(base_tok/base_res)-1:+.1%}, lost {best[2]:.1%})")
    pol = policy(test, mt, pt, tau)
    # group-p90 alarm on summed size (as alarm_runtime_r5, thresholds from this fit set)
    th = {g: q(v, .9) for g, v in gss.items()}
    per_b = {}
    for ri, (g, inst_, s, ok) in enumerate(test):
        tot = sum(s); cum = 0.0; al = (tot, ok)
        for c in s:
            cum += c
            if cum > th[g]:
                al = (cum, False); break
        pv = pol[ri]
        a = per_b.setdefault(inst_, [0.0] * 6)
        a[0] += tot; a[1] += ok; a[2] += pv[0]; a[3] += pv[1]; a[4] += al[0]; a[5] += al[1]
    T = [sum(v[i] for v in per_b.values()) for i in range(6)]
    tpr = lambda a, t, r: (a[t] / a[r]) if a[r] else float("inf")
    r_ns = tpr(T, 2, 3) / tpr(T, 0, 1); r_al = tpr(T, 2, 3) / tpr(T, 4, 5)
    lost = 1 - T[3] / T[1]; lost_al = 1 - T[5] / T[1]
    ci_ns = boot(per_b, 6, lambda a: tpr(a, 2, 3) / tpr(a, 0, 1))
    ci_al = boot(per_b, 6, lambda a: tpr(a, 2, 3) / tpr(a, 4, 5))
    okB = r_ns <= .9 and ci_ns[1] < .9 and lost <= MAX_LOST and ci_al[1] < 1.0
    yt = np.array([test[ri][3] for ri, _, _, _ in mt]); auc = _auc(pt, yt)
    print(f"  B classifier AUC on test update points (descriptive): {auc:.3f}")
    print(f"  B tokens/resolved vs no-stop {r_ns:.3f} [{ci_ns[0]:.3f},{ci_ns[1]:.3f}] | lost {lost:.1%} | "
          f"vs p90 alarm {r_al:.3f} [{ci_al[0]:.3f},{ci_al[1]:.3f}] (alarm vs no-stop {tpr(T,4,5)/tpr(T,0,1):.3f}, "
          f"alarm lost {lost_al:.1%}) | {'PASS' if okB else 'fail'}", flush=True)
    return {"fit_runs": len(fit), "test_runs": len(test), "skipped": skipped, "update_points": n,
            "A": {"mae_r8": tot_[0] / n, "mae_ref": tot_[1] / n, "ratio": ratio, "ratio_ci": list(rci),
                  "coverage": cov, "coverage_ci": list(cci), "pass": bool(okA),
                  "summed_ratio": tot_[4] / tot_[5], "summed_ratio_ci": list(sci), "turns": turns,
                  "vs_r7": (tot_[0] / n / r7["mae_r7"]) if not SMOKE else None},
            "B": {"tau": tau, "fit_lost": best[2], "auc": auc, "ratio_vs_nostop": r_ns, "ratio_vs_nostop_ci": list(ci_ns),
                  "lost": lost, "ratio_vs_alarm": r_al, "ratio_vs_alarm_ci": list(ci_al),
                  "alarm_vs_nostop": tpr(T, 4, 5) / tpr(T, 0, 1), "alarm_lost": lost_al, "pass": bool(okB)}}


def _auc(p, y):
    o = np.argsort(p, kind="mergesort"); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    # average ranks for ties
    ps = p[o]; i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    npos = y.sum(); nneg = len(y) - npos
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


if __name__ == "__main__":
    r7 = json.load(open(os.path.join(HERE, "forecast_r7.json")))
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    res = {ds: analyse(ds, r7[ds]) for ds in DATASETS if not only or ds in only}
    if SMOKE:
        print("\nSMOKE (fit-internal) done; no verdict."); sys.exit(0)
    for part in ("A", "B"):
        npass = sum(res[d][part]["pass"] for d in DATASETS)
        res[f"verdict_{part}"] = "PASS" if npass >= 2 else "FAIL"
        print(f"R8 part {part}: passes in {npass}/3 datasets -> {res[f'verdict_{part}']} (need >= 2)")
    json.dump(res, open(os.path.join(HERE, "forecast_r8.json"), "w"), indent=1)
    print("wrote forecast_r8.json")
