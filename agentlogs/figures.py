#!/usr/bin/env python3
"""Build every figure for the /journal article from committed/derived data.
Inputs: full_rows/*.jsonl (rebuilt by run_full.py), locked_test.out, h5_final.json,
confirm_r3.json. Outputs: figures/fig{1..5}.{svg,png} + figures/groups_full.csv.
Figures 1, 3, 4, 5 are descriptive (in-sample, all 424k priced sessions).
Figure 2 plots held-out results exactly as written by the pre-registered scripts."""
import csv, glob, json, os, re
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import lab

q = lab.quantile
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "figures"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "figure.dpi": 110})
INK, ACC, MUTED, BAR = "#222222", "#2b6cb0", "#999999", "#c53030"

rows = [r for f in sorted(glob.glob(os.path.join(ROOT, "full_rows", "*.jsonl"))) for r in map(json.loads, open(f))
        if r["calls"] > 3 and r["cost"] and r["model"] != "unknown"]
print(f"{len(rows)} priced sessions with >= 4 calls")
b = lambda L: 0 if (L or 0) < 200 else 1 if L < 1500 else 2
base = lambda r: (r["model"], r["event"], b(r["prompt_chars"]))
groups = defaultdict(list)
for r in rows:
    groups[base(r)].append(r)
big = {g: v for g, v in groups.items() if len(v) >= 100}


def save(fig, name):
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", name)


# ---- Fig 1: cost distribution per model (log scale) ----
bym = defaultdict(list)
for r in rows:
    bym[r["model"]].append(r["cost"])
models = [m for m, v in sorted(bym.items(), key=lambda kv: -len(kv[1])) if len(v) >= 1000]
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.boxplot([bym[m] for m in models], whis=(10, 90), showfliers=False, orientation="horizontal",
           medianprops={"color": ACC, "linewidth": 2}, boxprops={"color": INK}, whiskerprops={"color": INK}, capprops={"color": INK})
for i, m in enumerate(models, 1):
    mean = sum(bym[m]) / len(bym[m])
    ax.plot(mean, i, "D", color=BAR, ms=5, label="mean" if i == 1 else None)
ax.set_xscale("log")
t1 = [0.1, 0.2, 0.5, 1, 2, 5, 10]
ax.xaxis.set_major_locator(FixedLocator(t1)); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
ax.xaxis.set_minor_formatter(NullFormatter()); ax.grid(axis="x", alpha=.25)
ax.set_yticks(range(1, len(models) + 1), [f"{m} (n={len(bym[m]):,})" for m in models])
ax.set_xlabel("session cost, USD (log scale)\nbox = p25 to p75, blue line = median, whiskers = p10 to p90")
ax.legend(loc="lower right", frameon=False)
ax.set_title("Fig. 1  Cost per session by model\nthe mean sits well above the median for every model", loc="left")
save(fig, "fig1_cost_by_model")

# ---- Fig 2: width ratio per hypothesis vs the 0.8 bar ----
series = {}
h1 = []
for line in open(os.path.join(ROOT, "locked_test.out")):
    m = re.search(r"group\s+([\d.]+)x .*model\s+([\d.]+)x", line)
    if m:
        h1.append((float(m.group(1)) / float(m.group(2)), None))
series["H1  model x trigger\n(locked test, 15 groups)"] = h1
h5 = json.load(open(os.path.join(ROOT, "h5_final.json")))
series["H5  prompt-text clusters\n(10 clusters)"] = [(x["ratio"], x["ratio_ci_hi"]) for x in h5["results"]]
c3 = json.load(open(os.path.join(ROOT, "confirm_r3.json")))
series["H6  repo language\n(37 groups)"] = [(x["ratio"], x["ratio_ci_hi"]) for x in c3["H6"]["groups"]]
series["H7  user's past runs\n(37 groups)"] = [(x["ratio"], x["ratio_ci_hi"]) for x in c3["H7"]["groups"]]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for i, (name, pts) in enumerate(series.items()):
    y = len(series) - i
    for j, (r, hi) in enumerate(pts):
        yy = y + (j / max(1, len(pts) - 1) - .5) * .5
        if hi is not None:
            ax.plot([r, hi], [yy, yy], color=MUTED, lw=.8, alpha=.6)
        ok = hi is not None and hi < .8
        ax.plot(r, yy, "o", color=ACC if ok else INK, mfc=ACC if ok else ("none" if hi is None else INK), ms=3.5)
ax.axvline(1.0, color=MUTED, ls=":", lw=1)
ax.axvline(0.8, color=BAR, lw=1.5)
ax.set_ylim(0.6, len(series) + .75)
ax.text(0.79, len(series) + .62, "bar 0.8", color=BAR, ha="right", va="center", fontsize=8.5)
ax.text(1.01, len(series) + .62, "no gain", color=MUTED, ha="left", va="center", fontsize=8.5)
ax.set_yticks(range(len(series), 0, -1), list(series))
ax.set_xscale("log"); ax.xaxis.set_major_locator(FixedLocator([.3, .5, .8, 1, 1.5, 2, 3]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("range width relative to the simpler baseline (below 1 = narrower)\ndot = estimate, grey line = up to 90% CI upper bound; blue = width criteria met (ratio and CI upper below 0.8); both blue H5 clusters then failed held-out coverage\nH1 hollow: locked test reported point widths only (its bar was a separate pass rule)", fontsize=8.5)
ax.set_title("Fig. 2  Which inputs narrow the cost range on held-out repos", loc="left", pad=10)
save(fig, "fig2_hypotheses")

# ---- Fig 3: range around the median vs confidence ----
levels = [.2, .4, .5, .6, .8, .9, .95]
lo_pct, hi_pct = [], []
for c in levels:
    los, his = [], []
    for v in big.values():
        cs = [r["cost"] for r in v]; med = q(cs, .5)
        los.append(q(cs, .5 - c / 2) / med - 1); his.append(q(cs, .5 + c / 2) / med - 1)
    lo_pct.append(100 * q(los, .5)); hi_pct.append(100 * q(his, .5))
fig, ax = plt.subplots(figsize=(7.5, 3.8))
x = [100 * c for c in levels]
ax.fill_between(x, lo_pct, hi_pct, color=ACC, alpha=.2)
ax.plot(x, hi_pct, "o-", color=ACC, label="upper edge"); ax.plot(x, lo_pct, "o-", color=INK, label="lower edge")
for xi, l, h in zip(x, lo_pct, hi_pct):
    ax.annotate(f"+{h:.0f}%", (xi, h), textcoords="offset points", xytext=(-4 if xi > 20 else 6, 8), ha="right" if xi > 20 else "left", fontsize=8, color=ACC)
    ax.annotate(f"{l:.0f}%", (xi, l), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8)
ax.axhline(0, color=MUTED, lw=.8); ax.set_ylim(-140, 520); ax.set_xlim(15, 99)
ax.xaxis.set_major_locator(FixedLocator(x))
ax.set_xlabel("confidence that the cost lands inside the range (%)")
ax.set_ylabel("distance from typical cost (%)")
ax.set_title(f"Fig. 3  The price of confidence\nmedian over {len(big)} model × trigger groups", loc="left")
ax.legend(frameon=False, loc="upper left")
save(fig, "fig3_confidence_vs_range")

# ---- Fig 4: concentration of spend ----
cs = sorted((r["cost"] for r in rows), reverse=True); tot = sum(cs)
xs, ys, acc = [0], [0], 0
for i, c in enumerate(cs, 1):
    acc += c
    if i % 200 == 0 or i == len(cs):
        xs.append(100 * i / len(cs)); ys.append(100 * acc / tot)
tail = sum(r["cost"] for v in big.values() for p90 in [q([r["cost"] for r in v], .9)] for r in v if r["cost"] > p90)
bigtot = sum(r["cost"] for v in big.values() for r in v)
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.plot(xs, ys, color=ACC, lw=2, label="all sessions, most expensive first")
ax.plot([0, 100], [0, 100], color=MUTED, ls=":", label="equal spend per session")
for p in (1, 10):
    y = next(yy for xx, yy in zip(xs, ys) if xx >= p)
    ax.plot(p, y, "o", color=BAR); ax.annotate(f"top {p}% of sessions = {y:.0f}% of spend", (p, y),
                                               xytext=(45, 34 if p == 10 else 24), fontsize=9,
                                               arrowprops=dict(arrowstyle="-", color=MUTED, lw=.7))
ax.set_xlabel("share of sessions (%)"); ax.set_ylabel("share of total spend (%)")
ax.set_title(f"Fig. 4  Spend is concentrated\n${tot:,.0f} over {len(cs):,} sessions; within each model × trigger group,\n"
             f"sessions above the group's p90 carry {100*tail/bigtot:.0f}% of that group's spend", loc="left")
ax.legend(frameon=False, loc="lower right")
save(fig, "fig4_spend_concentration")

# ---- Fig 5: the ceiling (range width even with true call count known) ----
def deciles(v, key):
    xs = sorted(key(r) for r in v)
    return [xs[int(len(xs) * k / 10)] for k in range(1, 10)]
w_base, w_calls, w_tok = [], [], []
for v in big.values():
    c = [r["cost"] for r in v]; w_base.append(q(c, .9) / q(c, .1))
    for key, out in ((lambda r: r["calls"], w_calls), (lambda r: r["tokens"], w_tok)):
        cut = deciles(v, key); cells = defaultdict(list)
        for r in v:
            cells[sum(key(r) >= x for x in cut)].append(r["cost"])
        ws = [(len(cc), q(cc, .9) / q(cc, .1)) for cc in cells.values() if len(cc) >= 10]
        out.append(sum(n * w for n, w in ws) / sum(n for n, _ in ws))
labels = ["before the run:\nmodel × trigger", "in hindsight:\n+ true call-count decile", "in hindsight:\n+ true token-count decile"]
meds = [q(w_base, .5), q(w_calls, .5), q(w_tok, .5)]
fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.barh(labels[::-1], meds[::-1], color=[ACC, MUTED, MUTED])
for i, m in enumerate(meds[::-1]):
    ax.text(m, i, f"  {m:.1f}x", va="center")
ax.set_xlim(0, 11.5)
ax.set_xlabel("p90 / p10 cost ratio within group, median over groups (1x = exact)")
ax.set_title("Fig. 5  The ceiling\neven knowing the run's true size afterwards, cost still spreads about 2x", loc="left")
save(fig, "fig5_ceiling")

# ---- full group table ----
with open(os.path.join(OUT, "groups_full.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["model", "event", "prompt_bucket", "n", "at_least_p10", "typical_p50", "p90", "mean", "width_p90_over_p10"])
    for (m, e, pb), v in sorted(big.items(), key=lambda kv: -len(kv[1])):
        c = [r["cost"] for r in v]
        w.writerow([m, e, pb, len(c), f"{q(c,.1):.3f}", f"{q(c,.5):.3f}", f"{q(c,.9):.3f}", f"{sum(c)/len(c):.3f}", f"{q(c,.9)/q(c,.1):.1f}"])
print("wrote groups_full.csv,", len(big), "groups")
print("fig3", [(int(100*c), round(l), round(h)) for c, l, h in zip(levels, lo_pct, hi_pct)])
print("fig5 medians", [round(m, 1) for m in meds], "| fig4 tail share", round(100 * tail / bigtot))
