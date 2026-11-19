"""CLI: precompute tier outputs, then evaluate routing methods (run from anywhere; paths hang off the repo root).

    python src/run.py precompute --dataset hospital [--limit N]
    python src/run.py precompute --dataset hospital --refresh-column
    python src/run.py precompute --dataset hospital --refresh-cheap
    python src/run.py precompute --dataset hospital --refresh-llm-sem [--no-ask] [--direct]
    python src/run.py eval --dataset hospital
    python src/run.py eval --dataset all
    python src/run.py inject-semantic --dataset hospital|beers   # writes data/<name>_sem/
    python src/run.py sweep-summary [--sem]
    python src/run.py figure
    python src/run.py lambda-figure

Env: CACHE_DIR, RESULTS_DIR, LABEL_TUPLES, SPLIT_SEED, LABEL_EXCLUDE_SEM, SAMPLER (random|coverage) override the cache/results
dirs (default cache/b20_s0, results/b20_s0; sweep-summary writes results/summary/, or RESULTS_DIR if set), the label budget / split seed and (*_sem datasets) whether labeled rows avoid injected cells (precompute and eval must agree).
"""
import argparse
import csv
import os
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pipeline as pl

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", pl.BASE_DIR / "results" / "b20_s0"))  # per-dataset eval files of one config
# cross-config aggregates (budget_sweep*.csv, budget_lambda_star*.csv); an explicit RESULTS_DIR override redirects them too
SUMMARY_DIR = Path(os.environ["RESULTS_DIR"]) if "RESULTS_DIR" in os.environ else pl.BASE_DIR / "results" / "summary"
TAU_SWEEP = pl.TAU_GRID


def cmd_precompute(args):
    for name in (pl.DATASETS if args.dataset == "all" else [args.dataset]):
        if args.refresh_cheap:
            pl.refresh_cheap(name)
        elif args.refresh_column:
            pl.refresh_column(name)
        elif args.refresh_llm_sem:
            pl.refresh_llm_sem(name, ask=not args.no_ask, direct=args.direct)
        else:
            pl.precompute(name, limit=args.limit)


def cmd_inject_semantic(args):
    log = pl.write_semantic(args.dataset)
    by = {t: sum(x["type"] == t for x in log) for t in {x["type"] for x in log}}
    print(f"[{args.dataset}_sem] injected {len(log)} cells {by} in {len({x['entity'] for x in log})} entities")


def _extras(name, r, ece):
    """Columns added to every method row: repair accuracy is already in `r`; setup (FD/source proposal) calls, tokens and
    seconds apart and in total_calls; ECE/Brier per tier (raw, calibrated) are properties of the calibrators, so equal in every row."""
    sc = pl.setup_cost(name)
    out = {"tokens": r["prompt_tokens"] + r["completion_tokens"], "setup_llm_calls": sc["calls"], "setup_tokens": sc["tokens"],
           "setup_seconds": sc["seconds"], "total_calls": r["llm_calls"] + sc["calls"],
           "total_call_rate": (r["llm_calls"] + sc["calls"]) / r["n"] if r["n"] else 0.0}
    for tier, e in ece.items():
        out.update({f"ece_raw_{tier}": e["ece_before"], f"ece_cal_{tier}": e["ece_after"],
                    f"brier_raw_{tier}": e["brier_before"], f"brier_cal_{tier}": e["brier_after"], f"ece_n_{tier}": e["n"]})
    return out


def _eval_one(name, tau_default=0.8):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cells = pl.load_cache(name)
    if not cells:
        print(f"[{name}] no cached cells, run `precompute` first")
        return
    calib, test = pl.calib_test_split(cells)
    calibrators = pl.fit_calibrators(calib)

    taus = pl.method_taus(calib, calibrators, tau_default)  # auto taus are picked on calib cells only
    auto, insample = taus["ours_auto"], taus["ours_auto_insample"]
    print(f"\n=== {name}: methods @ tau={tau_default}, ours_auto tau={auto} (in-sample {insample}) ({len(test)} test cells) ===")
    ece = pl.ece_report(test, calibrators)
    rows = []
    for method in pl.METHODS:
        r = pl.route(test, method, taus[method], calibrators)
        r.update(_extras(name, r, ece))  # setup calls (FD/source proposals) are not part of llm_calls
        rows.append(r)
        print(f"{method:20s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
              f"llm_calls={r['llm_calls']:5d} llm_sec={r['llm_seconds']:.2f} "
              f"frac_to_llm={r['frac_to_llm']:.3f} latency={r['latency_seconds']:.2f}s")

    with open(RESULTS_DIR / f"{name}_methods.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # budget sweep: one row per lambda (tau_L chosen per lambda on the labeled rows) + which lambda ours_budget chose
    by = {r["method"]: r for r in rows}
    lam_rows = []
    for lam in pl.LAMBDAS:
        x = by[f"ours_lambda@{lam}"]
        lam_rows.append({"lambda": lam, "chosen": int(lam == calibrators["_lambda"]), "tau": x["tau"], "precision": x["precision"],
                         "recall": x["recall"], "f1": x["f1"], "llm_calls": x["llm_calls"], "total_calls": x["total_calls"],
                         "llm_call_rate": x["llm_call_rate"], "total_call_rate": x["total_call_rate"],
                         "prompt_tokens": x["prompt_tokens"], "completion_tokens": x["completion_tokens"], "tokens": x["tokens"],
                         "llm_seconds": x["llm_seconds"], "latency_per_cell": x["latency_per_cell"],
                         **{k: v for k, v in x.items() if k.startswith(("sem_", "nonsem_"))}})
    with open(RESULTS_DIR / f"{name}_lambda.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lam_rows[0].keys()))
        w.writeheader()
        w.writerows(lam_rows)
    print(f"ours_budget: lambda*={calibrators['_lambda']} tau={taus['ours_budget']}")

    print(f"\n=== {name}: ECE (10 bins) and Brier on test split ===")
    for tier, r in ece.items():
        print(f"{tier:8s} n={r['n']:5d} ece_before={r['ece_before']:.3f} ece_after={r['ece_after']:.3f} "
              f"brier_before={r['brier_before']:.3f} brier_after={r['brier_after']:.3f}")

    # tau sweep for `ours`, plus the other methods as fixed reference points
    tau_rows = []
    for tau in TAU_SWEEP:
        r = pl.route(test, "ours", tau, calibrators)
        tau_rows.append(r)
    with open(RESULTS_DIR / f"{name}_tau.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tau_rows[0].keys()))
        w.writeheader()
        w.writerows(tau_rows)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot([r["llm_calls"] for r in tau_rows], [r["f1"] for r in tau_rows], "o-", label="ours (tau sweep)")
    for r in rows:
        if r["method"] == "ours" or r["method"].startswith(("ours_sem_direct@", "ours_sem_guard@")):  # the variants stay out of the paper's pareto plots
            continue
        ax.scatter([r["llm_calls"]], [r["f1"]], marker="s")
        ax.annotate(r["method"], (r["llm_calls"], r["f1"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("LLM calls (test split)")
    ax.set_ylabel("F1")
    ax.set_title(f"{name}: cost vs. quality")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{name}_pareto.png", dpi=150)
    plt.close(fig)
    print(f"[{name}] wrote results/{name}_methods.csv, results/{name}_tau.csv, results/{name}_pareto.png")


def cmd_eval(args):
    names = pl.DATASETS if args.dataset == "all" else [args.dataset]
    for name in names:
        _eval_one(name, tau_default=args.tau)

BUDGETS, SEEDS = [20, 50], [0, 1, 2]
SFX = "_cov" if pl.SAMPLER == "coverage" else ""  # SAMPLER=coverage keeps its own cache/results dirs and figure files


def _cache_dir(n, s, xsem):
    """cache/b{N}_s{S}[_xsem][_cov]/; b20_s0 is the default (random 20-row seed-0 mixed) config."""
    return pl.BASE_DIR / "cache" / f"b{n}_s{s}{xsem}{SFX}"


def cmd_sweep_summary(args):
    """Mean/std over SEEDS per dataset x method x budget, recomputed from cache/b{N}_s{S}[_xsem][_cov]/ (tau=0.8).
    --sem: the *_sem datasets (with LABEL_EXCLUDE_SEM=1 their caches live in the *_xsem dirs)."""
    ms = lambda xs: (statistics.mean(xs), statistics.stdev(xs))
    out, table, lam_star = [], [], []
    xsem = "_xsem" if pl.LABEL_EXCLUDE_SEM else ""
    for name in (pl.SEM_DATASETS if args.sem else pl.DATASETS):
        for n in BUDGETS:
            runs = []
            for s in SEEDS:
                pl.CACHE_DIR = _cache_dir(n, s, xsem)
                calib, test = pl.calib_test_split(pl.load_cache(name), s, n)
                cals = pl.fit_calibrators(calib)
                taus = pl.method_taus(calib, cals)
                res = {m: pl.route(test, m, taus[m], cals) for m in pl.METHODS}
                ece = pl.ece_report(test, cals)
                for r in res.values():
                    r.update(_extras(name, r, ece))
                runs.append((len(calib), len(test), res, ece))
                lam_star.append({"dataset": name, "budget": n, "seed": s, "lambda": cals["_lambda"], "tau": taus["ours_budget"],
                                 "f1": res["ours_budget"]["f1"], "llm_calls": res["ours_budget"]["llm_calls"]})
            for m in pl.METHODS:
                row = {"dataset": name, "method": m, "budget": n, "n_seeds": len(SEEDS),
                       "n_calib": statistics.mean(r[0] for r in runs), "n_test": statistics.mean(r[1] for r in runs),
                       "setup_llm_calls": pl.setup_llm_calls(name)}
                for k in ("precision", "recall", "repair_accuracy", "f1", "llm_calls", "llm_call_rate", "total_calls", "total_call_rate",
                          "prompt_tokens", "completion_tokens", "tokens", "llm_seconds", "latency_seconds", "latency_per_cell",
                          *[k for k in runs[0][2][m] if k.startswith(("sem_", "nonsem_", "ece_", "brier_", "sys_"))]):
                    row[f"{k}_mean"], row[f"{k}_std"] = ms([r[2][m][k] for r in runs])
                row["tau_by_seed"] = " ".join(str(r[2][m]["tau"]) for r in runs) if m.startswith("ours") else ""
                out.append(row)
            f1 = {r["method"]: r for r in out[-len(pl.METHODS):]}
            table.append([name, n] + [f"{f1[m]['f1_mean']:.3f}+-{f1[m]['f1_std']:.3f}" for m in
                                      ("ours", "ours_auto", "ours_auto_insample", "ours_no_calib", "cascade", "llm_only", "rules_stats")]
                         + [f"{f1['ours']['llm_calls_mean']:.0f}", f"{f1['ours_auto']['llm_calls_mean']:.0f}",
                            f1["ours_auto"]["tau_by_seed"], f1["ours_auto_insample"]["tau_by_seed"]]
                         + [f"{statistics.mean(r[3][t]['ece_after'] for r in runs):.3f}" for t in pl.TIER_ORDER])
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUMMARY_DIR / f"budget_sweep{'_sem' if args.sem else ''}{xsem}{SFX}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    star_path = SUMMARY_DIR / f"budget_lambda_star{'_sem' if args.sem else ''}{xsem}{SFX}.csv"  # ours_budget's lambda* per dataset x budget x seed
    with open(star_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lam_star[0].keys()))
        w.writeheader()
        w.writerows(lam_star)
    hdr = ["dataset", "budget", "ours", "ours_auto", "insample", "no_calib", "cascade", "llm_only", "rules_stats", "ours_calls",
           "auto_calls", "auto_tau", "insample_tau", "ece_rules", "ece_fd", "ece_stats", "ece_col", "ece_llm"]
    print("F1 mean+-std over seeds (ours @ tau=0.8, ours_auto @ tau chosen per seed); *_calls = mean llm_calls; ece = calibrated, mean over seeds")
    for row in [hdr] + table:
        print("  ".join(f"{c!s:>13}" for c in row))
    print(f"wrote {out_path}")


def cmd_figure(args):
    out_dir = pl.BASE_DIR / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    floor = 1e-3  # rules_stats / zero-call points sit here, tick labelled "0"
    style = {"ours_auto": ("P", "#1f77b4"), "llm_only": ("s", "0.15"), "cascade": ("^", "0.40"), "ours_no_calib": ("D", "0.55"),
             "ours_no_column": ("v", "0.70"), "rules_stats": ("X", "0.30")}
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
    fig, axes = plt.subplots(1, len(pl.DATASETS), figsize=(1.75 * len(pl.DATASETS), 1.9), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], pl.DATASETS):
        with open(RESULTS_DIR / f"{name}_methods.csv") as f:
            m = {r["method"]: r for r in csv.DictReader(f)}
        with open(RESULTS_DIR / f"{name}_tau.csv") as f:
            tau = list(csv.DictReader(f))
        base = float(m["llm_only"]["llm_calls"])
        fx = lambda r: max(float(r["llm_calls"]) / base, floor)
        tau = [r for r in tau if float(r["tau"]) <= 0.85]  # tau >= 0.9 drop looks like a glitch: not plotted
        ax.plot([fx(r) for r in tau], [float(r["f1"]) for r in tau], "o-", color="#d1330f", ms=2.5, lw=1.2,
                label="ours (tau sweep)", zorder=3)
        t8 = next(r for r in tau if abs(float(r["tau"]) - 0.8) < 1e-9)
        ax.plot([fx(t8)], [float(t8["f1"])], "o", color="#d1330f", mec="white", ms=6, mew=0.6,
                label="ours, tau=0.8", zorder=4)
        for meth, (mk, c) in style.items():
            hollow = meth in ("rules_stats", "ours_no_column")  # these sit on top of other baselines
            ax.plot([fx(m[meth])], [float(m[meth]["f1"])], mk, color=c, ms=6 if hollow else 4.5, ls="none", label=meth,
                    mfc="none" if hollow else c, mew=0.9 if hollow else 1.0, zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(floor / 2, 3)
        ax.set_xticks([floor, 1e-2, 1e-1, 1])
        ax.set_xticklabels(["0", "1e-2", "0.1", "1"])
        ax.minorticks_off()
        ax.set_ylim(0, 1)
        ax.set_title(name.capitalize(), fontsize=7.5)
        ax.grid(axis="y", lw=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0][0].set_ylabel("Repair F1")
    fig.supxlabel("LLM calls / llm_only calls", fontsize=7)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, fontsize=6.5, handletextpad=0.3, columnspacing=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.9), pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"frontier{SFX}.{ext}", dpi=300)
    print(f"wrote {out_dir}/frontier{SFX}.pdf, .png")


LAMBDA_PANELS = [(d, "budget_sweep.csv") for d in pl.DATASETS] + [("beers_sem", "budget_sweep_sem_cov.csv"), ("hospital_sem", "budget_sweep_sem.csv")]


def cmd_lambda_figure(args):
    """paper/figures/lambda_budget.{pdf,png}: F1 vs LLM calls per error cell (log x, 0 on the floor) along the lambda grid of the final model ours_sem_guard@L ("Ours"),
    mean over seeds at the 20-row budget, from the sweep-summary csvs: one panel per dataset (beers_sem: coverage sampler; hospital_sem:
    random; both mixed), 4 + 3 panels with the legend in the free cell. Only lambda in {-inf, 0, 0.5, inf} is labelled (a label that would
    collide with a marker or another label is skipped); the star is ours_budget (lambda* chosen per seed on the labeled rows, listed in each
    title); dashed = semantic-cell F1 on the *_sem panels. On the beers_sem panel the older ours_lambda@L curve is drawn grey dash-dotted for
    comparison (the guard differs from it only there, at lambda > 0); ours_budget stays in the ours_lambda family."""
    out_dir = pl.BASE_DIR / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6})
    sweeps = {f: list(csv.DictReader(open(SUMMARY_DIR / f))) for f in {f for _, f in LAMBDA_PANELS} if (SUMMARY_DIR / f).exists()}
    panels = [(n, f) for n, f in LAMBDA_PANELS if any(r["dataset"] == n for r in sweeps.get(f, []))]
    ncol = 4
    fig, axes = plt.subplots(2, ncol, figsize=(7.0, 3.6), sharey=True, squeeze=False)
    axes = axes.ravel()
    floor = 1e-3
    marks = []  # (ax, x, y, markersize in pt) of every marker, for the label collision test
    wanted = []  # (ax, x, y, text)
    for ax, (name, fn) in zip(axes, panels):
        m = {r["method"]: r for r in sweeps[fn] if r["dataset"] == name and int(r["budget"]) == 20}
        star = SUMMARY_DIR / ("budget_lambda_star" + fn[len("budget_sweep"):])
        ls = "/".join(s["lambda"] for s in csv.DictReader(open(star)) if s["dataset"] == name and s["budget"] == "20") if star.exists() else ""
        fx = lambda r: max(float(r["llm_call_rate_mean"]), floor)
        fy = lambda r: float(r["f1_mean"])
        rows = [m[f"ours_sem_guard@{l}"] for l in pl.LAMBDAS]
        if name == "beers_sem":  # faint comparison: the ours_lambda@L curve the guard replaced
            old = [m[f"ours_lambda@{l}"] for l in pl.LAMBDAS]
            ax.plot([fx(r) for r in old], [fy(r) for r in old], "-.", color="0.5", lw=0.8, alpha=0.8, label="ours_lambda@L (no guard)", zorder=2)
            ax.plot([fx(r) for r in old], [float(r["sem_f1_mean"]) for r in old], "-.", color="0.5", lw=0.8, alpha=0.8, zorder=2)  # its semantic F1: where the guard differs
        ax.plot([fx(r) for r in rows], [fy(r) for r in rows], "o-", color="#d1330f", ms=2.2, lw=1.1, label="Ours (ours_sem_guard@L)", zorder=3)
        marks += [(ax, fx(r), fy(r), 2.2) for r in rows]
        if "sem_f1_mean" in rows[0]:
            ax.plot([fx(r) for r in rows], [float(r["sem_f1_mean"]) for r in rows], "o--", color="#d1330f", ms=1.6, lw=0.8, alpha=0.6,
                    label="semantic-cell F1", zorder=2)
            marks += [(ax, fx(r), float(r["sem_f1_mean"]), 1.6) for r in rows]
        for l, r in zip(pl.LAMBDAS, rows):
            if l in ("-inf", "0", "0.5", "inf"):
                wanted.append((ax, fx(r), fy(r), l.replace("inf", "\u221e")))
        b = m["ours_budget"]
        ax.plot([fx(b)], [fy(b)], "*", color="#f2b705", mec="black", mew=0.6, ms=9, label="ours_budget (\u03bb* in title)", zorder=5)
        marks.append((ax, fx(b), fy(b), 9))
        for meth, mk, c in (("ours_auto", "P", "#1f77b4"), ("llm_only", "s", "0.15"), ("static_gate", "^", "0.4")):
            x, y = fx(m[meth]), fy(m[meth])
            if meth == "llm_only" and abs(x / fx(m["ours_lambda@-inf"]) - 1) < 0.01 and abs(y - fy(m["ours_lambda@-inf"])) < 0.01:
                continue  # coincides with lambda=-inf
            ax.plot([x], [y], mk, color=c, ms=4.2, ls="none", label=meth, mfc="none")
            marks.append((ax, x, y, 4.2))
        ax.set_xscale("log")
        ax.set_xlim(floor / 2, 3)
        ax.set_xticks([floor, 1e-2, 1e-1, 1])
        ax.set_xticklabels(["0", "1e-2", "0.1", "1"])
        ax.minorticks_off()
        ax.set_ylim(0, 1.02)
        ax.set_title(name + (" (cov)" if "cov" in fn else "") + (f"\n\u03bb*={ls}" if ls else ""), fontsize=6.5)
        ax.grid(axis="y", lw=0.4, alpha=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[len(panels):]:
        ax.axis("off")
    hl = {}
    for ax in axes[:len(panels)]:
        hl.update(zip(*reversed(ax.get_legend_handles_labels())))
    axes[-1].legend(list(hl.values()), list(hl.keys()), loc="center left", frameon=False, fontsize=6, borderaxespad=0)
    for r in range(2):
        axes[r * ncol].set_ylabel("Repair F1")
    fig.supxlabel("LLM calls per error cell (log scale, 0 on the left)", fontsize=6.5)
    fig.tight_layout(pad=0.4, h_pad=0.8)
    fig.canvas.draw()
    # labels: small offset text that must not touch a marker or an earlier label; try below, above, right, left, else skip
    def disp(ax, x, y): return ax.transData.transform((x, y))
    placed = []
    for ax, x, y, text in wanted:
        for dx, dy, ha, va in ((0, -6, "center", "top"), (0, 6, "center", "bottom"), (6, 0, "left", "center"), (-6, 0, "right", "center"),
                               (0, -13, "center", "top"), (0, 13, "center", "bottom")):
            t = ax.annotate(text, (x, y), fontsize=5.5, xytext=(dx, dy), textcoords="offset points", ha=ha, va=va)
            bb = t.get_window_extent().expanded(1.05, 1.15)
            hit = False
            for a, mx, my, ms in marks:
                if a is not ax:
                    continue
                px, py = disp(a, mx, my)
                r = ms / 2 * fig.dpi / 72 + 1.5  # marker radius in px
                if bb.x0 - r < px < bb.x1 + r and bb.y0 - r < py < bb.y1 + r:
                    hit = True
                    break
            axb = ax.get_window_extent()
            hit = hit or any(bb.overlaps(o) for o in placed) or not (axb.x0 <= bb.x0 and bb.x1 <= axb.x1 and axb.y0 <= bb.y0 and bb.y1 <= axb.y1)
            if hit:
                t.remove()
                continue
            placed.append(bb)
            break
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"lambda_budget.{ext}", dpi=300)
    print(f"wrote {out_dir}/lambda_budget.pdf, .png ({[n for n, _ in panels]})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("inject-semantic")
    p0.add_argument("--dataset", required=True, choices=list(pl.SEM_CFG))
    p0.set_defaults(func=cmd_inject_semantic)

    p1 = sub.add_parser("precompute")
    p1.add_argument("--dataset", required=True, choices=pl.DATASETS + pl.SEM_DATASETS + ["all"])
    p1.add_argument("--limit", type=int, default=None)
    p1.add_argument("--refresh-column", action="store_true",
                    help="recompute only the column tier of the existing cache, in place")
    p1.add_argument("--refresh-cheap", action="store_true",
                    help="recompute only rules/fd/stats/error_type/group of the existing cache, in place")
    p1.add_argument("--refresh-llm-sem", action="store_true",
                    help="compute only the llm_sem tier (cells no other tier resolves) in the existing cache, in place")
    p1.add_argument("--no-ask", action="store_true",
                    help="with --refresh-llm-sem: make no LLM call, only reuse the stored answers")
    p1.add_argument("--direct", action="store_true",
                    help="with --refresh-llm-sem: also ask the cells only ours_sem_direct reaches (direct-eligible, not old-eligible)")
    p1.set_defaults(func=cmd_precompute)

    p2 = sub.add_parser("eval")
    p2.add_argument("--dataset", required=True, choices=pl.DATASETS + pl.SEM_DATASETS + ["all"])
    p2.add_argument("--tau", type=float, default=0.8)
    p2.set_defaults(func=cmd_eval)

    p3 = sub.add_parser("sweep-summary")
    p3.add_argument("--sem", action="store_true")
    p3.set_defaults(func=cmd_sweep_summary)
    sub.add_parser("lambda-figure").set_defaults(func=cmd_lambda_figure)
    sub.add_parser("figure").set_defaults(func=cmd_figure)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
