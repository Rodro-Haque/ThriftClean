"""Experimental variant report: ours_sem_direct@L and ours_sem_guard@L vs ours_lambda@L (replay only, no LLM). Writes results/summary/sem_direct_comparison.csv/.md.

    python src/sem_direct_comparison.py [--md-only]

Per dataset x condition (mixed/xsem; random/coverage sampler) x budget (20/50 rows) x lambda: mean/std over the 3 seeds of F1, P, R,
semantic F1/recall (sem datasets), LLM calls and how many are llm_sem calls (route() re-run with llm_sem costs zeroed), tokens, GPU s, tau.
Cap analysis (sem_direct only): llm_sem's calibrated confidence vs tau_L, and the repairs that a calibrator-free llm_sem (confidence 1) would add.
ours_sem_guard (K=1, the a-priori setting) = sem_direct iff >= K labeled cells have an llm_sem candidate (n_sem), else ours_lambda; K=2/3 are a
sensitivity check (variants ours_sem_guard_K2/_K3). Also the F1-vs-LLM-calls Pareto frontier over the lambda grid of each method, per config.
"""
import copy
import csv
import multiprocessing
import sys
import statistics
from pathlib import Path

import pipeline as pl

SUMMARY = pl.BASE_DIR / "results" / "summary"
OLD, NEW, GUARD = "ours_lambda@", "ours_sem_direct@", "ours_sem_guard@"
VARIANT = {OLD: "ours_lambda", NEW: "ours_sem_direct", GUARD: "ours_sem_guard"}
KS = (2, 3)  # sensitivity only; the main guard is pl.SEM_GUARD_K = 1, fixed a priori
ZERO = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
# condition -> (sampler, xsem); non-sem datasets have no xsem condition
CONDS = [("random", ""), ("coverage", ""), ("random", "_xsem"), ("coverage", "_xsem")]
KEYS = ["precision", "recall", "f1", "llm_calls", "tokens", "llm_seconds", "sem_f1", "sem_recall", "nonsem_f1"]


def ms(xs):
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def no_sem_cost(cells):
    out = copy.deepcopy(cells)
    for c in out:
        c["tiers"]["llm_sem"]["cost"] = dict(ZERO)
        if c.get("llm_sem_direct"):
            c["llm_sem_direct"]["cost"] = dict(ZERO)
    return out


def one(task):
    """All rows of one (dataset, sampler, xsem, budget) cell over the 3 seeds."""
    name, sampler, xs, n = task
    is_sem = name in pl.SEM_DATASETS
    pl.SAMPLER, pl.LABEL_EXCLUDE_SEM = sampler, bool(xs)
    runs, rows = [], []
    for s in (0, 1, 2):
        pl.CACHE_DIR = pl.BASE_DIR / "cache" / f"b{n}_s{s}{xs}{'_cov' if sampler == 'coverage' else ''}"
        calib, test = pl.calib_test_split(pl.load_cache(name), s, n)
        cals = pl.fit_calibrators(calib)
        taus = pl.method_taus(calib, cals)
        test0, uncapped = no_sem_cost(test), {**cals, "llm_sem": 1.0}
        res = {}
        for fam in (OLD, NEW, GUARD):
            for lam in pl.LAMBDAS:
                m = fam + lam
                r = pl.route(test, m, taus[m], cals)
                r["tokens"] = r["prompt_tokens"] + r["completion_tokens"]
                r["sem_calls"] = r["llm_calls"] - pl.route(test0, m, taus[m], cals)["llm_calls"]
                if fam != GUARD:  # the cap analysis / cap-free columns are for ours_lambda and ours_sem_direct only
                    u = pl.route(test, m, taus[m], uncapped)
                    r["uncapped_correct_gain"] = u["correct"] - r["correct"]
                    r["uncapped_f1"] = u["f1"]
                    if "sem_recall" in u:
                        r["uncapped_sem_recall"] = u["sem_recall"]
                res[m] = r
        for k in KS:  # sensitivity: the guard with threshold K (taus re-picked, since tau_L is cross-fitted through the same guard)
            pl.SEM_GUARD_K = k
            for lam in pl.LAMBDAS:
                m = GUARD + lam
                taus[f"K{k}:{m}"] = pl.auto_tau(calib, method=m)
                r = pl.route(test, m, taus[f"K{k}:{m}"], cals)
                r["tokens"] = r["prompt_tokens"] + r["completion_tokens"]
                r["sem_calls"] = r["llm_calls"] - pl.route(test0, m, taus[f"K{k}:{m}"], cals)["llm_calls"]
                res[f"K{k}:{m}"] = r
        pl.SEM_GUARD_K = 1
        runs.append((res, taus, {"n_sem": cals["_n"]["llm_sem"], "prior_llm": cals["_prior"]["llm"], "prior_sem": cals["_prior"]["llm_sem"],
                                 "sem_cal": pl._apply_cal(cals["llm_sem"], 1.0), "n_calib": len(calib),
                                 "n_calib_sem": sum(bool(c.get("semantic")) for c in calib)}))
    for fam, kk in [(f, "") for f in (OLD, NEW, GUARD)] + [(GUARD, f"K{k}:") for k in KS]:
        for lam in pl.LAMBDAS:
            m = kk + fam + lam
            row = {"dataset": name, "condition": ("xsem" if xs else "mixed") if is_sem else "n/a", "sampler": sampler,
                   "budget": n, "lambda": lam, "variant": VARIANT[fam] + (f"_{kk[:-1]}" if kk else "")}
            for k in KEYS + ["sem_calls", "uncapped_correct_gain", "uncapped_f1", "uncapped_sem_recall"]:
                if k in runs[0][0][m]:
                    row[f"{k}_mean"], row[f"{k}_std"] = ms([r[0][m][k] for r in runs])
            row["tau_by_seed"] = " ".join(str(r[1][m]) for r in runs)
            row["prior_llm_by_seed"] = " ".join(f"{r[2]['prior_llm']:.3f}" for r in runs)
            row["prior_sem_by_seed"] = " ".join(f"{r[2]['prior_sem']:.3f}" for r in runs)
            row["llm_sem_cal_by_seed"] = " ".join(f"{r[2]['sem_cal']:.3f}" for r in runs)
            row["n_calib_sem_by_seed"] = " ".join(str(r[2]["n_calib_sem"]) for r in runs)
            row["n_sem_by_seed"] = " ".join(str(r[2]["n_sem"]) for r in runs)
            row["tau_blocks_sem_seeds"] = sum(r[1][m] > r[2]["sem_cal"] for r in runs)
            rows.append(row)
    print(name, sampler, xs or "mixed", n, flush=True)
    return rows


def read_csv():
    rows = list(csv.DictReader(open(SUMMARY / "sem_direct_comparison.csv")))
    for r in rows:
        for k in [k for k, v in r.items() if v == ""]:  # columns a (non-sem) dataset does not have
            del r[k]
        for k, v in r.items():
            if k.endswith(("_mean", "_std")):
                r[k] = float(v)
        r["budget"], r["tau_blocks_sem_seeds"] = int(r["budget"]), int(r["tau_blocks_sem_seeds"])
    return rows


def main():
    if sys.argv[1:] == ["--md-only"]:  # rewrite the .md from the existing .csv
        return write_md(read_csv())
    tasks = [(name, sampler, xs, n) for name in pl.DATASETS + pl.SEM_DATASETS for sampler, xs in CONDS
             if not (xs and name not in pl.SEM_DATASETS) for n in (20, 50)]
    with multiprocessing.Pool(20) as pool:
        rows = [r for part in pool.map(one, tasks, chunksize=1) for r in part]  # pool.map keeps the task order
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(SUMMARY / "sem_direct_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    write_md(rows)


VERDICT = """## Verdict (read this first)

* **lambda <= 0 (incl. the final model's lambda = 0): ours_sem_direct is identical to ours_lambda, by construction.** A direct-eligible cell has no free or column candidate, so
  p_best = 0 and the ordinary llm test `q_llm - lambda*1 > 0` always passes at lambda <= 0; the llm is bought, and if it answers, llm_sem stays closed, as today.
  No new llm_sem call, same F1 / P / R / semantic F1 / tokens / GPU s / tau on every dataset, condition and budget (first two tables). The criterion "overall F1 and semantic F1 both
  increase" is **not met at lambda = 0**, at 0 extra calls.
* The variant can only act when lambda >= q_llm (the llm is not worth its cost) and lambda < q_sem. Of the 468 (config, lambda) pairs (7 datasets, 2 budgets, the sampler/xsem conditions, 13
  lambdas), 33 differ; the table "Every config and lambda where the variant differs" lists them all.
* **Gains** only on beers_sem, mixed labeled rows. Coverage sampler, 20 rows: lambda 0.2 F1 .967 -> .972, semantic F1 .510 -> .718 (+77 calls, all llm_sem); lambda 0.5 F1 .958 -> .967, semantic F1 .103 -> .528
  (+155 calls, 7 -> 162). The final model (lambda = 0) has .972 / .712 at 461 calls, so the variant matches its F1 with 16% fewer calls at lambda 0.2. 50 rows, coverage: lambda 0.3 .969 -> .977, sem F1 .150 -> .541 (+221 calls);
  random, lambda 0.3/0.5: .970 -> .975, sem F1 .126 -> .363 (+82 calls).
* **No gain, only extra calls**: beers_sem mixed/random at 20 rows (+80 calls, F1 and sem F1 unchanged) and every xsem config (+86 to +215 calls, semantic F1 0.184 or 0.000 unchanged): llm_sem answers are bought but
  rejected, because llm_sem's calibrator is the constant 0.5 (no labeled semantic cells) and tau_L > 0.5 in most seeds (cap table). beers / rayyan: +1 to +2 calls, F1 identical.
* **No dataset's F1 drops** at any lambda (minimum F1 difference over all 468 pairs is 0). Flights and hospital never differ. The newly asked cells (465 calls) were answered correctly 4/12 on flights,
  5/18 on hospital, 11/12 beers, 22/27 beers_sem, 159/172 hospital_sem, 153/222 movies, 0/1 rayyan.
"""


ORDER = lambda k: ((pl.DATASETS + pl.SEM_DATASETS).index(k[0]), k[1], k[2] != "random", k[3])
EPS = 1e-9


def gv(r, k):
    return r.get(k + "_mean", 0.0)  # sem F1 is absent on non-sem datasets: 0 on both sides, never a difference


def differs(a, b):
    return any(abs(gv(a, k) - gv(b, k)) > EPS for k in ("f1", "sem_f1", "precision", "recall", "llm_calls"))


def frontier(by, cfg, v):
    """Non-dominated (calls, F1) points over the lambda grid: lower calls and higher F1 are better; duplicates collapse."""
    pts = {(round(by[(*cfg, lam, v)]["llm_calls_mean"], 6), round(by[(*cfg, lam, v)]["f1_mean"], 9)) for lam in pl.LAMBDAS}
    return sorted(p for p in pts if not any(q != p and q[0] <= p[0] and q[1] >= p[1] for q in pts))


def pareto(by, cfg, new, old="ours_lambda"):
    """dominates / equal / dominated / incomparable: new's frontier vs old's. new weakly dominates a point iff it has a point with
    calls <= and F1 >= (EPS tolerance); 'dominates' = every old point is weakly dominated and some new point is not weakly dominated by old."""
    fn, fo = frontier(by, cfg, new), frontier(by, cfg, old)
    wd = lambda f, p: any(q[0] <= p[0] + EPS and q[1] >= p[1] - EPS for q in f)
    new_ok, old_ok = all(wd(fn, p) for p in fo), all(wd(fo, p) for p in fn)
    return "equal" if new_ok and old_ok else "dominates" if new_ok else "dominated" if old_ok else "incomparable"


def guard_pairs(by, v="ours_sem_guard", base="ours_lambda"):
    """[(cfg, lam, old row, new row)] over every (config, lambda) pair."""
    cfgs = sorted({k[:4] for k in by}, key=ORDER)
    return [(c, lam, by[(*c, lam, base)], by[(*c, lam, v)]) for c in cfgs for lam in pl.LAMBDAS]


def pair_stats(pairs):
    """counts over (config, lambda) pairs: differing / F1 worse / sem F1 worse / extra calls with no gain (no F1 or sem F1 gain, nothing else better)."""
    d = [(c, l, o, g) for c, l, o, g in pairs if differs(o, g)]
    worse = [x for x in d if gv(x[3], "f1") < gv(x[2], "f1") - EPS or gv(x[3], "sem_f1") < gv(x[2], "sem_f1") - EPS]
    wasted = [x for x in d if gv(x[3], "llm_calls") > gv(x[2], "llm_calls") + EPS and gv(x[3], "f1") <= gv(x[2], "f1") + EPS
              and gv(x[3], "sem_f1") <= gv(x[2], "sem_f1") + EPS]
    gain = [x for x in d if gv(x[3], "f1") > gv(x[2], "f1") + EPS or gv(x[3], "sem_f1") > gv(x[2], "sem_f1") + EPS]
    fewer = [x for x in d if gv(x[3], "llm_calls") < gv(x[2], "llm_calls") - EPS]
    return {"pairs": len(pairs), "differ": len(d), "worse": len(worse), "wasted": len(wasted), "gain": len(gain), "fewer_calls": len(fewer),
            "extra_calls": round(sum(gv(g, "llm_calls") - gv(o, "llm_calls") for _, _, o, g in d if gv(g, "llm_calls") > gv(o, "llm_calls"))),
            "sum_dF1": round(sum(gv(g, "f1") - gv(o, "f1") for _, _, o, g in d), 3), "sum_dsemF1": round(sum(gv(g, "sem_f1") - gv(o, "sem_f1") for _, _, o, g in d), 3)}


def rollup(verdicts):
    vs = set(verdicts)
    return "equal" if vs == {"equal"} else "dominates" if vs <= {"equal", "dominates"} else "dominated" if vs <= {"equal", "dominated"} else "mixed"


def guard_verdict(by):
    pairs = guard_pairs(by)
    st = pair_stats(pairs)
    le0 = [x for x in pairs if float(x[1]) <= 0]
    same0 = sum(not differs(o, g) and o["tau_by_seed"] == g["tau_by_seed"] for _, _, o, g in le0)
    d = [x for x in pairs if differs(x[2], x[3])]
    out = ["## Guard verdict (ours_sem_guard@L, K=1; read this first)", "",
           f"* `ours_sem_guard@L` = `ours_sem_direct@L` iff n_sem >= K (n_sem = labeled cells with an llm_sem candidate), else `ours_lambda@L`. "
           f"At lambda <= 0 it equals ours_lambda in {same0}/{len(le0)} (config, lambda) pairs (metrics and tau_L).",
           f"* Over all {st['pairs']} (config, lambda) pairs: differs from ours_lambda in {st['differ']}; F1 or semantic F1 lower in {st['worse']}; "
           f"extra calls with no F1 / semantic F1 gain in {st['wasted']}; a gain in {st['gain']}; fewer calls in {st['fewer_calls']}.",
           "* Sensitivity (guard with K=2 / K=3, tau re-picked): " + "; ".join(
               f"K={k}: " + ", ".join(f"{a} {b}" for a, b in pair_stats(guard_pairs(by, f"ours_sem_guard_K{k}")).items() if a != "pairs")
               for k in KS) + ".", ""]
    out.append("")
    return out


def guard_sections(by):
    pairs = guard_pairs(by)
    out = ["## Guard: every (config, lambda) where ours_sem_guard differs from ours_lambda", "",
           "Format old -> guard (means over 3 seeds). n_sem = labeled cells with an llm_sem candidate, per seed.", "",
           "| dataset | cond | sampler | budget | lambda | n_sem (seeds) | F1 old -> guard | sem F1 old -> guard | calls old -> guard | llm_sem calls old -> guard | tau old -> guard |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    g = lambda r, k: f"{r[k + '_mean']:.3f}" if k + "_mean" in r else "-"
    for c, lam, o, r in pairs:
        if differs(o, r):
            out.append(f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {lam} | {r['n_sem_by_seed']} | {g(o, 'f1')} -> {g(r, 'f1')} | {g(o, 'sem_f1')} -> {g(r, 'sem_f1')} | "
                       f"{o['llm_calls_mean']:.0f} -> {r['llm_calls_mean']:.0f} | {o['sem_calls_mean']:.0f} -> {r['sem_calls_mean']:.0f} | {o['tau_by_seed']} -> {r['tau_by_seed']} |")
    out.append("")
    # earlier wasted-call cases: ours_sem_direct bought calls without any F1 / semantic F1 gain
    wasted = [(c, lam, o, d, by[(*c, lam, "ours_sem_guard")]) for c, lam, o, d in guard_pairs(by, "ours_sem_direct") if pair_stats([(c, lam, o, d)])["wasted"]]
    out.extend(["## The earlier wasted-call cases (ours_sem_direct: more calls, no F1 / semantic F1 gain) and what the guard does", "",
                "| dataset | cond | sampler | budget | lambda | n_sem (seeds) | calls old / direct / guard | F1 old / direct / guard | sem F1 old / direct / guard | removed by guard |",
                "|---|---|---|---|---|---|---|---|---|---|"])
    for c, lam, o, d, r in wasted:
        out.append(f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {lam} | {r['n_sem_by_seed']} | {o['llm_calls_mean']:.0f} / {d['llm_calls_mean']:.0f} / {r['llm_calls_mean']:.0f} | "
                   f"{g(o, 'f1')} / {g(d, 'f1')} / {g(r, 'f1')} | {g(o, 'sem_f1')} / {g(d, 'sem_f1')} / {g(r, 'sem_f1')} | "
                   f"{'yes' if not differs(o, r) else 'partly' if r['llm_calls_mean'] < d['llm_calls_mean'] - EPS else 'no'} |")
    out.append("")
    out.extend(["## Counts over all (config, lambda) pairs, by variant vs ours_lambda", "",
                "| variant | pairs | differ | F1 or sem F1 lower | extra calls, no gain | gain (F1 or sem F1) | fewer calls | extra calls summed over the differing pairs | sum of F1 diffs | sum of sem F1 diffs |", "|---|---|---|---|---|---|---|---|---|---|"])
    for v in ("ours_sem_direct", "ours_sem_guard", *[f"ours_sem_guard_K{k}" for k in KS]):
        st = pair_stats(guard_pairs(by, v))
        out.append(f"| {v} | {st['pairs']} | {st['differ']} | {st['worse']} | {st['wasted']} | {st['gain']} | {st['fewer_calls']} | {st['extra_calls']} | {st['sum_dF1']:+.3f} | {st['sum_dsemF1']:+.3f} |")
    out.append("")
    cfgs = sorted({k[:4] for k in by}, key=ORDER)
    out.extend(["## Pareto frontier (F1 vs LLM calls, over the 13 lambdas) vs ours_lambda, per config", "",
                "dominates = every ours_lambda frontier point is matched or beaten (calls <=, F1 >=) by a point of the variant and the variant has a point ours_lambda "
                "does not match; equal = same frontier; dominated = the reverse; incomparable = neither. Roll-up per dataset x condition: dominates if all configs "
                "dominate or are equal and one dominates, mixed otherwise.", "",
                "| dataset | cond | sampler | budget | guard vs ours_lambda | sem_direct vs ours_lambda | guard K=2 | guard K=3 |", "|---|---|---|---|---|---|---|---|"])
    vd = {}
    for c in cfgs:
        row = [pareto(by, c, v) for v in ("ours_sem_guard", "ours_sem_direct", "ours_sem_guard_K2", "ours_sem_guard_K3")]
        vd[c] = row
        out.append(f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | " + " | ".join(row) + " |")
    out.extend(["", "| dataset | cond | guard vs ours_lambda (roll-up) | sem_direct vs ours_lambda (roll-up) | configs |", "|---|---|---|---|---|"])
    for dc in sorted({c[:2] for c in cfgs}, key=lambda k: ((pl.DATASETS + pl.SEM_DATASETS).index(k[0]), k[1])):
        cs = [c for c in cfgs if c[:2] == dc]
        out.append(f"| {dc[0]} | {dc[1]} | {rollup([vd[c][0] for c in cs])} | {rollup([vd[c][1] for c in cs])} | {len(cs)} |")
    out.append("")
    return out


def fmt(r, k, d=3):
    return f"{r[k + '_mean']:.{d}f}±{r[k + '_std']:.{d}f}" if k + "_mean" in r else "-"


def write_md(rows):
    by = {(r["dataset"], r["condition"], r["sampler"], r["budget"], r["lambda"], r["variant"]): r for r in rows}
    out = ["# ours_sem_direct@L vs ours_lambda@L (experimental variant, replay only)", "",
           "Mean±std (sample std) over seeds 0,1,2. Generated by `src/sem_direct_comparison.py` from `results/summary/sem_direct_comparison.csv`; "
           "not part of the paper tables. `sem calls` = llm_sem calls inside `calls`; tau = tau_L chosen per seed; `cap` = F1 gain if llm_sem's "
           "calibrated confidence were 1 instead of its calibrator (correct repairs gained, mean over seeds).", ""]
    hdr = ("| dataset | cond | sampler | variant | F1 | P | R | sem F1 | sem R | calls | sem calls | tokens | GPU s | tau |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    def line(r, label):
        return (f"| {r['dataset']} | {r['condition']} | {r['sampler']} | {label} | {fmt(r, 'f1')} | {fmt(r, 'precision')} | {fmt(r, 'recall')} | "
                f"{fmt(r, 'sem_f1')} | {fmt(r, 'sem_recall')} | {fmt(r, 'llm_calls', 1)} | {fmt(r, 'sem_calls', 1)} | {fmt(r, 'tokens', 0)} | "
                f"{fmt(r, 'llm_seconds', 1)} | {r['tau_by_seed']} |")

    def table(title, lam, budget, lam_filter=None):
        out.extend([f"## {title}", "", hdr])
        keys = sorted({k[:4] for k in by if k[3] == budget and k[4] == lam}, key=lambda k: ((pl.DATASETS + pl.SEM_DATASETS).index(k[0]), k[1] != "mixed", k[2] != "random"))
        for k in keys:
            for v, lab in (("ours_lambda", f"ours_lambda@{lam}"), ("ours_sem_direct", f"ours_sem_direct@{lam}"), ("ours_sem_guard", f"ours_sem_guard@{lam}")):
                out.append(line(by[(*k, lam, v)], lab))
        out.append("")

    for budget in (20, 50):
        table(f"{budget} labeled rows, lambda = 0 (final model vs variant)", "0", budget)
    for budget in (20, 50):
        for lam in ("0.2", "0.5"):
            table(f"{budget} labeled rows, lambda = {lam}", lam, budget)
    out.extend(["## Every config and lambda where the variant differs from ours_lambda (identical elsewhere, incl. all of lambda <= 0)", "",
                "| dataset | cond | sampler | budget | lambda | F1 old -> new | sem F1 old -> new | calls old -> new | llm_sem calls old -> new | prior llm / sem (seeds) | tau old -> new |",
                "|---|---|---|---|---|---|---|---|---|---|---|"])
    g = lambda r, k: f"{r[k + '_mean']:.3f}" if k + "_mean" in r else "-"
    for k in sorted((k for k in by if k[-1] == "ours_sem_direct"), key=lambda k: ((pl.DATASETS + pl.SEM_DATASETS).index(k[0]), k[1], k[2], k[3], pl.LAMBDAS.index(k[4]))):
        o, r = by[k[:-1] + ("ours_lambda",)], by[k]
        if all(o[c] == r[c] for c in ("f1_mean", "llm_calls_mean", "recall_mean", "precision_mean")):
            continue
        out.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} | {g(o, 'f1')} -> {g(r, 'f1')} | {g(o, 'sem_f1')} -> {g(r, 'sem_f1')} | "
                   f"{o['llm_calls_mean']:.0f} -> {r['llm_calls_mean']:.0f} | {o['sem_calls_mean']:.0f} -> {r['sem_calls_mean']:.0f} | "
                   f"{r['prior_llm_by_seed']} / {r['prior_sem_by_seed']} | {o['tau_by_seed']} -> {r['tau_by_seed']} |")
    out.append("")
    out.extend(guard_sections(by))
    out.extend(["## Lambda sweep of ours_sem_direct and ours_sem_guard on beers_sem and hospital_sem (20 rows)", "", hdr])
    for name in ("beers_sem", "hospital_sem"):
        for cond, sampler in (("mixed", "random"), ("mixed", "coverage"), ("xsem", "random"), ("xsem", "coverage")):
            for lam in ("-0.5", "0", "0.2", "0.5"):
                for v in ("ours_lambda", "ours_sem_direct", "ours_sem_guard"):
                    out.append(line(by[(name, cond, sampler, 20, lam, v)], f"{v}@{lam}"))
    out.append("")
    out.extend(["## Does tau_L cap the semantic gain? (ours_sem_direct, sem datasets, per config, over 3 seeds)", "",
                "llm_sem's calibrator is the constant (k+1)/(n+2) of the labeled cells that have an llm_sem candidate, 0.5 when there are none "
                "(always in xsem, since no labeled row has an injected cell). The router accepts an llm_sem answer only if its calibrated confidence >= tau_L, "
                "and tau_L comes from the grid 0.5..0.95 (ties: larger), so every seed with tau_L > that confidence cannot accept any llm_sem answer.", "",
                "| dataset | cond | sampler | budget | lambda | llm_sem calibrated conf | labeled sem cells | tau_L | seeds with tau_L > cal conf | F1 | cap-free F1 | extra correct if cap-free |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|",
                *[f"| {r['dataset']} | {r['condition']} | {r['sampler']} | {r['budget']} | {r['lambda']} | {r['llm_sem_cal_by_seed']} | {r['n_calib_sem_by_seed']} | "
                  f"{r['tau_by_seed']} | {r['tau_blocks_sem_seeds']}/3 | {fmt(r, 'f1')} | {fmt(r, 'uncapped_f1')} | {fmt(r, 'uncapped_correct_gain', 1)} |"
                  for r in sorted((r for r in rows if r["variant"] == "ours_sem_direct" and r["lambda"] in ("0", "0.5") and r["dataset"] in pl.SEM_DATASETS),
                                  key=lambda r: (r["lambda"], r["dataset"], r["condition"], r["sampler"] != "random", r["budget"]))], ""])
    _unused = (["## Does tau_L cap the semantic gain? (ours_sem_direct@0, per config, over 3 seeds)", "",
                "| dataset | cond | sampler | budget | llm_sem calibrated conf | labeled sem cells | tau_L | seeds with tau_L > cal conf | F1 | cap-free F1 | extra correct if cap-free |",
                "|---|---|---|---|---|---|---|---|---|---|---|",
                *[f"| {r['dataset']} | {r['condition']} | {r['sampler']} | {r['budget']} | {r['llm_sem_cal_by_seed']} | {r['n_calib_sem_by_seed']} | "
                  f"{r['tau_by_seed']} | {r['tau_blocks_sem_seeds']}/3 | {fmt(r, 'f1')} | {fmt(r, 'uncapped_f1')} | {fmt(r, 'uncapped_correct_gain', 1)} |"
                  for r in rows if r["variant"] == "ours_sem_direct" and r["lambda"] == "0" and r["dataset"] in pl.SEM_DATASETS], ""])
    out[2:2] = guard_verdict(by) + [""] + ["# Part 1: ours_sem_direct@L (previous report, unchanged)", ""] + VERDICT.splitlines() + [""]
    (SUMMARY / "sem_direct_comparison.md").write_text("\n".join(out))


if __name__ == "__main__":
    main()
