#!/usr/bin/env python
"""Aggregate the existing result CSVs into the paper tables. No new pipeline methods (T9 replays cached tier outputs via pipeline helpers), no LLM calls, deterministic.

    .venv/bin/python src/make_tables.py    # writes paper/tables/{final_tables.md, tables.tex, T*.csv}; needs the repo caches (T9 replays cache/b20_s0, b20_s1, b20_s2)

Sources are listed in paper/tables/README.md. "Ours" is ours_sem_guard@0 everywhere (at lambda=0 it equals ours_lambda@0 of the earlier router, asserted below).
LaTeX needs: booktabs, longtable, graphicx (amsmath/amssymb for \\lambda, \\pm, \\dagger).
"""
import re
import statistics
from collections import namedtuple
from pathlib import Path

import pandas as pd

import pipeline as pl  # T9 only: load_cache / calib_test_split / _prf for the replay-only baselines

BASE = Path(__file__).resolve().parent.parent
RES, OUT = BASE / "results" / "summary", BASE / "paper" / "tables"
Ms = namedtuple("Ms", "mean std")  # std None = single value

DS5 = ["hospital", "flights", "beers", "rayyan", "movies"]
SEM = ["hospital_sem", "beers_sem"]
DS7 = DS5 + SEM
DNAME = {"hospital": "Hospital", "flights": "Flights", "beers": "Beers", "rayyan": "Rayyan", "movies": "Movies (held-out)",
         "hospital_sem": "Hospital-sem (row conflict)", "beers_sem": "Beers-sem (world knowledge)"}
SHORT = {"hospital": "H", "flights": "F", "beers": "B", "rayyan": "R", "movies": "M"}
M = {  # sweep method -> paper name
    "ours_sem_guard@0": "Ours (λ=0)", "ours_budget": "Ours-auto-λ", "ours_auto": "Ours-auto-τ", "static_gate": "Static gate",
    "raw_threshold": "Raw threshold", "cascade": "Cascade",
    "llm_only": "LLM-only (Qwen3-8B, context prompt)", "qwen_strong": "LLM-only (Qwen3-8B, knowledge prompt)",
    "jellyfish": "Jellyfish-8B (our run)"}
BARAN_OURS, BARAN_PAPER = "Baran (perfect detection, our protocol)", "Baran (paper protocol)"
STD_NOTE = "mean±std over seeds 0/1/2 (sample std; Baran: population std as in logs/baran_summary_with_movies.txt)"

# ---------------------------------------------------------------- data
def load(f):
    d = pd.read_csv(RES / f)
    return {(r.dataset, r.method, int(r.budget)): r for r in d.itertuples(index=False)}

SW = {}  # SW[cond][sampler] -> {(dataset, method, budget): row}; cond: main (5 datasets) | mixed | xsem (sem datasets)
SW["main"] = {"random": load("budget_sweep.csv"), "coverage": load("budget_sweep_cov.csv")}
SW["mixed"] = {"random": load("budget_sweep_sem.csv"), "coverage": load("budget_sweep_sem_cov.csv")}
SW["xsem"] = {"random": load("budget_sweep_sem_xsem.csv"), "coverage": load("budget_sweep_sem_xsem_cov.csv")}
RND = {**SW["main"]["random"], **SW["mixed"]["random"]}  # all 7 datasets, mixed labels, random sampler
LLM = {(r.dataset, r.sampler, r.system, int(r.budget)): r for r in pd.read_csv(RES / "llm_only_baselines_summary.csv").itertuples(index=False)}
LLM.update({(r.dataset, r.sampler, r.system, int(r.budget)): r for r in pd.read_csv(RES / "llm_only_baselines_cov_summary.csv").itertuples(index=False)})
LSTAR = pd.concat([pd.read_csv(RES / "budget_lambda_star.csv"), pd.read_csv(RES / "budget_lambda_star_sem.csv")])
LIT = (RES / "llm_only_literature.md").read_text()
_VP = BASE / "paper" / "reviews" / "final_verdict_fable.md"
VERDICT = _VP.read_text() if _VP.exists() else None  # internal notes are not shipped; the two cross-checks on VERDICT are skipped


def ms(row, col):
    """(mean, std) of `col` from a sweep row (`col_mean/col_std`) or an llm-only summary row; None if absent/NaN."""
    m = getattr(row, f"{col}_mean", None)
    if m is None or pd.isna(m):
        return None
    s = getattr(row, f"{col}_std", None)
    return Ms(float(m), None if s is None or pd.isna(s) else float(s))


def baran():
    """{(dataset, 'paper'|'ours'): F1 Ms} from logs/baran_summary_with_movies.txt (metric 'ours'; A all = paper protocol, B test = our rows)."""
    out = {}
    for ln in (BASE / "logs" / "baran_summary_with_movies.txt").read_text().splitlines():
        m = re.match(r"(\w+)\s+([AB])\s+(all|test)\s+ours\s*:\s*(?:[\d.]+\+-[\d.]+ / ){2}([\d.]+)\+-([\d.]+)", ln)
        if m and (m[2], m[3]) in (("A", "all"), ("B", "test")):
            out[(m[1], "paper" if m[2] == "A" else "ours")] = Ms(float(m[4]), float(m[5]))
    return out

BARAN = baran()

# the final model equals the earlier router at lambda=0: same metrics, every column (tau, calls, tokens, ECE, ...)
for (d, m, b), o in RND.items():
    if m == "ours_sem_guard@0":
        x, y = o._asdict(), RND[(d, "ours_lambda@0", b)]._asdict()
        assert all(x[k] == y[k] or (pd.isna(x[k]) and pd.isna(y[k])) for k in x if k != "method"), (d, b)

# cross-check: the llm-only summary repeats Ours/LLM-only(context) from the sweeps; they must agree
for (d, s, sysm, b), r in LLM.items():
    if sysm in ("ours_lambda@0", "llm_only") and s == "random" and (d, sysm, b) in RND:  # that csv still carries Ours as ours_lambda@0 (= the guard at lambda=0)
        o = RND[(d, sysm, b)]
        for c in ("f1", "llm_calls", "tokens", "llm_seconds"):
            assert abs(getattr(r, c + "_mean") - getattr(o, c + "_mean")) < 1e-9, (d, sysm, b, c)

# ---------------------------------------------------------------- formatting
def num(v, kind, tex, dec=0):
    if v is None:
        return "–"
    if kind == "f": return f"{v:.{2 if tex else 3}f}"
    if kind == "e": return f"{v:.3f}"
    if kind == "f2": return f"{v:.2f}"
    if kind == "i" or kind == "s": return f"{v:.0f}"
    if kind == "k": return f"{v / 1000:.{dec}f}"
    if kind == "p": return f"{v * 100:.1f}%"
    if kind == "x": return f"{v:.1f}×"
    raise ValueError(kind)


def cell(v, kind, tex):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "–"
    if isinstance(v, str):
        return v
    if isinstance(v, dict):  # compound cells
        if kind == "fc":  # F1 [calls]
            return f"{cell(v['f1'], 'f', tex)} [{cell(v['calls'], 'i', tex)}]"
        if kind == "arrow":  # raw → calibrated
            return f"{num(v['raw'], 'e', tex)}→{num(v['cal'], 'e', tex)}"
    dec = 1 if kind == "k" and abs(v.mean if isinstance(v, Ms) else v) < 9500 else 0  # tokens (K): one decimal below 9.5K
    if isinstance(v, Ms):
        if v.std is None:
            return num(v.mean, kind, tex, dec)
        return f"{num(v.mean, kind, tex, dec)}±{num(v.std, kind, tex, dec)}" if kind != "p" else f"{v.mean * 100:.1f}±{v.std * 100:.1f}%"
    return num(v, kind, tex, dec)


def flat(key, v):
    """CSV: Ms -> key, key_std; compound dict -> key_sub(_std)."""
    if isinstance(v, dict):
        return {k2: x for sub, sv in v.items() for k2, x in flat(f"{key}_{sub}", sv).items()}
    if isinstance(v, Ms):
        return {key: v.mean, f"{key}_std": v.std}
    return {key: v}


def tex_esc(s):
    s = str(s)
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_")):
        s = s.replace(a, b)
    for a, b in (("λ", "$\\lambda$"), ("τ", "$\\tau$"), ("±", "$\\pm$"), ("−", "$-$"), ("∞", "$\\infty$"), ("→", "$\\to$"), ("×", "$\\times$"),
                 ("†", "$^\\dagger$"), ("–", "--"), ("≥", "$\\geq$"), ("★", "$^\\star$")):
        s = s.replace(a, b)
    return s


class Table:
    def __init__(self, tid, title, cols, rows, notes=(), nb=1, widths=None, group=False):
        """cols: [(key, header, kind)], kind in s(tring) f e i k p x s-seconds ... ; rows: [{key: value}]; nb: leading label columns whose
        repeats are blanked; widths: {key: 'p{3cm}'} for text columns."""
        self.tid, self.title, self.cols, self.rows, self.notes, self.nb, self.widths, self.group = tid, title, cols, rows, list(notes), nb, widths or {}, group

    def grid(self, tex):
        out, prev = [], None
        for r in self.rows:
            line = [cell(r.get(k), r.get("_kind", kind) if kind != "t" else kind, tex) for k, _, kind in self.cols]  # _kind: per-row number format
            lab = line[:self.nb]
            blank = [all(lab[j] == prev[j] for j in range(i + 1)) if prev else False for i in range(self.nb)]
            out.append((bool(prev) and lab[0] != prev[0], [("" if blank[i] else x) if i < self.nb else x for i, x in enumerate(line)]))
            prev = lab
        return out

    def md(self):
        h = [c[1] for c in self.cols]
        s = [f"### {self.tid}. {self.title}", "", "| " + " | ".join(h) + " |", "|" + "|".join("---" for _ in h) + "|"]
        s += ["| " + " | ".join(x.replace("|", "/") for x in g) + " |" for _, g in self.grid(False)]
        return "\n".join(s + [""] + [f"* {n}" for n in self.notes] + [""])

    def csv(self):
        rows = []
        for r in self.rows:
            d = {}
            for k, _, _ in self.cols:
                d.update(flat(k, r.get(k)))
            rows.append(d)
        return pd.DataFrame(rows)

    def tex(self):
        spec = "".join((self.widths.get(k, "l") if kind == "t" else "r") for k, _, kind in self.cols)
        hdr = " & ".join(tex_esc(c[1]) for c in self.cols) + r" \\"
        body = []
        for new_group, g in self.grid(True):
            if new_group and self.group:
                body.append(r"\midrule")
            body.append(" & ".join(tex_esc(x) for x in g) + r" \\")
        notes = (r"\par\smallskip{\footnotesize " + " ".join(tex_esc(n) + "." if not n.endswith(".") else tex_esc(n) for n in self.notes) + "}") if self.notes else ""
        cap, lab = tex_esc(f"{self.tid}. {self.title}"), f"tab:{self.tid}"
        long = len(self.rows) > 24
        if long:
            return "\n".join([r"{\footnotesize", rf"\begin{{longtable}}{{{spec}}}", rf"\caption{{{cap}}}\label{{{lab}}}\\", r"\toprule", hdr, r"\midrule",
                              r"\endfirsthead", r"\toprule", hdr, r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot", *body,
                              r"\end{longtable}", notes, "}", ""])
        wrap = (r"\resizebox{\linewidth}{!}{%", "}")
        return "\n".join([r"\begin{table}[t]\centering\footnotesize", rf"\caption{{{cap}}}\label{{{lab}}}", wrap[0],
                          rf"\begin{{tabular}}{{{spec}}}", r"\toprule", hdr, r"\midrule", *body, r"\bottomrule", r"\end{tabular}" + wrap[1], notes,
                          r"\end{table}", ""])

# ---------------------------------------------------------------- tables
def T1():
    pub = re.search(r"Table III[^\n]*Jellyfish-7B[^\n]*?\| ([\d.]+) / ([\d.]+) / ([\d.]+) / not run / not run", LIT)  # LasRepair, published
    jf7 = dict(zip(DS5[:3], map(float, pub.groups())))
    rows = []
    for d in DS5:
        o, lo = RND[(d, "ours_sem_guard@0", 20)], RND[(d, "llm_only", 20)]
        rows.append({"dataset": DNAME[d], "ours": ms(o, "f1"), "ctx": ms(lo, "f1"), "know": ms(LLM[(d, "random", "qwen_strong", 20)], "f1"),
                     "jf8": ms(LLM[(d, "random", "jellyfish", 20)], "f1"), "baran_ours": BARAN.get((d, "ours")), "baran_paper": BARAN.get((d, "paper")),
                     "jf7": jf7.get(d), "ours_calls": ms(o, "llm_calls"), "llm_calls": ms(lo, "llm_calls"),
                     "reduction": lo.llm_calls_mean / o.llm_calls_mean})
    cols = [("dataset", "Dataset", "t"), ("ours", "Ours (λ=0) F1", "f"), ("ctx", "LLM-only (context) F1", "f"), ("know", "LLM-only (knowledge) F1", "f"),
            ("jf8", "Jellyfish-8B (our run) F1", "f"), ("baran_ours", "Baran (our protocol) F1", "f"), ("baran_paper", "Baran (paper protocol) F1", "f"),
            ("jf7", "Jellyfish-7B† F1", "f"), ("ours_calls", "Ours LLM calls", "i"), ("llm_calls", "LLM-only LLM calls", "i"), ("reduction", "Reduction", "x")]
    return Table("T1", "Main results, 20 labeled rows, random sampler, perfect detection (F1 on unlabeled rows)", cols, rows, [
        STD_NOTE,
        "Ours = ours_sem_guard@0; LLM-only (context) = Qwen3-8B per-cell prompt with candidate context, LLM-only (knowledge) = Qwen3-8B knowledge prompt (n=3, T=0.7), both ask every error cell once; Jellyfish-8B = our run of the model-card imputation prompt",
        "Baran (our protocol): its 20 labeled tuples forced to our random rows, scored on our test split (config B). Baran (paper protocol): its own tuple sampler, scored on all error cells (config A); both no transfer learning, perfect detection",
        "† Jellyfish-7B is a published number (LasRepair, arXiv 2606.17582, Table III; oracle detection, no labels, not our run); '–' = not reported",
        "Reduction = LLM-only calls / Ours calls (ratio of 3-seed means); Movies is the held-out dataset (frozen code run once)"])


def T2():
    cols = [("dataset", "Dataset", "t"), ("system", "System", "t"), ("p", "P", "f"), ("r", "R = repair acc.", "f"), ("f1", "F1", "f"),
            ("calls", "LLM calls", "i"), ("total", "Total calls", "i"), ("rate", "Call rate", "p"), ("tok", "Tokens (K)", "k"),
            ("gpu", "GPU s", "i"), ("lat", "Latency / cell (s)", "e"), ("sece", "Sys ECE", "e"), ("sbr", "Sys Brier", "e")]
    rows = []
    for d in DS7:
        for m in ("ours_sem_guard@0", "llm_only"):
            x = RND[(d, m, 20)]
            rows.append({"dataset": DNAME[d], "system": M[m].replace(" (Qwen3-8B, context prompt)", " (context)"), "p": ms(x, "precision"),
                         "r": ms(x, "recall"), "f1": ms(x, "f1"), "calls": ms(x, "llm_calls"), "total": ms(x, "total_calls"),
                         "rate": ms(x, "llm_call_rate"), "tok": ms(x, "tokens"), "gpu": ms(x, "llm_seconds"), "lat": ms(x, "latency_per_cell"),
                         "sece": ms(x, "sys_ece"), "sbr": ms(x, "sys_brier")})
    return Table("T2", "All deck metrics, Ours (λ=0) vs LLM-only (context), 20 labeled rows, random sampler", cols, rows, [
        STD_NOTE,
        "P = correct repairs / repairs with output != dirty; R = correct repairs / error cells, which equals repair accuracy under perfect detection (every error cell counts once)",
        "LLM calls = routed calls (per-cell llm, llm_sem, each column group once); Total calls = LLM calls + 2 per-dataset setup calls (FD and source proposal); Call rate = LLM calls / error cells",
        "Tokens = prompt + completion tokens of the routed calls (K = thousands). GPU s = sum of recorded LLM request seconds of the tiers reached; per-cell requests were issued 8 at a time and each request's seconds include vLLM queueing/batching, so it overstates a dedicated GPU",
        "Latency / cell = sequential-equivalent end-to-end compute of the routed path (sum over cells of the seconds of tiers reached) / error cells; the README does not describe an idle-machine measurement, so none is claimed; per-dataset setup is not included",
        "Sys ECE / Sys Brier = 10-bin ECE and Brier of the final decision's confidence (calibrated) over the repaired cells; LLM-only has no calibrator, its sys metrics use its raw confidence"], nb=1, group=True)


ABL = ["ours_sem_guard@0", "ours_budget", "ours_auto", "static_gate", "raw_threshold", "cascade", "ours_no_column", "ours_no_llm", "ours_rules_llm", "ours_stats_llm"]


def wide(tid, title, methods, budget, notes, extra=None):
    cols = [("method", "Method", "t")] + [(d, DNAME[d], "fc") for d in DS7]
    rows = []
    for m in methods:
        r = {"method": (extra or {}).get(m) or M.get(m, m)}
        for d in DS7:
            x = RND.get((d, m, budget))
            r[d] = {"f1": ms(x, "f1"), "calls": ms(x, "llm_calls")} if x is not None else None
        rows.append(r)
    return Table(tid, title, cols, rows, notes)


def T3():
    t = wide("T3", "Ablations, 20 labeled rows, random sampler: F1 [LLM calls]", ABL, 20, [
        STD_NOTE,
        "Ours-auto-λ = ours_budget (λ* chosen by 2-fold cross-fit on the labeled rows); Ours-auto-τ = ours_auto (τ chosen on the labeled rows); Static gate = tier kept iff its Laplace accuracy >= 0.8, no per-cell score; Raw threshold = τ=0.8 on uncalibrated confidences (= ours_no_calib); Cascade = no escalation gating (= no_escalation)",
        "ours_no_column / ours_no_llm / ours_rules_llm / ours_stats_llm = Ours (λ=0) with the column tier skipped / no LLM tiers / only rules+LLM / only stats+LLM tiers (replay only)",
        "Hospital-sem / Beers-sem: injected semantic errors, labeled rows drawn at random (mixed condition)"])
    return t


def T4a():
    rows = []
    for d in DS7:
        r = pd.read_csv(BASE / "results" / "b20_s0" / f"{d}_methods.csv").set_index("method").loc["ours_sem_guard@0"]
        for t in ("rules", "fd", "stats", "column", "llm", "llm_sem"):
            if r[f"ece_n_{t}"] >= 20:
                rows.append({"dataset": DNAME[d], "tier": t, "n": int(r[f"ece_n_{t}"]),
                             "ece": {"raw": r[f"ece_raw_{t}"], "cal": r[f"ece_cal_{t}"]}, "brier": {"raw": r[f"brier_raw_{t}"], "cal": r[f"brier_cal_{t}"]}})
    cols = [("dataset", "Dataset", "t"), ("tier", "Tier", "t"), ("n", "n test cells", "i"), ("ece", "ECE raw→cal", "arrow"), ("brier", "Brier raw→cal", "arrow")]
    return Table("T4a", "Per-tier calibration, seed 0, 20 labeled rows (test split; tiers with n >= 20 only)", cols, rows, [
        "10-bin ECE and Brier of each tier's confidence for its own candidate, raw → after the calibrator; single seed (results/<name>_methods.csv), tiers with fewer than 20 reached test cells (fd, llm_sem on most datasets) omitted",
        "The column tier has no calibrator (identity), so raw = calibrated there; calibration can also worsen a tier's ECE"], group=True)


def T4b():
    cols = [("dataset", "Dataset", "t"), ("ece", "Ours ECE (calibrated)", "e"), ("ece_raw", "Ours ECE (raw conf.)", "e"), ("rt_ece", "Raw threshold ECE", "e"),
            ("brier", "Ours Brier (calibrated)", "e"), ("brier_raw", "Ours Brier (raw conf.)", "e"), ("rt_brier", "Raw threshold Brier", "e"),
            ("n", "n repaired cells", "i")]
    rows = []
    for d in DS7:
        o, rt = RND[(d, "ours_sem_guard@0", 20)], RND[(d, "raw_threshold", 20)]
        rows.append({"dataset": DNAME[d], "ece": ms(o, "sys_ece"), "ece_raw": ms(o, "sys_ece_raw"), "rt_ece": ms(rt, "sys_ece"), "brier": ms(o, "sys_brier"),
                     "brier_raw": ms(o, "sys_brier_raw"), "rt_brier": ms(rt, "sys_brier"), "n": ms(o, "sys_n")})
    return Table("T4b", "System-level calibration of the final decision, Ours vs Raw threshold, 20 labeled rows", cols, rows, [
        STD_NOTE,
        "ECE/Brier of the confidence attached to the final repair, over the repaired cells (sys_n). Ours (calibrated) uses the calibrated confidence, 'raw conf.' the same decisions with uncalibrated tier confidences; Raw threshold has no calibrator, so its two versions coincide",
        "Lower is better"])


def T5():
    cols = [("dataset", "Dataset", "t"), ("cond", "Condition", "t"), ("method", "Method", "t"), ("f1", "Overall F1", "f"), ("sem_f1", "Semantic F1", "f"),
            ("sem_rec", "Semantic recall", "f"), ("calls", "LLM calls", "i"), ("share", "Calls on semantic cells", "p")]
    rows = []
    for d in SEM:
        for cond, sampler in (("mixed", "random"), ("mixed", "coverage"), ("xsem", "random"), ("xsem", "coverage")):
            sw = SW[cond][sampler]
            samp = "random" if sampler == "random" else "coverage"
            label = f"{'mixed' if cond == 'mixed' else 'xsem (RQ3 shift)'}, {samp}"
            for m in ("ours_sem_guard@0", "static_gate", "raw_threshold", "cascade", "llm_only", "qwen_strong"):
                if m == "qwen_strong":
                    x = LLM.get((d, samp, "qwen_strong", 20)) if cond == "mixed" else None
                    ctx = sw[(d, "llm_only", 20)]  # asks every error cell: share of semantic cells = same as the context prompt on this split
                    r = {"f1": ms(x, "f1"), "sem_f1": ms(x, "sem_f1"), "sem_rec": ms(x, "sem_recall"), "calls": ms(x, "llm_calls"),
                         "share": ms(ctx, "sem_call_share") if x is not None else None}
                else:
                    x = sw[(d, m, 20)]
                    r = {"f1": ms(x, "f1"), "sem_f1": ms(x, "sem_f1"), "sem_rec": ms(x, "sem_recall"), "calls": ms(x, "llm_calls"), "share": ms(x, "sem_call_share")}
                rows.append({"dataset": DNAME[d], "cond": label, "method": M[m], **r})
    return Table("T5", "Semantic splits, 20 labeled rows: mixed vs xsem labels, random vs coverage sampler", cols, rows, [
        STD_NOTE,
        "Hospital-sem = row_conflict errors (wrong state contradicted by the same row); Beers-sem = world_knowledge errors (only the brewery's city gives the state). Semantic F1/recall are over the injected cells only; Calls on semantic cells = share of a method's LLM calls spent on injected cells",
        "mixed: labeled rows drawn from all rows; xsem: labeled rows exclude every row with an injected cell (the RQ3 distribution shift). random vs coverage = labeled-row sampler (coverage represents cells no label-free tier resolves)",
        "LLM-only (knowledge) was only evaluated on the mixed split (random: both datasets; coverage: Beers-sem only), '–' elsewhere. LLM-only asks every error cell, so its share is the fraction of test cells that are semantic (for LLM-only (knowledge) taken from LLM-only (context) on the same split)",
        "Hospital-sem xsem random and xsem coverage give identical numbers in the source CSVs"], nb=2, group=True)


def T6():
    lam = ["-inf", "-0.5", "0", "0.2", "0.5", "1", "inf"]
    nm = lambda l: {"-inf": "−∞", "inf": "+∞"}.get(l, l.replace("-", "−"))
    cols = [("dataset", "Dataset", "t"), ("lam", "λ", "t"), ("f1", "F1", "f"), ("calls", "LLM calls", "i"), ("tok", "Tokens (K)", "k"), ("gpu", "GPU s", "i"),
            ("lat", "Latency / cell (s)", "e"), ("sem", "Semantic F1", "f"), ("ol", "ours_lambda F1 [calls]", "fc"), ("ol_sem", "ours_lambda semantic F1", "f")]
    rows = []
    for d in DS7:
        star = LSTAR[(LSTAR.dataset == d) & (LSTAR.budget == 20)].sort_values("seed")["lambda"].tolist()
        star = "/".join(("+∞" if v == float("inf") else "−∞" if v == float("-inf") else f"{v:g}") for v in star)
        # main columns: ours_sem_guard@L (the final model's family); λ* auto stays ours_budget (ours_lambda family, no guard-family auto-λ exists), so its two column groups coincide
        for m, name in [(f"ours_sem_guard@{l}", nm(l) + (" (Ours)" if l == "0" else "")) for l in lam] + [("ours_budget", f"λ* auto (ours_lambda family; seeds 0/1/2: {star})")]:
            x = RND[(d, m, 20)]
            o = RND[(d, m.replace("ours_sem_guard", "ours_lambda"), 20)]
            rows.append({"dataset": DNAME[d], "lam": name, "f1": ms(x, "f1"), "calls": ms(x, "llm_calls"), "tok": ms(x, "tokens"), "gpu": ms(x, "llm_seconds"),
                         "lat": ms(x, "latency_per_cell"), "sem": ms(x, "sem_f1") if d in SEM else None,
                         "ol": {"f1": ms(o, "f1"), "calls": ms(o, "llm_calls")}, "ol_sem": ms(o, "sem_f1") if d in SEM else None})
    return Table("T6", "λ budget ablation, 20 labeled rows, random sampler", cols, rows, [
        STD_NOTE,
        "Main columns: ours_sem_guard@λ, the final model's family (Ours = λ=0). It buys a paid tier iff its prior accuracy minus λ·cost exceeds the best free candidate; −∞ buys every applicable paid tier (max budget), +∞ makes no LLM call (min budget; zero-cost DSL column programs stay free). τ_λ is chosen per λ on the labeled rows",
        "The last two columns repeat the same λ for ours_lambda@λ (the earlier router), so the guard's effect is visible: on this configuration (20 rows, random sampler) the two are identical at every λ shown; the guard differs from ours_lambda only at λ > 0 on Beers-sem with other settings (coverage sampler, 50 rows), see results/summary/sem_direct_comparison.md, and never at λ ≤ 0",
        "Decoupled llm_sem with the calibration guard: llm_sem gets its own test q_sem − λC > p_best for cells no free tier resolves (direct-eligible), also when the per-cell LLM was not bought; it is enabled only if llm_sem has at least one labeled calibration cell (n_sem ≥ K = 1), otherwise the router is ours_lambda@λ",
        "λ* auto = ours_budget (Ours-auto-λ), λ* picked per seed by cross-fit over the ours_lambda@λ family on the labeled rows (values in the row label, results/budget_lambda_star.csv); no guard-family auto-λ exists (it would need a new method and a full sweep re-run), so both column groups show ours_budget there",
        "Tokens, GPU s and latency as in T2; Semantic F1 only for the injected-error splits (mixed labels)"], nb=1, group=True)


def T7():
    t = wide("T7", "50 labeled rows, random sampler: F1 [LLM calls]", ["ours_sem_guard@0", "ours_budget", "static_gate", "llm_only"], 50, [
        STD_NOTE,
        "Baran was run with 20 labeled tuples only (logs/baran_summary_with_movies.txt); no 50-row Baran result exists, so it is '–'",
        "LLM-only (context) asks every error cell, so its calls equal the number of test error cells"])
    t.rows.append({"method": "Baran (perfect detection, our protocol)", **{d: None for d in DS7}})
    return t


def published():
    """Published rows (copied numbers, other protocols), shared by T8 and T9: [(system, model, detection, labels, {ds: F1}, source)]."""
    def chk(src, *nums):
        for n in nums:
            assert n in src, n
    chk(LIT, "0.8871", "0.6324", "0.7832", "0.9632", "0.8798", "0.8894", "0.90 / 0.57 / 0.97 / 0.86 / 0.87", "H 1.00/0.91/0.95", "B 1.00/0.60/0.75",
        "R 0.01/0.00/0.00", "M 0.21/0.05/0.08", "0.87 / 0.81 / 0.94 / 0.92 / 0.91", "0.70 / 0.32 / 0.88 / 0.36 / 0.80")
    if VERDICT:
        chk(VERDICT, ".87/1.00/.90/.52", "Baran-perfect-ED")
    f = lambda *v: dict(zip(DS5, [None if x is None else float(x) for x in v]))
    rows = [
        ("Cocoon", "Claude 3.5", "own (statistical + LLM semantic); human-in-the-loop replaced by LLM-provided ground truth", "none", f(0.90, 0.57, 0.97, 0.86, 0.87), "Cocoon, arXiv 2410.15547, Table 1"),
        ("TabClean", "GPT-5 family (verdict: GPT-5.x)", "errors given (Table VI, per final_verdict_fable.md); llm_only_literature.md: oracle-like / own", "adaptive dev set 110-369 rows", f(0.87, 0.81, 0.94, 0.92, 0.91), "TabClean, arXiv 2606.25388, Table VI/VII"),
        ("LasRepair++", "GPT-5 instructor + T5-large", "oracle", "fine-tunes on sampled tuples (2,000 for large datasets)", f(0.9632, 0.8798, 0.8894, None, None), "LasRepair, arXiv 2606.17582, Table III"),
        ("IterClean (re-run by TabClean)", "gpt-5-mini", "own (LLM detector + verifier)", "5 seed tuples", f(0.95, 0.00, 0.75, 0.00, 0.08), "TabClean, arXiv 2606.25388, Table VI/VII"),
        ("Jellyfish-7B (as repair baseline)", "Jellyfish-7B (Mistral-based), no fine-tuning", "oracle", "none", f(0.8871, 0.6324, 0.7832, None, None), "LasRepair, arXiv 2606.17582, Table III"),
        ("Baran (paper, perfect detection)", "no LLM", "perfect error detection", "20 tuples", f(0.87, 1.00, 0.90, 0.52, None), "published Baran perfect-ED numbers as cited in paper/final_verdict_fable.md"),
        ("Baran (re-run in TabClean)", "no LLM", "oracle-like error set / own (as in llm_only_literature.md)", "20 labeled tuples", f(0.70, 0.32, 0.88, 0.36, 0.80), "TabClean, arXiv 2606.25388, Table VI/VII"),
    ]
    return rows


def T8():
    rows = published()
    cols = [("system", "System", "t"), ("model", "Model", "t"), ("det", "Detection assumption", "t"), ("labels", "Label budget", "t")] + \
           [(d, SHORT[d] + " F1", "f2") for d in DS5] + [("src", "Source", "t")]
    w = {"system": "p{2.1cm}", "model": "p{2.3cm}", "det": "p{3.4cm}", "labels": "p{2.3cm}", "src": "p{2.6cm}"}
    return Table("T8", "Published comparison (context only; numbers copied from the papers, not our runs, not comparable protocols)", cols,
                 [{"system": a, "model": b, "det": c, "labels": e, **g, "src": s} for a, b, c, e, g, s in rows], [
        "F1 as published; '–' = not run / not reported. Detection assumption and label budget differ from ours (oracle detection, 20 labeled rows, Qwen3-8B), so these rows give context, not a ranking",
        "Sources: results/llm_only_literature.md and paper/final_verdict_fable.md. IterClean and Cocoon-in-TabClean (gpt-5-mini) are TabClean's re-runs; IterClean F is 0.00 on Flights and Rayyan there",
        "Ours (λ=0) for reference, perfect detection, 20 rows: " + ", ".join(f"{SHORT[d]} {RND[(d, 'ours_sem_guard@0', 20)].f1_mean:.2f}" for d in DS5)],
        widths=w)


# T9 replay-only single-family baselines: per test cell the first non-null value among the listed cached tiers; a repair counts when output != dirty.
# Same cells as the sweeps (pl.load_cache + pl.calib_test_split(seed, n_tuples=20)); cache/b20_s{0,1,2}/.
SINGLE = {"Rules only": ("rules",), "Statistics only": ("stats",), "Rules + statistics (no LLM)": ("rules", "stats"),
          "Rules + LLM-proposed FDs + statistics": ("rules", "fd", "stats")}
# seed-mean F1 from the check run before T9 was written (reproduced to 1e-3)
SINGLE_EXPECT = {"Rules only": (.016, .000, .484, .000, .160), "Statistics only": (.950, .000, .065, .000, .005),
                 "Rules + statistics (no LLM)": (.915, .000, .518, .000, .098), "Rules + LLM-proposed FDs + statistics": (.915, .932, .538, .000, .098)}


def replay_single():
    """{system: {dataset: Ms(F1)}}, sample std over seeds 0/1/2 like the sweeps."""
    assert pl.SAMPLER == "random" and not pl.LABEL_EXCLUDE_SEM  # env must not change the split
    f1 = {k: {d: [] for d in DS5} for k in SINGLE}
    keep = pl.CACHE_DIR
    try:
        for d in DS5:
            for seed, cdir in ((0, BASE / "cache" / "b20_s0"), (1, BASE / "cache" / "b20_s1"), (2, BASE / "cache" / "b20_s2")):
                pl.CACHE_DIR = cdir
                _, test = pl.calib_test_split(pl.load_cache(d), seed, n_tuples=20)
                for k, tiers in SINGLE.items():
                    correct = repairs = 0
                    for c in test:
                        out = next((c["tiers"][t]["value"] for t in tiers if c["tiers"][t]["value"] is not None), c["dirty"]).strip()
                        rep = out != c["dirty"].strip()
                        repairs += rep
                        correct += rep and out == c["clean"].strip()
                    f1[k][d].append(pl._prf(correct, repairs, len(test))[2])
    finally:
        pl.CACHE_DIR = keep
    res = {k: {d: Ms(statistics.mean(v), statistics.stdev(v)) for d, v in dd.items()} for k, dd in f1.items()}
    for k, exp in SINGLE_EXPECT.items():
        for d, e in zip(DS5, exp):
            assert abs(res[k][d].mean - e) < 1e-3, (k, d, res[k][d].mean, e)
    for d in DS5:  # the 3-tier replay is the sweep's rules_stats row
        assert abs(res["Rules + LLM-proposed FDs + statistics"][d].mean - RND[(d, "rules_stats", 20)].f1_mean) < 1e-9, d
        assert abs(res["Rules + LLM-proposed FDs + statistics"][d].std - RND[(d, "rules_stats", 20)].f1_std) < 1e-9, d
    return res


def T9():
    pub = {r[0]: r for r in published()}
    for d in DS5:
        assert RND[(d, "ours_no_llm", 20)].llm_calls_mean == 0, d  # "0 per-cell LLM calls"
    assert VERDICT is None or re.search(r"Hospital P 1\.00 / R \.713 → F1 ≈ \.83", VERDICT)  # HoloClean, as cited in final_verdict_fable.md
    single = replay_single()
    P1, P2 = "A. Same protocol (ours)", "B. Published (copied; other protocols)"
    A = lambda sys, model, vals, **kw: {"part": P1, "system": sys, "model": model, "det": "perfect", "labels": "20 rows, random", **dict(zip(DS5, vals)), **kw}
    sw = lambda m: [ms(RND[(d, m, 20)], "f1") for d in DS5]
    rows = [A(k, "–" if "FDs" not in k else "Qwen3-8B (setup calls)", [single[k][d] for d in DS5]) for k in SINGLE]
    rows += [
        A("All free tiers incl. program search (0 per-cell LLM calls)", "Qwen3-8B (setup calls)", sw("ours_no_llm")),
        A("Baran (reproduced, our protocol)", "none", [BARAN.get((d, "ours")) for d in DS5]),
        A("LLM-only (Qwen3-8B, context prompt)", "Qwen3-8B", sw("llm_only")),
        A("LLM-only (Qwen3-8B, knowledge prompt)", "Qwen3-8B", [ms(LLM[(d, "random", "qwen_strong", 20)], "f1") for d in DS5]),
        A("Jellyfish-8B (our run) [a]", "Jellyfish-8B", [ms(LLM[(d, "random", "jellyfish", 20)], "f1") for d in DS5]),
        A("Plain cascade", "Qwen3-8B", sw("cascade")),
        A("Ours (λ=0)", "Qwen3-8B", sw("ours_sem_guard@0")),
        A("LLM calls: Ours (λ=0)", "Qwen3-8B", [ms(RND[(d, "ours_sem_guard@0", 20)], "llm_calls") for d in DS5], _kind="i"),
        A("LLM calls: LLM-only (context)", "Qwen3-8B", [ms(RND[(d, "llm_only", 20)], "llm_calls") for d in DS5], _kind="i"),
    ]
    B = lambda sys, model, det, labels, vals: {"part": P2, "system": sys, "model": model, "det": det, "labels": labels,
                                                **dict(zip(DS5, vals)), "_kind": "f2"}
    hc = [0.83] + [None] * 4  # F1 = 2PR/(P+R) for P 1.00 / R .713 = .832; the verdict gives only Hospital P/R (Flights P .888, recall not read)
    rows.append(B("HoloClean", "–", "own (denial constraints)", "–", hc))
    for key, name in (("Baran (paper, perfect detection)", "Baran (paper)"), ("Jellyfish-7B (as repair baseline)", "Jellyfish-7B (via LasRepair)"),
                      ("LasRepair++", "LasRepair++"), ("IterClean (re-run by TabClean)", "IterClean (via TabClean)"), ("Cocoon", "Cocoon"), ("TabClean", "TabClean")):
        _, model, det, labels, vals, _ = pub[key]
        rows.append(B(name, model, det, labels, [vals[d] for d in DS5]))
    rows.append(B("Ours (λ=0), for reference", "Qwen3-8B", "perfect", "20 rows", [RND[(d, "ours_sem_guard@0", 20)].f1_mean for d in DS5]))
    cols = [("part", "Part", "t"), ("system", "System", "t"), ("model", "Model", "t"), ("det", "Detection", "t"), ("labels", "Labels", "t")] + \
           [(d, SHORT[d] + " F1", "f") for d in DS5]
    w = {"part": "p{1.9cm}", "system": "p{3.8cm}", "model": "p{2.2cm}", "det": "p{2.4cm}", "labels": "p{2.2cm}"}
    return Table("T9", "Full comparison vs SOTA: same-protocol runs (A) and published numbers (B), F1", cols, rows, [
        "Part A: perfect detection, 20 labeled rows (random sampler), F1 on the unlabeled rows; " + STD_NOTE + ". Datasets: H Hospital, F Flights, B Beers, R Rayyan, M Movies (held-out). LLM calls rows: routed calls per run, mean±std (setup calls excluded)",
        "Rules only / Statistics only / Rules + statistics / Rules + FDs + statistics are replay-only single-family baselines computed by make_tables.py from the cached tier outputs: per test cell the first non-null value among the listed tiers, a repair counts when output != dirty, P/R/F1 as in the pipeline; the last equals the sweep's rules_stats row (checked to 1e-9). They make no per-cell LLM call",
        "Setup: every row built on the pipeline caches (all but Baran, the LLM-only rows and the published rows) was computed after the same 2 per-dataset setup LLM calls (FD proposal, source proposal); only 'Rules + LLM-proposed FDs + statistics' and the full pipeline use their output in the fd tier. The cached stats tier has no candidate on columns with a kept FD (hence Flights F1 0.000 without the fd tier), so 'Statistics only' and 'Rules + statistics' are the stats tier as cached, not a generic statistics baseline run with FDs off",
        "[a] Jellyfish-8B (our run) used the model-card data-imputation prompt with the dirty value hidden (a different task from repair); cite the published Jellyfish-7B number (Part B, LasRepair) instead",
        "Part B: numbers copied from the papers (T8 sources: results/llm_only_literature.md, paper/final_verdict_fable.md); detection assumption, label budget and model differ from ours, so this is context, not a ranking; '–' = not run / not reported. HoloClean Hospital 0.83 = F1 of the published P 1.00 / R 0.713 (final_verdict_fable.md: denial constraints, own detection, Flights recall not read)"],
        group=True, widths=w)



def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tabs = [T1(), T2(), T3(), T4a(), T4b(), T5(), T6(), T7(), T8(), T9()]
    (OUT / "final_tables.md").write_text("# Final tables (generated by make_tables.py; do not edit)\n\n" + "\n".join(t.md() for t in tabs))
    (OUT / "tables.tex").write_text("% generated by make_tables.py; needs booktabs, longtable, graphicx, amsmath\n" + "\n".join(t.tex() for t in tabs))
    for t in tabs:
        t.csv().to_csv(OUT / f"{t.tid}.csv", index=False, float_format="%.10g")


if __name__ == "__main__":
    main()
