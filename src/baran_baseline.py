"""Baran (Mahdavi & Abedjan, PVLDB 2020) error CORRECTION under PERFECT detection, no transfer learning.

Run with the separate env:  .venv-raha/bin/python src/baran_baseline.py --datasets hospital flights beers rayyan
  Config A: Baran's own tuple sampler (numpy seed 0/1/2), 20 labels; scored on all error cells (paper protocol)
            and on the test split (error cells outside the tuples Baran labeled).
  Config B: our protocol - the labeled tuples are forced to pipeline.calib_test_split(seed) rows (SAMPLER=random), test split.
Metrics: Baran's own (Dataset.get_data_cleaning_evaluation) and ours (P = correct / repairs with output != dirty,
R = correct / error cells, values compared after .strip()). One JSON line per run appended to logs/baran_results.jsonl.
`--summary` prints mean/std over seeds.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import raha.correction, raha.dataset

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "logs" / "baran_results.jsonl"


def our_data(name):
    """Same loader/error definition as pipeline.load(): raw strings, error = strip(dirty) != strip(clean)."""
    kw = dict(dtype=str, keep_default_na=False)
    d = pd.read_csv(ROOT / "data" / name / "dirty.csv", **kw)
    c = pd.read_csv(ROOT / "data" / name / "clean.csv", **kw)
    errs = {(i, j) for j in range(d.shape[1]) for i, (a, b) in enumerate(zip(d.iloc[:, j], c.iloc[:, j])) if a.strip() != b.strip()}
    return d, c, errs


def our_labeled_rows(name, seed):
    """Labeled rows of our pipeline (calib_test_split, SAMPLER=random, 20 tuples), computed in the main .venv."""
    code = (f"import json,pipeline as p; d,c,e=p.load({name!r}); "
            f"calib,_=p.calib_test_split(e, seed={seed}, n_tuples=20); print('ROWS'+json.dumps(sorted({{r for r,_ in calib}})))")
    env = dict(os.environ, SAMPLER="random", LABEL_EXCLUDE_SEM="0", PYTHONPATH=str(SRC))  # pipeline.py lives in src/
    out = subprocess.run([str(ROOT / ".venv/bin/python"), "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout
    return json.loads(out.split("ROWS")[-1])


def ours_metric(corr, dirty, clean, errs, excl, norm=None):
    """P/R/F1 on rows not in `excl`. norm: optional value normalizer applied to clean (Baran normalizes html/whitespace)."""
    f = (lambda s: s.strip()) if norm is None else norm
    E = {c for c in errs if c[0] not in excl}
    reps = {c: v for c, v in corr.items() if c[0] not in excl and f(v) != dirty.iloc[c].strip()}
    ok = sum(1 for c, v in reps.items() if c in E and f(v) == f(clean.iloc[c]))
    p, r = ok / len(reps) if reps else 0.0, ok / len(E) if E else 0.0
    return dict(P=p, R=r, F1=2 * p * r / (p + r) if p + r else 0.0, repairs=len(reps), correct=ok, n_err=len(E))


class ForcedBaran(raha.correction.Correction):
    """Baran with the tuple sampler replaced by a fixed labeled-row list (module-level so Baran's Pool can pickle it)."""
    forced_rows = ()

    def sample_tuple(self, d):
        d.sampled_tuple = self.forced_rows[len(d.labeled_tuples)]


class BlankedBaran(raha.correction.Correction):
    """Diagnostic: vicinity models keyed on the columns in `blank` (column names) produce no candidates."""
    blank = ()

    def _vicinity_based_corrector(self, models, ed):
        out = super()._vicinity_based_corrector(models, ed)
        return [{} if (self.blank == "all" or j in self.blank) else r for j, r in enumerate(out)]


def run(name, cfg, seed, blank=()):
    dirty, clean, errs = our_data(name)
    ds = raha.dataset.Dataset({"name": name, "path": str(ROOT / "data" / name / "dirty.csv"),
                               "clean_path": str(ROOT / "data" / name / "clean.csv")})
    baran_errs = ds.get_actual_errors_dictionary()
    ds.detected_cells = {c: "JUST A DUMMY VALUE" for c in errs}  # perfect detection = OUR error set, positional (row, col_idx)
    app = (ForcedBaran if cfg == "B" else BlankedBaran if blank else raha.correction.Correction)()
    if blank:  # column names -> indices ("all" = every vicinity model)
        app.blank = "all" if blank == ["all"] else {list(dirty.columns).index(c) for c in blank}
    app.SAVE_RESULTS = False  # otherwise Baran writes a results folder into data/<name>/; pretrained model path stays "" = no transfer
    app.LABELING_BUDGET = 20
    if cfg == "B":
        app.forced_rows = our_labeled_rows(name, seed)
    else:
        np.random.seed(seed)  # Baran samples with numpy.random.choice over max-score ties
    t0 = time.time()
    corr = app.run(ds)
    secs = time.time() - t0
    labeled = set(ds.labeled_tuples)
    assert len(labeled) == 20
    rest = {i: 1 for i in range(ds.dataframe.shape[0]) if i not in labeled}
    norm = raha.dataset.Dataset.value_normalizer
    views = {"all": set()} if cfg == "A" else {}
    views["test"] = labeled
    res = dict(blank=blank, dataset=name, cfg=cfg, seed=seed, secs=secs, n_err_ours=len(errs), n_err_baran=len(baran_errs),
               n_corrected=len(corr), labeled=sorted(int(i) for i in labeled))
    for v, excl in views.items():
        b = ds.get_data_cleaning_evaluation(corr, sampled_rows_dictionary=rest if excl else False)
        res[v] = dict(baran=dict(zip(["ed_P", "ed_R", "ed_F1", "P", "R", "F1"], b)),
                      ours=ours_metric(corr, dirty, clean, errs, excl),
                      ours_norm=ours_metric(corr, dirty, clean, errs, excl, norm=norm))
    return res


def summary():
    rows = [json.loads(l) for l in open(OUT)]
    df = pd.DataFrame(rows)
    print("dataset cfg view metric-def : P / R / F1 mean+-std over seeds, mean secs")
    for (n, cfg), g in df.groupby(["dataset", "cfg"], sort=False):
        for v in ("all", "test"):
            if v not in g or g[v].isna().all():
                continue
            for m in ("baran", "ours", "ours_norm"):
                x = pd.DataFrame([r[m] for r in g[v]])
                print(f"{n:9s} {cfg} {v:4s} {m:9s}: " + " / ".join(f"{x[k].mean():.3f}+-{x[k].std(ddof=0):.3f}" for k in ("P", "R", "F1"))
                      + f"   n_seeds={len(g)} secs={g.secs.mean():.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["hospital", "flights", "beers", "rayyan"])
    ap.add_argument("--configs", nargs="+", default=["A", "B"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--blank-vicinity", nargs="+", default=[], help="ablation (config A): column names whose vicinity models are muted, or 'all'")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary(); sys.exit()
    OUT.parent.mkdir(exist_ok=True)
    for name in a.datasets:
        for cfg in a.configs:
            for seed in a.seeds:
                r = run(name, cfg, seed, a.blank_vicinity)
                print(json.dumps(r), flush=True)
                with open(OUT if not a.blank_vicinity else OUT.with_name("baran_ablation.jsonl"), "a") as f:
                    f.write(json.dumps(r) + "\n")
