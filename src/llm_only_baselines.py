"""Pure-LLM repair baselines under our protocol (oracle detection, every error cell asked once, test split of calib_test_split).

    LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=Qwen/Qwen3-8B python src/llm_only_baselines.py ask --model qwen_strong --datasets all
    LLM_BASE_URL=... LLM_MODEL=NECOUDBFM/Jellyfish-8B           python src/llm_only_baselines.py ask --model jellyfish   --datasets all
    python src/llm_only_baselines.py eval                          # random sampler, all datasets -> results/summary/llm_only_baselines*.csv
    SAMPLER=coverage python src/llm_only_baselines.py eval --datasets beers_sem   # extra config (beers_sem coverage)

qwen_strong = pl.tier_llm_sem (knowledge prompt, n=3, T=0.7, thinking off, majority vote; a sample equal to the dirty value is dropped,
no sample left = no repair), asked for EVERY error cell. jellyfish = Jellyfish-8B, data-imputation prompt of its model card with the
erroneous cell omitted, T=0, n=1. Answers cache to cache/b20_s0/llm_only_<model>_<name>.jsonl (one line per cell, resumable).
"""
import argparse
import concurrent.futures
import csv
import json
import os
import statistics
import time
from pathlib import Path

import pipeline as pl

NAMES = pl.DATASETS + pl.SEM_DATASETS
CACHE = pl.SHARED_CACHE  # cache/b20_s0/
RESULTS = pl.BASE_DIR / "results" / "summary"
KEYWORD = {"hospital": "hospital", "flights": "flight", "beers": "beer", "rayyan": "article", "movies": "movie"}  # {keyword} of the card's prompt
JF_SYSTEM = "You are an AI assistant that follows instruction extremely well.\nUser will give you a question. Your task is to answer as faithfully as you can."
JF_MAX_TOKENS = 128
ZERO = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}


def jellyfish_prompt(dirty, row, col, name):
    """Jellyfish model card, 'For Data Imputation' user prompt + its template; the cell's own (erroneous) value is omitted."""
    rec = ", ".join(f"{c}: {dirty.at[row, c]}" for c in dirty.columns if c != col)
    kw = KEYWORD[name.removesuffix("_sem")]
    user = (f"You are presented with a {kw} record that is missing a specific attribute: {col}.\n"
            f"Your task is to deduce or infer the value of {col} using the available information in the record.\n"
            f"You may be provided with fields like {', '.join(c for c in dirty.columns if c != col)} to help you in the inference.\n"
            f"Record: [{rec}]\n"
            f"Based on the provided record, what would you infer is the value for the missing attribute {col}?\n"
            f"Answer only the value of {col}.")
    # the card's template drops the "\n\n" after each header that Llama-3 expects: without it the model answers with a stray "\n\n"/">" prefix
    # (checked on a capital-of-France and a restaurant-city probe), so the standard Llama-3 whitespace is used
    return (f"<|start_header_id|>system<|end_header_id|>\n\n{JF_SYSTEM}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


def ask_jellyfish(dirty, row, col, name):
    t0 = time.time()
    resp = pl._get_client().completions.create(model=os.environ["LLM_MODEL"], prompt=jellyfish_prompt(dirty, row, col, name), temperature=0,
                                               max_tokens=JF_MAX_TOKENS, stop=["<|eot_id|>"])
    u = resp.usage
    cost = {"calls": 1, "prompt_tokens": u.prompt_tokens or 0, "completion_tokens": u.completion_tokens or 0, "seconds": time.time() - t0}
    text = resp.choices[0].text.strip()
    return (text.splitlines() or [""])[0].strip() or None, text, cost


def cache_path(model, name):
    return CACHE / f"llm_only_{model}_{name}.jsonl"


def read_cache(model, name):
    p = cache_path(model, name)
    return {(o["row"], o["col"]): o for o in map(json.loads, filter(str.strip, p.open()))} if p.exists() else {}


def cmd_ask(a):
    from openai import BadRequestError
    for name in NAMES if a.datasets == ["all"] else a.datasets:
        dirty, _, error_cells = pl.load(name)
        err_set, done = set(error_cells), read_cache(a.model, name)
        todo = [rc for rc in error_cells if rc not in done]
        print(f"[{a.model}/{name}] {len(error_cells)} error cells, {len(todo)} to ask", flush=True)
        if not todo:
            continue
        ex = {c: pl._format_examples(dirty, err_set, c) for c in dirty.columns}

        def one(rc):
            row, col = rc
            if a.model == "qwen_strong":
                v, _, cost = pl.tier_llm_sem(dirty, row, col, ex[col])  # exceptions are swallowed inside _llm_vote
                return {"row": row, "col": col, "value": v, "cost": cost} if cost["completion_tokens"] else None
            try:
                v, raw, cost = ask_jellyfish(dirty, row, col, name)
            except BadRequestError as e:  # deterministic (context length): record as no answer
                return {"row": row, "col": col, "value": None, "raw": None, "err": str(e)[:200], "cost": dict(ZERO, calls=1)}
            except Exception:
                return None  # transient: not cached, asked again on resume
            return {"row": row, "col": col, "value": v, "raw": raw, "cost": cost}

        t0, n_ok = time.time(), 0
        with open(cache_path(a.model, name), "a") as f, concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for i, fut in enumerate(concurrent.futures.as_completed([pool.submit(one, rc) for rc in todo]), 1):
                o = fut.result()
                if o:
                    f.write(json.dumps(o, ensure_ascii=False) + "\n")
                    f.flush()
                    n_ok += 1
                if i % 500 == 0:
                    print(f"  {i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
        wall = time.time() - t0
        tp = CACHE / f"llm_only_{a.model}_{name}_timing.json"
        prev = json.loads(tp.read_text()) if tp.exists() else {"wall_seconds": 0.0, "runs": []}
        prev["wall_seconds"] += wall
        prev["runs"].append({"asked": len(todo), "cached": n_ok, "wall_seconds": wall})
        tp.write_text(json.dumps(prev))
        print(f"[{a.model}/{name}] cached {n_ok}/{len(todo)} in {wall:.0f}s", flush=True)


def score(dirty, clean, cells, answers):
    """Metrics of one system on `cells` ((row, col) tuples). A cell repairs iff the answer is non-empty and != dirty (strip-compared)."""
    c = dict.fromkeys(("n", "repairs", "correct", "calls", "prompt_tokens", "completion_tokens", "seconds", "empty"), 0)
    seg = {True: dict(n=0, repairs=0, correct=0), False: dict(n=0, repairs=0, correct=0)}
    for rc in cells:
        o = answers[rc]
        d, t = dirty.at[rc[0], rc[1]].strip(), clean.at[rc[0], rc[1]].strip()
        v = o["value"]
        rep = v is not None and v.strip() != d
        ok = rep and v.strip() == t
        c["n"] += 1
        c["repairs"] += rep
        c["correct"] += ok
        c["empty"] += not rep
        for k in ("calls", "prompt_tokens", "completion_tokens", "seconds"):
            c[k] += o["cost"].get(k, 0)
        s = seg[rc in pl._SEM_CELLS]
        s["n"] += 1
        s["repairs"] += rep
        s["correct"] += ok
    c["precision"], c["recall"], c["f1"] = pl._prf(c["correct"], c["repairs"], c["n"])
    c["tokens"] = c["prompt_tokens"] + c["completion_tokens"]
    c["llm_seconds"], c["latency_per_cell"] = c["seconds"], c["seconds"] / c["n"]
    if pl._SEM_CELLS:
        _, c["sem_recall"], c["sem_f1"] = pl._prf(seg[True]["correct"], seg[True]["repairs"], seg[True]["n"])
        c["nonsem_f1"] = pl._prf(seg[False]["correct"], seg[False]["repairs"], seg[False]["n"])[2]
    return c


METRICS = ["precision", "recall", "f1", "llm_calls", "prompt_tokens", "completion_tokens", "tokens", "llm_seconds", "latency_per_cell", "sem_f1",
           "sem_recall", "nonsem_f1", "total_calls"]


def sweep_file(name):
    return "budget_sweep.csv" if name in pl.DATASETS else "budget_sweep_sem.csv"


def cmd_eval(a):
    sfx = "_cov" if pl.SAMPLER == "coverage" else ""
    runs, summary = [], []
    for name in NAMES if a.datasets == ["all"] else a.datasets:
        dirty, clean, error_cells = pl.load(name)
        answers = {m: read_cache(m, name) for m in a.models}
        sweep = {(r["method"], int(r["budget"])): r for r in csv.DictReader(open(RESULTS / (sweep_file(name)[:-4] + sfx + ".csv")))
                 if r["dataset"] == name}
        for n in (20, 50):
            per = {m: [] for m in a.models}
            for s in (0, 1, 2):
                calib, test = pl.calib_test_split(error_cells, s, n)
                for m in a.models:
                    missing = [rc for rc in test if rc not in answers[m]]
                    assert not missing, f"{m}/{name}: {len(missing)} test cells not asked"
                    r = score(dirty, clean, test, answers[m])
                    r["llm_calls"] = r.pop("calls")
                    per[m].append(r)
                    runs.append({"dataset": name, "sampler": pl.SAMPLER, "budget": n, "seed": s, "system": m, "n_calib": len(calib), "n_test": len(test),
                                 **{k: v for k, v in r.items() if k != "seconds"}})
            for system, rows in [*per.items(), *[(x, None) for x in ("ours_lambda@0", "llm_only")]]:
                row = {"dataset": name, "sampler": pl.SAMPLER, "budget": n, "system": system, "n_seeds": 3}
                if rows is not None:
                    row["n_test"] = statistics.mean(r["n"] for r in rows)
                    for k in METRICS:
                        if k in rows[0]:
                            row[f"{k}_mean"], row[f"{k}_std"] = statistics.mean(r[k] for r in rows), statistics.stdev(r[k] for r in rows)
                else:  # ours / existing llm_only: the sweep csv (same splits, same seeds)
                    x = sweep[(system, n)]
                    row["n_test"] = float(x["n_test"])
                    for k in METRICS:
                        if f"{k}_mean" in x:
                            row[f"{k}_mean"], row[f"{k}_std"] = float(x[f"{k}_mean"]), float(x[f"{k}_std"])
                summary.append(row)
            nt = {r["system"]: r["n_test"] for r in summary[-(len(a.models) + 2):]}
            assert len(set(nt.values())) == 1, f"{name} b{n}: n_test differs between baselines and the sweep: {nt}"  # same splits as the sweep
    out = RESULTS / f"llm_only_baselines{sfx}"
    for suffix, rows in (("_runs.csv", runs), ("_summary.csv", summary)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(f"{out}{suffix}", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {out}_runs.csv, {out}_summary.csv")
    for r in summary:
        print(f"{r['dataset']:13s} b{r['budget']} {r['system']:14s} P={r.get('precision_mean', float('nan')):.3f} R={r.get('recall_mean', float('nan')):.3f} "
              f"F1={r['f1_mean']:.3f}+-{r['f1_std']:.3f} calls={r['llm_calls_mean']:.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ask")
    p.add_argument("--model", required=True, choices=["qwen_strong", "jellyfish"])
    p.add_argument("--datasets", nargs="+", default=["all"])
    p.set_defaults(func=cmd_ask)
    p = sub.add_parser("eval")
    p.add_argument("--models", nargs="+", default=["qwen_strong", "jellyfish"])
    p.add_argument("--datasets", nargs="+", default=["all"])
    p.set_defaults(func=cmd_eval)
    args = ap.parse_args()
    args.func(args)
