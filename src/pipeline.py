"""Confidence-calibrated routing for hybrid tabular data cleaning.

Six repair tiers (rules -> fd -> stats -> column -> llm -> llm_sem). Every cell always starts
at the three free tiers (rules, fd, stats); only once those miss tau does an
error-type profiler pick which paid tier to escalate to (column for
systematic errors, llm otherwise; llm_sem, the world-knowledge LLM operator, comes last). Isotonic calibration gates each tier's
confidence, and a handful of routing policies replay over a cached
precompute pass.
"""
import builtins as _builtins
import concurrent.futures
import functools
import itertools
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ponytail: difflib is O(n*m) per pair; swap to rapidfuzz if slow
import difflib

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root (this file lives in src/)
DATA_DIR = BASE_DIR / "data"
SHARED_CACHE = BASE_DIR / "cache" / "b20_s0"  # default config (20 labeled rows, seed 0, random); also holds the split-free shared caches
CACHE_DIR = Path(os.environ.get("CACHE_DIR", SHARED_CACHE))

DATASETS = ["hospital", "flights", "beers", "rayyan", "movies"]  # movies: held-out, added later (data plumbing only)
SEM_DATASETS = ["hospital_sem", "beers_sem"]  # controlled semantic-error variants (see inject_semantic); not part of "all"

TIER_ORDER = ["rules", "fd", "stats", "column", "llm", "llm_sem"]


# ---------------------------------------------------------------- data -----

def load(name):
    """Load dirty/clean csvs for `name`, return (dirty_df, clean_df, error_cells)."""
    d = pd.read_csv(DATA_DIR / name / "dirty.csv", dtype=str, keep_default_na=False)
    c = pd.read_csv(DATA_DIR / name / "clean.csv", dtype=str, keep_default_na=False)
    assert d.shape == c.shape, f"{name}: shape mismatch {d.shape} vs {c.shape}"
    if list(d.columns) != list(c.columns):
        # ponytail: raha's clean/dirty csvs for hospital/beers use different
        # column-name casing/style for the *same* positional columns (checked
        # by hand) - align clean's header onto dirty's by position.
        c = c.copy()
        c.columns = d.columns
    error_cells = []
    for col in d.columns:
        dv, cv = d[col].tolist(), c[col].tolist()
        for i, (a, b) in enumerate(zip(dv, cv)):
            if a.strip() != b.strip():
                error_cells.append((i, col))
    global _SEM_CELLS
    sem_path = DATA_DIR / name / "semantic_cells.json"
    _SEM_CELLS = {(r["row"], r["col"]) for r in json.loads(sem_path.read_text())} if sem_path.exists() else set()
    global _UNRESOLVED
    if SAMPLER == "coverage":
        _UNRESOLVED = _unresolved_set(name, d, error_cells)
    print(f"[{name}] rows={len(d)} cols={len(d.columns)} error_cells={len(error_cells)}")
    return d, c, error_cells


_UNRESOLVED = set()  # coverage sampler: (row, col) cells of the dataset last passed to load() that no label-free tier resolves
_SEM_CELLS = set()  # injected (row, col) cells of the dataset last passed to load(); empty for the original datasets


# Semantic-error injection: WORLD-KNOWLEDGE errors the table cannot resolve. Every row of an entity gets the same wrong
# `state` (so majority votes and dependencies agree with the error), only for entities whose city is theirs alone.
SEM_CFG = {"hospital": dict(entity="provider_number", n_state=150, n_city=50),
           "beers": dict(entity="brewery_id", n_state=250, n_city=0)}
_AMBIGUOUS_CITIES = {"springfield", "portland", "columbus", "jackson", "franklin", "greenville", "clinton", "salem",
                     "madison", "washington", "richmond", "arlington", "aurora", "kansas city", "charleston"}


def inject_semantic(dirty, clean, entity, n_state, n_city=0, seed=0, state="state", city="city"):
    """(dirty copy with injected errors, [{"row", "col", "type", "entity"}]). Entities are drawn in seeded random
    order until n_state cells carry a wrong `state` (uniform over the OTHER states of the table), then n_city cells
    a wrong `city` (a city of another state in the table; hospital only). An entity qualifies iff its rows agree on
    a clean city that belongs to no other entity, is not in _AMBIGUOUS_CITIES and appears with one state only, and
    none of its cells in the injected column is already dirty (existing errors are never touched)."""
    strip = lambda df, c: df[c].str.strip()
    cs = clean.groupby(strip(clean, city))[state].agg(lambda x: set(v.strip() for v in x))
    ok_city = {c for c, st in cs.items() if len(st) == 1 and c.lower() not in _AMBIGUOUS_CITIES}
    owners = clean.groupby(strip(clean, city))[entity].nunique()
    states = sorted(set(strip(clean, state)))
    rng, out, log, used = random.Random(seed), dirty.copy(), [], set()
    for col, n in ((state, n_state), (city, n_city)):
        ents = sorted(e for e, g in clean.groupby(entity) if e not in used and len(set(strip(g, city))) == 1
                      and strip(g, city).iloc[0] in ok_city and owners[strip(g, city).iloc[0]] == 1
                      and (strip(dirty.loc[g.index], col) == strip(g, col)).all())
        rng.shuffle(ents)
        done = 0
        for e in ents:
            if done >= n:
                break
            rows = list(clean.index[clean[entity] == e])
            st = strip(clean, state)[rows[0]]
            wrong = rng.choice([s for s in states if s != st] if col == state else sorted(c for c in ok_city if next(iter(cs[c])) != st))
            for r in rows:
                out.at[r, col] = wrong
                log.append({"row": r, "col": col, "type": col, "entity": e})
            done += len(rows)
            used.add(e)
    return out, log


def write_semantic(name):
    """data/<name>_sem/{dirty.csv, clean.csv, semantic_cells.json} from data/<name>; the original files are only read."""
    import shutil
    src, dst = DATA_DIR / name, DATA_DIR / f"{name}_sem"
    d = pd.read_csv(src / "dirty.csv", dtype=str, keep_default_na=False)
    c = pd.read_csv(src / "clean.csv", dtype=str, keep_default_na=False)
    c.columns = d.columns
    out, log = inject_semantic(d, c, **SEM_CFG[name])
    dst.mkdir(exist_ok=True)
    shutil.copyfile(src / "clean.csv", dst / "clean.csv")
    out.to_csv(dst / "dirty.csv", index=False)
    (dst / "semantic_cells.json").write_text(json.dumps(log))
    return log


# ----------------------------------------------------- column profiling ----

def _generalize(s):
    out = []
    for ch in s:
        if ch.isalpha():
            out.append("A" if ch.isupper() else "a")
        elif ch.isdigit():
            out.append("9")
        else:
            out.append(ch)
    return "".join(out)


def _dominant_case(pattern):
    alpha = [ch for ch in pattern if ch in ("A", "a")]
    if not alpha:
        return None
    if all(ch == "A" for ch in alpha):
        return "upper"
    if all(ch == "a" for ch in alpha):
        return "lower"
    return "title"  # ponytail: mixed-case patterns approximated as title case


_NULL_TOKENS = {"n/a", "na", "null", "none", "nan", "-", "--", "?"}
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_number(val):
    m = _NUM_RE.search(val)
    if not m:
        return None
    text = m.group(0)
    f = float(text)
    return str(int(f)) if f == int(f) else text


def _error_groups(dirty_df, error_cells):
    """Group error cells by (col, generalized pattern of the dirty value)."""
    groups = defaultdict(list)
    for row, col in error_cells:
        val = dirty_df.at[row, col].strip()
        groups[(col, _generalize(val))].append((row, col))
    return groups


def _build_profile(df, error_cells=None, labeled=None):
    profile = {"pattern": {}, "freq": {}, "groups": {}, "numeric": {}, "good_pattern": {}, "good_freq": {}}
    for col in df.columns:
        vals = [v.strip() for v in df[col].tolist()]
        # ponytail: our setting assumes perfect (oracle) error detection, so the "known good" value set and
        # dominant pattern that score repair candidates come from non-error cells only: a systematically dirty
        # value ("12.0 oz.") is frequent but must not vouch for itself. Plus the CLEAN values of the labeled calib
        # cells (a column with no non-error cells, e.g. Beers `ounces`, is otherwise all dirty). ponytail: these
        # labels are the same 20/50 labeled rows every other component uses. <2 distinct good values: all cells.
        good = [v for i, v in enumerate(vals) if v and (i, col) not in (error_cells or ())]
        good += [v for (_, c), v in (labeled or {}).items() if c == col and v]
        if len(set(good)) < 2:
            good = [v for v in vals if v]
        profile["good_freq"][col] = Counter(good)
        profile["good_pattern"][col] = Counter(_generalize(v) for v in good).most_common(1)[0][0] if good else ""
        gen = [_generalize(v) for v in vals if v]
        pattern = Counter(gen).most_common(1)[0][0] if gen else ""
        profile["pattern"][col] = pattern
        profile["freq"][col] = Counter(v for v in vals if v)
        nonempty = [v for v in vals if v]
        n_num = sum(1 for v in nonempty if _NUM_RE.search(v))
        profile["numeric"][col] = bool(nonempty) and n_num / len(nonempty) >= 0.8
        groups = defaultdict(list)
        for i, v in enumerate(vals):
            if v:
                groups[v].append(i)
        profile["groups"][col] = groups
    return profile


# ponytail: id()-keyed cache (df, error_cells); entries hold strong refs to both so a live
# key's id can never be reused by a different (freed) object - not safe
# across processes, but safe for many short-lived DataFrames in one process.
_PROFILE_CACHE = {}


def _get_profile(df, error_cells=None, labeled=None):
    key = (id(df), id(error_cells), id(labeled))
    cached = _PROFILE_CACHE.get(key)
    if cached is None or cached[0] is not df or cached[1] is not error_cells or cached[3] is not labeled:
        cached = (df, error_cells, _build_profile(df, error_cells, labeled), labeled)
        _PROFILE_CACHE[key] = cached
    return cached[2]


def _labeled_clean(clean, error_cells):
    """{(row, col): clean value} of the labeled calib-split cells only (same split as everything else)."""
    calib, test = calib_test_split(error_cells)
    labeled = {(r, c): clean.at[r, c].strip() for r, c in calib}
    assert not labeled.keys() & set(test), "leakage: a test-split cell's clean value would be used"
    return labeled


def _reformat_digits(val, pattern):
    if any(ch in ("A", "a") for ch in pattern):
        return None
    digits = [ch for ch in val if ch.isdigit()]
    needed = sum(1 for ch in pattern if ch == "9")
    if len(digits) != needed:
        return None
    out, di = [], 0
    for ch in pattern:
        if ch == "9":
            out.append(digits[di])
            di += 1
        else:
            out.append(ch)
    return "".join(out)


def _normalize_candidates(val, pattern):
    cands = []
    stripped = val.strip()
    cands.append(stripped)
    cands.append(re.sub(r"\s+", " ", stripped))
    case = _dominant_case(pattern)
    if case == "upper":
        cands.append(stripped.upper())
    elif case == "lower":
        cands.append(stripped.lower())
    elif case == "title":
        cands.append(stripped.title())
    cands.append(re.sub(r"[^\w\s.\-]", "", stripped))
    digit_cand = _reformat_digits(stripped, pattern)
    if digit_cand:
        cands.append(digit_cand)
    seen, out = set(), []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def error_type(dirty_df, row, col, error_groups=None):
    val = dirty_df.at[row, col].strip()
    if error_groups is not None:
        # Systematic, column-wide transformations (e.g. every `ounces` cell
        # dirty in the same way) - checked first, ahead of syntactic/statistical.
        group = error_groups.get((col, _generalize(val)))
        if group and len(group) >= 10:
            return "systematic"
    profile = _get_profile(dirty_df)
    pattern = profile["pattern"][col]
    freq = profile["freq"][col]
    if not val or _generalize(val) != pattern:
        return "syntactic"
    if freq.get(val, 0) <= 1:
        return "statistical"
    return "semantic"


# -------------------------------------------------------------- tier 1 -----

def tier_rules(dirty_df, row, col, error_cells=None, labeled=None):
    t0 = time.time()
    profile = _get_profile(dirty_df, error_cells, labeled)
    pattern = profile["good_pattern"][col]
    freq = profile["good_freq"][col]
    value_set = {v for v, n in freq.items() if n >= 2}
    val = dirty_df.at[row, col].strip()

    if val.lower() in _NULL_TOKENS:
        candidate = ""
    else:
        candidate = None
        if val and _generalize(val) == pattern:
            candidate = val
        else:
            for cand in _normalize_candidates(val, pattern):
                if _generalize(cand) == pattern:
                    candidate = cand
                    break
        # Systematically-dirty numeric column (Beers-style "12.0 oz", "0.05%"):
        # pattern-matching alone just echoes the uniformly-dirty format back
        # unchanged. Only step in then - if pattern-matching instead found
        # nothing at all (candidate is None), this is more likely a one-off,
        # non-numeric corruption, and stats/llm's fallback is a safer bet.
        # Require exactly one number in the value - a value with two numbers
        # (Flights times like "7:16 a.m.") isn't a single-number-plus-unit
        # slip and extraction would just pick one arbitrarily.
        if candidate == val and profile["numeric"][col] and len(_NUM_RE.findall(val)) == 1:
            num = _extract_number(val)
            if num is not None and num != candidate:
                candidate = num

    if candidate is None:
        raw_conf = 0.0
    elif candidate in value_set:
        raw_conf = 1.0
    else:
        raw_conf = 0.5
    cost = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": time.time() - t0}
    return candidate, raw_conf, cost


# -------------------------------------------------------------- tier 2 -----

def _voters(dirty_df, idxs, row, col, error_cells):
    """Rows of `idxs` other than `row` with a non-empty `col` value that vote in a dependency vote: those whose `col`
    cell is not a known error cell (one is enough: perfect detection makes it strong evidence), else all of them,
    but then only with >= 2 (an unverified lone voter is not a vote). No voters: [] = abstain."""
    rows = [r for r in idxs if r != row and dirty_df.at[r, col].strip()]
    # ponytail: perfect detection => known-clean voters; Baran's vicinity models use the same signal
    return [r for r in rows if (r, col) not in (error_cells or ())] or (rows if len(rows) >= 2 else [])


def _stats_candidates(dirty_df, row, col, fds=None, error_cells=None):
    """Shared by tier_stats and tier_llm; dirty table only.
    Returns (deps, sims): deps = [(value, conf, support_count)] one per other column B
    (majority value of `col` among other rows sharing this row's B value), sims = [(value,
    difflib ratio)] over frequent (>=2) values of `col`. A column with a kept FD (`fds`, see kept_fds)
    belongs to tier_fd: stats has no candidate there."""
    profile = _get_profile(dirty_df)
    freq = profile["freq"][col]
    groups = profile["groups"]
    val = dirty_df.at[row, col].strip()

    if fds and col in fds:
        return [], []

    deps = []
    for b in dirty_df.columns:
        if b == col:
            continue
        b_val = dirty_df.at[row, b].strip()
        if not b_val:
            continue
        idxs = groups[b].get(b_val)
        if not idxs or len(idxs) < 2:
            continue
        vals = [dirty_df.at[r2, col].strip() for r2 in _voters(dirty_df, idxs, row, col, error_cells)]
        if not vals:
            continue
        maj_val, maj_count = Counter(vals).most_common(1)[0]
        support = len(vals)
        deps.append((maj_val, (maj_count / support) * min(support, 5) / 5, maj_count))

    # ponytail: no `fv != val` filter, on purpose. An already-frequent dirty value matches itself at
    # ratio 1.0, wins, and load_cache drops it as a no-op: deliberate abstention (stats must never
    # "correct" a value that is already frequent), leaving the cell to the paid tiers.
    sims = [(fv, difflib.SequenceMatcher(None, val, fv).ratio()) for fv, n in freq.items() if n >= 2]
    return deps, sims


def tier_stats(dirty_df, row, col, fds=None, error_cells=None):
    # ponytail: no dedicated numeric-outlier handling (e.g. z-score repair);
    # dependency vote + nearest-frequent-value covers the benchmarks fine.
    t0 = time.time()
    deps, sims = _stats_candidates(dirty_df, row, col, fds, error_cells)
    candidates = []
    if deps:
        d = max(deps, key=lambda x: x[1])
        candidates.append((d[0], d[1]))
    if sims:
        candidates.append(max(sims, key=lambda x: x[1]))

    cost = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": time.time() - t0}
    if not candidates:
        return None, 0.0, cost
    candidate, raw_conf = max(candidates, key=lambda x: x[1])
    return candidate, raw_conf, cost


def tier_fd(dirty_df, row, col, fds=None, error_cells=None):
    """Vote of the kept FD for `col` (see kept_fds): the majority value among the other rows sharing this row's
    lhs (source-reliability weighted when the calib split preferred that, see kept_fds); raw_conf = its share,
    times min(support, 5) / 5 when the calib split preferred that variant. No kept FD, no vote or an echo of
    the dirty value: no candidate."""
    t0 = time.time()
    vote = _fd_vote(dirty_df, row, fds[col][0], col, fds[col][6], error_cells) if fds and col in fds else None
    cost = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": time.time() - t0}
    if vote is None or vote[0] == dirty_df.at[row, col].strip():
        return None, 0.0, cost
    return vote[0], _fd_conf(vote, fds[col][4]), cost


def _fd_conf(vote, conf):
    return vote[1] * (min(vote[3], 5) / 5 if conf == "support" else 1.0)


# ------------------------------------------------------- LLM-proposed FDs ---
# Multi-source tables (Flights) need a vote over rows sharing the *right* key column, and picking it from
# dirty data alone fails (correlated time columns win). So the LLM proposes lhs -> rhs dependencies once
# per dataset (dirty table only, like Cocoon/LLMClean) and each split keeps only those whose vote is
# accurate on its labeled (calib) cells.

FDS_DIR = SHARED_CACHE  # shared by all split configs: the proposal depends only on the dirty table
FD_SAMPLE_ROWS, FD_SEED = 15, 0


def _fd_prompt(dirty_df):
    sample = dirty_df.sample(min(FD_SAMPLE_ROWS, len(dirty_df)), random_state=FD_SEED)
    return (
        "You are analysing a possibly dirty table; it may combine several sources describing the same entity.\n"
        f"Columns: {list(dirty_df.columns)}\n"
        f"Sample rows (CSV):\n{sample.to_csv(index=False)}\n"
        "Propose functional dependencies that should hold in the clean table: rows that agree on the lhs "
        "columns must agree on the rhs column. The lhs must identify a real-world entity that several rows "
        "describe (a code or name), never a row id or any column that is unique per row. Reply with ONLY a JSON list, no prose: "
        '[{"lhs": ["<col>"], "rhs": "<col>"}, ...] with 1-2 lhs columns and a single rhs column.'
    )


def _parse_fds(text, columns):
    """[(lhs tuple, rhs)] from the model's JSON; entries with unknown columns or rhs in lhs are dropped."""
    try:
        raw = json.loads(text[text.index("["):text.rindex("]") + 1])
    except Exception:
        return []
    fds = []
    for e in raw if isinstance(raw, list) else []:
        try:
            lhs, rhs = e["lhs"], e["rhs"]
            lhs = tuple([lhs] if isinstance(lhs, str) else lhs)
            if 1 <= len(lhs) <= 2 and rhs in columns and all(c in columns for c in lhs) and rhs not in lhs \
                    and (lhs, rhs) not in fds:
                fds.append((lhs, rhs))
        except Exception:
            continue
    return fds


def propose_fds(name, dirty_df):
    """ONE LLM call per dataset (cached in cache/<name>_fds.json with raw response + cost) -> (fds, cost)."""
    path = FDS_DIR / f"{name}_fds.json"
    if not path.exists():
        if os.environ.get("LLM_STUB") == "1":
            return [], {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
        t0 = time.time()
        resp = _get_client().chat.completions.create(
            model=os.environ.get("LLM_MODEL", "Qwen/Qwen3-8B"), messages=[{"role": "user", "content": _fd_prompt(dirty_df)}],
            temperature=0, n=1, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        usage = getattr(resp, "usage", None)
        cost = {"calls": 1, "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0, "seconds": time.time() - t0}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"raw": resp.choices[0].message.content, "cost": cost}, ensure_ascii=False, indent=1))
    rec = json.loads(path.read_text())
    return _parse_fds(rec["raw"] or "", list(dirty_df.columns)), rec["cost"]


def _source_prompt(dirty_df):
    sample = dirty_df.sample(min(FD_SAMPLE_ROWS, len(dirty_df)), random_state=FD_SEED)
    return (
        "You are analysing a possibly dirty table.\n"
        f"Columns: {list(dirty_df.columns)}\n"
        f"Sample rows (CSV):\n{sample.to_csv(index=False)}\n"
        "If the rows come from several data sources (different sources describing the same real-world entity), "
        "which single column identifies the source of each row? Reply with ONLY that column name, or null if "
        "there is no such column."
    )


def propose_source(name, dirty_df):
    """ONE more LLM call per dataset (cached in cache/<name>_source.json, separate from the FD proposal) ->
    (source column or None, cost). Only an exact column name counts; anything else (null, prose) is None."""
    path = FDS_DIR / f"{name}_source.json"
    if not path.exists():
        if os.environ.get("LLM_STUB") == "1":
            return None, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
        t0 = time.time()
        resp = _get_client().chat.completions.create(
            model=os.environ.get("LLM_MODEL", "Qwen/Qwen3-8B"), messages=[{"role": "user", "content": _source_prompt(dirty_df)}],
            temperature=0, n=1, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        usage = getattr(resp, "usage", None)
        cost = {"calls": 1, "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0, "seconds": time.time() - t0}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"raw": resp.choices[0].message.content, "cost": cost}, ensure_ascii=False, indent=1))
    rec = json.loads(path.read_text())
    answer = (rec["raw"] or "").strip().strip("`'\". \n")
    return (answer if answer in dirty_df.columns else None), rec["cost"]


def setup_cost(name):
    """{calls, tokens, seconds} spent once per dataset before routing (the FD and source proposals); reported apart from per-method llm_calls."""
    costs = [json.loads(path.read_text())["cost"] for path in (FDS_DIR / f"{name}_fds.json", FDS_DIR / f"{name}_source.json") if path.exists()]
    return {"calls": sum(c["calls"] for c in costs), "tokens": sum(c["prompt_tokens"] + c["completion_tokens"] for c in costs),
            "seconds": sum(c["seconds"] for c in costs)}


def setup_llm_calls(name):
    return setup_cost(name)["calls"]


def _truth_discovery(dirty_df, lhs, rhs, src, w0=None, iters=5):
    """{source: weight} for the votes of lhs -> rhs: iterative weighted voting in the spirit of ACCU (Dong et
    al., "Integrating conflicting data: the role of source dependence", PVLDB 2009), minus the source-dependence
    part. Start from equal weights (or `w0`); repeat `iters` times: (a) every entity group (rows sharing the lhs
    values) takes the weighted-majority value, (b) a source's weight becomes the Laplace-smoothed rate at which
    its claims agree with those values, over ALL its claims in the dirty table. Dirty data only. Groups with a
    single claim carry no evidence and are skipped."""
    keys = list(zip(*[dirty_df[c].str.strip() for c in lhs]))
    groups = defaultdict(list)
    for k, v, s in zip(keys, dirty_df[rhs].str.strip(), dirty_df[src].str.strip()):
        if v and s and all(k):
            groups[k].append((s, v))
    groups = [g for g in groups.values() if len(g) >= 2]
    w = dict(w0 or {})
    for _ in range(iters):
        agree, total = Counter(), Counter()
        for g in groups:
            score = defaultdict(float)
            for s, v in g:
                score[v] += w.get(s, 0.5)
            best = max(score, key=score.get)
            for s, v in g:
                total[s] += 1
                agree[s] += v == best
        w = {s: (agree[s] + 1) / (total[s] + 2) for s in total}
    return w


def _fd_vote(dirty_df, row, lhs, rhs, truth=None, error_cells=None):
    """(majority value of rhs, its share, its count, support = number of voting rows) among the OTHER rows
    sharing `row`'s lhs values, or None without voters (see _voters). `truth` = (source column, source weights):
    the majority and share are then weighted by each row's source (unseen source: 0.5). Voters exclude known
    error cells of `rhs` (`error_cells`) unless all of them are (see _voters)."""
    keys = [dirty_df.at[row, c].strip() for c in lhs]
    if not all(keys):
        return None
    idxs = _get_profile(dirty_df)["groups"][lhs[0]].get(keys[0], [])
    match = [r2 for r2 in idxs if all(dirty_df.at[r2, c].strip() == k for c, k in zip(lhs, keys))]
    vals = [(dirty_df.at[r2, rhs].strip(), truth[1].get(dirty_df.at[r2, truth[0]].strip(), 0.5) if truth else 1.0)
            for r2 in _voters(dirty_df, match, row, rhs, error_cells)]
    if not vals:
        return None
    score, count = defaultdict(float), Counter()
    for v, w in vals:
        score[v] += w
        count[v] += 1
    v = max(score, key=lambda x: (score[x], count[x]))
    return v, score[v] / sum(score.values()), count[v], len(vals)


def kept_fds(name, dirty, clean, error_cells):
    """{rhs: (lhs, score, answered, ok, conf, variant, truth)}: per rhs the best proposed FD whose vote is right
    on the calib error cells (score (ok+1)/(answered+2) >= 0.5 with >= 2 answered). With a source column
    (propose_source) the vote is tried unweighted, source-weighted (_truth_discovery) and weighted with weights
    initialised from the calib rows' labels; `variant` is the one with the best calib score (ties: the simpler),
    `truth` its (source column, weights) or None. `conf` = "share" or "support", the raw confidence (see
    tier_fd) with the higher AUC on those calib cells (ties: share). Labels of the test split are never read."""
    calib, test = calib_test_split(error_cells)
    errs = set(error_cells)
    src = propose_source(name, dirty)[0]
    calib_rows = {r for r, _ in calib}
    assert not calib_rows & {r for r, _ in test}, "leakage: labeled rows overlap the test split"
    used, kept = [], {}
    for lhs, rhs in propose_fds(name, dirty)[0]:
        variants = [("unweighted", None)]
        if src:
            w1 = _truth_discovery(dirty, lhs, rhs, src)
            ok_r, n_r = Counter(), Counter()  # calib-row labels only: per source, claims on rhs that are right
            for r in calib_rows:
                s = dirty.at[r, src].strip()
                n_r[s] += 1
                ok_r[s] += dirty.at[r, rhs].strip() == clean.at[r, rhs].strip()
            w2 = _truth_discovery(dirty, lhs, rhs, src, {s: (ok_r[s] + 1) / (n_r[s] + 2) for s in n_r})
            variants += [("weighted", (src, w1)), ("calib", (src, w2))]
        best = None
        for variant, truth in variants:
            pts = []  # (vote, correct) per answered calib cell
            for r, k in calib:
                vote = _fd_vote(dirty, r, lhs, rhs, truth, errs) if k == rhs else None
                if vote is None or vote[0] == dirty.at[r, k].strip():
                    continue
                used.append((r, k))
                pts.append((vote, vote[0] == clean.at[r, k].strip()))
            ok, answered = sum(y for _, y in pts), len(pts)
            score = (ok + 1) / (answered + 2)
            if best is None or score > best[0]:
                best = (score, answered, ok, pts, variant, truth)
        score, answered, ok, pts, variant, truth = best
        keep = answered >= 2 and score >= 0.5 and (rhs not in kept or score > kept[rhs][1])
        auc = {c: roc_auc_score([y for _, y in pts], [_fd_conf(v, c) for v, _ in pts]) if 0 < ok < answered else 0.5
               for c in ("share", "support")}
        conf = "support" if auc["support"] > auc["share"] else "share"
        print(f"[{name}] FD {list(lhs)} -> {rhs}: {ok}/{answered} score={score:.2f} auc={auc}"
              f"{' kept ' + conf + ' ' + variant if keep else ''}")
        if keep:
            kept[rhs] = (lhs, score, answered, ok, conf, variant, truth)
    assert not set(used) & set(test), "leakage: FD validation used a test-split cell"
    return kept


# ------------------------------------------------------------ tier col -----
# One LLM call per systematic error group (not per cell): ask the model for a
# small Python `fix()` function and replay it locally on every cell of the
# group. Group-level caching lives in its own file so it survives resumption
# independently of the per-cell cache.

_SAFE_BUILTIN_NAMES = ("len", "int", "float", "str", "round", "range", "min", "max", "abs")
_SAFE_BUILTINS = {n: getattr(_builtins, n) for n in _SAFE_BUILTIN_NAMES}


def _safe_import(name, *args, **kwargs):
    # ponytail: real models habitually write a redundant `import re` even
    # though `re` is already a global - allow only that, nothing else.
    if name == "re":
        return re
    raise ImportError(f"import of {name!r} is not allowed")


_SAFE_BUILTINS["__import__"] = _safe_import


def _exec_fix(program_src, val):
    # ponytail: exec of local-LLM code with restricted builtins, no process sandbox;
    # use a subprocess sandbox if the model ever comes from an untrusted source
    ns = {"re": re, "__builtins__": _SAFE_BUILTINS}
    try:
        exec(program_src, ns)
        return ns["fix"](val)
    except Exception:
        return None


def _parse_python_block(text):
    if not text:
        return None
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return m.group(1) if m else text


def _column_prompt_inputs(dirty_df, clean_df, col, group_cells, calib_set, error_cell_set):
    # all labeled cells of the group; capped only for prompt length
    labeled = [(r, c) for r, c in group_cells if (r, c) in calib_set][:15]
    assert all((r, c) in calib_set for r, c in labeled), "leakage: example cell not in calib split"
    examples = [(dirty_df.at[r, c], clean_df.at[r, c]) for r, c in labeled]
    labeled_set = set(labeled)
    other_dirty = [dirty_df.at[r, c] for r, c in group_cells if (r, c) not in labeled_set][:10]
    non_error_vals = []
    for i in range(len(dirty_df)):
        if (i, col) in error_cell_set:
            continue
        v = dirty_df.at[i, col].strip()
        if v and v not in non_error_vals:
            non_error_vals.append(v)
        if len(non_error_vals) >= 10:
            break
    return examples, other_dirty, non_error_vals


def _column_examples(dirty_df, clean_df, col, error_groups, calib_set):
    """Labeled (dirty, clean) pairs of ALL systematic groups (>= 10 error cells, i.e. those the column tier
    serves) of `col`; calib split only."""
    cells = [rc for (c, _), g in error_groups.items() if c == col and len(g) >= 10 for rc in g if rc in calib_set]
    assert all(rc in calib_set for rc in cells), "leakage: column example cell not in calib split"
    return [(dirty_df.at[r, c], clean_df.at[r, c]) for r, c in cells]


# Column-level character-substitution program (mojibake): (dirty substring -> clean substring) rules, each at
# most 3 chars long, read off difflib replace/delete opcodes of the column's labeled pairs.
# ponytail: off by default - over 6 configs x 4 datasets it never changed `ours` (tau 0.8: the LOO Laplace score of a
# column with a handful of labeled pairs stays < 0.8), moved ours_auto by -0.003..+0.009 (the auto tau is picked on
# calib cells the rules were fit on) and cost hospital cascade 0.013 at 50 tuples. CHARSUB=1 turns it on.
CHARSUB = os.environ.get("CHARSUB", "0") == "1"

def _pair_rules(d, c):
    ops = difflib.SequenceMatcher(None, d.strip(), c.strip(), autojunk=False).get_opcodes()
    return {(d.strip()[i1:i2], c.strip()[j1:j2]) for tag, i1, i2, j1, j2 in ops
            if tag in ("replace", "delete") and i2 - i1 <= 3 and j2 - j1 <= 3}


def _merge_rules(rule_sets):
    """{src: dst}: srcs seen in >= 2 examples or holding a non-ASCII / control char; the most common dst wins."""
    seen = defaultdict(Counter)
    n_ex = Counter()
    for rules in rule_sets:
        for src in {s for s, _ in rules}:
            n_ex[src] += 1
        for src, dst in rules:
            seen[src][dst] += 1
    odd = lambda s: any(ord(ch) > 127 or not ch.isprintable() for ch in s)
    return {s: cnt.most_common(1)[0][0] for s, cnt in seen.items() if n_ex[s] >= 2 or odd(s)}


def _apply_rules(rules, val):
    for src in sorted(rules, key=len, reverse=True):
        val = val.replace(src, rules[src])
    return val


def _charsub_program(pairs):
    """(rules, LOO Laplace score) over the column's labeled (dirty, clean) pairs: each example is predicted
    with the rules learned from the OTHER examples."""
    sets = [_pair_rules(d, c) for d, c in pairs]
    ok = sum(_apply_rules(_merge_rules(sets[:i] + sets[i + 1:]), d.strip()) == c.strip() for i, (d, c) in enumerate(pairs))
    return _merge_rules(sets), (ok + 1) / (len(pairs) + 2)


def _column_prompt(col, examples, other_dirty, non_error_vals, feedback=None):
    ex_lines = "\n".join(f"  {d!r} -> {c!r}" for d, c in examples) or "  (none)"
    fb = ""
    if feedback:
        prog, fails = feedback
        fb_lines = "\n".join(f"  {d!r} -> {o!r} (expected {c!r})" for d, o, c in fails)
        fb = (f"A previous attempt failed on some labeled examples:\n```python\n{prog}\n```\n"
              f"Its outputs on the failing examples:\n{fb_lines}\n"
              "Fix these failures; the new function must reproduce ALL labeled examples.\n\n")
    return fb + (
        f"You are writing a Python data-cleaning function for column '{col}'.\n"
        f"Labeled examples (dirty -> clean):\n{ex_lines}\n\n"
        f"Other dirty values in this group: {other_dirty}\n\n"
        f"Non-error values seen in this column: {non_error_vals}\n\n"
        "Write a single Python function `def fix(s: str) -> str:` that maps a dirty "
        "value to its cleaned form, using only the `re` module. Reply with ONLY the "
        "function in a ```python code block, no prose."
    )


def _column_llm_sample(prompt, k=5, temperature=0.7):
    """The LLM call for a column-group program: k sampled completions -> (texts, cost)."""
    model = os.environ.get("LLM_MODEL", "Qwen/Qwen3-8B")
    messages = [{"role": "user", "content": prompt}]
    client = _get_client()
    cost = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
    texts = []

    def _record_usage(resp):
        usage = getattr(resp, "usage", None)
        if usage:
            cost["prompt_tokens"] += usage.prompt_tokens or 0
            cost["completion_tokens"] += usage.completion_tokens or 0

    # Qwen3 "thinking" mode wraps replies in <think>...</think> before the
    # actual content; we want the raw fix()-function text, so switch it off.
    no_think = {"chat_template_kwargs": {"enable_thinking": False}}

    t0 = time.time()
    try:
        resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, n=k,
                                               extra_body=no_think)
        cost["calls"] += 1
        _record_usage(resp)
        texts = [choice.message.content for choice in resp.choices]
    except Exception:
        for _ in range(k):
            try:
                resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, n=1,
                                                       extra_body=no_think)
                cost["calls"] += 1
                _record_usage(resp)
                texts.append(resp.choices[0].message.content)
            except Exception:
                cost["calls"] += 1
    cost["seconds"] = time.time() - t0
    return texts, cost


def _column_cache_path(name):
    sub = CACHE_DIR / "stub" if os.environ.get("LLM_STUB") == "1" else CACHE_DIR
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"{name}_column.jsonl"


def _load_column_groups(name):
    path = _column_cache_path(name)
    groups = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    groups[(obj["col"], obj["pattern"])] = obj
    return groups


def _example_failures(program, examples):
    """[(dirty, program output, expected clean)] for every labeled example the program misses."""
    fails = []
    for d, c in examples:
        out = _exec_fix(program, d.strip())
        if out is None or str(out).strip() != c.strip():
            fails.append((d, out, c))
    return fails


def _norm(o):
    return None if o is None else str(o).strip()


def _group_score(outs, cleans, first=False):
    """Leave-one-out Laplace estimate of P(group's repair is right) - no extra LLM calls.
    outs[p][i] = normalized output of candidate program p on labeled example i. For each i, the
    programs that pass all OTHER examples vote on i (majority, or the first/simplest if `first`);
    score = (ok + 1) / (n + 2), 0.5 with no examples."""
    n = len(cleans)
    miss = [[j for j in range(n) if o[j] != cleans[j]] for o in outs]
    ok = 0
    for i in range(n):
        votes = [o[i] for o, m in zip(outs, miss) if m in ([], [i]) and o[i] is not None]
        if votes and (votes[0] if first else Counter(votes).most_common(1)[0][0]) == cleans[i]:
            ok += 1
    return (ok + 1) / (n + 2)


# A tiny FlashFill-style DSL tried before any LLM sampling: split a value into tokens (a whole number such
# as "12.0" is one token, letter runs are tokens, everything between them is a remembered delimiter), select
# 1..4 of them (permutations allowed) or an all-but-last / all-but-first slice, format each, join with a
# delimiter taken from the clean examples. Programs are plain dicts {"idx", "fmt", "delim"} so they serialize
# into the group record.

_DSL_FORMATS = ("asis", "int", "pad2", "last2", "num", "round1", "round2", "round3", "round4", "mon")
_DSL_CAP = 20000
_DSL_TOK = re.compile(r"\d+(?:\.\d+)?|[^\W\d_]+")  # numbers | letter runs (unicode letters, unlike [A-Za-z])
_MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def _dsl_tokens(s):
    """(tokens, delimiters): delimiters[i] is the original text between token i and i+1."""
    ms = list(_DSL_TOK.finditer(s))
    return [m.group() for m in ms], [s[a.end():b.start()] for a, b in zip(ms, ms[1:])]


@functools.lru_cache(maxsize=None)
def _dsl_fmt(tok, f):
    if f == "asis":
        return tok
    if not tok[0].isdigit():
        return None
    if f == "mon":
        return _MONTHS[int(tok) - 1] if tok.isdigit() and 1 <= int(tok) <= 12 else None
    if f in ("int", "pad2", "last2"):
        return {"int": str(int(tok)), "pad2": tok.zfill(2), "last2": tok[-2:].zfill(2)}[f] if tok.isdigit() else None
    # num / roundK: canonical float text, trailing zeros (and a bare ".0") dropped
    s = f"{float(tok):.{f[5]}f}" if f.startswith("round") else format(Decimal(tok).normalize(), "f")
    return s.rstrip("0").rstrip(".") if "." in s else s


def _dsl_run(prog, toks, delims):
    idx = prog["idx"]
    if isinstance(idx, str):  # "all-but-last" / "all-but-first": consecutive tokens, original delimiters
        if len(toks) < 2:
            return None
        a, b = (0, len(toks) - 1) if idx == "all-but-last" else (1, len(toks))
        return toks[a] + "".join(delims[i - 1] + toks[i] for i in range(a + 1, b))
    try:
        parts = [_dsl_fmt(toks[i], f) for i, f in zip(idx, prog["fmt"])]
    except IndexError:
        return None
    return None if None in parts else prog["delim"].join(parts)


def _dsl_apply(prog, val):
    return _dsl_run(prog, *_dsl_tokens(val))


def _dsl_search(examples, plaus=None):
    """Enumerate DSL programs (<= _DSL_CAP), simplest first. Returns (simplest program consistent with
    ALL labeled examples, n_enumerated, n_consistent, LOO group score) or None: needs >= 2 examples and
    at least one consistent program. Zero LLM calls. `plaus` = (patterns, dirty values): among the consistent
    programs prefer the simplest one whose outputs on those values all have a `_generalize` pattern in
    `patterns` (the column's non-error values), so "0.08199999999999999" is not left as-is when the labeled
    examples cannot tell as-is from round3."""
    if len(examples) < 2:
        return None
    tds = [_dsl_tokens(d.strip()) for d, _ in examples]
    cleans = [c.strip() for _, c in examples]
    n_tok = min(len(t) for t, _ in tds)
    # a format that fails on some example's token can never be part of a consistent program
    ok = [[f for f in _DSL_FORMATS if all(_dsl_fmt(t[i], f) is not None for t, _ in tds)] for i in range(n_tok)]
    delims = sorted({d for c in cleans for d in _dsl_tokens(c)[1]})
    progs = itertools.chain(
        ({"idx": [i], "fmt": [f], "delim": ""} for i in range(n_tok) for f in ok[i]),
        ({"idx": s, "fmt": [], "delim": ""} for s in ("all-but-last", "all-but-first")),
        ({"idx": list(idx), "fmt": list(fmt), "delim": delim}
         for k in range(2, min(n_tok, 4) + 1)
         for idx in itertools.permutations(range(n_tok), k)
         for fmt in itertools.product(*[ok[i] for i in idx])
         for delim in delims))
    progs = list(itertools.islice(progs, _DSL_CAP))
    # simplest first: fewest tokens (a slice counts as one), then fewest non-as-is formats (sort is stable)
    progs.sort(key=lambda g: (len(g["idx"]) if isinstance(g["idx"], list) else 1, sum(f != "asis" for f in g["fmt"])))
    outs = [[_norm(_dsl_run(g, *t)) for t in tds] for g in progs]
    consistent = [g for g, o in zip(progs, outs) if o == cleans]
    if not consistent:
        return None
    if plaus:
        seen = lambda g: all((o := _norm(_dsl_apply(g, v))) is not None and _generalize(o) in plaus[0] for v in plaus[1])
        consistent = [g for g in consistent if seen(g)] or consistent
    # ponytail: LOO uses the simplest program passing the other examples (what we deploy), not the
    # majority over all of them; the cap can cut off complex programs on wide values (many tokens).
    return consistent[0], len(progs), len(consistent), _group_score(outs, cleans, first=True)


def _get_or_make_column_group(cache, f, col, pattern, examples, other_dirty, non_error_vals, col_examples=(), reuse=None,
                              plaus=None):
    """Generate (once) and cache a group's fix programs: the DSL search first (source "dsl", no LLM
    call), LLM sampling only when it finds nothing. `f` is the open group cache file.
    A group with < 2 labeled examples of its own searches the DSL on `col_examples`, the labeled examples
    of ALL the column's groups (source "dsl_column", scored by LOO over that union). `reuse` = earlier
    group records: an "llm" one is reused instead of sampling again.
    Record: `programs` = those passing ALL labeled examples, `all_programs` = every sampled
    program (both rounds), `group_score` = leave-one-out Laplace score over all_programs."""
    key = (col, pattern)
    if key in cache:
        return cache[key]
    t_group = time.time()
    source = "dsl"
    found = _dsl_search(examples, plaus)
    if not found and len(examples) < 2:
        source, found = "dsl_column", _dsl_search(col_examples, plaus)
    if found:
        prog, n_enum, n_ok, score = found
        rec = {"col": col, "pattern": pattern, "source": source, "programs": [prog], "all_programs": [],
               "group_score": score, "n_sampled": n_enum, "n_verified": n_ok, "retry": False,
               "examples": [list(e) for e in examples],
               "cost": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}}
    elif reuse and reuse.get(key, {}).get("source") == "llm":
        rec = reuse[key]
    elif os.environ.get("LLM_STUB") == "1":
        rec = {"col": col, "pattern": pattern, "source": "llm", "programs": [], "all_programs": [],
               "group_score": 0.5, "n_sampled": 0, "n_verified": 0,
               "retry": False, "examples": [list(e) for e in examples],
               "cost": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}}
    else:
        texts, cost = _column_llm_sample(_column_prompt(col, examples, other_dirty, non_error_vals))
        programs = [p for p in map(_parse_python_block, texts) if p]
        all_programs, n_sampled, retry = list(programs), len(texts), False
        # Keep only programs that reproduce ALL labeled examples (0 examples: can't verify, keep all).
        fails = {p: _example_failures(p, examples) for p in programs}
        kept = [p for p in programs if not fails[p]]
        if examples and not kept:
            # One retry: show the best program's (fewest failures) wrong outputs, then verify again.
            retry = True
            best = min(programs, key=lambda p: len(fails[p]), default="")
            fb = (best, fails.get(best) or [(d, None, c) for d, c in examples])
            texts, cost2 = _column_llm_sample(_column_prompt(col, examples, other_dirty, non_error_vals, fb))
            for k_ in cost:
                cost[k_] += cost2[k_]
            n_sampled += len(texts)
            programs2 = [p for p in map(_parse_python_block, texts) if p]
            all_programs += programs2
            kept = [p for p in programs2 if not _example_failures(p, examples)]
        outs = [[_norm(_exec_fix(p, d.strip())) for d, _ in examples] for p in all_programs]
        rec = {"col": col, "pattern": pattern, "source": "llm", "programs": kept, "all_programs": all_programs,
               "group_score": _group_score(outs, [c.strip() for _, c in examples]),
               "n_sampled": n_sampled, "n_verified": len(kept), "retry": retry,
               "examples": [list(e) for e in examples], "cost": cost}
    if not (rec is reuse.get(key) if reuse else False):  # a reused LLM record keeps the wall time it was made with
        rec["cost"]["wall_seconds"] = time.time() - t_group  # DSL search or LLM sampling+verification: paid once per group
    cache[key] = rec
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    f.flush()
    return rec


def tier_column(dirty_df, row, col, group_rec, examples=None):
    if os.environ.get("LLM_STUB") == "1":
        return None, 0.0, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
    val = dirty_df.at[row, col].strip()
    programs = group_rec.get("programs") or []
    cost = group_rec.get("cost", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0})
    run = _dsl_apply if group_rec.get("source", "").startswith("dsl") else _exec_fix
    outputs = [_norm(run(p, val)) for p in programs if p]
    non_null = [o for o in outputs if o is not None]
    if not non_null:
        return None, 0.0, cost
    majority_val, majority_count = Counter(non_null).most_common(1)[0]
    # group-level honest P(correct) x agreement among the kept (verified) programs
    raw_conf = group_rec["group_score"] * majority_count / len(programs)
    return majority_val, raw_conf, cost


# -------------------------------------------------------------- tier 3 -----

_LLM_CLIENT = None


def _get_client():
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        from openai import OpenAI
        base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
        _LLM_CLIENT = OpenAI(base_url=base_url, api_key="EMPTY")
    return _LLM_CLIENT


def _parse_llm_json(text):
    if not text:
        return None, 0.0
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        obj = json.loads(text[start:end])
        v = obj.get("value")
        if v is None:
            return None, 0.0
        return str(v).strip(), float(obj.get("confidence", 0.0))
    except Exception:
        return None, 0.0


def tier_llm(dirty_df, row, col, k=3, temperature=0.7):
    if os.environ.get("LLM_STUB") == "1":
        return None, 0.0, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    t_prep = time.time()
    profile = _get_profile(dirty_df)
    top_values = [v for v, _ in profile["freq"][col].most_common(10)]
    deps, sims = _stats_candidates(dirty_df, row, col)
    best_dep = {}
    for v, _, n in deps:
        best_dep[v] = max(best_dep.get(v, 0), n)
    dep_c = [(v, n) for v, n in sorted(best_dep.items(), key=lambda x: -x[1])[:5]]
    sim_c = [v for v, _ in sorted(sims, key=lambda x: -x[1])[:5]]
    row_json = json.dumps({c: dirty_df.at[row, c] for c in dirty_df.columns}, ensure_ascii=False)
    prompt = (
        f"You are cleaning tabular data. Column: {col}\n"
        f"Full row (JSON): {row_json}\n"
        f"Frequent values seen in this column: {top_values}\n"
        f"Candidate values (from this table only):\n"
        f"- implied by other columns, as [value, supporting rows]: {dep_c}\n"
        f"- nearest frequent values: {sim_c}\n"
        f"Pick a candidate or answer another value if none fits. "
        f"The value in column '{col}' may be erroneous. Reply with ONLY JSON, "
        f'no prose: {{"value": "<corrected value>", "confidence": <0-1 float>}}'
    )
    val, conf, cost = _llm_vote(prompt, k, temperature)
    cost["prep_seconds"] = time.time() - t_prep - cost["seconds"]  # candidate/prompt building, apart from the LLM request time
    return val, conf, cost


def _llm_vote(prompt, k, temperature, drop=None):
    """k sampled {"value", "confidence"} answers -> (majority value, raw_conf, cost); raw_conf = (agreement +
    mean verbalized confidence of the agreeing samples) / 2. An answer equal to `drop` counts as no answer."""
    t0 = time.time()
    model = os.environ.get("LLM_MODEL", "Qwen/Qwen3-8B")
    messages = [{"role": "user", "content": prompt}]
    client = _get_client()
    cost = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
    samples = []

    def _record_usage(resp):
        usage = getattr(resp, "usage", None)
        if usage:
            cost["prompt_tokens"] += usage.prompt_tokens or 0
            cost["completion_tokens"] += usage.completion_tokens or 0

    # Qwen3 "thinking" mode wraps replies in <think>...</think> before the
    # actual content; we need bare JSON, so switch it off.
    no_think = {"chat_template_kwargs": {"enable_thinking": False}}

    try:
        resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, n=k,
                                               extra_body=no_think)
        cost["calls"] += 1
        _record_usage(resp)
        for choice in resp.choices:
            v, c = _parse_llm_json(choice.message.content)
            if v is not None and v != drop:
                samples.append((v, c))
    except Exception:
        for _ in range(k):
            try:
                resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, n=1,
                                                       extra_body=no_think)
                cost["calls"] += 1
                _record_usage(resp)
                v, c = _parse_llm_json(resp.choices[0].message.content)
                if v is not None and v != drop:
                    samples.append((v, c))
            except Exception:
                cost["calls"] += 1

    cost["seconds"] = time.time() - t0
    if not samples:
        return None, 0.0, cost
    votes = Counter(v for v, _ in samples)
    majority_val, majority_count = votes.most_common(1)[0]
    agreement = majority_count / k
    agreeing_confs = [c for v, c in samples if v == majority_val]
    mean_conf = sum(agreeing_confs) / len(agreeing_confs) if agreeing_confs else 0.0
    raw_conf = (agreement + mean_conf) / 2
    return majority_val, raw_conf, cost


def _format_examples(dirty_df, error_cell_set, col, n=10):
    """The n most frequent non-error values of `col` (all cells if that leaves none): FORMAT examples for llm_sem."""
    vals = [v.strip() for i, v in enumerate(dirty_df[col]) if (i, col) not in error_cell_set]
    return [v for v, _ in Counter(v for v in vals if v).most_common(n)] or [v for v, _ in Counter(dirty_df[col].str.strip()).most_common(n)]


def tier_llm_sem(dirty_df, row, col, examples, k=3, temperature=0.7):
    """Semantic LLM operator: the cell is KNOWN wrong (oracle detection), so ask for the real-world-correct value
    without table candidates (they anchor the model on a consistently corrupted value); `examples` fix the format."""
    if os.environ.get("LLM_STUB") == "1":
        return None, 0.0, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
    val = dirty_df.at[row, col].strip()
    row_json = json.dumps({c: dirty_df.at[row, c] for c in dirty_df.columns}, ensure_ascii=False)
    prompt = (
        f"You are cleaning tabular data. Row (JSON): {row_json}\n"
        f"The current value '{val}' of column '{col}' is known to be wrong. Using real-world knowledge and the "
        f"other columns of this row, give the correct value of '{col}'.\n"
        f"Other values of this column: {examples}. Answer in exactly this format. Do not repeat '{val}'.\n"
        f'Reply with ONLY JSON, no prose: {{"value": "<correct value>", "confidence": <0-1 float>}}'
    )
    return _llm_vote(prompt, k, temperature, drop=val)


# --------------------------------------------------------- precompute ------

def _cache_path(name):
    sub = CACHE_DIR / "stub" if os.environ.get("LLM_STUB") == "1" else CACHE_DIR
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"{name}.jsonl"


def _column_fields(dirty, clean, row, col, et, error_groups, calib_set, error_cell_set, col_cache, cf, reuse=None,
                   charsub=None):
    """(group_id, value, raw_conf, cost) of the column tier for one cell. A systematic cell first tries its group
    program; a cell it does not cover (any error type) falls back to the column's character substitutions when
    their LOO score over the column's labeled pairs is >= 0.5 (`charsub` = per-column memo dict)."""
    gid, val, conf, cost = None, None, 0.0, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
    if et == "systematic":
        pattern = _generalize(dirty.at[row, col].strip())
        group_cells = error_groups[(col, pattern)]
        examples, other_dirty, non_error_vals = _column_prompt_inputs(
            dirty, clean, col, group_cells, calib_set, error_cell_set)
        col_examples = [] if (col, pattern) in col_cache else _column_examples(dirty, clean, col, error_groups, calib_set)
        plaus = None if (col, pattern) in col_cache else (
            {_generalize(dirty.at[i, col].strip()) for i in range(len(dirty)) if (i, col) not in error_cell_set},
            list(dict.fromkeys(dirty.at[r, c].strip() for r, c in group_cells))[:50])
        group_rec = _get_or_make_column_group(col_cache, cf, col, pattern, examples, other_dirty, non_error_vals,
                                              col_examples, reuse, plaus)
        gid = [col, pattern]
        t_cell = time.time()
        val, conf, cost = tier_column(dirty, row, col, group_rec, examples)
        cost = {**cost, "cell_seconds": time.time() - t_cell}  # this cell's own apply time; the group's is cost["wall_seconds"]
    if val is None and charsub is not None:
        if col not in charsub:
            pairs = [(dirty.at[r, c], clean.at[r, c]) for r, c in sorted(calib_set) if c == col]
            charsub[col] = _charsub_program(pairs) if pairs else ({}, 0.0)
        rules, score = charsub[col]
        if score >= 0.5:
            val, conf = _apply_rules(rules, dirty.at[row, col].strip()), score
    return gid, val, conf, cost


def refresh_column(name):
    """Recompute ONLY the column tier (+ group id) of the existing cell cache in place;
    llm/rules/stats fields are never touched. The group file is rebuilt: DSL programs are re-searched, earlier
    LLM group records are reused (no new LLM call unless a group has no record)."""
    dirty, clean, error_cells = load(name)
    error_cell_set = set(error_cells)
    error_groups = _error_groups(dirty, error_cells)
    calib_set = set(calib_test_split(error_cells)[0])
    path = _cache_path(name)
    tmp = path.with_suffix(".jsonl.tmp")
    old, col_cache, charsub = _load_column_groups(name), {}, {} if CHARSUB else None
    with open(path) as src, open(tmp, "w") as out, open(_column_cache_path(name), "w") as cf:
        for line in src:
            if not line.strip():
                continue
            rec = json.loads(line)
            gid, val, conf, cost = _column_fields(dirty, clean, rec["row"], rec["col"], rec["error_type"],
                                                  error_groups, calib_set, error_cell_set, col_cache, cf, old, charsub)
            rec["group"] = gid
            rec["tiers"]["column"] = {"value": val, "raw_conf": conf, "cost": cost}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(f"[{name}] refreshed column tier -> {path}")


def refresh_cheap(name):
    """Recompute ONLY rules/fd/stats/error_type/group of the existing cell cache in place; the llm and
    column tier fields are copied through untouched (byte-identical)."""
    dirty, clean, error_cells = load(name)
    error_cell_set = set(error_cells)
    error_groups = _error_groups(dirty, error_cells)
    fds = kept_fds(name, dirty, clean, error_cells)
    labeled = _labeled_clean(clean, error_cells)
    path = _cache_path(name)
    tmp = path.with_suffix(".jsonl.tmp")
    with open(path) as src, open(tmp, "w") as out:
        for line in src:
            if not line.strip():
                continue
            rec = json.loads(line)
            row, col = rec["row"], rec["col"]
            rec["error_type"] = error_type(dirty, row, col, error_groups)
            rec["group"] = [col, _generalize(dirty.at[row, col].strip())] if rec["error_type"] == "systematic" else None
            for tier, fn in (("rules", lambda d, r, c: tier_rules(d, r, c, error_cell_set, labeled)), ("fd", lambda d, r, c: tier_fd(d, r, c, fds, error_cell_set)),
                             ("stats", lambda d, r, c: tier_stats(d, r, c, fds, error_cell_set))):
                v, c, cost = fn(dirty, row, col)
                rec["tiers"][tier] = {"value": v, "raw_conf": c, "cost": cost}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(f"[{name}] refreshed rules/fd/stats/error_type -> {path}")


LLM_SEM_DIR = SHARED_CACHE  # one llm_sem answer per (dataset, row, col), shared by all split configs (the prompt is split-free)


_DIRECT_FREE = ("rules", "fd", "stats", "column")  # ours_sem_direct: llm_sem is directly reachable iff all of these have no non-echo candidate


def refresh_llm_sem(name, ask=True, direct=False):
    """Compute ONLY the llm_sem tier of the existing cell cache in place. A cell is eligible iff no other tier has a
    non-echo candidate (load_cache's rule): only those are asked (8 threads), each (row, col) once ever, in
    cache/<name>_llm_sem.jsonl; other cells get an empty tier. Every other field is left untouched.
    ask=False: no LLM call at all, eligible cells without a stored answer stay empty (counted in the printout).
    Also sets, per cell, `llm_sem_direct_eligible` (ours_sem_direct: rules/fd/stats/column have no non-echo candidate, the llm may
    have one; a superset of `llm_sem_eligible`) and `llm_sem_direct` = the stored answer of such a cell (or absent). direct=True also
    asks the direct-eligible cells that are not old-eligible and lack an answer (same file, once per (row, col)). `tiers["llm_sem"]`
    and `llm_sem_eligible` keep the old definition, so every other method replays exactly as before."""
    dirty, _, error_cells = load(name)
    error_cell_set, path = set(error_cells), LLM_SEM_DIR / f"{name}_llm_sem.jsonl"
    done = {}
    if path.exists():
        done = {(o["row"], o["col"]): o for o in map(json.loads, filter(str.strip, path.open()))}
    cells_path = _cache_path(name)
    recs = [json.loads(line) for line in open(cells_path) if line.strip()]
    for rec in recs:
        rec["llm_sem_eligible"] = all(t["value"] is None or t["value"].strip() == rec["dirty"].strip()
                                      for k, t in rec["tiers"].items() if k != "llm_sem")
        rec["llm_sem_direct_eligible"] = all(t is None or t["value"] is None or t["value"].strip() == rec["dirty"].strip()
                                             for t in map(rec["tiers"].get, _DIRECT_FREE))
    todo = [r for r in recs if r["llm_sem_eligible"] and (r["row"], r["col"]) not in done]
    todo_direct = [r for r in recs if r["llm_sem_direct_eligible"] and not r["llm_sem_eligible"] and (r["row"], r["col"]) not in done]
    n_missing = sum(r["llm_sem_direct_eligible"] and (r["row"], r["col"]) not in done for r in recs)
    print(f"[{name}] llm_sem direct-eligible: {sum(r['llm_sem_direct_eligible'] for r in recs)} of {len(recs)}, "
          f"{n_missing} lack a stored answer ({len(todo_direct)} of them not old-eligible)")
    todo = todo + (todo_direct if direct else [])
    examples = {}
    with open(path, "a") as f, concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {}
        for r in todo if ask else []:
            ex = examples.setdefault(r["col"], _format_examples(dirty, error_cell_set, r["col"]))
            futures[pool.submit(tier_llm_sem, dirty, r["row"], r["col"], ex)] = r
        for fut in concurrent.futures.as_completed(futures):
            r = futures[fut]
            v, c, cost = fut.result()
            done[(r["row"], r["col"])] = o = {"row": r["row"], "col": r["col"], "value": v, "raw_conf": c, "cost": cost}
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
            f.flush()
    empty = {"value": None, "raw_conf": 0.0, "cost": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}}
    tmp = cells_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as out:
        for rec in recs:
            o = done.get((rec["row"], rec["col"])) if rec["llm_sem_eligible"] else None
            rec["tiers"]["llm_sem"] = {k: o[k] for k in ("value", "raw_conf", "cost")} if o else empty
            od = done.get((rec["row"], rec["col"])) if rec["llm_sem_direct_eligible"] else None
            rec.pop("llm_sem_direct", None)
            if od:
                rec["llm_sem_direct"] = {k: od[k] for k in ("value", "raw_conf", "cost")}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(cells_path)
    print(f"[{name}] llm_sem: {sum(r['llm_sem_eligible'] for r in recs)} eligible of {len(recs)}, {len(todo)} {'newly asked' if ask else 'unanswered, not asked'} -> {cells_path}")
    return {"direct_eligible": sum(r["llm_sem_direct_eligible"] for r in recs), "missing": n_missing, "missing_new": len(todo_direct)}


def precompute(name, limit=None):
    dirty, clean, error_cells = load(name)
    if limit:
        error_cells = error_cells[:limit]
    error_cell_set = set(error_cells)
    error_groups = _error_groups(dirty, error_cells)
    # Same seed/split fn eval() will use later (len(error_cells) determines it,
    # not content) - so a group's labeled examples never leak the test split.
    calib_pairs, _test_pairs = calib_test_split(error_cells)
    calib_set = set(calib_pairs)
    fds = kept_fds(name, dirty, clean, error_cells)
    labeled = _labeled_clean(clean, error_cells)

    path = _cache_path(name)
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    done.add((obj["row"], obj["col"]))

    col_cache, charsub = _load_column_groups(name), {} if CHARSUB else None
    col_path = _column_cache_path(name)
    n_new = 0
    with open(path, "a") as f, open(col_path, "a") as cf:
        # Phase 1 (sequential): rules/fd/stats are cheap, and column-group setup
        # touches shared state (col_cache dict, cf file) - build every cell's
        # base record and only leave the per-cell tier_llm call for phase 2.
        pending = {}
        t_phase1 = time.time()
        for row, col in error_cells:
            if (row, col) in done:
                continue
            et = error_type(dirty, row, col, error_groups)
            r_val, r_conf, r_cost = tier_rules(dirty, row, col, error_cell_set, labeled)
            f_val, f_conf, f_cost = tier_fd(dirty, row, col, fds, error_cell_set)
            s_val, s_conf, s_cost = tier_stats(dirty, row, col, fds, error_cell_set)

            group_id, col_val, col_conf, col_cost = _column_fields(
                dirty, clean, row, col, et, error_groups, calib_set, error_cell_set, col_cache, cf, charsub=charsub)

            pending[(row, col)] = {
                "row": row, "col": col,
                "dirty": dirty.at[row, col], "clean": clean.at[row, col],
                **({"semantic": (row, col) in _SEM_CELLS} if _SEM_CELLS else {}),
                "error_type": et,
                "group": group_id,
                "tiers": {
                    "rules": {"value": r_val, "raw_conf": r_conf, "cost": r_cost},
                    "fd": {"value": f_val, "raw_conf": f_conf, "cost": f_cost},
                    "stats": {"value": s_val, "raw_conf": s_conf, "cost": s_cost},
                    "column": {"value": col_val, "raw_conf": col_conf, "cost": col_cost},
                },
            }

        # Phase 2 (concurrent): fire tier_llm calls off a small thread pool so
        # vLLM sees several requests at once and can batch them on the GPU.
        # ponytail: max_workers=8 is a ceiling sized for one vLLM replica on
        # one GPU; raise it (or shard across a pool per replica) if serving
        # scales out to multiple GPUs/instances.
        t_phase2 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(tier_llm, dirty, row, col): (row, col) for row, col in pending}
            for future in concurrent.futures.as_completed(futures):
                row, col = futures[future]
                l_val, l_conf, l_cost = future.result()
                rec = pending[(row, col)]
                rec["tiers"]["llm"] = {"value": l_val, "raw_conf": l_conf, "cost": l_cost}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_new += 1
    if n_new:  # wall-clock of the two phases (rules/fd/stats/column, sequential; per-cell llm, 8 threads)
        (path.parent / f"{name}_timing.json").write_text(json.dumps(
            {"n_cells": n_new, "phase1_seconds": t_phase2 - t_phase1, "phase2_llm_wall_seconds": time.time() - t_phase2}))
    print(f"[{name}] precompute done: {len(error_cells)} error cells, {n_new} newly computed -> {path}")
    return path


def load_cache(name):
    path = _cache_path(name)
    cells = []
    with open(path) as f:
        for line in f:
            if line.strip():
                cells.append(json.loads(line))
    if SAMPLER == "coverage":
        unresolved = _unresolved_set(name)
        for cell in cells:
            cell["unresolved"] = (cell["row"], cell["col"]) in unresolved
    gsize = Counter(tuple(c["group"]) for c in cells if c.get("group"))  # whole-dataset group sizes: amortizes a column program's calls
    for cell in cells:
        if cell.get("group"):
            cell["group_size"] = gsize[tuple(cell["group"])]
        dirty = cell["dirty"].strip()
        for t in cell["tiers"].values():
            # ponytail: detection here is oracle (every cell in the cache is a
            # known-dirty error cell), so a "candidate" that just echoes the
            # dirty value back can never be a correct repair. Left in, it's
            # accepted as a (wrong) fix and its raw_conf poisons calibration
            # and blocks escalation to later tiers - drop it at the one place
            # every reader (calibration, routing) goes through.
            if t["value"] is not None and t["value"].strip() == dirty:
                t["value"] = None
                t["raw_conf"] = 0.0
        t = cell.get("llm_sem_direct")
        if t and t["value"] is not None and t["value"].strip() == dirty:
            t["value"], t["raw_conf"] = None, 0.0
    return cells


# --------------------------------------------------------- calibration -----

N_LABEL_TUPLES = int(os.environ.get("LABEL_TUPLES", 20))
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", 0))
# RQ3 shift: LABEL_EXCLUDE_SEM=1 samples labeled rows only from rows WITHOUT an injected semantic cell (labeled data
# "mostly syntactic"); the default mixes them in. No effect on datasets without injected cells.
LABEL_EXCLUDE_SEM = os.environ.get("LABEL_EXCLUDE_SEM", "0") == "1"


SAMPLER = os.environ.get("SAMPLER", "random")  # "random" (default) | "coverage": see _coverage_rows
Q_UNRESOLVED = 5  # coverage sampler: target number of labeled unresolved cells
UNRESOLVED_DIR = SHARED_CACHE  # one label-free proxy per dataset, shared by all split configs


def _unresolved_set(name, dirty=None, error_cells=None):
    """Label-free proxy for cells the cheap/table tiers cannot fix: the generic stats candidate (fds=None: FD validation
    reads labels) AND the per-cell llm value (label-free prompt) are both None or echo the dirty value. Read from the
    cached llm tier once, kept in cache/<name>_unresolved.json. Never reads clean, nor the rules tier (its known-good
    set holds labeled clean values)."""
    path = UNRESOLVED_DIR / f"{name}_unresolved.json"
    if not path.exists():
        if dirty is None:
            dirty, _, error_cells = load(name)  # under SAMPLER=coverage load() already wrote `path`
    if not path.exists():
        llm = {(r["row"], r["col"]): r["tiers"]["llm"]["value"] for r in map(json.loads, filter(str.strip, open(_cache_path(name))))}
        echo = lambda v, r, c: v is None or v.strip() == dirty.at[r, c].strip()
        path.write_text(json.dumps([[r, c] for r, c in error_cells
                                    if echo(llm[(r, c)], r, c) and echo(tier_stats(dirty, r, c)[0], r, c)]))
    return {(r, c) for r, c in json.loads(path.read_text())}


def _coverage_rows(rows, n_unresolved, seed, n_tuples):
    """`rows` (all contain an error) + {row: number of unresolved cells} -> n_tuples rows: first rows holding unresolved
    cells, uniformly with the seed, until Q_UNRESOLVED unresolved cells are labeled (at most n_tuples // 2 rows), then
    the rest uniformly from the remaining rows. Sees row ids and counts only: no clean, no labels."""
    rng = random.Random(seed)
    cand = [r for r in rows if n_unresolved.get(r)]
    rng.shuffle(cand)
    chosen, q = [], 0
    for r in cand:
        if q >= Q_UNRESOLVED or len(chosen) >= n_tuples // 2:
            break
        chosen.append(r)
        q += n_unresolved[r]
    rest = sorted(set(rows) - set(chosen))
    return set(chosen) | set(rng.sample(rest, min(n_tuples - len(chosen), len(rest))))


def calib_test_split(cells, seed=SPLIT_SEED, n_tuples=N_LABEL_TUPLES):
    # ponytail: Baran-style label budget - every error cell in n_tuples random
    # rows that contain an error (oracle detection tells us which rows those are).
    # Works on (row, col) tuples (precompute) and cached cell dicts (eval) alike.
    # ponytail: Raha/Baran also pick their labeled tuples non-randomly (clustering); this sampler only uses dirty-data signals
    # (SAMPLER=coverage, see _coverage_rows; default stays random).
    row_of = lambda c: c["row"] if isinstance(c, dict) else c[0]
    is_sem = lambda c: c.get("semantic") if isinstance(c, dict) else tuple(c) in _SEM_CELLS
    is_unresolved = lambda c: c.get("unresolved") if isinstance(c, dict) else tuple(c) in _UNRESOLVED
    rows = sorted({row_of(c) for c in cells})
    if LABEL_EXCLUDE_SEM:
        rows = sorted(set(rows) - {row_of(c) for c in cells if is_sem(c)})
    if SAMPLER == "coverage":
        labeled = _coverage_rows(rows, Counter(row_of(c) for c in cells if is_unresolved(c)), seed, n_tuples)
    else:
        labeled = set(random.Random(seed).sample(rows, min(n_tuples, len(rows))))
    calib = [c for c in cells if row_of(c) in labeled]
    test = [c for c in cells if row_of(c) not in labeled]
    return calib, test


MIN_ISOTONIC = 30  # calibration points a tier needs (both labels present) to keep isotonic
TAU_GRID = [round(0.5 + 0.05 * i, 2) for i in range(10)]


def fit_calibrators(calib_cells):
    # cals["_prior"][tier]: Laplace accuracy of the tier's candidates on these labeled cells, known before paying
    # for the tier (used by ours_lambda's escalation test and by static_gate).
    cals = {"_prior": {}, "_n": {}}  # _n[tier]: labeled cells with a candidate from the tier (what _prior is fit on); ours_sem_guard reads _n["llm_sem"]
    for tier in TIER_ORDER:
        ys = [1.0 if c["tiers"][tier]["value"].strip() == c["clean"].strip() else 0.0
              for c in calib_cells if c["tiers"][tier]["value"] is not None]
        cals["_prior"][tier] = (sum(ys) + 1) / (len(ys) + 2)
        cals["_n"][tier] = len(ys)
    for tier in TIER_ORDER:
        if tier == "column":
            # ponytail: the column raw_conf is already a held-out (leave-one-out, Laplace) P(correct)
            # estimate, so identity calibration; refitting on the same calib cells would re-introduce
            # the leak. None = pass-through (clipped) in _apply_cal.
            cals[tier] = None
            continue
        xs, ys = [], []
        for cell in calib_cells:
            t = cell["tiers"][tier]
            if t["value"] is None:
                continue
            xs.append(t["raw_conf"])
            ys.append(1.0 if t["value"].strip() == cell["clean"].strip() else 0.0)
        if len(set(ys)) < 2:
            # ponytail: single-class (or no) calibration labels can't fit anything. Store the
            # Laplace-smoothed accuracy (k+1)/(n+2) as a constant - the raw mean would claim
            # certainty (0 or 1) from a handful of cells, and falling back to raw_conf would just
            # re-expose the uncalibrated (usually overconfident) score.
            cals[tier] = (sum(ys) + 1) / (len(ys) + 2)
            continue
        if len(xs) >= MIN_ISOTONIC:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(xs, ys)
            cals[tier] = iso
            continue
        # ponytail: isotonic overfits a step at small n (2 wrong points zeroed a whole raw range),
        # so use Platt scaling; C=10 (light L2) only stops separable data from diverging. A
        # non-increasing slope would break route()'s "calibrated(1.0) is the max" skip -> constant.
        platt = LogisticRegression(C=10.0).fit([[x] for x in xs], ys)
        cals[tier] = platt if platt.coef_[0][0] > 0 else (sum(ys) + 1) / (len(ys) + 2)
    return cals


def _apply_cal(cal, raw):
    if cal is None:
        return min(max(raw, 0.0), 1.0)
    if isinstance(cal, float):
        return cal
    if isinstance(cal, LogisticRegression):
        return float(cal.predict_proba([[raw]])[0][1])
    return float(cal.predict([raw])[0])


def _ece(confs, corrects, bins=10):
    if not confs:
        return 0.0
    total = len(confs)
    err = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        if i == bins - 1:
            sel = [j for j, c in enumerate(confs) if lo <= c <= hi]
        else:
            sel = [j for j, c in enumerate(confs) if lo <= c < hi]
        if not sel:
            continue
        avg_conf = sum(confs[j] for j in sel) / len(sel)
        avg_acc = sum(corrects[j] for j in sel) / len(sel)
        err += (len(sel) / total) * abs(avg_conf - avg_acc)
    return err


def _brier(confs, corrects):
    return sum((c - y) ** 2 for c, y in zip(confs, corrects)) / len(confs) if confs else 0.0


def ece_report(test_cells, calibrators):
    report = {}
    for tier in TIER_ORDER:
        raw_confs, cal_confs, corrects = [], [], []
        for cell in test_cells:
            t = cell["tiers"][tier]
            if t["value"] is None:
                continue
            raw_confs.append(t["raw_conf"])
            cal = calibrators.get(tier)
            cal_confs.append(_apply_cal(cal, t["raw_conf"]))
            corrects.append(1.0 if t["value"].strip() == cell["clean"].strip() else 0.0)
        report[tier] = {
            "ece_before": _ece(raw_confs, corrects),
            "ece_after": _ece(cal_confs, corrects),
            "brier_before": _brier(raw_confs, corrects),
            "brier_after": _brier(cal_confs, corrects),
            "n": len(raw_confs),
        }
    return report


# --------------------------------------------------------------- route -----

LAMBDAS = ["-inf", "-1", "-0.5", "-0.2", "-0.1", "0", "0.1", "0.2", "0.3", "0.5", "0.7", "1", "inf"]  # names; float() parses them
METHODS = ["rules_stats", "llm_only", "cascade", "ours", "ours_auto", "ours_auto_insample", "ours_no_calib", "ours_no_column",
           "static_gate", "raw_threshold", "no_escalation", "ours_rules_llm", "ours_stats_llm", "ours_no_llm",
           "ours_budget", *[f"ours_lambda@{l}" for l in LAMBDAS], *[f"ours_sem_direct@{l}" for l in LAMBDAS],
           *[f"ours_sem_guard@{l}" for l in LAMBDAS]]
LAMBDA_FAMILIES = ("ours_lambda@", "ours_sem_direct@", "ours_sem_guard@")
SEM_GUARD_K = 1  # ours_sem_guard: llm_sem's own path is on iff >= K labeled cells have an llm_sem candidate (a-priori K=1; the comparison script varies it)
# ablations, all replay-only: static_gate keeps a tier iff its labeled Laplace accuracy >= STATIC_GATE_MIN, then plain
# cascade; raw_threshold = ours_no_calib (raw confidence >= tau); no_escalation = first non-null tier of ours' path with no
# confidence gate; ours_* subsets = ours_auto restricted to these tiers (no_llm: also column programs from the DSL only).
STATIC_GATE_MIN = 0.8
_SUBSETS = {"ours_rules_llm": {"rules", "llm", "llm_sem"}, "ours_stats_llm": {"fd", "stats", "llm", "llm_sem"},
            "ours_no_llm": {"rules", "fd", "stats", "column"}}


def method_taus(calib, calibrators, tau_default=0.8):
    """{method: tau}: cross-fitted auto tau (F1) for ours_auto and every ours_lambda@L, the in-sample one for ours_auto_insample,
    tau_default for the other ours* methods, None for the tau-free baselines. ours_budget: its (lambda*, tau) come from
    auto_budget; lambda* is stored in calibrators["_lambda"] (a parameter fitted on the labeled rows, like "_prior")."""
    taus = {}
    for m in METHODS:
        if m == "ours_budget":
            calibrators["_lambda"], taus[m] = auto_budget(calib)
            continue
        taus[m] = (auto_tau_insample(calib, calibrators) if m == "ours_auto_insample"
                   else auto_tau(calib, method=m) if m == "ours_auto" or m in _SUBSETS or m.startswith(LAMBDA_FAMILIES)
                   else tau_default if m.startswith("ours") or m == "raw_threshold" else None)
    return taus


def auto_tau_insample(calib_cells, calibrators):
    """tau in TAU_GRID maximizing F1 of `ours` on the calib split, with calibrators fit on those same cells."""
    # ponytail: optimistic - the calibrators were fit on these very cells, so the in-sample F1 favours
    # low tau. Kept as `ours_auto_insample` for comparison with the cross-fitted auto_tau. Ties -> higher tau.
    return max(reversed(TAU_GRID), key=lambda t: route(calib_cells, "ours", t, calibrators)["f1"])


def _cv_tau(calib_cells, seed=0, method="ours"):
    """(tau, pooled cross-fitted F1 at that tau): tau in TAU_GRID maximizing the pooled F1 of `method` over a 2-fold cross-fit of the
    calib split: each fold's cells are routed with calibrators fit on the other fold. Folds split labeled ROWS (cells of one
    row are correlated), fixed seed; ties go to the higher tau. A tier with no cells in a training fold
    gets fit_calibrators' constant fallback."""
    rows = sorted({c["row"] for c in calib_cells})
    random.Random(seed).shuffle(rows)
    folds = [set(rows[0::2]), set(rows[1::2])]
    parts = []  # (held-out cells, calibrators fit on the other fold)
    for held in folds:
        test = [c for c in calib_cells if c["row"] in held]
        if test:
            parts.append((test, fit_calibrators([c for c in calib_cells if c["row"] not in held])))

    def pooled_f1(tau):
        rs = [route(t, method, tau, cals) for t, cals in parts]
        return _prf(*(sum(r[k] for r in rs) for k in ("correct", "repairs", "n")))[2]

    tau = max(reversed(TAU_GRID), key=pooled_f1)
    return tau, pooled_f1(tau)


def auto_tau(calib_cells, seed=0, method="ours"):
    """tau in TAU_GRID maximizing the pooled cross-fitted F1 of `method` (see _cv_tau); ties -> higher tau. method="ours_lambda@L"
    routes with budget L and picks its own tau_L."""
    return _cv_tau(calib_cells, seed, method)[0]


def auto_budget(calib_cells, seed=0):
    """(lambda*, tau_L*) for ours_budget: the lambda in LAMBDAS whose cross-fitted pooled F1 (at its own tau_L) is highest; ties go to
    the LARGER lambda (the cheaper one). Labeled rows only."""
    res = {l: _cv_tau(calib_cells, seed, f"ours_lambda@{l}") for l in reversed(LAMBDAS)}  # large lambda first: max keeps the first tie
    lam = max(res, key=lambda l: res[l][1])
    return lam, res[lam][0]


def _prf(correct, repairs, n):
    p, r = correct / repairs if repairs else 0.0, correct / n if n else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def route(cells, method, tau, calibrators):
    """Replay cached tier outputs under `method`. No new LLM calls are made;
    LLM cost is charged only for cells whose routing actually reaches tier_llm.
    `ours_lambda@L` (L = the service-level budget knob, see below) escalates per cell to a paid tier t iff q_t - L*C_t > the best
    calibrated confidence so far; `ours_budget` = ours_lambda@calibrators["_lambda"] (chosen on the labeled rows by auto_budget)."""
    name, method = method, {"raw_threshold": "ours_no_calib", "ours_sem_direct": "ours_sem_direct@0", "ours_sem_guard": "ours_sem_guard@0"}.get(method, method)
    if method == "ours_budget":
        method = f"ours_lambda@{calibrators['_lambda']}"
    sem_direct = method.startswith("ours_sem_direct@")  # ours_lambda@L, except llm_sem has its own test and eligibility (see below)
    if method.startswith("ours_sem_guard@"):  # = ours_sem_direct@L iff llm_sem has labeled evidence (>= SEM_GUARD_K cells), else ours_lambda@L
        sem_direct = calibrators["_n"]["llm_sem"] >= SEM_GUARD_K
    if method.startswith(LAMBDA_FAMILIES):
        lam = float(method.split("@")[1])
        gsize = Counter(tuple(c["group"]) for c in cells if c.get("group"))  # fallback when cells carry no whole-dataset group_size
    prior = (calibrators or {}).get("_prior", {})

    def calibrated_conf(tier, raw):
        if method == "ours_no_calib":
            return raw
        cal = calibrators.get(tier) if calibrators else None
        return _apply_cal(cal, raw)

    agg = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "llm_seconds": 0.0, "latency_seconds": 0.0,
           "repairs": 0, "correct": 0, "n": len(cells), "to_llm": 0, "charged_groups": set()}
    sys_conf = []  # (calibrated conf, raw conf, correct) of the accepted candidate of every repaired cell: system-level calibration
    seg = defaultdict(Counter)  # by the cell's `semantic` flag: n, repairs, correct, calls (only when cells carry it)

    for cell in cells:
        dirty_val = cell["dirty"].strip()
        clean_val = cell["clean"].strip()
        tiers = cell["tiers"]
        output = dirty_val
        used_llm = used_sem = False
        used_column = False
        sem_rec = tiers.get("llm_sem")  # the llm_sem answer this cell is charged for (ours_sem_direct may use its `llm_sem_direct` one)
        acc = None  # (calibrated, raw) confidence of the accepted candidate; calibrated = raw for methods that do not calibrate
        cell_calls = 0
        reached = []  # tiers this cell's routed path computed (latency = sum of their seconds)

        if method == "rules_stats":
            for t in ("rules", "fd", "stats"):
                reached.append(t)
                if tiers[t]["value"] is not None:
                    output = tiers[t]["value"]
                    acc = (tiers[t]["raw_conf"],) * 2
                    break

        elif method == "llm_only":
            used_llm = True
            reached.append("llm")
            l = tiers["llm"]
            if l["value"] is not None:
                output = l["value"]
                acc = (l["raw_conf"],) * 2

        elif method in ("cascade", "static_gate"):
            for t in TIER_ORDER:
                if method == "static_gate" and prior[t] < STATIC_GATE_MIN:
                    continue
                reached.append(t)
                used_llm |= t == "llm"
                used_sem |= t == "llm_sem"
                if t == "column":
                    used_column = True
                v = tiers[t]
                if v["value"] is not None:
                    output = v["value"]
                    acc = (v["raw_conf"],) * 2
                    break

        elif method in ("ours", "ours_auto", "ours_auto_insample", "ours_no_calib", "ours_no_column", "no_escalation", *_SUBSETS):
            # Every cell always starts free: rules, fd, then stats. Only once
            # all miss tau does error_type pick the paid escalation path -
            # systematic errors go through the (one-call-per-group) column
            # tier before per-cell llm; everything else goes straight to llm.
            tier_sequence = ["rules", "fd", "stats"]
            if method != "ours_no_column" and (cell["error_type"] == "systematic" or tiers["column"]["value"] is not None) \
                    and not (method == "ours_no_llm" and tiers["column"]["cost"].get("calls", 0)):  # LLM-made program
                tier_sequence.append("column")
            tier_sequence += ["llm", "llm_sem"]
            tier_sequence = [t for t in tier_sequence if t in _SUBSETS.get(method, tier_sequence)]
            gated = method != "no_escalation"
            for t in tier_sequence:
                # ponytail: isotonic is monotone, so calibrated(raw=1.0) is the tier's max
                # possible confidence; below tau it can never be accepted - skip it (no call,
                # no cost). Same output, cheaper. Exact only while calibrators are monotone.
                if gated and method != "ours_no_calib" and calibrated_conf(t, 1.0) < tau:
                    continue
                reached.append(t)
                used_llm |= t == "llm"
                used_sem |= t == "llm_sem"
                if t == "column":
                    used_column = True
                v = tiers[t]
                if v["value"] is None:
                    continue
                if not gated or calibrated_conf(t, v["raw_conf"]) >= tau:
                    output = v["value"]
                    acc = (calibrated_conf(t, v["raw_conf"]) if gated else v["raw_conf"], v["raw_conf"])
                    break
            # falling off the end (never accepted) -> abstain, keep dirty_val

        elif method.startswith(LAMBDA_FAMILIES):
            # r* = argmax_r [P(correct|x,r) - lam*C(r)], greedily over the paid tiers in cost order. Free tiers first: the best
            # calibrated candidate (p_best, v_best). A paid tier t is bought iff its prior accuracy q_t (labeled Laplace) minus lam*C_t beats
            # p_best; C = 1 per llm/llm_sem call, group calls / group size for column. lam=-inf: buy every applicable tier; +inf: no LLM call (0-cost tiers stay free).
            # llm_sem applies only to eligible cells (no other tier has a candidate) and only after the llm was bought.
            # ours_sem_direct@L: the same, but llm_sem is tested on its own (q_sem - L*1 > p_best) whenever the free tiers and column
            # have no candidate (`llm_sem_direct_eligible`), also when the llm was not bought; if it was bought and found a candidate, no.
            p_best, v_best, raw_best = 0.0, None, 0.0
            for t in ("rules", "fd", "stats"):
                reached.append(t)
                if tiers[t]["value"] is not None and calibrated_conf(t, tiers[t]["raw_conf"]) > p_best:
                    p_best, v_best, raw_best = calibrated_conf(t, tiers[t]["raw_conf"]), tiers[t]["value"], tiers[t]["raw_conf"]
            grp = tuple(cell["group"]) if cell.get("group") else None
            bought_llm = False
            for t in ("column", "llm", "llm_sem"):
                if t == "column":
                    if not (cell["error_type"] == "systematic" or tiers["column"]["value"] is not None):
                        continue
                    c = tiers["column"]["cost"].get("calls", 0) / (cell.get("group_size") or gsize[grp]) if grp else 0.0
                elif t == "llm_sem" and sem_direct:
                    if not (cell.get("llm_sem_direct_eligible", all(tiers[k]["value"] is None for k in _DIRECT_FREE))
                            and not (bought_llm and tiers["llm"]["value"] is not None)):
                        continue
                    sem_rec = cell.get("llm_sem_direct") or tiers["llm_sem"]
                    c = 1.0
                elif t == "llm_sem":
                    if not (bought_llm and cell.get("llm_sem_eligible", tiers["llm_sem"]["value"] is not None)):
                        continue
                    c = 1.0
                else:
                    c = 1.0
                # a tier with C_t == 0 (e.g. a DSL column program) is free like rules/fd/stats: no lambda term, so no inf * 0
                if lam != -float("inf") and prior[t] - (lam * c if c else 0.0) <= p_best:
                    continue
                reached.append(t)
                bought_llm |= t == "llm"
                used_llm |= t == "llm"
                used_sem |= t == "llm_sem"
                used_column |= t == "column"
                tv = sem_rec if t == "llm_sem" else tiers[t]
                if tv["value"] is not None and calibrated_conf(t, tv["raw_conf"]) > p_best:
                    p_best, v_best, raw_best = calibrated_conf(t, tv["raw_conf"]), tv["value"], tv["raw_conf"]
            if v_best is not None and p_best >= tau:
                output, acc = v_best, (p_best, raw_best)  # else abstain: keep dirty_val

        else:
            raise ValueError(f"unknown method {method}")

        for t in [t for t, u in (("llm", used_llm), ("llm_sem", used_sem)) if u]:  # llm_sem: cost 0 on cells it was not asked
            lc = (sem_rec if t == "llm_sem" else tiers[t])["cost"]
            agg["llm_calls"] += lc.get("calls", 0)
            cell_calls += lc.get("calls", 0)
            agg["prompt_tokens"] += lc.get("prompt_tokens", 0)
            agg["completion_tokens"] += lc.get("completion_tokens", 0)
            agg["llm_seconds"] += lc.get("seconds", 0.0)
            agg["latency_seconds"] += lc.get("seconds", 0.0) + lc.get("prep_seconds", 0.0)
        for t in reached:  # the free tiers' own compute, and the column tier's per-cell apply time
            if t in ("rules", "fd", "stats", "column"):
                agg["latency_seconds"] += tiers[t]["cost"].get("cell_seconds" if t == "column" else "seconds", 0.0)
        if used_llm or used_sem:
            agg["to_llm"] += 1

        if used_column and cell.get("group"):
            grp = tuple(cell["group"])
            if grp not in agg["charged_groups"]:
                agg["charged_groups"].add(grp)
                cc = tiers["column"]["cost"]
                agg["llm_calls"] += cc.get("calls", 0)
                cell_calls += cc.get("calls", 0)
                agg["prompt_tokens"] += cc.get("prompt_tokens", 0)
                agg["completion_tokens"] += cc.get("completion_tokens", 0)
                agg["llm_seconds"] += cc.get("seconds", 0.0)
                agg["latency_seconds"] += cc.get("wall_seconds", cc.get("seconds", 0.0))  # group program made once: DSL search or LLM
            agg["to_llm"] += 1

        repaired, right = output.strip() != dirty_val, output.strip() == clean_val
        agg["repairs"] += repaired
        agg["correct"] += repaired and right
        if repaired and acc:
            sys_conf.append((*acc, right))
        if "semantic" in cell:
            s_ = seg[bool(cell["semantic"])]
            s_.update(n=1, repairs=repaired, correct=repaired and right, calls=cell_calls)

    precision, recall, f1 = _prf(agg["correct"], agg["repairs"], agg["n"])
    cal, raw, ok = ([x[i] for x in sys_conf] for i in range(3))
    sem = {"sys_ece": _ece(cal, ok), "sys_brier": _brier(cal, ok), "sys_ece_raw": _ece(raw, ok), "sys_brier_raw": _brier(raw, ok),
           "sys_n": len(ok)}  # system-level calibration of the final decision, over the repaired cells
    if seg:  # *_sem datasets: semantic (injected) vs non-semantic cells
        _, sem["sem_recall"], sem["sem_f1"] = _prf(seg[True]["correct"], seg[True]["repairs"], seg[True]["n"])
        sem["nonsem_f1"] = _prf(seg[False]["correct"], seg[False]["repairs"], seg[False]["n"])[2]
        sem["sem_call_share"] = seg[True]["calls"] / agg["llm_calls"] if agg["llm_calls"] else 0.0
    return {
        "method": name, "tau": tau,
        "precision": precision, "recall": recall, "f1": f1,
        "repair_accuracy": recall,  # correct repairs / error cells: the same number as recall under our definitions
        "llm_calls": agg["llm_calls"], "llm_call_rate": agg["llm_calls"] / agg["n"] if agg["n"] else 0.0,
        "prompt_tokens": agg["prompt_tokens"], "completion_tokens": agg["completion_tokens"],
        # GPU time = sum of LLM request seconds (per-cell calls ran 8 at a time, column groups one at a time)
        "llm_seconds": agg["llm_seconds"],
        # sequential-equivalent end-to-end time of the routed path: sum of the seconds of the tiers each cell reached
        "latency_seconds": agg["latency_seconds"],
        "latency_per_cell": agg["latency_seconds"] / agg["n"] if agg["n"] else 0.0,
        "frac_to_llm": agg["to_llm"] / agg["n"] if agg["n"] else 0.0,
        "n": agg["n"], "repairs": agg["repairs"], "correct": agg["correct"], **sem,
        # mean recorded LLM latency (measured under 8 concurrent requests) per repaired cell
        "llm_sec_per_repair": agg["llm_seconds"] / agg["repairs"] if agg["repairs"] else 0.0,
    }
