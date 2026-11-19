"""Plain-assert smoke tests. Run from the repo root with: .venv/bin/python tests/test_pipeline.py"""
import io
import json
import random
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import pipeline as pl  # noqa: E402

ZERO_COST = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}


def test_tier_rules_fixes_format_error():
    # "10-001" is a punctuation/format slip on an otherwise-uniform zip column.
    df = pd.DataFrame({"zip": ["10001", "10001", "10-001"]})
    value, conf, cost = pl.tier_rules(df, 2, "zip")
    assert value == "10001", f"expected reformatted zip, got {value!r}"
    assert conf == 1.0, f"expected high confidence (value is in the freq set), got {conf}"
    assert et_is(df, 2, "zip", "syntactic")


def test_tier_rules_known_good_values_exclude_error_cells():
    # "12.0 oz." is frequent only because it is itself a systematic dirty value; with oracle detection it
    # must not vouch for the candidate "12.0 oz." for the dirty "12.0 OZ.".
    df = pd.DataFrame({"oz": ["12", "12", "9", "12.0 oz.", "12.0 oz.", "12.0 oz.", "12.0 OZ."]})
    assert pl.tier_rules(df, 6, "oz")[:2] == ("12.0 oz.", 1.0), "no error set: dirty value is its own evidence"
    errs = {(i, "oz") for i in range(3, 7)}
    assert pl.tier_rules(df, 6, "oz", errs)[:2] == (None, 0.0), "error cells are not known-good values"
    tiny = pd.DataFrame({"oz": ["12", "12.0 oz.", "12.0 oz.", "12.0 OZ."]})  # <2 distinct good values -> all cells
    assert pl.tier_rules(tiny, 3, "oz", {(1, "oz"), (2, "oz"), (3, "oz")})[:2] == pl.tier_rules(tiny, 3, "oz")[:2]


def test_tier_rules_known_good_includes_labeled_clean_values():
    # Beers `ounces`: every cell is an error, so there are no non-error cells; the labeled calib cells' CLEAN
    # values ("12", "16") are the known-good set, and the frequent dirty "12.0 oz." must not vouch for itself.
    df = pd.DataFrame({"oz": ["12.0 oz."] * 5 + ["12.0 OZ."] * 2 + ["16.0 oz."]})
    errs = {(i, "oz") for i in range(len(df))}
    assert pl.tier_rules(df, 5, "oz", errs)[:2] == ("12.0 oz.", 1.0), "no labels: falls back to all dirty values"
    labeled = {(0, "oz"): "12", (1, "oz"): "12", (7, "oz"): "16"}
    value, conf, _ = pl.tier_rules(df, 5, "oz", errs, labeled)
    assert (value, conf) == (None, 0.0), (value, conf)  # good pattern is "99": "12.0 oz." no longer vouches for itself
    only_one = {(0, "oz"): "12"}  # <2 distinct good values -> fallback to all dirty values
    assert pl.tier_rules(df, 5, "oz", errs, only_one)[:2] == ("12.0 oz.", 1.0)
    clean = pd.DataFrame({"oz": ["12"] * 8})  # only calib-split clean values are read (leakage assert inside)
    lab = pl._labeled_clean(clean, sorted(errs))
    calib, test = pl.calib_test_split(sorted(errs))
    assert set(lab) == set(calib) and not set(lab) & set(test)


def test_tier_stats_fixes_dependency_error():
    # row 2's zip disagrees with the other two NYC rows that share its city.
    df = pd.DataFrame({
        "city": ["NYC", "NYC", "NYC"],
        "zip": ["10001", "10001", "99999"],
    })
    value, conf, cost = pl.tier_stats(df, 2, "zip")
    assert value == "10001", f"expected dependency-vote majority, got {value!r}"
    assert conf > 0, "dependency vote should carry positive confidence"
    assert et_is(df, 2, "zip", "statistical")


def test_tier_stats_fixes_typo_via_fuzzy_match():
    # "rad" is a one-letter typo of the column's frequent value "red".
    df = pd.DataFrame({"color": ["red", "red", "rad"]})
    value, conf, cost = pl.tier_stats(df, 2, "color")
    assert value == "red", f"expected nearest frequent value, got {value!r}"
    assert conf > 0
    assert et_is(df, 2, "color", "statistical")


def test_tier_stats_self_match_is_deliberate_abstention_for_frequent_values():
    # "x" is frequent, so it matches itself at ratio 1.0 and wins; load_cache then drops that echo, so stats
    # abstains and never "corrects" an already-frequent value into a look-alike ("xy").
    df = pd.DataFrame({"v": ["x", "x", "xy", "xy"]})
    deps, sims = pl._stats_candidates(df, 0, "v")
    assert ("x", 1.0) in sims
    assert pl.tier_stats(df, 0, "v")[:2] == ("x", 1.0)  # echo of the dirty value


def _flights(monkeypatch):
    """5 flights x 4 source rows; the 4th row of each flight has a bogus time. Calib = flights 0-2's cells."""
    n = 5
    rows = [(f"F{f}", f"s{j}", f"t{f}" if j < 3 else f"bad{f}", "z") for f in range(n) for j in range(4)]
    dirty = pd.DataFrame(rows, columns=["flight", "src", "time", "junk"])
    clean = dirty.copy()
    clean["time"] = [f"t{f}" for f in range(n) for _ in range(4)]
    errs = [(4 * f + 3, "time") for f in range(n)]
    monkeypatch.setattr(pl, "calib_test_split", lambda cells, **kw: (cells[:3], cells[3:]))
    monkeypatch.setattr(pl, "propose_source", lambda name, df: (None, ZERO_COST))
    return dirty, clean, errs


def test_llm_proposed_fd_is_kept_after_calib_validation_and_drives_stats(monkeypatch):
    dirty, clean, errs = _flights(monkeypatch)
    calls = []

    def create(**kw):
        calls.append(kw)
        content = ('```json\n[{"lhs": ["flight"], "rhs": "time"}, {"lhs": ["nope"], "rhs": "time"}, '
                   '{"lhs": ["time"], "rhs": "time"}]\n```')
        usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()
        return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": content})()})()],
                              "usage": usage})()

    client = type("Cl", (), {"chat": type("Ch", (), {"completions": type("Co", (), {"create": staticmethod(create)})()})()})()
    monkeypatch.setattr(pl, "_get_client", lambda: client)
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(pl, "FDS_DIR", Path(d))
        assert pl.propose_fds("fl", dirty)[0] == [(("flight",), "time")]  # bogus entries dropped
        pl.propose_fds("fl", dirty)
        assert len(calls) == 1 and calls[0]["temperature"] == 0 and calls[0]["n"] == 1  # cached after 1 call
        assert pl.setup_llm_calls("fl") == 1
        fds = pl.kept_fds("fl", dirty, clean, errs)
    assert fds["time"][0] == ("flight",) and fds["time"][1:6] == (0.8, 3, 3, "share", "unweighted")  # (3+1)/(3+2)
    # test-split cell (flight 4): the fd tier votes the entity majority with conf = share 1.0 (generic vote would
    # give 3/3*3/5 = 0.6); stats has no candidate for a column with a kept FD
    assert pl.tier_fd(dirty, 19, "time", fds)[:2] == ("t4", 1.0)
    assert pl.tier_stats(dirty, 19, "time", fds)[0] is None
    # a vote echoing the dirty value (or no vote) gives None
    dirty.at[19, "time"] = "t4"
    assert pl.tier_fd(dirty, 19, "time", fds)[0] is None
    assert pl.tier_fd(dirty.assign(flight=[f"u{i}" for i in range(20)]), 19, "time", fds)[0] is None


def test_fd_tier_picks_raw_confidence_variant_on_calib_split(monkeypatch):
    # entities A: 5 good rows + 1 error row (vote right, support 5); B: 2 wrong rows + 1 error row (vote wrong,
    # support 2). Share is 1.0 for every vote (AUC 0.5); share x min(support,5)/5 separates them (AUC 1.0).
    rows, errs = [], []
    for e, (v, n_other) in {"A1": ("a", 5), "A2": ("a", 5), "B1": ("x", 2), "B2": ("x", 2)}.items():
        rows += [(e, v)] * n_other
        errs.append((len(rows), "val"))
        rows.append((e, "bad"))
    dirty = pd.DataFrame(rows, columns=["ent", "val"])
    clean = dirty.copy()
    for r, _ in errs:
        clean.at[r, "val"] = "a" if dirty.at[r, "ent"].startswith("A") else "b"
    monkeypatch.setattr(pl, "calib_test_split", lambda cells, **kw: (cells, []))
    monkeypatch.setattr(pl, "propose_fds", lambda name, df: ([(("ent",), "val")], ZERO_COST))
    monkeypatch.setattr(pl, "propose_source", lambda name, df: (None, ZERO_COST))
    fds = pl.kept_fds("fl", dirty, clean, errs)
    assert fds["val"][3:6] == (2, "support", "unweighted"), fds
    r_a, r_b = errs[0][0], errs[2][0]
    assert pl.tier_fd(dirty, r_a, "val", fds)[:2] == ("a", 1.0)
    assert pl.tier_fd(dirty, r_b, "val", fds)[:2] == ("x", 0.4)


def test_source_weighted_fd_vote_beats_copied_errors_and_is_chosen_on_calib(monkeypatch):
    # 12 entities, good sources g1-g3 always right; bad sources b1-b4 each give their own wrong value, except on
    # entities E0/E1 where they all copy the same wrong one -> the plain majority is wrong there. Calib = the two
    # error cells (g1's rows on E0/E1); truth discovery learns the bad sources' low weights from E2..E11.
    rows, errs = [], []
    for e in range(12):
        for s in ("g1", "g2", "g3"):
            if s == "g1" and e < 2:
                errs.append((len(rows), "val"))
            rows.append((f"E{e}", s, "zz" if s == "g1" and e < 2 else f"v{e}"))
        rows += [(f"E{e}", b, f"x{e}" if e < 2 else f"{b}_{e}") for b in ("b1", "b2", "b3", "b4")]
    dirty = pd.DataFrame(rows, columns=["ent", "src", "val"])
    clean = dirty.assign(val=[f"v{r[0][1:]}" for r in rows])
    w = pl._truth_discovery(dirty, ("ent",), "val", "src")
    assert min(w["g1"], w["g2"], w["g3"]) > max(w[b] for b in ("b1", "b2", "b3", "b4")), w
    r = errs[0][0]
    assert pl._fd_vote(dirty, r, ("ent",), "val")[0] == "x0", "unweighted: 4 copied votes beat 2 good ones"
    assert pl._fd_vote(dirty, r, ("ent",), "val", ("src", w))[0] == "v0"
    monkeypatch.setattr(pl, "calib_test_split", lambda cells, **kw: (cells, []))
    monkeypatch.setattr(pl, "propose_fds", lambda name, df: ([(("ent",), "val")], ZERO_COST))
    monkeypatch.setattr(pl, "propose_source", lambda name, df: ("src", ZERO_COST))
    fds = pl.kept_fds("fl", dirty, clean, errs)
    assert fds["val"][3] == 2 and fds["val"][5] in ("weighted", "calib"), fds["val"][:6]  # 2/2 vs 0/2 unweighted
    assert pl.tier_fd(dirty, r, "val", fds)[0] == "v0"


def test_dependency_votes_count_only_known_clean_voters_and_fall_back_when_none():
    # key group "a": 4 error rows copy "bad", 2 clean rows say "good" -> the plain majority is wrong
    df = pd.DataFrame({"k": ["a"] * 6, "v": ["bad"] * 4 + ["good"] * 2})
    errs = {(i, "v") for i in range(4)}
    assert pl._fd_vote(df, 0, ("k",), "v")[0] == "bad"
    assert pl._fd_vote(df, 0, ("k",), "v", None, errs)[0] == "good"
    assert pl._stats_candidates(df, 0, "v")[0][0][0] == "bad" and pl._stats_candidates(df, 0, "v", None, errs)[0][0][0] == "good"
    assert pl.tier_fd(df, 0, "v", {"v": (("k",), 1, 1, 1, "share", "unweighted", None)}, errs)[0] == "good"
    assert pl._fd_vote(df, 0, ("k",), "v", None, {(i, "v") for i in range(6)})[0] == "bad"  # no clean voter: as before


def test_one_known_clean_voter_is_enough_but_fallback_needs_two():
    df = pd.DataFrame({"k": ["a"] * 3, "v": ["bad", "bad", "good"]})
    one_clean = {(0, "v"), (1, "v")}  # row 0's only clean voter is row 2
    assert pl._fd_vote(df, 0, ("k",), "v", None, one_clean)[:2] == ("good", 1.0)
    assert pl._stats_candidates(df, 0, "v", None, one_clean)[0][0][0] == "good"
    assert pl._fd_vote(df, 0, ("k",), "v", None, {(i, "v") for i in range(3)})[0] == "bad"  # no clean: 2 all-voters
    two = df.iloc[:2].reset_index(drop=True)
    assert pl._fd_vote(two, 0, ("k",), "v", None, {(0, "v"), (1, "v")}) is None  # fallback with 1 voter: abstain
    assert pl._stats_candidates(two, 0, "v", None, {(0, "v"), (1, "v")})[0] == []


def test_propose_source_is_one_cached_call_and_only_exact_column_names_count(monkeypatch):
    calls = []

    def create(**kw):
        calls.append(kw)
        content = "`src`\n"
        usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()
        return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": content})()})()],
                              "usage": usage})()

    client = type("Cl", (), {"chat": type("Ch", (), {"completions": type("Co", (), {"create": staticmethod(create)})()})()})()
    monkeypatch.setattr(pl, "_get_client", lambda: client)
    df = pd.DataFrame({"src": ["a", "b"], "v": ["1", "2"]})
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(pl, "FDS_DIR", Path(d))
        assert pl.propose_source("x", df)[0] == "src" and pl.propose_source("x", df)[0] == "src"
        assert len(calls) == 1 and calls[0]["temperature"] == 0 and pl.setup_llm_calls("x") == 1
        (Path(d) / "x_source.json").write_text(json.dumps({"raw": "null", "cost": ZERO_COST}))
        assert pl.propose_source("x", df)[0] is None


def test_bogus_fd_with_low_calib_score_is_dropped(monkeypatch):
    dirty, clean, errs = _flights(monkeypatch)
    monkeypatch.setattr(pl, "propose_fds", lambda name, df: ([(("junk",), "time")], ZERO_COST))
    assert pl.kept_fds("fl", dirty, clean, errs) == {}  # constant key: 0/3 right, score 0.2
    monkeypatch.setattr(pl, "propose_fds", lambda name, df: ([(("junk",), "time"), (("flight",), "time")], ZERO_COST))
    assert pl.kept_fds("fl", dirty, clean, errs)["time"][0] == ("flight",)


def test_refresh_cheap_leaves_llm_and_column_fields_untouched(monkeypatch):
    df = pd.DataFrame({"k": ["a"] * 3, "v": ["x", "y", "y"]})
    clean = pd.DataFrame({"k": ["a"] * 3, "v": ["y", "y", "y"]})
    monkeypatch.setattr(pl, "load", lambda name: (df, clean, [(0, "v")]))
    monkeypatch.setattr(pl, "propose_fds", lambda name, d: ([], ZERO_COST))
    monkeypatch.setattr(pl, "propose_source", lambda name, d: (None, ZERO_COST))
    marker = {"value": "KEEP", "raw_conf": 0.123, "cost": ZERO_COST}
    rec = {"row": 0, "col": "v", "dirty": "x", "clean": "y", "error_type": "old", "group": None,
           "tiers": {"rules": marker, "stats": marker, "column": marker, "llm": marker}}
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "c.jsonl"
        path.write_text(json.dumps(rec) + "\n")
        monkeypatch.setattr(pl, "_cache_path", lambda name: path)
        pl.refresh_cheap("fake")
        new = json.loads(path.read_text())
    assert new["tiers"]["llm"] == marker and new["tiers"]["column"] == marker
    assert new["tiers"]["stats"]["value"] == "y" and new["error_type"] != "old"


def test_refresh_llm_sem_asks_only_unresolved_cells_once_and_keeps_other_fields(monkeypatch):
    df = pd.DataFrame({"k": ["a"] * 3, "v": ["x", "x", "y"]})
    clean = pd.DataFrame({"k": ["a"] * 3, "v": ["z", "z", "z"]})
    monkeypatch.setattr(pl, "load", lambda name: (df, clean, [(0, "v"), (1, "v"), (2, "v")]))
    none = {"value": None, "raw_conf": 0.0, "cost": ZERO_COST}
    echo = {"value": "x", "raw_conf": 0.9, "cost": ZERO_COST}  # a no-op candidate does not resolve the cell
    recs = [{"row": 0, "col": "v", "dirty": "x", "clean": "z", "tiers": {"rules": echo, "llm": none}},
            {"row": 1, "col": "v", "dirty": "x", "clean": "z", "tiers": {"rules": {**echo, "value": "y"}, "llm": none}},
            {"row": 2, "col": "v", "dirty": "y", "clean": "z", "tiers": {"rules": none, "llm": {**echo, "value": "y"}}}]
    asked = []
    monkeypatch.setattr(pl, "tier_llm_sem", lambda d, r, c, ex, **kw: asked.append((r, ex)) or ("z", 0.8, {**ZERO_COST, "calls": 1}))
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "c.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in recs))
        monkeypatch.setattr(pl, "_cache_path", lambda name: path)
        monkeypatch.setattr(pl, "LLM_SEM_DIR", Path(d))
        pl.refresh_llm_sem("fake")
        new = [json.loads(line) for line in path.read_text().splitlines()]
        pl.refresh_llm_sem("fake")  # shared answers are reused: nothing is asked again
    assert sorted(a[0] for a in asked) == [0, 2] and all(a[1] == ["x", "y"] for a in asked), asked
    assert [r["llm_sem_eligible"] for r in new] == [True, False, True]
    assert new[0]["tiers"]["llm_sem"] == {"value": "z", "raw_conf": 0.8, "cost": {**ZERO_COST, "calls": 1}}
    assert new[1]["tiers"]["llm_sem"] == {"value": None, "raw_conf": 0.0, "cost": ZERO_COST}
    assert new[0]["tiers"]["rules"] == echo and new[1]["tiers"]["rules"]["value"] == "y"


def test_llm_sem_prompt_forbids_the_dirty_value_and_dirty_answers_count_as_none(monkeypatch):
    seen = []

    class Client:  # 3 samples: two repeat the dirty value, one gives 'z'
        chat = type("C", (), {"completions": type("P", (), {"create": lambda self, **kw: seen.append(kw) or type("R", (), {
            "usage": None, "choices": [type("Ch", (), {"message": type("M", (), {"content": c})()})() for c in (
                '{"value": "x", "confidence": 1}', '{"value": "x", "confidence": 1}', '{"value": "z", "confidence": 0.9}')]})()})()})()

    monkeypatch.setattr(pl, "_get_client", lambda: Client())
    df = pd.DataFrame({"k": ["a"], "v": ["x"]})
    val, conf, cost = pl.tier_llm_sem(df, 0, "v", ["y", "z"])
    prompt = seen[0]["messages"][0]["content"]
    assert "'x' of column 'v' is known to be wrong" in prompt and "Do not repeat 'x'" in prompt and "['y', 'z']" in prompt
    assert seen[0]["n"] == 3 and seen[0]["temperature"] == 0.7 and seen[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert (val, cost["calls"]) == ("z", 1) and abs(conf - (1 / 3 + 0.9) / 2) < 1e-9, "dirty answers are no votes, agreement stays /k"
    assert pl._format_examples(pd.DataFrame({"v": ["a", "b", "b", "c", "bad"]}), {(4, "v")}, "v", 2) == ["b", "a"]


def et_is(df, row, col, expected):
    got = pl.error_type(df, row, col)
    assert got == expected, f"error_type({row},{col}) = {got!r}, expected {expected!r}"
    return True


def test_ours_abstains_below_tau(monkeypatch):
    def stub_tier_llm(dirty_df, row, col, k=3, temperature=0.7):
        return "STUBVAL", 0.3, {"calls": 1, "prompt_tokens": 10, "completion_tokens": 5, "seconds": 0.01}

    monkeypatch.setattr(pl, "tier_llm", stub_tier_llm)
    llm_val, llm_conf, llm_cost = pl.tier_llm(None, 0, "x")  # exercises the stub, like precompute would

    cell = {
        "row": 0, "col": "x", "dirty": "Foo", "clean": "Bar", "error_type": "semantic",
        "tiers": {
            "rules": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
            "fd": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
            "stats": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
            "column": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
            "llm": {"value": llm_val, "raw_conf": llm_conf, "cost": llm_cost},
            "llm_sem": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
        },
    }
    result = pl.route([cell], "ours", 0.8, {})
    assert result["repairs"] == 0, "confidence 0.3 < tau 0.8 should abstain (no repair made)"
    assert result["llm_calls"] == 1, "the llm tier was still reached, so its cost must be charged"


def test_precision_recall_known_case():
    def cell(dirty, clean, rules_val, stats_val):
        return {
            "row": 0, "col": "c", "dirty": dirty, "clean": clean, "error_type": "syntactic",
            "tiers": {
                "rules": {"value": rules_val, "raw_conf": 1.0 if rules_val else 0.0, "cost": ZERO_COST},
                "fd": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
                "stats": {"value": stats_val, "raw_conf": 1.0 if stats_val else 0.0, "cost": ZERO_COST},
                "llm": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
            },
        }

    cells = [
        cell("a", "A", "A", None),   # rules fires, correct repair
        cell("b", "Y", None, "Z"),   # rules None -> falls to stats, wrong repair
        cell("c", "C", None, None),  # no candidate anywhere -> no repair
        cell("d", "D", "d", None),   # candidate equals dirty value -> not counted as a repair
    ]
    r = pl.route(cells, "rules_stats", None, {})
    assert r["n"] == 4
    assert r["repairs"] == 2, r
    assert r["correct"] == 1, r
    assert abs(r["precision"] - 0.5) < 1e-9
    assert abs(r["recall"] - 0.25) < 1e-9
    assert abs(r["f1"] - (2 * 0.5 * 0.25 / 0.75)) < 1e-9


def test_tier_rules_numeric_extraction_integer():
    # every "ounces" cell is dirty the same way -> pattern-matching alone is a
    # no-op, so numeric extraction has to step in.
    df = pd.DataFrame({"ounces": ["12.0 oz", "12.0 oz", "16.0 oz", "16.0 oz", "12.0 oz"]})
    value, conf, cost = pl.tier_rules(df, 0, "ounces")
    assert value == "12", f"expected extracted integer, got {value!r}"


def test_tier_rules_numeric_extraction_float():
    df = pd.DataFrame({"abv": ["0.05%", "0.05%", "0.06%", "0.06%"]})
    value, conf, cost = pl.tier_rules(df, 0, "abv")
    assert value == "0.05", f"expected extracted float text, got {value!r}"


def test_tier_rules_numeric_extraction_skips_multi_number_value():
    # Flights-style times like "7:16 a.m." contain two number matches (7 and
    # 16) - numeric extraction must not fire and collapse this to "7".
    df = pd.DataFrame({"time": ["7:16 a.m.", "7:16 a.m.", "8:20 a.m.", "8:20 a.m."]})
    value, conf, cost = pl.tier_rules(df, 0, "time")
    assert value != "7", f"numeric extraction should not fire on a multi-number value, got {value!r}"


def test_tier_rules_null_token():
    df = pd.DataFrame({"ibu": ["N/A", "N/A", "20", "20"]})
    value, conf, cost = pl.tier_rules(df, 0, "ibu")
    assert value == "", f"expected null token mapped to empty string, got {value!r}"


def _rayyan_like_group(n=12):
    """n error cells sharing one dirty pattern (Y/M/D unpadded -> M/D/YY)."""
    dirty_vals = [f"9/{i % 3 + 1}/{10 + i:02d}" for i in range(n)]
    clean_vals = [f"{i % 3 + 1}/{10 + i:02d}/09" for i in range(n)]
    dirty_df = pd.DataFrame({"date": dirty_vals})
    clean_df = pd.DataFrame({"date": clean_vals})
    error_cells = [(i, "date") for i in range(n)]
    return dirty_df, clean_df, error_cells


def test_column_tier_classifies_and_fixes_systematic_group(monkeypatch):
    dirty_df, clean_df, error_cells = _rayyan_like_group()
    error_cell_set = set(error_cells)
    error_groups = pl._error_groups(dirty_df, error_cells)

    assert pl.error_type(dirty_df, 0, "date", error_groups) == "systematic"

    calib_pairs, _test_pairs = pl.calib_test_split(error_cells)
    calib_set = set(calib_pairs)
    pattern = pl._generalize(dirty_df.at[0, "date"].strip())
    group_cells = error_groups[("date", pattern)]
    assert len(group_cells) == 12

    examples, other_dirty, non_error_vals = pl._column_prompt_inputs(
        dirty_df, clean_df, "date", group_cells, calib_set, error_cell_set)

    fixed_program = (
        "```python\n"
        "def fix(s):\n"
        "    y, m, d = s.split('/')\n"
        "    return f\"{m}/{d}/{int(y):02d}\"\n"
        "```"
    )

    def stub_sample(prompt, k=3, temperature=0.7):
        return [fixed_program] * 3, {"calls": 1, "prompt_tokens": 5, "completion_tokens": 5, "seconds": 0.01}

    monkeypatch.setattr(pl, "_column_llm_sample", stub_sample)
    monkeypatch.setattr(pl, "_dsl_search", lambda examples, plaus=None: None)  # exercise the LLM path

    col_cache = {}
    group_rec = pl._get_or_make_column_group(
        col_cache, io.StringIO(), "date", pattern, examples, other_dirty, non_error_vals)
    assert group_rec["cost"]["calls"] == 1

    cells = []
    for row, col in group_cells:
        value, conf, cost = pl.tier_column(dirty_df, row, col, group_rec)
        assert value == clean_df.at[row, col], f"row {row}: expected {clean_df.at[row, col]!r}, got {value!r}"
        # 12 labeled examples, all LOO-correct -> (12+1)/(12+2); all 3 programs agree -> x 1.0
        assert abs(conf - 13 / 14) < 1e-9, f"expected Laplace group score 13/14, got {conf}"
        cells.append({
            "row": row, "col": col, "dirty": dirty_df.at[row, col], "clean": clean_df.at[row, col],
            "error_type": "systematic", "group": [col, pattern],
            "tiers": {
                "rules": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
                "fd": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
                "stats": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
                "column": {"value": value, "raw_conf": conf, "cost": cost},
                "llm": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
            },
        })

    result = pl.route(cells, "cascade", None, {})
    assert result["repairs"] == 12 and result["correct"] == 12, result
    assert result["llm_calls"] == 1, "the group's LLM cost must be charged exactly once, not per cell"


def test_column_tier_examples_from_calib_split_only():
    dirty_df, clean_df, error_cells = _rayyan_like_group()
    error_cell_set = set(error_cells)
    calib_pairs, _test_pairs = pl.calib_test_split(error_cells)
    calib_set = set(calib_pairs)

    examples, _other_dirty, _non_error_vals = pl._column_prompt_inputs(
        dirty_df, clean_df, "date", error_cells, calib_set, error_cell_set)

    dirty_to_row = {dirty_df.at[r, "date"]: r for r, _ in error_cells}
    assert examples, "expected at least one labeled example for this leakage check to be meaningful"
    for dirty_val, _clean_val in examples:
        row = dirty_to_row[dirty_val]
        assert (row, "date") in calib_set, "leaked a test-split cell as a labeled example"


def test_load_cache_drops_noop_candidate(monkeypatch):
    # Oracle detection means every cached cell is a known error - a tier
    # candidate that just echoes the dirty value back can never be a repair
    # and must be dropped by load_cache, not merely down-weighted.
    rec = {
        "row": 0, "col": "c", "dirty": "wrongish", "clean": "FIXED", "error_type": "syntactic",
        "group": None,
        "tiers": {
            "rules": {"value": "wrongish", "raw_conf": 1.0, "cost": ZERO_COST},   # no-op -> must be dropped
            "stats": {"value": "FIXED", "raw_conf": 0.6, "cost": ZERO_COST},       # real candidate -> untouched
            "column": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
            "llm": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
        },
    }
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "fake_cache.jsonl"
        path.write_text(json.dumps(rec) + "\n")
        monkeypatch.setattr(pl, "_cache_path", lambda name: path)
        cells = pl.load_cache("fake")

    tiers = cells[0]["tiers"]
    assert tiers["rules"]["value"] is None, "no-op candidate (== dirty) must be dropped"
    assert tiers["rules"]["raw_conf"] == 0.0
    assert tiers["stats"]["value"] == "FIXED", "a genuine candidate must survive untouched"
    assert tiers["stats"]["raw_conf"] == 0.6


def test_calibrator_fallback_on_single_class_labels():
    # Every calibration example for "rules" is wrong despite raw_conf 1.0 ->
    # single-class labels, isotonic regression can't fit. The fallback must
    # be the Laplace-smoothed accuracy (0.2 here), never a silent pass-through of
    # raw_conf (which would keep looking like 1.0, i.e. maximally confident).
    calib_cells = [
        {"row": i, "col": "c", "dirty": "d", "clean": "CLEAN", "error_type": "syntactic",
         "tiers": {
             "rules": {"value": "WRONG", "raw_conf": 1.0, "cost": ZERO_COST},
             "fd": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
             "stats": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
             "column": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
             "llm": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
             "llm_sem": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
         }}
        for i in range(3)
    ]
    calibrators = pl.fit_calibrators(calib_cells)
    assert calibrators["rules"] == 1 / 5, f"expected Laplace constant (0+1)/(3+2), got {calibrators['rules']!r}"

    # route("ours") must not accept the high-raw_conf-but-miscalibrated rules
    # candidate - it should escalate past the tier instead of repairing wrong.
    cell = {
        "row": 0, "col": "c", "dirty": "d", "clean": "CLEAN", "error_type": "syntactic",
        "tiers": {
            "rules": {"value": "WRONG", "raw_conf": 1.0, "cost": ZERO_COST},
            "fd": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
            "stats": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
            "column": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
            "llm": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
            "llm_sem": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
        },
    }
    result = pl.route([cell], "ours", 0.8, calibrators)
    assert result["repairs"] == 0, "calibrated conf 0.0 < tau must not be accepted at the rules tier"


def test_column_group_keeps_only_verified_programs(monkeypatch):
    dirty_df, clean_df, error_cells = _rayyan_like_group()
    pattern = pl._generalize(dirty_df.at[0, "date"].strip())
    examples = [(dirty_df.at[r, "date"], clean_df.at[r, "date"]) for r in range(3)]
    good = "def fix(s):\n    y, m, d = s.split('/')\n    return f\"{m}/{d}/{int(y):02d}\"\n"
    bad1 = "def fix(s):\n    return s\n"                     # echoes the dirty value
    bad2 = "def fix(s):\n    y, m, d = s.split('/')\n    return f\"{d}/{m}/{y}\"\n"  # swaps day/month
    monkeypatch.setattr(pl, "_dsl_search", lambda examples, plaus=None: None)
    monkeypatch.setattr(pl, "_column_llm_sample", lambda prompt, k=5, temperature=0.7: (
        [bad1, bad2, good], {"calls": 1, "prompt_tokens": 5, "completion_tokens": 5, "seconds": 0.01}))

    rec = pl._get_or_make_column_group({}, io.StringIO(), "date", pattern, examples, [], [])
    assert rec["programs"] == [good], "only the program reproducing ALL examples is kept"
    assert (rec["n_sampled"], rec["n_verified"], rec["retry"]) == (3, 1, False)
    value, conf, _cost = pl.tier_column(dirty_df, 5, "date", rec)
    assert value == clean_df.at[5, "date"] and abs(conf - 4 / 5) < 1e-9, (value, conf)
    assert rec["all_programs"] == [bad1, bad2, good], "unverified programs are stored too"


def test_route_treats_llm_sem_as_the_last_paid_tier():
    cost = lambda n: {"calls": n, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
    none = {"value": None, "raw_conf": 0.0, "cost": ZERO_COST}
    mk = lambda i: {"row": i, "col": "s", "dirty": "x", "clean": "y", "error_type": "semantic", "group": None,
                    "tiers": {**{t: none for t in pl.TIER_ORDER}, "llm": {"value": None, "raw_conf": 0.0, "cost": cost(1)},
                              "llm_sem": {"value": "y", "raw_conf": 0.9, "cost": cost(1)}}}  # llm only echoed (dropped), llm_sem knows
    cells = [mk(i) for i in range(20)]
    cals = pl.fit_calibrators(cells)
    assert pl.TIER_ORDER[-1] == "llm_sem" and cals["_prior"]["llm_sem"] == 21 / 22
    for m in ("ours_auto", "cascade", "static_gate", "ours_lambda@0.05"):
        r = pl.route(cells, m, 0.8, cals)
        # llm (prior 0.5, no candidate) is skipped by the gate/static_gate; the ungated cascade and the budget router (0.5 - 0.05 > 0) still pay it
        assert (r["f1"], r["llm_calls"]) == (1.0, 40 if m in ("cascade", "ours_lambda@0.05") else 20), (m, r)
    assert pl.route(cells, "ours_lambda@0.6", 0.8, cals)["llm_calls"] == 0, "0.5 - 0.6 < 0: the llm is not worth a call, llm_sem is unreachable"
    assert pl.route(cells, "llm_only", None, cals)["f1"] == 0.0, "llm_only never reaches llm_sem"
    assert pl.route(cells, "ours_no_llm", 0.8, cals)["llm_calls"] == 0


def test_route_skips_tier_that_can_never_reach_tau():
    # constant calibrator 0.1 < tau 0.8: the llm tier could never be accepted, so it
    # must not be called (no cost) even though it holds a candidate with raw_conf 1.0.
    cell = {
        "row": 0, "col": "c", "dirty": "d", "clean": "CLEAN", "error_type": "semantic",
        "tiers": {
            "rules": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
            "fd": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
            "stats": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
            "column": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
            "llm": {"value": "CLEAN", "raw_conf": 1.0,
                    "cost": {"calls": 3, "prompt_tokens": 1, "completion_tokens": 1, "seconds": 0.1}},
            "llm_sem": {"value": None, "raw_conf": 0.0, "cost": ZERO_COST},
        },
    }
    r = pl.route([cell], "ours", 0.8, {"llm": 0.1})
    assert r["llm_calls"] == 0 and r["repairs"] == 0, r
    r = pl.route([cell], "ours", 0.8, {})  # no calibrator for the tier -> not skipped
    assert r["llm_calls"] == 3 and r["correct"] == 1, r


def test_group_score_leave_one_out_and_column_cap():
    cleans = ["a", "b", "c"]
    good, bad_on_2 = ["a", "b", "c"], ["a", "b", "X"]
    # i=2: the programs passing examples 0,1 are good+bad_on_2+good -> majority "c" (right)
    assert abs(pl._group_score([good, bad_on_2, good], cleans) - 4 / 5) < 1e-9
    # only bad_on_2 exists: on i=2 it passes the others and votes "X" (wrong); i=0,1 have no voter
    assert abs(pl._group_score([bad_on_2], cleans) - 1 / 5) < 1e-9
    assert pl._group_score([good], []) == 0.5, "no examples -> prior 0.5"
    # no [:5] cap any more: 20 labeled cells -> 15 examples (prompt-length cap)
    df = pd.DataFrame({"c": [f"{i}x" for i in range(20)]})
    cells = [(i, "c") for i in range(20)]
    ex, _, _ = pl._column_prompt_inputs(df, df, "c", cells, set(cells), set(cells))
    assert len(ex) == 15


def test_column_calibrator_is_identity():
    cell = {"row": 0, "col": "c", "dirty": "d", "clean": "CLEAN", "error_type": "systematic",
            "tiers": {t: {"value": "CLEAN", "raw_conf": 0.7, "cost": ZERO_COST} for t in pl.TIER_ORDER}}
    cals = pl.fit_calibrators([cell] * 3)
    assert cals["column"] is None and pl._apply_cal(cals["column"], 0.7) == 0.7
    assert pl._apply_cal(None, 1.3) == 1.0, "clipped to [0,1]"


def _cal_cells(n, n_wrong):
    """n stats-tier cells, raw_conf spread over [0.6, 1.0]; the n_wrong lowest-raw ones are wrong."""
    return [{"row": i, "col": "c", "dirty": "d", "clean": "CLEAN", "error_type": "statistical",
             "tiers": {**{t: {"value": None, "raw_conf": 0.0, "cost": ZERO_COST} for t in pl.TIER_ORDER},
                       "stats": {"value": "WRONG" if i < n_wrong else "CLEAN", "raw_conf": 0.6 + 0.4 * i / (n - 1),
                                 "cost": ZERO_COST}}} for i in range(n)]


def test_calibrators_platt_small_n_isotonic_large_n():
    small = pl.fit_calibrators(_cal_cells(20, 4))["stats"]
    assert hasattr(small, "predict_proba"), "n<30 with both labels -> Platt"
    assert pl._apply_cal(small, 0.6) < pl._apply_cal(small, 1.0), "Platt is increasing"
    assert 0.5 < pl._apply_cal(small, 1.0) < 1.0
    big = pl.fit_calibrators(_cal_cells(40, 8))["stats"]
    assert isinstance(big, pl.IsotonicRegression), "n>=30 with both labels -> isotonic"


def test_auto_tau_uses_calib_cells_only_and_prefers_higher_tau_on_ties():
    def cell(v):
        return {"row": 0, "col": "c", "dirty": "d", "clean": "CLEAN", "error_type": "syntactic",
                "tiers": {**{t: {"value": None, "raw_conf": 0.0, "cost": ZERO_COST} for t in pl.TIER_ORDER},
                          "rules": {"value": v, "raw_conf": 1.0, "cost": ZERO_COST}}}
    calib = [cell("CLEAN"), cell("CLEAN"), cell("WRONG")]
    # rules calibrated to 0.6: accepted at tau<=0.6 (F1 > 0), rejected above (F1 0) -> best tau 0.6
    assert pl.auto_tau_insample(calib, {"rules": 0.6}) == 0.6
    assert pl.route(calib, "ours_auto", 0.6, {"rules": 0.6})["correct"] == 2
    # calibrated 0.99: accepted everywhere, all ties -> highest tau in the grid
    assert pl.auto_tau_insample(calib, {"rules": 0.99}) == 0.95


def test_auto_tau_is_cross_fitted_over_rows():
    def cell(row):
        return {"row": row, "col": "c", "dirty": "d", "clean": "CLEAN", "error_type": "syntactic",
                "tiers": {**{t: {"value": None, "raw_conf": 0.0, "cost": ZERO_COST} for t in pl.TIER_ORDER},
                          "rules": {"value": "CLEAN", "raw_conf": 1.0, "cost": ZERO_COST}}}
    calib = [cell(r) for r in range(8)]  # 2 folds x 4 rows, all correct
    # each fold is routed by a calibrator fit on only 4 correct cells: constant (4+1)/(4+2) = 0.83 -> tau 0.8;
    # in-sample the same calibrator sees 8 cells: (8+1)/(8+2) = 0.9 -> tau 0.9
    assert pl.auto_tau(calib) == 0.8
    assert pl.auto_tau_insample(calib, pl.fit_calibrators(calib)) == 0.9
    # all cells in one row -> one empty fold, the other trains on nothing: constant 0.5 -> only tau 0.5 accepts
    assert pl.auto_tau([cell(0)] * 4) == 0.5


def _no_llm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("LLM must not be called when the DSL finds a program")
    monkeypatch.setattr(pl, "_column_llm_sample", boom)


def test_dsl_solves_rayyan_like_dates_without_llm(monkeypatch):
    # Y/M/D -> M/D/YY with a 2-digit year: the padding detail the 8B model keeps getting wrong
    dirty_df, clean_df, error_cells = _rayyan_like_group()
    pattern = pl._generalize(dirty_df.at[0, "date"].strip())
    examples = [(dirty_df.at[r, "date"], clean_df.at[r, "date"]) for r in range(4)]
    _no_llm(monkeypatch)
    rec = pl._get_or_make_column_group({}, io.StringIO(), "date", pattern, examples, [], [])
    assert rec["source"] == "dsl" and rec["cost"]["calls"] == 0
    assert abs(rec["group_score"] - 5 / 6) < 1e-9, "4 LOO-correct examples -> (4+1)/(4+2)"
    for r in range(4, 12):
        value, conf, _ = pl.tier_column(dirty_df, r, "date", rec)
        assert value == clean_df.at[r, "date"] and abs(conf - 5 / 6) < 1e-9, (value, conf)


def test_dsl_solves_beers_like_ounces_and_needs_two_examples(monkeypatch):
    examples = [("12.0 oz", "12"), ("16.0 oz", "16"), ("8.0 oz", "8")]
    _no_llm(monkeypatch)
    rec = pl._get_or_make_column_group({}, io.StringIO(), "ounces", "99.9 aa", examples, [], [])
    assert rec["source"] == "dsl" and rec["programs"][0]["fmt"] == ["num"], "simplest program: one number token, canonical float"
    assert pl._dsl_apply(rec["programs"][0], "24.0 oz") == "24"
    assert pl._dsl_search(examples[:1]) is None, "needs >= 2 labeled examples"
    assert pl._dsl_search([("a-b", "zz"), ("c-d", "yy")]) is None, "nothing consistent -> fall back to LLM"


def test_dsl_richer_formats_slices_and_column_transfer(monkeypatch):
    _no_llm(monkeypatch)
    dsl = lambda ex: pl._dsl_search(ex)[0]
    assert pl._dsl_apply(dsl([("0.082499%", "0.082"), ("0.05%", "0.05")]), "0.0819999%") == "0.082"  # round3
    assert pl._dsl_apply(dsl([("71-7", "Jul-71"), ("3191-7", "Jul-91")]), "5-12") == "Dec-05"  # mon + last2
    ex = [("Edwards CO", "Edwards"), ("Salt Lake City UT", "Salt Lake City")]
    assert pl._dsl_apply(dsl(ex), "Fort Collins CO") == "Fort Collins"  # all-but-last, original delimiters
    assert pl._dsl_tokens("12.0 oz.") == (["12.0", "oz"], [" "])
    # examples cannot tell as-is from round3; the column's non-error values are 3-decimal -> round3 wins
    ex = [("0.045%", "0.045"), ("0.05%", "0.05")]
    plaus = ({pl._generalize("0.082")}, ["0.08199999999999999%"])
    assert pl._dsl_apply(pl._dsl_search(ex)[0], "0.08199999999999999%") == "0.08199999999999999"
    assert pl._dsl_apply(pl._dsl_search(ex, plaus)[0], "0.08199999999999999%") == "0.082"
    # column transfer: group "99.9 aaaaa" has 1 own label; the union over the column's groups pins the program
    own = [("12.0 ounce", "12")]
    union = own + [("16.0 oz", "16"), ("8.0 oz", "8"), ("12.0 oz.", "12")]
    rec = pl._get_or_make_column_group({}, io.StringIO(), "ounces", "99.9 aaaaa", own, [], [], union)
    assert rec["source"] == "dsl_column" and rec["cost"]["calls"] == 0
    assert abs(rec["group_score"] - 5 / 6) < 1e-9, "LOO Laplace over the 4 union examples"
    assert pl.tier_column(pd.DataFrame({"ounces": ["19.2 ounce"]}), 0, "ounces", rec)[0] == "19.2"
    # no examples at all in the column -> nothing to transfer, falls to the LLM path (stubbed here)
    monkeypatch.setattr(pl, "_column_llm_sample", lambda *a, **k: ([], ZERO_COST))
    assert pl._get_or_make_column_group({}, io.StringIO(), "c", "p", [], [], [], [])["source"] == "llm"


def test_charsub_learns_mojibake_rules_from_labeled_pairs_and_needs_loo_score():
    pairs = [('{"Akin Uz\ufffd_mc\ufffd_gil","Bo Li"}', '{"Akin Uz_mc_gil","Bo Li"}'),
             ("Dani\ufffdl", "Danil"), ("Pawe\ufffdska", "Paweska")]
    rules, score = pl._charsub_program(pairs)
    assert rules == {"\ufffd": ""}, "one rule; plain 1-example rules (like 'l') would be dropped"
    assert pl._apply_rules(rules, "Uz\ufffd_mc") == "Uz_mc"
    assert abs(score - 4 / 5) < 1e-9, "every pair is reproduced from the other two -> (3+1)/(3+2)"
    # _column_fields: a non-systematic cell gets the substitution only when the column's LOO score passes 0.5
    d = pd.DataFrame({"a": ["x\ufffdy", "p\ufffdq", "m\ufffdn", "r\ufffds"]})
    c = pd.DataFrame({"a": ["xy", "pq", "mn", "rs"]})
    calib = {(0, "a"), (1, "a"), (2, "a")}
    fields = lambda: pl._column_fields(d, c, 3, "a", "statistical", {}, calib, calib, {}, io.StringIO(), charsub={})
    assert fields()[1:3] == ("rs", 4 / 5)
    c.at[0, "a"] = c.at[1, "a"] = "zzz"  # labels no longer follow one rule -> low LOO score -> abstain
    assert fields()[1] is None


class _MonkeyPatch:
    """Tiny stand-in for pytest's monkeypatch fixture (attr set + auto restore)."""

    def __init__(self):
        self._saved = []

    def setattr(self, obj, name, value):
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, value in reversed(self._saved):
            setattr(obj, name, value)


def test_inject_semantic_corrupts_whole_entities_with_unambiguous_unique_city():
    rows = [("e1", "Bend", "OR"), ("e1", "Bend", "OR"), ("e2", "Salem", "OR"), ("e3", "Reno", "NV"), ("e3", "Reno", "NV"),
            ("e4", "Boise", "ID"), ("e5", "Boise", "ID"), ("e6", "Ely", "NV"), ("e7", "Ely", "NV")]
    clean = pd.DataFrame(rows, columns=["eid", "city", "state"])
    dirty = clean.copy()
    dirty.at[3, "state"] = "xx"  # e3 has an existing error: never touched
    out, log = pl.inject_semantic(dirty, clean, "eid", n_state=99, seed=1)
    hit = {x["entity"] for x in log}
    assert hit == {"e1"}, "Salem ambiguous, e3 already dirty, e4/e5 and e6/e7 share a city"
    assert {(x["row"], x["col"]) for x in log} == {(r, "state") for r in clean.index[clean.eid.isin(hit)]}
    for x in log:  # whole entity, a different state that exists in the table
        assert out.at[x["row"], "state"] != clean.at[x["row"], "state"] and out.at[x["row"], "state"] in {"OR", "NV", "ID"}
    assert len(set(out.state[clean.eid == "e1"])) == 1
    assert (out.drop(index=[x["row"] for x in log]) == dirty.drop(index=[x["row"] for x in log])).all().all()
    assert pl.inject_semantic(dirty, clean, "eid", n_state=99, seed=1)[1] == log, "fixed seed"


def test_semantic_flag_cached_and_label_exclusion_avoids_injected_rows(monkeypatch):
    n = 40
    clean = pd.DataFrame({"k": [f"k{i}" for i in range(n)], "v": ["y"] * n})
    dirty = clean.copy()
    dirty["v"] = "x"  # every row has an error
    sem = [{"row": i, "col": "v", "type": "v", "entity": "e"} for i in range(0, n, 2)]
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "toy_sem").mkdir()
        dirty.to_csv(Path(d) / "toy_sem" / "dirty.csv", index=False)
        clean.to_csv(Path(d) / "toy_sem" / "clean.csv", index=False)
        (Path(d) / "toy_sem" / "semantic_cells.json").write_text(json.dumps(sem))
        monkeypatch.setattr(pl, "DATA_DIR", Path(d))
        monkeypatch.setattr(pl, "propose_fds", lambda name, df: ([], ZERO_COST))
        monkeypatch.setattr(pl, "propose_source", lambda name, df: (None, ZERO_COST))
        monkeypatch.setattr(pl, "tier_llm", lambda df, r, c, **kw: ("y", 0.9, ZERO_COST))
        monkeypatch.setattr(pl, "_cache_path", lambda name: Path(d) / "c.jsonl")
        monkeypatch.setattr(pl, "_column_cache_path", lambda name: Path(d) / "col.jsonl")
        pl.precompute("toy_sem")
        cells = pl.load_cache("toy_sem")
        assert {c["row"] for c in cells if c["semantic"]} == {i for i in range(0, n, 2)}
        assert any(c["semantic"] for c in pl.calib_test_split(cells)[0]), "default mixes injected rows in"
        monkeypatch.setattr(pl, "LABEL_EXCLUDE_SEM", True)
        calib, test = pl.calib_test_split(cells)
        assert len(calib) == pl.N_LABEL_TUPLES and not any(c["semantic"] for c in calib)
        assert not any(r % 2 == 0 for r, _ in pl.calib_test_split([(r, "v") for r in range(n)])[0]), "tuples too"


def test_route_reports_semantic_split_and_call_share():
    cost = lambda n: {"calls": n, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
    none = {"value": None, "raw_conf": 0.0, "cost": ZERO_COST}
    mk = lambda i, sem, llm_val: {"row": i, "col": "s", "dirty": "x", "clean": "y", "error_type": "semantic", "group": None,
                                  "semantic": sem, "tiers": {**{t: none for t in pl.TIER_ORDER},
                                                             "llm": {"value": llm_val, "raw_conf": 1.0, "cost": cost(1)}}}
    cells = [mk(0, True, "y"), mk(1, True, "z"), mk(2, False, "y"), mk(3, False, "y")]
    r = pl.route(cells, "llm_only", None, {})
    assert (r["sem_recall"], r["sem_f1"]) == (0.5, 0.5) and r["nonsem_f1"] == 1.0
    assert r["sem_call_share"] == 0.5 and r["llm_calls"] == 4 and r["f1"] == 0.75
    assert "sem_f1" not in pl.route([{k: v for k, v in c.items() if k != "semantic"} for c in cells], "llm_only", None, {})


def _lambda_cells(n=20):
    """20 cells: a cheap-but-unsure `stats` guess (right 3 of 4) and a per-cell llm (1 call, always right)."""
    cost = lambda n: {"calls": n, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
    none = {"value": None, "raw_conf": 0.0, "cost": ZERO_COST}
    return [{"row": i, "col": "s", "dirty": "x", "clean": "y", "error_type": "semantic", "group": None,
             "tiers": {**{t: none for t in pl.TIER_ORDER},
                       "stats": {"value": "y" if i % 4 else "z", "raw_conf": 0.7, "cost": ZERO_COST},
                       "llm": {"value": "y", "raw_conf": 0.95, "cost": cost(1)}}} for i in range(n)]


def test_ours_lambda_is_a_per_cell_budget_router_and_calls_are_monotone_in_lambda():
    cells = _lambda_cells(40)  # stats: calibrated 0.75, llm: prior 41/42; the llm beats the free guess iff 41/42 - lambda > 0.75
    cals = pl.fit_calibrators(cells)
    res = {l: pl.route(cells, f"ours_lambda@{l}", pl.auto_tau(cells, method=f"ours_lambda@{l}"), cals) for l in pl.LAMBDAS}
    assert pl.LAMBDAS[0] == "-inf" and pl.LAMBDAS[-1] == "inf" and float("-inf") < float(pl.LAMBDAS[1])
    assert res["-inf"]["llm_calls"] == 40 and res["-inf"]["f1"] == 1.0, "-inf: always buy every applicable tier"
    assert res["0.2"]["llm_calls"] == 40 and res["0.3"]["llm_calls"] == 0, "buys iff 0.976 - lambda > 0.75"
    assert res["inf"]["llm_calls"] == 0 and res["inf"]["f1"] < 1.0, "+inf: no LLM calls"
    calls = [res[l]["llm_calls"] for l in pl.LAMBDAS]
    assert calls == sorted(calls, reverse=True), f"calls never grow with lambda: {calls}"
    # a column group amortizes its program over the group: 2 calls / 4 cells = 0.5 per cell, so it is bought for lambda < 0.5 * ... only if q_col - lam*0.5 > p_best
    z = lambda **k: {**ZERO_COST, **k}
    none = {"value": None, "raw_conf": 0.0, "cost": ZERO_COST}
    mk = lambda i: {"row": i, "col": "s", "dirty": "x", "clean": "y", "error_type": "systematic", "group": ["s", "9"], "group_size": 4,
                    "tiers": {**{t: none for t in pl.TIER_ORDER}, "column": {"value": "y", "raw_conf": 1.0, "cost": z(calls=2)},
                              "llm": {"value": "y", "raw_conf": 1.0, "cost": z(calls=1)}}}
    gc = [mk(i) for i in range(4)]
    gcals = pl.fit_calibrators(gc)
    q = gcals["_prior"]["column"]  # 5/6
    assert abs(q - 5 / 6) < 1e-9
    r = pl.route(gc, "ours_lambda@0.5", 0.5, gcals)  # column: 5/6 - 0.5*0.5 > 0 -> bought; llm: 5/6 - 0.5 = 0.33 < p_best(1.0) -> not
    assert r["llm_calls"] == 2 and r["f1"] == 1.0, r  # the group's 2 calls, once
    assert pl.route(gc, "ours_lambda@2", 0.5, gcals)["llm_calls"] == 0, "5/6 - 2*0.5 < 0"
    assert pl.route(gc, "ours_lambda@inf", 0.5, gcals)["llm_calls"] == 0 and pl.route(gc, "ours_lambda@-inf", 0.5, gcals)["llm_calls"] == 6
    dsl = [{**c, "tiers": {**c["tiers"], "column": {"value": "y", "raw_conf": 1.0, "cost": z(calls=0)}}} for c in gc]  # DSL program: 0 calls
    r = pl.route(dsl, "ours_lambda@inf", 0.5, pl.fit_calibrators(dsl))
    assert r["llm_calls"] == 0 and r["f1"] == 1.0, "+inf: a 0-cost column program is still used"


def test_ours_budget_picks_lambda_by_cross_fitted_f1_and_ties_go_to_the_cheaper_lambda():
    cells = _lambda_cells(40)
    lam, tau = pl.auto_budget(cells)
    f1 = {l: pl._cv_tau(cells, 0, f"ours_lambda@{l}")[1] for l in pl.LAMBDAS}  # lambdas that still buy the llm tie at the top
    best = max(f1.values())
    assert f1[lam] == best and all(f1[l] < best for l in pl.LAMBDAS if float(l) > float(lam)) and float(lam) >= 0.1, (lam, f1)
    assert [l for l in pl.LAMBDAS if f1[l] == best][-1] == lam and tau in pl.TAU_GRID
    cals = pl.fit_calibrators(cells)
    taus = pl.method_taus(cells, cals)
    assert cals["_lambda"] == lam and taus["ours_budget"] == tau
    a, b = pl.route(cells, "ours_budget", tau, cals), pl.route(cells, f"ours_lambda@{lam}", tau, cals)
    assert {k: v for k, v in a.items() if k != "method"} == {k: v for k, v in b.items() if k != "method"} and a["method"] == "ours_budget"


def test_system_level_calibration_is_over_repaired_cells_with_accepted_confidence():
    none = {"value": None, "raw_conf": 0.0, "cost": ZERO_COST}
    mk = lambda i, v, c: {"row": i, "col": "s", "dirty": "x", "clean": "y", "error_type": "semantic", "group": None,
                          "tiers": {**{t: none for t in pl.TIER_ORDER}, "llm": {"value": v, "raw_conf": c, "cost": ZERO_COST}}}
    cells = [mk(0, "y", 0.9), mk(1, "z", 0.9), mk(2, "y", 0.6), mk(3, "y", 0.6), mk(4, None, 0.0)]  # the last cell is never repaired
    r = pl.route(cells, "llm_only", None, {})
    assert r["sys_n"] == 4 and abs(r["sys_ece"] - 0.4) < 1e-9 and abs(r["sys_brier"] - (0.01 + 0.81 + 0.16 + 0.16) / 4) < 1e-9
    assert (r["sys_ece_raw"], r["sys_brier_raw"]) == (r["sys_ece"], r["sys_brier"]), "llm_only does not calibrate: calibrated = raw"
    r = pl.route(cells, "ours", 0.5, {"llm": 0.7})  # calibrated conf 0.7 for every accepted candidate, 3 of 4 right
    assert r["sys_n"] == 4 and abs(r["sys_ece"] - 0.05) < 1e-9 and abs(r["sys_brier"] - (3 * 0.09 + 0.49) / 4) < 1e-9
    assert abs(r["sys_ece_raw"] - 0.4) < 1e-9 and abs(r["sys_brier_raw"] - 0.285) < 1e-9, "raw = the accepted tier's raw confidence"
    assert pl.route(cells, "rules_stats", None, {})["sys_n"] == 0


def test_ablation_methods_static_gate_no_escalation_raw_threshold_and_operator_subsets():
    cells = _lambda_cells()  # no free-tier value in _lambda_cells: rebuild with a rules tier right on half the cells
    for i, c in enumerate(cells):
        c["tiers"]["stats"] = {"value": None, "raw_conf": 0.0, "cost": ZERO_COST}
        c["tiers"]["rules"] = {"value": "y" if i < 10 else "z", "raw_conf": 1.0 if i < 10 else 0.3, "cost": ZERO_COST}
    cals = pl.fit_calibrators(cells)
    assert cals["_prior"]["rules"] == 0.5  # < STATIC_GATE_MIN
    r = {m: pl.route(cells, m, 0.8, cals) for m in pl.METHODS if m not in ("ours_auto_insample", "ours_budget")}
    assert (r["static_gate"]["llm_calls"], r["static_gate"]["f1"]) == (20, 1.0), "rules gated out for good, llm kept"
    assert (r["cascade"]["llm_calls"], r["cascade"]["f1"]) == (0, 0.5) and r["no_escalation"]["f1"] == 0.5
    assert r["ours_rules_llm"]["llm_calls"] == 10 and r["ours_rules_llm"]["f1"] == 1.0, "confidence gate escalates the bad half"
    assert r["ours_stats_llm"]["llm_calls"] == 20, "no rules tier: everything goes to the llm"
    assert r["ours_no_llm"]["llm_calls"] == 0 and r["ours_no_llm"]["recall"] == 0.5
    assert r["raw_threshold"]["method"] == "raw_threshold"
    assert {k: v for k, v in r["raw_threshold"].items() if k != "method"} == \
        {k: v for k, v in pl.route(cells, "ours_no_calib", 0.8, cals).items() if k != "method"}


def test_brier_after_calibration_and_latency_per_repair():
    cells = _lambda_cells()
    for c in cells:
        c["tiers"]["llm"]["cost"] = {"calls": 1, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 2.0}
    cals = pl.fit_calibrators(cells)
    rep = pl.ece_report(cells, cals)
    c = pl._apply_cal(cals["stats"], 0.7)  # one raw value: calibration is a constant, 15 of 20 right
    assert abs(rep["stats"]["brier_after"] - (15 * (1 - c) ** 2 + 5 * c ** 2) / 20) < 1e-9
    assert abs(rep["stats"]["brier_before"] - (0.3 ** 2 * 15 + 0.7 ** 2 * 5) / 20) < 1e-9
    assert abs(rep["llm"]["brier_after"] - (1 / 22) ** 2) < 1e-9, "all right: Laplace constant 21/22"
    r = pl.route(cells, "llm_only", None, cals)
    assert r["llm_sec_per_repair"] == 2.0 and r["llm_seconds"] == 40.0


def test_route_reports_latency_call_rate_and_repair_accuracy():
    z = lambda **k: {**ZERO_COST, **k}
    none = {"value": None, "raw_conf": 0.0, "cost": ZERO_COST}
    mk = lambda i: {"row": i, "col": "s", "dirty": "x", "clean": "y", "error_type": "systematic", "group": ["s", "9"],
                    "tiers": {**{t: none for t in pl.TIER_ORDER},
                              "rules": {"value": None, "raw_conf": 0.0, "cost": z(seconds=0.5)},
                              "column": {"value": "y", "raw_conf": 1.0, "cost": z(calls=2, seconds=3.0, wall_seconds=4.0, cell_seconds=0.25)},
                              "llm": {"value": "z", "raw_conf": 1.0, "cost": z(calls=1, seconds=2.0, prep_seconds=0.1)}}}
    cells = [mk(i) for i in range(4)]
    r = pl.route(cells, "cascade", None, {})  # rules (none) -> fd -> stats -> column accepts: llm never reached
    assert r["llm_calls"] == 2 and abs(r["llm_call_rate"] - 0.5) < 1e-9 and r["repair_accuracy"] == r["recall"] == 1.0
    assert abs(r["latency_seconds"] - (4 * (0.5 + 0.25) + 4.0)) < 1e-9, "group program time is paid once, per-cell apply time each cell"
    assert abs(r["latency_per_cell"] * 4 - r["latency_seconds"]) < 1e-9 and r["llm_seconds"] == 3.0
    r = pl.route(cells, "llm_only", None, {})
    assert abs(r["latency_seconds"] - 4 * 2.1) < 1e-9 and r["llm_seconds"] == 8.0, "llm tier: request seconds + prompt prep"


def test_eval_writes_lambda_sweep_and_lambda_figure(monkeypatch):
    import run
    monkeypatch.setattr(pl, "load_cache", lambda name: _lambda_cells(60))
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(run, "RESULTS_DIR", Path(d))
        monkeypatch.setattr(run, "SUMMARY_DIR", Path(d))
        monkeypatch.setattr(pl, "BASE_DIR", Path(d))
        run._eval_one("toy")
        rows = list(__import__("csv").DictReader(open(Path(d) / "toy_lambda.csv")))
        assert [r["lambda"] for r in rows] == pl.LAMBDAS and {"chosen", "tau", "precision", "recall", "f1", "llm_calls", "total_calls", "llm_call_rate",
                                                              "prompt_tokens", "completion_tokens", "tokens", "llm_seconds", "latency_per_cell"} <= set(rows[0])
        assert int(rows[-1]["llm_calls"]) == 0 < int(rows[0]["llm_calls"]), "lambda=inf makes no LLM call, lambda=-inf always does"
        assert [r["lambda"] for r in rows if r["chosen"] == "1"] == ["0.2"]
        m = list(__import__("csv").DictReader(open(Path(d) / "toy_methods.csv")))[0]
        assert {"repair_accuracy", "llm_call_rate", "total_calls", "tokens", "llm_seconds", "latency_seconds", "latency_per_cell",
                "ece_raw_llm", "ece_cal_llm", "brier_raw_stats", "brier_cal_stats"} <= set(m), sorted(m)
        import csv as _csv
        with open(Path(d) / "budget_sweep.csv", "w", newline="") as f:  # toy sweep summary: the figure reads these
            w = _csv.DictWriter(f, fieldnames=["dataset", "method", "budget", "llm_call_rate_mean", "f1_mean"])
            w.writeheader()
            for meth in ["llm_only", "ours_auto", "static_gate", "ours_budget", *[f"ours_lambda@{l}" for l in pl.LAMBDAS], *[f"ours_sem_guard@{l}" for l in pl.LAMBDAS]]:  # the figure plots the guard family as Ours
                w.writerow({"dataset": "hospital", "method": meth, "budget": 20, "llm_call_rate_mean": 0.1, "f1_mean": 0.5})
        run.cmd_lambda_figure(None)
        assert (Path(d) / "paper" / "figures" / "lambda_budget.png").stat().st_size > 0


def test_coverage_sampler_includes_unresolved_rows_same_n_deterministic_and_clean_free(monkeypatch):
    n, unres = 200, {7, 91, 130}  # 3 rows, 2 unresolved cells each (>= Q_UNRESOLVED cells needs all of them)
    mk = lambda clean: [{"row": r, "col": c, "clean": clean(r), "unresolved": r in unres and c in "ab"} for r in range(n) for c in "abc"]
    monkeypatch.setattr(pl, "SAMPLER", "coverage")
    calib, test = pl.calib_test_split(mk(lambda r: "z"), seed=3, n_tuples=20)
    assert len({c["row"] for c in calib}) == 20 and len(calib) + len(test) == 3 * n
    assert unres <= {c["row"] for c in calib}, "unresolved rows are represented"
    again = pl.calib_test_split(mk(lambda r: f"poison{r}"), seed=3, n_tuples=20)[0]  # other clean values: same rows
    assert [(c["row"], c["col"]) for c in calib] == [(c["row"], c["col"]) for c in again]
    assert {c["row"] for c in calib} != {c["row"] for c in pl.calib_test_split(mk(lambda r: "z"), seed=4, n_tuples=20)[0]}
    assert len(pl.calib_test_split(mk(lambda r: "z"), seed=3, n_tuples=6)[0]) == 6 * 3, "cap n/2 rows, then filled to N"
    monkeypatch.setattr(pl, "_UNRESOLVED", {(7, "a")})
    assert 7 in {r for r, _ in pl.calib_test_split([(r, c) for r in range(n) for c in "abc"], 1, 20)[0]}, "tuples too"
    monkeypatch.setattr(pl, "SAMPLER", "random")
    assert {c["row"] for c in pl.calib_test_split(mk(lambda r: "z"), seed=3, n_tuples=20)[0]} == set(random.Random(3).sample(range(n), 20))


def test_unresolved_proxy_needs_no_stats_and_no_llm_answer_and_ignores_clean_and_fds(monkeypatch):
    df = pd.DataFrame({"k": ["a", "a", "a", "b", "c"], "v": ["x", "x", "x", "q", "y"], "w": ["1", "2", "3", "4", "5"]})
    rec = lambda r, c, llm: {"row": r, "col": c, "dirty": df.at[r, c], "clean": "POISON", "tiers": {"llm": {"value": llm}}}
    cells = [(0, "w"), (2, "v"), (3, "w"), (4, "w")]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "c.jsonl"  # (0,w): stats votes '2' -> resolved; (2,v): stats and llm echo 'x'; (3,w): nothing at all; (4,w): llm answers
        path.write_text("".join(json.dumps(rec(r, c, v)) + "\n" for (r, c), v in zip(cells, (None, "x", None, "z"))))
        monkeypatch.setattr(pl, "_cache_path", lambda name: path)
        monkeypatch.setattr(pl, "UNRESOLVED_DIR", Path(d))
        assert pl._unresolved_set("toy", df, cells) == {(2, "v"), (3, "w")}
        assert (Path(d) / "toy_unresolved.json").exists() and pl._unresolved_set("toy") == {(2, "v"), (3, "w")}


def test_refresh_llm_sem_no_ask_reuses_stored_answers_and_never_calls_the_llm(monkeypatch):
    df = pd.DataFrame({"v": ["x", "y"]})
    monkeypatch.setattr(pl, "load", lambda name: (df, df, [(0, "v"), (1, "v")]))
    none = {"value": None, "raw_conf": 0.0, "cost": ZERO_COST}
    recs = [{"row": r, "col": "v", "dirty": df.at[r, "v"], "clean": "z", "tiers": {"rules": none, "llm": none}} for r in (0, 1)]
    monkeypatch.setattr(pl, "tier_llm_sem", lambda *a, **kw: 1 / 0)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "c.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in recs))
        (Path(d) / "fake_llm_sem.jsonl").write_text(json.dumps({"row": 0, "col": "v", "value": "z", "raw_conf": 0.8, "cost": ZERO_COST}) + "\n")
        monkeypatch.setattr(pl, "_cache_path", lambda name: path)
        monkeypatch.setattr(pl, "LLM_SEM_DIR", Path(d))
        pl.refresh_llm_sem("fake", ask=False)
        new = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["llm_sem_eligible"] for r in new] == [True, True]
    assert new[0]["tiers"]["llm_sem"]["value"] == "z" and new[1]["tiers"]["llm_sem"]["value"] is None


def test_ours_sem_direct_tests_llm_sem_on_its_own_and_is_ours_lambda_otherwise():
    c = lambda n: {**ZERO_COST, "calls": n}
    none = {"value": None, "raw_conf": 0.0, "cost": ZERO_COST}
    # free tiers fail; the llm answers (wrongly) but would not be bought at lambda>=q_llm; llm_sem (a separate answer) knows the truth
    mk = lambda i: {"row": i, "col": "s", "dirty": "x", "clean": "y", "error_type": "semantic", "group": None,
                    "llm_sem_eligible": False, "llm_sem_direct_eligible": True,
                    "llm_sem_direct": {"value": "y", "raw_conf": 0.9, "cost": c(1)},
                    "tiers": {**{t: none for t in pl.TIER_ORDER}, "llm": {"value": "w", "raw_conf": 0.9, "cost": c(1)},
                              "llm_sem": none}}
    cells = [mk(i) for i in range(20)]
    # the labeled cells carry llm_sem answers too (prior 21/22 as in the old-eligible test); llm is always wrong (prior 1/22)
    prior = pl.fit_calibrators([{**x, "tiers": {**x["tiers"], "llm_sem": {"value": "y", "raw_conf": 0.9, "cost": c(1)}}} for x in cells])
    cals = {**pl.fit_calibrators(cells), "_prior": prior["_prior"]}
    assert cals["_prior"]["llm"] < 0.1 < cals["_prior"]["llm_sem"]
    r0, d0 = pl.route(cells, "ours_lambda@0.5", 0.5, cals), pl.route(cells, "ours_sem_direct@0.5", 0.5, cals)
    assert r0["llm_calls"] == 0 and r0["f1"] == 0.0, "old: llm not worth 0.5, so llm_sem is never asked"
    assert d0["llm_calls"] == 20 and d0["f1"] == 1.0, "direct: llm_sem alone is worth it (0.95 - 0.5 > 0), the llm is not bought"
    assert pl.route(cells, "ours_sem_direct@0.96", 0.5, cals)["llm_calls"] == 0, "21/22 - 0.96 < 0"
    for l in ("0", "-inf"):  # llm bought (q_llm > 0) and non-echo -> not eligible: identical to ours_lambda
        a, b = pl.route(cells, f"ours_lambda@{l}", 0.5, cals), pl.route(cells, f"ours_sem_direct@{l}", 0.5, cals)
        assert (a["f1"], a["llm_calls"]) == (b["f1"], b["llm_calls"]) == (0.0, 20), (l, a, b)
    # a non-echo free candidate makes the cell not direct-eligible
    ne = [{**x, "llm_sem_direct_eligible": False} for x in cells]
    assert pl.route(ne, "ours_sem_direct@0.5", 0.5, cals)["llm_calls"] == 0
    # the alias ours_sem_direct is lambda=0
    assert pl.route(cells, "ours_sem_direct", 0.5, cals)["llm_calls"] == pl.route(cells, "ours_sem_direct@0", 0.5, cals)["llm_calls"]


def test_ours_sem_guard_is_sem_direct_only_with_labeled_llm_sem_evidence():
    c = lambda n: {**ZERO_COST, "calls": n}
    none = {"value": None, "raw_conf": 0.0, "cost": ZERO_COST}
    mk = lambda i: {"row": i, "col": "s", "dirty": "x", "clean": "y", "error_type": "semantic", "group": None,
                    "llm_sem_eligible": False, "llm_sem_direct_eligible": True,
                    "llm_sem_direct": {"value": "y", "raw_conf": 0.9, "cost": c(1)},
                    "tiers": {**{t: none for t in pl.TIER_ORDER}, "llm": {"value": "w", "raw_conf": 0.9, "cost": c(1)}, "llm_sem": none}}
    cells = [mk(i) for i in range(20)]
    sem = lambda n: [{**x, "tiers": {**x["tiers"], "llm_sem": {"value": "y", "raw_conf": 0.9, "cost": c(1)}}} for x in cells[:n]]
    cals0 = pl.fit_calibrators(cells)  # no labeled llm_sem candidate: n_sem = 0, prior 0.5
    cals2 = {**cals0, "_prior": pl.fit_calibrators(sem(20))["_prior"], "_n": {**cals0["_n"], "llm_sem": 2}}  # evidence: prior 21/22, n_sem = 2
    assert cals0["_n"]["llm_sem"] == 0 and pl.fit_calibrators(sem(3))["_n"]["llm_sem"] == 3
    f = lambda r: (r["f1"], r["llm_calls"])
    for cals, direct in ((cals0, False), (cals2, True)):
        for l in ("-inf", "0", "0.2", "0.5", "inf"):
            g, o, d = (pl.route(cells, f"{m}@{l}", 0.5, cals) for m in ("ours_sem_guard", "ours_lambda", "ours_sem_direct"))
            assert f(g) == f(d if direct else o), (l, direct, f(g), f(o), f(d))
    assert f(pl.route(cells, "ours_sem_guard@0.5", 0.5, cals2)) == (1.0, 20), "evidence: llm_sem alone is bought"
    assert f(pl.route(cells, "ours_sem_guard@0.5", 0.5, cals0)) == (0.0, 0), "no evidence: same as ours_lambda"
    assert f(pl.route(cells, "ours_sem_guard", 0.5, cals2)) == f(pl.route(cells, "ours_sem_guard@0", 0.5, cals2)), "alias = lambda 0"
    pl.SEM_GUARD_K = 3  # n_sem = 2 < K
    try:
        assert f(pl.route(cells, "ours_sem_guard@0.5", 0.5, cals2)) == (0.0, 0)
    finally:
        pl.SEM_GUARD_K = 1


def test_refresh_llm_sem_direct_flag_and_missing_count(monkeypatch):
    df = pd.DataFrame({"v": ["x", "y", "z"]})
    monkeypatch.setattr(pl, "load", lambda name: (df, df, [(0, "v"), (1, "v"), (2, "v")]))
    none = {"value": None, "raw_conf": 0.0, "cost": ZERO_COST}
    cand = {"value": "q", "raw_conf": 0.9, "cost": ZERO_COST}
    recs = [{"row": 0, "col": "v", "dirty": "x", "clean": "q", "tiers": {"rules": none, "llm": none}},       # old-eligible
            {"row": 1, "col": "v", "dirty": "y", "clean": "q", "tiers": {"rules": none, "llm": cand}},       # direct only
            {"row": 2, "col": "v", "dirty": "z", "clean": "q", "tiers": {"rules": cand, "llm": cand}}]       # neither
    monkeypatch.setattr(pl, "tier_llm_sem", lambda *a, **kw: ("q", 0.7, {**ZERO_COST, "calls": 1}))
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "c.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in recs))
        monkeypatch.setattr(pl, "_cache_path", lambda name: path)
        monkeypatch.setattr(pl, "LLM_SEM_DIR", Path(d))
        assert pl.refresh_llm_sem("fake", ask=False) == {"direct_eligible": 2, "missing": 2, "missing_new": 1}
        assert pl.refresh_llm_sem("fake", ask=True) == {"direct_eligible": 2, "missing": 2, "missing_new": 1}  # asks only old-eligible
        assert pl.refresh_llm_sem("fake", ask=True, direct=True) == {"direct_eligible": 2, "missing": 1, "missing_new": 1}  # now the direct-only one
        assert pl.refresh_llm_sem("fake", ask=False)["missing"] == 0
        new = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["llm_sem_eligible"] for r in new] == [True, False, False] and [r["llm_sem_direct_eligible"] for r in new] == [True, True, False]
    assert new[1]["tiers"]["llm_sem"]["value"] is None and new[1]["llm_sem_direct"]["value"] == "q" and "llm_sem_direct" not in new[2]


def test_run_cache_dir_suffix_only_for_coverage_and_random_keeps_old_dirs(monkeypatch):
    import run
    assert [run._cache_dir(20, 0, "").name, run._cache_dir(50, 2, "_xsem").name] == ["b20_s0", "b50_s2_xsem"]
    monkeypatch.setattr(run, "SFX", "_cov")
    assert [run._cache_dir(20, 0, "").name, run._cache_dir(50, 2, "_xsem").name] == ["b20_s0_cov", "b50_s2_xsem_cov"]


def main():
    tests = [
        test_route_treats_llm_sem_as_the_last_paid_tier,
        test_refresh_llm_sem_asks_only_unresolved_cells_once_and_keeps_other_fields,
        test_llm_sem_prompt_forbids_the_dirty_value_and_dirty_answers_count_as_none,
        test_eval_writes_lambda_sweep_and_lambda_figure,
        test_brier_after_calibration_and_latency_per_repair,
        test_ablation_methods_static_gate_no_escalation_raw_threshold_and_operator_subsets,
        test_ours_lambda_is_a_per_cell_budget_router_and_calls_are_monotone_in_lambda,
        test_ours_sem_direct_tests_llm_sem_on_its_own_and_is_ours_lambda_otherwise,
        test_ours_sem_guard_is_sem_direct_only_with_labeled_llm_sem_evidence,
        test_refresh_llm_sem_direct_flag_and_missing_count,
        test_system_level_calibration_is_over_repaired_cells_with_accepted_confidence,
        test_ours_budget_picks_lambda_by_cross_fitted_f1_and_ties_go_to_the_cheaper_lambda,
        test_route_reports_latency_call_rate_and_repair_accuracy,
        test_route_reports_semantic_split_and_call_share,
        test_semantic_flag_cached_and_label_exclusion_avoids_injected_rows,
        test_inject_semantic_corrupts_whole_entities_with_unambiguous_unique_city,
        test_tier_rules_fixes_format_error,
        test_tier_rules_known_good_values_exclude_error_cells,
        test_tier_rules_known_good_includes_labeled_clean_values,
        test_tier_stats_fixes_dependency_error,
        test_tier_stats_fixes_typo_via_fuzzy_match,
        test_tier_stats_self_match_is_deliberate_abstention_for_frequent_values,
        test_llm_proposed_fd_is_kept_after_calib_validation_and_drives_stats,
        test_fd_tier_picks_raw_confidence_variant_on_calib_split,
        test_source_weighted_fd_vote_beats_copied_errors_and_is_chosen_on_calib,
        test_dependency_votes_count_only_known_clean_voters_and_fall_back_when_none,
        test_one_known_clean_voter_is_enough_but_fallback_needs_two,
        test_propose_source_is_one_cached_call_and_only_exact_column_names_count,
        test_bogus_fd_with_low_calib_score_is_dropped,
        test_refresh_cheap_leaves_llm_and_column_fields_untouched,
        test_ours_abstains_below_tau,
        test_precision_recall_known_case,
        test_tier_rules_numeric_extraction_integer,
        test_tier_rules_numeric_extraction_float,
        test_tier_rules_numeric_extraction_skips_multi_number_value,
        test_tier_rules_null_token,
        test_column_tier_classifies_and_fixes_systematic_group,
        test_column_tier_examples_from_calib_split_only,
        test_column_group_keeps_only_verified_programs,
        test_group_score_leave_one_out_and_column_cap,
        test_column_calibrator_is_identity,
        test_calibrators_platt_small_n_isotonic_large_n,
        test_dsl_solves_rayyan_like_dates_without_llm,
        test_dsl_solves_beers_like_ounces_and_needs_two_examples,
        test_dsl_richer_formats_slices_and_column_transfer,
        test_charsub_learns_mojibake_rules_from_labeled_pairs_and_needs_loo_score,
        test_auto_tau_uses_calib_cells_only_and_prefers_higher_tau_on_ties,
        test_auto_tau_is_cross_fitted_over_rows,
        test_route_skips_tier_that_can_never_reach_tau,
        test_load_cache_drops_noop_candidate,
        test_calibrator_fallback_on_single_class_labels,
        test_coverage_sampler_includes_unresolved_rows_same_n_deterministic_and_clean_free,
        test_unresolved_proxy_needs_no_stats_and_no_llm_answer_and_ignores_clean_and_fds,
        test_refresh_llm_sem_no_ask_reuses_stored_answers_and_never_calls_the_llm,
        test_run_cache_dir_suffix_only_for_coverage_and_random_keeps_old_dirs,
    ]
    for t in tests:
        mp = _MonkeyPatch()
        try:
            if "monkeypatch" in t.__code__.co_varnames[: t.__code__.co_argcount]:
                t(mp)
            else:
                t()
            print(f"ok  - {t.__name__}")
        finally:
            mp.undo()
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
