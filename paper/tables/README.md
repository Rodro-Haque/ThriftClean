# paper/tables: final paper tables

Regenerate (deterministic, CPU, no LLM calls): `.venv/bin/python src/make_tables.py`. Output: `final_tables.md` (human-readable),
`tables.tex` (booktabs/longtable, needs graphicx), and one CSV per table (`T*.csv`, full precision, `_std` columns).
"Ours" = `ours_sem_guard@0` (equal to the earlier router's `ours_lambda@0` at λ=0; `make_tables.py` asserts it); mean±std over seeds 0/1/2 unless noted. Metric definitions: top-level README.md (sweep/summary files live in results/summary/, per-dataset seed-0 files in results/b20_s0/).

| table | content | source files |
|---|---|---|
| T1 | main results, 20 rows, random sampler, 5 datasets | results/summary/budget_sweep.csv, results/summary/llm_only_baselines_summary.csv, logs/baran_summary_with_movies.txt, results/summary/llm_only_literature.md (Jellyfish-7B, published) |
| T2 | all deck metrics, Ours vs LLM-only (context), 20 rows | results/summary/budget_sweep.csv, budget_sweep_sem.csv |
| T3 | ablations, F1 [calls], 20 rows | results/summary/budget_sweep.csv, budget_sweep_sem.csv |
| T4a | per-tier ECE/Brier raw -> calibrated, seed 0, n >= 20 | results/b20_s0/<name>_methods.csv |
| T4b | system-level ECE/Brier, Ours vs Raw threshold | results/summary/budget_sweep.csv, budget_sweep_sem.csv |
| T5 | semantic splits (mixed/xsem x random/coverage) | results/summary/budget_sweep_sem{,_cov,_xsem,_xsem_cov}.csv, llm_only_baselines{,_cov}_summary.csv |
| T6 | lambda budget ablation, 20 rows: `ours_sem_guard@λ` plus `ours_lambda@λ` comparison columns (λ* auto = `ours_budget`, ours_lambda family) | results/summary/budget_sweep.csv, budget_sweep_sem.csv, budget_lambda_star{,_sem}.csv |
| T7 | 50 labeled rows, F1 [calls] (no Baran at 50 rows) | results/summary/budget_sweep.csv, budget_sweep_sem.csv |
| T8 | published numbers, context only | results/summary/llm_only_literature.md |
| T9 | full comparison vs SOTA: (A) same protocol, perfect detection, 20 rows, random, 3 seeds: replay-only single-family baselines (rules / stats / rules+stats / rules+fd+stats, computed in make_tables.py from cache/b20_s{0,1,2} via pl.load_cache + pl.calib_test_split; the last asserted equal to the sweep's rules_stats row, all four asserted to the values of the earlier check), ours_no_llm, Baran, LLM-only x3, cascade, Ours, LLM calls Ours vs LLM-only; (B) published numbers (T8 rows + HoloClean Hospital from P 1.00 / R .713), Ours last | results/summary/budget_sweep.csv, results/summary/llm_only_baselines_summary.csv, logs/baran_summary_with_movies.txt, cache/b20_s{0,1,2}/<name>.jsonl, results/summary/llm_only_literature.md |

Caveats: LLM-only (knowledge) exists only for the mixed split (T5 shows '-' elsewhere); Baran was run with 20 labels only; T8 rows and T9 part B use other
detection assumptions, models and label budgets than ours. The seed-0 results/b20_s0/<name>_lambda.csv and results/b20_s0/<name>_methods.csv are the seed-0 views of
the sweeps (checked identical to a fresh replay; unchanged for Ours at λ=0).
