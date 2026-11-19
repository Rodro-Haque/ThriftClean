#!/usr/bin/env bash
# Full reproduction of the final model (ours_sem_guard@0; = ours_lambda@0 of the earlier router at lambda=0). ORDER MATTERS. Needs: vLLM running (scripts/serve_vllm.sh) and
# LLM_BASE_URL / LLM_MODEL exported; .venv and .venv-raha set up (README). cache/, results/, logs/ are git-ignored.
# The precompute steps CALL THE LLM (hours); everything from "eval" on is replay-only. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

# 1. cache/b20_s0 = seed 0, 20 rows, random (also holds the split-free shared caches: fds, source, llm_sem, unresolved)
$PY src/run.py precompute --dataset all && $PY src/run.py precompute --dataset all --refresh-llm-sem
for d in hospital beers; do
  $PY src/run.py inject-semantic --dataset $d
  for s in fds source; do cp cache/b20_s0/${d}_$s.json cache/b20_s0/${d}_sem_$s.json; done
  $PY src/run.py precompute --dataset ${d}_sem && $PY src/run.py precompute --dataset ${d}_sem --refresh-llm-sem
done

# 2. per-config caches cache/b{N}_s{S}{,_cov,_xsem,_xsem_cov}/ (llm/llm_sem answers are split-free, so copy them and refresh only the
#    split-dependent tiers; may call the LLM for new column groups)
for N in 20 50; do for S in 0 1 2; do for V in "" _cov _xsem _xsem_cov; do
  C=b${N}_s${S}$V; if [ $C = b20_s0 ]; then continue; fi
  (
    case $V in *cov) SM=coverage;; *) SM=random;; esac
    case $V in _xsem*) EX=1; DS="hospital_sem beers_sem";; *) EX=0; DS="hospital flights beers rayyan movies hospital_sem beers_sem";; esac
    export CACHE_DIR=cache/$C LABEL_TUPLES=$N SPLIT_SEED=$S SAMPLER=$SM LABEL_EXCLUDE_SEM=$EX; mkdir -p $CACHE_DIR
    for n in $DS; do
      cp cache/b20_s0/$n.jsonl cache/b20_s0/${n}_column.jsonl $CACHE_DIR/
      $PY src/run.py precompute --dataset $n --refresh-cheap; $PY src/run.py precompute --dataset $n --refresh-column
      if [ $SM = coverage ]; then $PY src/run.py precompute --dataset $n --refresh-llm-sem --no-ask; fi
    done
  )
done; done; done

# 3. replay-only evaluation
$PY src/run.py eval --dataset all; for d in hospital_sem beers_sem; do $PY src/run.py eval --dataset $d; done   # results/b20_s0/<name>_methods.csv
$PY src/run.py sweep-summary; $PY src/run.py sweep-summary --sem; LABEL_EXCLUDE_SEM=1 $PY src/run.py sweep-summary --sem   # results/summary/budget_sweep{,_sem,_sem_xsem}.csv
SAMPLER=coverage $PY src/run.py sweep-summary; SAMPLER=coverage $PY src/run.py sweep-summary --sem; SAMPLER=coverage LABEL_EXCLUDE_SEM=1 $PY src/run.py sweep-summary --sem   # *_cov

# 4. LLM-only baselines (asks every error cell once; resumable). Qwen3-8B is the served model; then serve Jellyfish-8B
#    (the script pauses for it).
LLM_MODEL=Qwen/Qwen3-8B $PY src/llm_only_baselines.py ask --model qwen_strong
read -rp 'Now serve Jellyfish-8B (MODEL=NECOUDBFM/Jellyfish-8B scripts/serve_vllm.sh), then press Enter: '
LLM_MODEL=NECOUDBFM/Jellyfish-8B $PY src/llm_only_baselines.py ask --model jellyfish
$PY src/llm_only_baselines.py eval; SAMPLER=coverage $PY src/llm_only_baselines.py eval --datasets beers_sem   # results/summary/llm_only_baselines*.csv

# 5. Baran (own env). logs/baran_results.jsonl is APPENDED to: re-running without clearing it double-counts seeds, so it is moved aside.
if [ -f logs/baran_results.jsonl ]; then mv logs/baran_results.jsonl logs/baran_results.$(date +%Y%m%d-%H%M%S).bak; fi
.venv-raha/bin/python src/baran_baseline.py --datasets hospital flights beers rayyan movies
.venv-raha/bin/python src/baran_baseline.py --summary > logs/baran_summary_with_movies.txt

# 6. paper tables and figures (deterministic, no LLM)
$PY src/make_tables.py                                   # paper/tables/ T1-T9
$PY src/run.py figure; $PY src/run.py lambda-figure      # paper/figures/frontier.*, lambda_budget.*

# 7. EXPERIMENTAL VARIANT ours_sem_direct@L (not part of the final model; every output above stays byte-identical). Pass --direct asks the
#    llm_sem cells only this variant reaches (needs vLLM; once per (dataset,row,col), into cache/b20_s0/<name>_llm_sem.jsonl); pass --no-ask then
#    copies the answers into every config. The comparison is replay-only: results/summary/sem_direct_comparison.{csv,md}.
for PASS in --direct --no-ask; do for N in 20 50; do for S in 0 1 2; do for V in "" _cov _xsem _xsem_cov; do
  (
    case $V in *cov) SM=coverage;; *) SM=random;; esac
    case $V in _xsem*) EX=1; DS="hospital_sem beers_sem";; *) EX=0; DS="hospital flights beers rayyan movies hospital_sem beers_sem";; esac
    export CACHE_DIR=cache/b${N}_s${S}$V LABEL_TUPLES=$N SPLIT_SEED=$S SAMPLER=$SM LABEL_EXCLUDE_SEM=$EX
    for n in $DS; do $PY src/run.py precompute --dataset $n --refresh-llm-sem $PASS; done
  )
done; done; done; done
$PY src/sem_direct_comparison.py
