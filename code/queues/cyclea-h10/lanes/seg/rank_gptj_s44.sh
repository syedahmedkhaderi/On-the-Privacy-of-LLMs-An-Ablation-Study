#!/bin/bash
T=work/cyclea-h10/tools; P=work/venvs/cyclea-dea/bin/python; NEW=work/rs-1/cycleA/bundles/dea-ext-20261110
J=work/cyclea-h10/results/rank_gptj_s44/jobs.csv
for cond in generic_prefix pii_leading true_prefix; do for seed in 44; do
  $P $T/h10_dea.py --bundle $NEW --model gpt-j-6b --cond $cond --seed $seed --block rank_gptj_s44 --tokens
done; done
# scoring pass for every completed cell of this model in this block
grep -o 'ca-h10-rank_gptj_s44-gpt-j-6b-[A-Za-z_]*-s[0-9]*-[a-z0-9]*-[0-9TZ]*\(-retry1\)\?,complete' $J | cut -d, -f1 | while read rid; do
  ( cd work && CUDA_VISIBLE_DEVICES=$GPU_INDEX $P $T/h10_rank_score.py $rid ) >> work/cyclea-h10/results/rank_gptj_s44/scores.log 2>&1
done
exit 0
