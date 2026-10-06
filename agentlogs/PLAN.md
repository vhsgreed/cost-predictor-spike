# Pre-registered plan: agent cost range + loop warning (AgentLogs)

Written 2026-10-06 before any confirmatory run. Changes after this commit must be logged
below with date and reason; bars may not be moved after seeing validation or test data.

## Data
- risenlab/agentlogs, HF revision `04013a44d3432c6654bca1dcd4a01218a9406b80`, CC BY 4.0.
- Development: log shards 00000-00004 (explore and tune freely).
- Validation: shards 00100-00104 (one look per decision).
- Locked test: 10 shards drawn with `random.Random(20261006).sample(range(5, 276), 10)`
  excluding validation; run once, at the end.
- Numbers and labels only. Tool arguments hashed on read; no prompt or message text stored.

## Hypotheses and pass bars
Scored on a two-sided p10-p90 interval (nominal 80%), split by repository, 90% bootstrap CIs.

- **H1 archetype range.** Per-group intervals (model x trigger event x prompt-length bucket,
  groups with >= 100 sessions in training) beat the per-model baseline.
  Pass: width at least 20% narrower than per-model baseline with CI upper bound below it,
  coverage CI containing 80%, in >= 3 groups.
- **H2 loop warning.** Sessions flagged by the loop detector are over-represented in the
  top cost decile.
  Pass: ratio >= 2.0 (CI lower bound >= 1.5) AND >= 20% of top-decile dollar cost falls
  after onset AND hand-check false-positive rate <= 30% on 30 flagged sessions.
- **H3 loops predict failure (exploratory).** Failed/cancelled/timed-out rate in flagged
  vs unflagged sessions. Reported, no bar.

## Controls (must hold or the run is void)
- Shuffled-target control: features may not beat the baseline.
- Ceiling: model given the true call count, reported as the best achievable width.
- Baseline is the per-model average, not the global average.

## Targets
Both tokens and dollar cost (OpenRouter list prices 2026-10-06, cached input priced
separately). Unpriced models excluded from dollar results and counted.

## Reproduction
`build_dataset.py OUT SHARDS...` then the scoring scripts; `test_lab.py` must pass first.

## Change log
- 2026-10-06: initial version.
- 2026-10-06: H2 hand-check run (blind, 1 rater, 58/60 answered). Flagged: 9 yes / 20 no /
  1 unsure -> false-positive rate 67%, fails the <= 30% bar. Unflagged: 10/28 judged loops.
  Rater answers drift strongly with case order (cases 1-30: 3 yes; 31-60: 16 yes) while
  flags are spread evenly, so the reference itself is unreliable. H2 NOT carried to
  validation. Signal renamed "repeat/no-edit stretch" (cost signal, not a verified loop).
- 2026-10-06: correction to the H2 entry above: flagged cases were 18 in positions 1-30 and
  12 in 31-60, not evenly spread. Conclusion (H2 fails) unchanged.
- 2026-10-06: H1 validation run once (validate_h1.py, val shards 00100-00104, 7,687 priced
  sessions, all 15 groups with >= 100 dev sessions pre-selected by size, not by dev result).
  6 of 15 groups pass -> H1 PASS. Output in validate_h1.out.
- 2026-10-06 (before locked test): ADD H4 floor. "At least" bound = p10 of the training
  group's dollar cost (90% one-sided). Pass: overall floor coverage 90% CI contains 90%, AND
  floor coverage CI contains 90% in >= 2/3 of the 15 H1 groups with >= 30 test sessions.
  Locked test = shards in test_shards.txt, training = dev only (identical to validation),
  same 15 groups, run once.
- 2026-10-06: LOCKED TEST run once (locked_test.py; shards in test_shards.txt; train = dev
  only). 15,349 priced sessions, 6,509 repos. H1: 5/15 groups pass -> PASS (need 3).
  H4: floor overall 90% [90,90]; 10/15 groups ok -> PASS. Output: locked_test.out.
  This confirms H1+H4 on unseen data. No further group selection or bar changes permitted.
- 2026-10-06: ADD H5 (exploratory, dev shards only, repo-split): prompt-text clusters
  (nomic-embed-text, k-means, k=10, L2-normalized) separate cost. Bar: >= 2 clusters with
  >= 30 held-out sessions whose p10-p90 dollar width is <= 0.8 x the model-only width,
  with 90% CI (bootstrap over training repos) of the width ratio entirely below 0.8, and
  coverage 90% CI containing 80%. Also report whether clusters beat model x event groups
  (informational, no bar). Bar set before running.
- 2026-10-06: H5 canary run (canary_h5.py; 1,725 dev sessions with prompt text, 1,157 repos,
  k-means k=10 on nomic-embed-text). FIRST run under-implemented the bar (checked width
  point estimate only, omitted the ratio CI) and appeared to pass 4/7 clusters. Script
  corrected to the full pre-registered bar before reporting: 0/7 clusters pass -> H5 FAIL.
  Best cluster ratio CI upper 0.88 (point ratio 0.67). Informational: model x event group
  width median 10.6x vs cluster 16.1x; clusters weaker than trigger groups.
  Interpretation: canary underpowered (n_test 36-65 per cluster); 4 clusters have point
  ratios 0.63-0.79 but wide CIs. Final H5 verdict deferred to the full 56 GB run, same bar,
  one look. Output: canary_h5.out.
- 2026-10-06: FULL 56 GB RUN pre-registered before data. Source: risenlab/agentlogs (HF),
  dataset sha 04013a44d3432c6654bca1dcd4a01218a9406b80; all 276 agent_session_logs shards
  processed fresh -> full_sessions.jsonl (same row schema as dev.jsonl; parsing identical
  to build_dataset.py). Outputs: (a) production cost tables per (model, event, prompt-length
  bucket) group, descriptive only (n, p10/p50/p90, mean, floor = p10 = "at least X" at 90%
  confidence), no new hypothesis tests; (b) H5 FINAL verdict, bar UNCHANGED from the canary:
    prompt = session-table prompt, >= 4 words; sessions with >= 4 calls and priced cost;
    embeddings: nomic-embed-text 768d via local Ollama, L2-normalized;
    k-means k=10, k-means++ init, 40 iters, seed 20261006, centroids FIT ON TRAIN PROMPTS
    ONLY (canary fit on all rows; final is leakage-free), test prompts assigned to nearest
    centroid;
    repo split: unique repos shuffled by random.Random(20261006), first 70% train, 30% test;
    pass per cluster (>= 30 train and >= 30 test sessions): train p10-p90 $ width <= 0.8 x
    the width of the cluster's dominant model (train sessions of that model), 90% CI of the
    width ratio (200 bootstrap resamples over training repos; upper bound = 95th percentile)
    entirely below 0.8, AND test coverage 90% CI (500 membership resamples) contains 80%.
    H5 passes if >= 2 clusters pass. ONE LOOK at the verdict; no re-tuning after.
  H1/H4 locked test untouched: no re-selection of test shards, no re-scoring.

## Full-run results (2026-10-06, one look)

- Data: 276/276 shards (2 parsed in a pre-launch smoke test, 274 in the main run, 61 min), 471,067 sessions, 424,108 priced with >= 4 calls. Shards deleted after parsing.
- Production tables (descriptive): `tables_full.json`, `full_tables.out`. 175 model x event x prompt-length groups with >= 100 sessions. Floor (p10) sits at 27-44% of the median in the 15 largest groups.
- **H5 FINAL: 0/10 clusters pass -> FAIL.** 92,429 sessions with prompt text, 13,535 repos (train 66,535 / test 25,894). Clusters k0 (ratio 0.62, CI hi 0.76) and k2 (0.68, CI hi 0.76) clear both width criteria but fail coverage on held-out repos (CI [72%,78%] and [76%,78%], must contain 80%). Bar not moved. Model x event median width 11.3x (n=90) vs cluster median 16.4x (n=10).
- Disclosure: a smoke test of `h5_final.py` on shards 00005-00006 (about 1,100 prompt sessions) was run before launch to verify the code; it printed a verdict (0 pass). No code, parameter or bar was changed after it except an unrelated print fix in `run_full.py`.
