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

## Round 3 pre-registration: H6 (repo), H7 (user history), prompt-category table (2026-10-06, before any cost-by-repo or cost-by-user look)

Data: `full_rows/` (424,108 priced sessions, >= 4 calls, 30,143 repos) joined to `risenlab/agentlogs` `repositories/` (same dataset sha `04013a44d3432c6654bca1dcd4a01218a9406b80`, join verified 100% on repo name; only join counts were inspected, no cost) and session-table `user.id` + `created_at`.

Split: repos shuffled with `random.Random(20261007)`; first 50% = EXPLORE, rest = CONFIRM. Inside CONFIRM, repos split again 70% fit / 30% test with `random.Random(20261008)`.

Baseline: H1 groups (model x event x prompt-length bucket), p10-p90 interval fit on CONFIRM-fit.

H6, repo context: refine each baseline group by a repo bucket built from `code_lines` (size) and `main_language`.
H7, user history: refine each baseline group by a user bucket built from the user's prior priced sessions. Prior = same `user.id`, `created_at` strictly earlier than the predicted session (no future leakage); sessions with too few priors fall into a "no history" bucket and stay in the evaluation.

EXPLORE phase (free, unbounded): choose cutoffs, number of buckets, language grouping, history window and minimum prior count. These choices are frozen in a separate commit BEFORE the CONFIRM script runs. Nothing chosen after that commit.

CONFIRM bar (one look, per hypothesis): eligible = baseline groups with >= 300 CONFIRM-test sessions. Per eligible group: session-weighted mean width (p90/p10) of the refined intervals vs baseline width; pass if ratio <= 0.8 AND 90% CI upper of the ratio < 0.8 (200 bootstrap resamples over CONFIRM-fit repos, upper = 95th percentile) AND pooled test coverage 90% CI (500 resamples) contains 80%. Hypothesis passes if >= 1/3 of eligible groups pass (same fraction as H1's 5/15 standard).

Prompt-category reference table (descriptive, no hypothesis): named categories (fix CI, dependency update, tests, refactor, docs, review feedback, question/chat, other) assigned by keyword rules written on EXPLORE; Karl rates a blind sample of 30 labels; agreement reported as-is. Table axes: category x repo-size bucket x model; every cell publishes n, p10/p50/p90, width. Cells with n < 100 suppressed.

### Round 3 frozen choices (from EXPLORE only, `explore_r3.out`; committed before CONFIRM runs)
Selection rule applied to explore output: most groups with ratio <= 0.8, tiebreak lowest median ratio, then fewest buckets.
- H6 repo bucket = main_language in {TypeScript, Python, JavaScript, Go, Rust, C#, Java}, else "other" (missing = "na"). Explore: 3/54 groups <= 0.8, median ratio 0.99. Size cutoffs did no better (2-3/54, median 1.00).
- H7 user bucket = median cost of the user's last 10 strictly-earlier priced sessions, requiring >= 3 such sessions (else "none"); buckets cut at $0.25 / $0.60 / $1.20 / $3.00 (5 buckets). Explore: 9/54 groups <= 0.8, median ratio 0.91.
- Refined cell with < 50 fit sessions falls back to the baseline group interval.
- Stated expectation before CONFIRM: both likely FAIL the >= 1/3 bar (explore best H6 3/54, H7 9/54 = 17%, before CIs).

### Round 3 CONFIRM results (one look, `confirm_r3.out` / `confirm_r3.json`)
- H6 language: FAIL (see confirm_r3.json). H7 user history: 0/37 groups pass -> FAIL. Both as predicted before CONFIRM.

## Round 4 pre-registration: external replication of H1 and H4 (2026-10-06, before any cost from these sets is computed)

Disclosure of prior contact: (a) `melissapan/swe-bench-lite-agent-traces-v14`: replicate cost spread already computed during the survey (median max/min across 3 identical runs 1.55x, p90 2.86x); it is therefore EXPLORATORY and used only as the noise-floor descriptive, never in the replication tests. (b) One `Exgentic/agent-llm-traces-v2` sample row was seen in the survey (agent_cost 0.89); no distribution computed.

Datasets (HF revisions pinned):
- E1 `Exgentic/agent-llm-traces-v2` @ 4b8ad4ab198438e5a170f9171c19c6a2cf7c1814 (10,056 runs; agent_cost, success, harness, benchmark, models)
- E2 `open-agent-leaderboard/traces` @ fcb6c1a6f5b649cde515b6f787e8bdbaa11d626c (~10k session files; token usage)
- E3 `MaxDevv/real-pi-coding-agent-traces-sessions` @ 8c593252ddad7dca08a0afc07896195fa73f2d6e (~1.3k developer sessions)
- Noise floor (exploratory): `melissapan/swe-bench-lite-agent-traces-v14` @ 2cdf7f3c08052508330cf99dc600c6a00ea89bff

Procedure:
1. Schema-only inspection (column names, value types, category counts; no cost or token sums). Grouping columns, split unit and cost source per dataset are then frozen in a separate commit BEFORE any cost is computed.
2. Cost = dataset-reported USD where present. Where only tokens exist, price with the same OpenRouter list prices as `lab.call_cost` (2026-10-06); unpriceable models excluded and counted. Sessions with cost <= 0 or missing excluded and counted.
3. Group = model x task source x harness (the external analog of model x trigger; task source = benchmark/subset or, for E3, project). Baseline = model only. Caveat stated in advance: across different benchmarks this grouping may narrow ranges trivially; results are reported per benchmark as well.
4. Split by task unit (task_id / benchmark instance; E3: project = repo prefix of file name) shuffled with `random.Random(20261010)`, 70% fit / 30% test.

Bars (same as the originals):
- R-H4 floor: floor = fit-group p10. Pass in a dataset if pooled test floor coverage 90% CI (500 bootstrap) contains 90% AND the CI contains 90% in >= 2/3 of eligible groups (>= 100 fit, >= 30 test).
- R-H1 range: per eligible group, p10-p90 width <= 0.8x the model-only width AND 90% CI upper of the ratio < 0.8 (200 bootstraps over fit task units, upper = 95th percentile) AND test coverage 90% CI contains 80%. Pass in a dataset if >= 20% of eligible groups pass (H1 original: 3/15), minimum 1.
- A dataset with < 3 eligible groups is reported "not evaluable", not counted.
- Replication claim for each hypothesis: passes in every evaluable dataset AND >= 2 datasets evaluable. Otherwise "not replicated" (or "not evaluable"), reported as-is. One look.
- 2026-10-06 (round 4, schema stage, before any cost computed): E2 `open-agent-leaderboard/traces` dropped as an independent replication set: 3,196 of 3,398 downloaded session ids (94%) also appear in E1 (same Exgentic runs, Claude Code subset). Counting it would double-count E1. Replication claim therefore rests on E1 + E3; the ">= 2 evaluable datasets" rule is unchanged.
- E1 schema facts (no cost read): 10,056 runs, 115 run configs, 5 harnesses, 6 benchmarks, 5 models, each run a single model. Task unit = sha1 of the first user text part in the first span with input messages -> 3,345 distinct tasks (1-5 configs each). Split unit = (benchmark, task hash).
- 2026-10-06 (round 4, counts only, still before any cost computed): two amendments, made from group SIZES and the split design, not from outcomes.
  1. Eligibility. Under the registered minimum (>= 100 fit, >= 30 test) E1 has 4 eligible groups (E1 groups are run configs of ~100 tasks each; 109 groups) and E3 has 2, so both would be "not evaluable". PRIMARY verdict stays as registered (expected: not evaluable). A SECONDARY analysis is added with minimum >= 50 fit and >= 20 test; same bars otherwise. Smaller groups make p10/p90 noisier and width CIs wider, so this does not favour passing R-H1; it is labelled secondary everywhere.
  2. E3 split. Registered split was by project, but the group includes project, so held-out groups could never be fit (design error). E3 split becomes chronological within each project: first 70% of a project's sessions by file timestamp = fit, last 30% = test (the realistic case: a project's past predicts its future). Bootstrap unit for E3 = session. E1 unchanged: split by (benchmark, task hash), seed 20261010.
  E3 facts (no cost read): 1,291 sessions, 21 projects, all carry per-message usage.cost.total; session cost = sum of cost.total; group = dominant model x project; sessions with cost 0 (local models) excluded and counted.

### Round 4 results (one look; `replicate_r4.out` / `replicate_r4.json`)
- Process note: the first execution computed all results, then crashed in the final verdict print (KeyError, iteration over `res` after adding a key). Fixed the print only and re-ran; seeds are fixed, so the numbers are identical to the first execution. No parameter changed.
- PRIMARY (>= 100 fit / 30 test): E1 2 eligible groups, E3 0 -> both not evaluable -> R-H1 and R-H4 NOT EVALUABLE.
- SECONDARY (>= 50 / 20): E1 R-H4 PASS (pooled floor held 90.1% [89.1, 91.4], 48/60 groups); E1 R-H1 PASS (29/60 groups, need 12). E3 1 eligible group -> not evaluable. Claim needs >= 2 evaluable datasets -> NOT EVALUABLE.
- Caveat on E1 R-H1 (stated in advance): model-only baselines mix 6 benchmarks and are very wide (Kimi 232x, DeepSeek 122x, Opus 29x, GPT-5.2 6x); most passing groups come from that mixing. For GPT-5.2, whose baseline is narrow (6.2x), 1/11 groups pass and 4 are wider than baseline. Read R-H1 on E1 as "task source matters", not as support for model x trigger ranges in general.

## Round 5 pre-registration: do expensive runs fail more? (2026-10-06, before any outcome or token data is loaded)

Question: within a comparable group, are runs in the cost tail less likely to succeed? This is gate 1 of the shortlisted tail-alarm / stopping-policy line.

Data (HF, revisions pinned at download and logged in the freeze commit): `SWE-bench/SWE-smith-trajectories` (resolved), `nebius/SWE-rebench-openhands-trajectories` (resolved), `nebius/SWE-agent-trajectories` (target = resolved label). Datasets without an outcome label are not used.

Cost proxy (no billed cost exists): estimated input+output volume = sum over assistant turns of (characters of the full conversation up to and including that turn) / 4. Labelled "estimated tokens" everywhere; never converted to dollars. Secondary proxy: number of assistant turns.

Group = dataset x model (x repo where present). Tail = run above its group's p90 of estimated tokens; body = at or below group p50. Groups need >= 200 runs.

O1 (primary): per dataset, pooled tail resolve rate / body resolve rate (risk ratio). Pass in a dataset if RR <= 0.8 AND 90% CI upper (1,000 bootstrap resamples over task instances) < 0.8. O1 holds if it passes in >= 2 of the evaluable datasets (evaluable = >= 3 groups).
Exploratory, no bar: (a) the same for the turns proxy; (b) stopping backtest: stop each run at its group p90/p95/p99 -> share of estimated tokens saved vs share of resolved runs lost (a resolved run that exceeds the threshold counts as lost); (c) resolve rate by group decile.
Stated caveat: task difficulty drives both cost and failure, so O1 cannot say stopping CAUSES nothing to be lost beyond what the backtest shows; it only says whether the tail is mostly failing runs.

Procedure: schema-only inspection first (column names, types, outcome label counts; no token sums by outcome). Column choices frozen in a separate commit. Then one run.

### Round 5 frozen choices (schema + label counts only; no token volume by outcome computed)
Revisions: SWE-smith @ 08e109b4a59eaeebf80e4675cd125d42e7ac99a4 (MIT), SWE-rebench-openhands @ 35455389ab51bf5e2306bfd436ef72d0f98bf882 (CC-BY-4.0), SWE-agent @ 68195a1450865274106246d0d0296a1d6807b88e (CC-BY-4.0).
- D1 SWE-smith: `train` files only (23,821 trajectories; `xml` is an identical copy, `tool`/`ticks` are other scaffold formats of overlapping or separate runs, excluded to avoid duplicates). Outcome = `resolved`. Group = `model` (3 models). Messages: list of {role, content}; assistant role = "assistant".
- D2 SWE-rebench-openhands: 67,074 runs, single agent setup, no model column. Outcome = `resolved` (0/1). Group = `repo`, groups >= 200 runs (43 repos, 18,781 runs). Assistant role = "assistant".
- D3 SWE-agent: 80,036 runs. Outcome = `target`. Group = `model_name` (3 models). Assistant role = "ai"; text field = `text`.
- Estimated tokens = sum over assistant turns of (cumulative characters of all messages up to and including that turn) / 4. Turns proxy = number of assistant turns.
- Bootstrap unit = instance_id. All else as registered.

### Round 5 results (one look; `outcome_r5.out` / `outcome_r5.json`)
- **O1 PASS, 3/3 datasets.** Tail (above group p90, est. tokens) vs body (<= p50) resolve-rate ratio: D1 SWE-smith 0.38 [0.36, 0.41]; D2 SWE-rebench-OH 0.43 [0.38, 0.47]; D3 SWE-agent 0.14 [0.12, 0.16]. The turns proxy gives the same (0.37 / 0.42 / 0.14).
- Data error found after the run: D1 `train` files contain 2,255 duplicate rows (26,076 rows, 23,821 distinct traj_id); the freeze said 23,821. Deduplicated re-run of D1 only: RR 0.40 [0.37, 0.43], PASS; backtest p90 saves 13.5% / loses 4.9%. Verdict unchanged. Both numbers reported.
- Stopping backtest (exploratory): stop at group p90 saves 13.5% / 2.6% / 21.8% of est. tokens and loses 4.7% (4.9% dedup) / 5.2% / 2.1% of resolved runs. p95: 4.1/1.1/14.1% saved, 1.9/2.3/0.8% lost. Saving exceeds loss in D1 and D3, not in D2.
- Resolve rate falls nearly monotonically with cost decile in all three (D1 62% -> 21%, D2 60% -> 23%, D3 peaks at 32% in decile 2, then falls to 3-4%).
- Limits: estimated tokens, not billed cost; benchmark runs, not real use; difficulty confounds (association, not cause).
