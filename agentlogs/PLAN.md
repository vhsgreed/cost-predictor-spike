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

## Success-label check (2026-10-07, one rater, blind)
30 runs (all 5 model-"corrected" + 25 random model-"accepted"), rater Karl, 1 unsure excluded.
- Agreement 23/29 (79%). Of 7 runs Karl marked corrected, the labeler caught 3 (recall 43%). Of 5 runs it marked corrected, 3 were (precision 60%).
- Among model-"accepted" runs, Karl marked 4/24 corrected (17%) -> estimated true correction rate ~20% of human-replied runs (~13/66), not the 8% the labeler reported.
- Verdict: labeler NOT trusted for automation. Not added to cron. Sheet with message excerpts deleted; key and answers (labels only) kept locally, gitignored.

## Supplementary analyses for article v0.2 (2026-10-07; POST-HOC, descriptive, no new hypothesis tests)
Prompted by an LLM-generated review of draft v0.1. Outputs: `rescore_dev.out`, `revision_v02.out`, `alarm_runtime_r5.out/.json`.
- H2/H3 numbers re-run from `rescore_dev.py` (dev, exploratory): flagged share top decile 40% vs rest 12%, ratio 3.3x [3.0, 3.6]; top-decile dollars after onset 26%; H3 not-completed 5.6% (n=1,206) flagged vs 5.1% (n=7,010) clean. H2 still fails on the hand-check bar (67% false positives).
- Ceiling pairing (held-out dev, `rescore_dev.out`): true call count known -> 4.3x (tokens), 5.3x (dollars); model+event+promptlen 14.3x / 10.7x; per-model 17.8x / 13.7x. Fig. 5 is a different, in-sample computation (dollars, deciles of true call count / tokens within group): 9.7x -> 3.0x -> 1.9x.
- Floor comparison on the locked test (15 H1 groups): pooled coverage group floor 89.5%, model-only floor 89.6%, global floor 89.5%. Per group: group floors 83-94%, model-only 54-99%, global 52-100%. Grouping buys per-group calibration and informative values ($0.07-$1.02), not pooled coverage. Floor/median 0.21-0.45.
- H1 passing groups cover 26.7% of locked-test sessions and 18.1% of spend.
- Excluded low end (full run): priced sessions with <= 3 calls = 15,849 (3.6%), $5,001 (0.7% of priced spend); unpriced sessions 31,110.
- Runtime alarm (SWE data, threshold = fit-instance group p90, held-out instances): fires on 9.9/10.8/10.9% of runs and on 4.5/5.3/2.4% of resolved runs; resolve rate fired 19.5/19.9/3.3% vs not fired 44.8/43.6/16.7%; median share of tokens still ahead when it fires 27/11/29%; est. tokens per resolved run when stopping at the alarm -9.2% / +2.6% / -20.3% (SWE-smith / SWE-rebench-OH / SWE-agent).

## Validation V1: list-price-equivalent cost vs billed cost (2026-10-07, before computing ratios)
- Data: the author's own OpenRouter-billed calls (gen ids + token counts from local agent.log*; numbers only, not published). Schema checked on one record: generation endpoint returns billed `usage` and native token counts that match the logged values.
- Estimate model under test: uncached input x prompt price + cached input x cache-read price + output x completion price, using OpenRouter list prices for the model from /api/v1/models fetched 2026-10-07 (price date noted; drift disclosed).
- Metric per call: billed usage / estimated cost. Report: median, p10-p90, share within +/-5% and +/-10%, split by cache share.
- Validation bar: median ratio in [0.95, 1.05] AND >= 90% of calls within +/-10%. Pass => "the pricing model (tokens x list price with cache split) matches billing on N generations of model X"; scope caveat stated: this validates the pricing model on one model/account, not the Claude/GPT price entries used for AgentLogs.
- Join integrity check: logged native tokens vs generation record tokens must match exactly.

### V1 results (2026-10-07)
1,496 logged OpenRouter calls with generation ids; 1,493 joined (3 ids returned 404, excluded). Join integrity: logged native tokens matched the billing record exactly on every joined call (0 mismatches). Model: xiaomi/mimo-v2.6-pro via OpenRouter (billing record model xiaomi/mimo-v2.6-pro-20260921).
- billed / list-price estimate: median 1.0000 | p10 0.9586 p90 1.0152 | within 5%: 98% | within 10%: 100%.
- Cache split correct: median 1.0000 in both high-cache (>50% cached, n=1,348) and low-cache groups (n=145).
- Totals: billed $7.0766 vs estimated $7.1827 (0.985). Bar: PASS (median in [0.95,1.05], 100% within 10%).
- Scope: validates the pricing MODEL (tokens x list price with cached-input split) against real billing on one model and one account. Does not validate the individual Claude/GPT price entries used for AgentLogs; those were snapshot from the same OpenRouter price list (2026-10-06) and AgentLogs has no billing records to check against. Per-call rows stay local (cost-predictor-local/billing_validation.json).

## Round 6 pre-registration: remaining-cost forecast during the run (2026-10-07, before any forecast is computed)

Question: given evidence from a run in progress (turns elapsed, cumulative estimated size), how accurately can the remaining and final cost be forecast, and how does accuracy evolve as the run unfolds? Complements O1 (threshold alarm) and TokenCast (arXiv:2609.35760), whose absolute numbers are not comparable (different token accounting); baselines here are internal.

Data: same three sets as O1 (SWE-smith train files deduped, SWE-rebench-OH, SWE-agent; revisions pinned in round 5's freeze). Estimated size = cumulative chars / 4 summed over assistant turns, as before.

Split: 70% fit / 30% test by instance_id, `random.Random(20261014)`. All thresholds, medians and cells fitted on fit instances only.

Method (table forecast): at each update point (after assistant turn k), x = cumulative so far / fit-group median final size. Cell = (k bucket: 1,2,3,4-5,6-10,11-20,21-40,41+; x bucket: <0.1,0.1-0.25,0.25-0.5,0.5-0.75,0.75-1,1-1.5,>=1.5). Point forecast of FINAL size = median of fit cell finals; interval = fit cell p10-p90. Fallback: cell < 50 fit rows -> k-only cell -> group table.

Baselines: (a) unconditional group median final; (b) k-only table (same method, x ignored). The better of the two is the reference.

Metrics (update points k >= 3, pooled): MAE on final size (bootstrap ratio over instances, 200 resamples, 90% CI); 80% interval coverage (fit p10-p90). Bar: MAE ratio vs best baseline <= 0.8 with CI upper < 0.8 AND pooled coverage CI contains 80%, in >= 2 of 3 datasets. One look.

Descriptive (no bar): MAE and MAPE by turn bucket (when does the forecast become useful); task-start (k=1) coverage and error; per-dataset tables.
- Round 6 process note: first execution killed before any evaluation output (quantile recomputation made it too slow). Predictions precomputed per cell (same quantiles on the same fit lists, identical math); no method, cell, threshold or bar changed.
- Round 6 second process note: first complete run crashed on SWE-rebench (test runs in repos with no fit rows had no group table). Fix: skip test runs whose group has < 50 fit rows (the fallback chain's own support rule). SWE-smith results had printed before the crash and are disclosed as seen; seeds fixed, so re-running all three gives identical smith numbers. No threshold or bar changed.

### Round 6 results (one look; `forecast_r6.out` / `forecast_r6.json`)
- Bar: FAIL, 0/3. MAE ratio vs best baseline (k-only table won everywhere): SWE-smith 0.897 [0.883, 0.909]; SWE-rebench-OH 1.006 [0.982, 1.025]; SWE-agent 0.984 [0.979, 0.990]. Bar <= 0.8 in >= 2 datasets not met.
- Interval calibration (the other half of the bar) held everywhere: 80% coverage 80.3% [79.5, 81.2] / 80.7% [79.6, 81.9] / 79.8% [79.1, 80.7]. Task-start coverage 82.2% / 74.8% / 78.8%.
- Descriptive: cumulative-size evidence helps only late (SWE-smith turns 15-20 ratio 0.91-0.98 vs 1.02-1.12 at turns 3-14). SWE-agent: k-only baseline very strong at turns 11+ (runs have step caps).
- Practical limit: 10,415 of 20,315 SWE-rebench test runs were skipped (repositories with < 50 fit runs); repo-level tables need prior history in that repo.
- Conclusion: a cheap table forecast gives honest calibrated intervals but little point accuracy over a turn-count-only baseline; TokenCast's learned compositional models (14.5% MAE reduction vs their comparators, different accounting) find more.

### Round 6 errata and POST-HOC re-tabulation (2026-10-08, written before the re-tabulation runs)
Found on code review of `forecast_r6.py`, after the R6 results were published (article v1.3):
1. **Per-turn breakdown used an oracle baseline.** Line 108 accumulated `min(e_a, e_b)` per update point, i.e. the better of the two baselines chosen separately at every point with knowledge of the true final. No forecaster can do that, so the per-turn "baseline" MAE in `forecast_r6.out` is too low and the per-turn ratios are biased against the table. Article §5.8's "slightly worse before that [turn 15]" rests on these numbers. The pooled verdict is NOT affected: it uses one baseline for all points (lines 113-114) and the same one in every bootstrap resample.
2. **Reference baseline was chosen on test data** (lines 113-114 compare the two baselines' test MAE). This can only favour the baseline (conservative for the table); it picked the k-only table in all three datasets. Verdict unaffected.
3. **Design limitation (not a code error, matches the registration):** the table's cells `(turn bucket, size bucket)` pool all groups, and predict finals in absolute tokens, while the k-only baseline's cells are `(group, turn bucket)`. The "richer" forecast therefore discarded the group that the baseline kept. This is the likely reason it loses early and loses ~2x to the baseline on SWE-agent from turn 11. Round 7 tests the corrected design.
4. **Task-start line:** the k=1 MAE printed is the group median's error, not the table's; the k=1 coverage is the table's. Label corrected in the re-tabulation.

POST-HOC re-tabulation (`posthoc_r6.py`, descriptive, no bar, R6 verdict unchanged): same data, split (seed 20261014), cells, fallbacks and skip rule as R6. Per turn bucket, report table MAE vs (a) the fixed k-only baseline and (b) the group median, separately; and the table's k=1 MAE. Output `posthoc_r6.out/.json`.

## Round 7 pre-registration: during-run forecast with the group kept (2026-10-08, before any R7 forecast is computed)

Question: does cumulative size add point accuracy over the turns-only table when the forecast keeps the group (model or repository)?

Data and split: identical to R6 (three SWE sets, SWE-smith deduped by traj_id, estimated size = cumulative chars / 4 at each assistant turn; 70/30 fit/test by instance_id, `random.Random(20261014)`; test runs whose group has < 50 fit runs skipped, as in R6). Same split on purpose: R7 is a paired replacement of R6's table on identical test runs.

Forecaster: x = cumulative so far / fit-group median final. Cell = (group, turn bucket, x bucket), same buckets as R6. Point forecast of final size = median of fit finals in the cell; interval = cell p10-p90. Fallback when a cell has < 50 fit rows: (group, turn bucket) cell, then group.

Reference baseline, FIXED IN ADVANCE: the k-only table `(group, turn bucket)` (R6's winner in all three sets). Not re-chosen on test data.

Metrics and bar (unchanged from R6): update points k >= 3, pooled; MAE ratio vs the reference with 200 bootstrap resamples over test instance_ids (`random.Random(20261015)`, CI = 11th and 190th sorted values, as R6); 80% coverage CI from the same resamples. Pass in a dataset: ratio <= 0.8 AND CI upper < 0.8 AND coverage CI contains 80%. R7 passes if >= 2 of 3 datasets pass. One look.

Descriptive (no bar): per-turn-bucket MAE vs the reference (no oracle); share of update points answered by the full cell vs a fallback; variant B = pooled (turn, x) cells predicting final / group median, rescaled by the group median (the normalised version of R6's pooling), same metrics, no verdict.

Disclosure: the R7 design was motivated by reading R6's per-turn output on this same test split; the split is reused rather than redrawn because every instance has already been seen in R6 as fit or test. Stated expectation before running: FAIL. R6's table sat at 0.90-1.01 of the baseline; keeping the group should improve that, but reaching 0.8 with CI would require cumulative size to carry far more information than R6 suggested.
