# What an AI coding-agent run will cost: what can be known before it starts, and what can only be seen while it runs

**Status:** DRAFT v0.1, 2026-10-07. Written by the assistant from the lab record; Karl Sundström to review, rewrite and approve before anything is public.
**Author:** Karl Sundström (Agent Recourse).
**Reproduction:** every number below is produced by a script in this repository, from public data at pinned revisions. The pre-registration file `agentlogs/PLAN.md` holds each bar with the commit in which it was written, before the data it was tested on.

---

## Abstract

We asked whether the dollar cost of an AI coding-agent run can be predicted before the run starts, and if not, what can be said instead. On 424,108 priced sessions from the public AgentLogs dataset ($743,263 of model spend), with all bars registered before testing and repositories held out, the answer is mostly no. Knowing the model and what triggered the run narrows the cost range for about a third of situations (5 of 15 groups on a locked test). Prompt text, repository size and language, and the user's own past runs do not narrow it further (0/10, 0/37, 0/37 groups pass). Even knowing the true number of calls afterwards leaves a 3x spread, and running the identical task three times on the identical setup spreads cost by a median 1.55x. Two things do hold. A floor ("this run will cost at least X") is right 90% of the time, on the locked test and again on a second, independent benchmark dataset. And during a run, crossing what 90% of similar runs cost is a strong warning sign: across 171,000 runs from three public software-engineering benchmarks, runs in the top tenth of cost resolved their task 2.5 to 7 times less often than typical runs. The practical consequence is a different instrument from the one we set out to build: not a price tag before you press send, but a floor before, and an alarm during.

## 1. Introduction

Agent harnesses show cost after the fact, if at all. Users of routing services such as OpenRouter often see $0, because the harness never receives the provider's price (we tested an upstream fix, Hermes PR #128755). The question that motivated this work is simpler and harder: when someone types a task into an agent, can we tell them what it will cost?

A weather forecast is a useful comparison. It is accurate for tomorrow and useless for next month, because small differences compound. An agent run is a sequence of model calls, each depending on the last: whether the first attempt works, whether a test fails, whether the agent gets stuck. If each step is uncertain, a forecast made before step one should be wide. This study measures how wide, and what, if anything, narrows it.

## 2. Hypotheses

All bars were committed before the data they were tested on (commit in brackets).

| ID | Claim | Pass bar | Registered |
|---|---|---|---|
| H1 | Model x trigger x prompt-length groups give narrower cost ranges than the model alone | width <= 0.8x model-only, CI upper < 0.8x, coverage CI contains 80%, in >= 3 of 15 groups | `fc4ec02` |
| H2 | A loop detector flags runs over-represented in the most expensive tenth | ratio >= 2.0 AND hand-check false-positive rate <= 30% | `fc4ec02` |
| H3 | Flagged loops predict failure | exploratory | `fc4ec02` |
| H4 | A floor (group p10) is a reliable "at least X" | floor coverage CI contains 90% overall and in >= 2/3 of groups | `88dfab6` |
| H5 | Prompt-text clusters narrow ranges | >= 2 clusters pass the H1 width bar | `78fa206`, final `cff0cea` |
| H6 | Repository size and language narrow ranges further | >= 1/3 of groups pass | `9638be8` |
| H7 | The user's own recent runs narrow ranges further | >= 1/3 of groups pass | `9638be8` |
| R4 | H1 and H4 replicate on other public datasets | pass in every evaluable dataset, >= 2 evaluable | `2b8c3f7` |
| O1 | Runs in the cost tail resolve less often | tail/typical resolve ratio <= 0.8, CI upper < 0.8, in >= 2 datasets | `272c139` |

## 3. Method

**Data.** `risenlab/agentlogs` (CC BY 4.0, revision `04013a44d3432c6654bca1dcd4a01218a9406b80`): step logs of GitHub-triggered coding-agent sessions. All 276 log shards were parsed; sessions with at least 4 priced calls were kept (424,108 sessions, 30,143 repositories). Cost = tokens x list price per model, with cached input priced separately. Repository metadata (1.8M repositories) joined to 100% of sessions.

**Splits.** Development shards were explored freely; five validation shards were used once per decision; ten locked-test shards, drawn with a fixed seed, were used exactly once. Later rounds split by repository (or by task for benchmarks) so that every tested range is scored on repositories the model never saw.

**Scoring.** A range is the p10 to p90 of training costs in a group (nominally 80%). Its width is p90/p10. A group passes if its width is at most 0.8x the model-only width, the 90% bootstrap confidence interval of that ratio (resamples over training repositories) stays below 0.8, and the share of held-out runs inside the range is consistent with 80%.

**External data.** Replication used `Exgentic/agent-llm-traces-v2` (10,056 benchmark runs with reported agent cost) and `MaxDevv/real-pi-coding-agent-traces-sessions` (1,291 developer sessions). The failure question used `SWE-bench/SWE-smith-trajectories`, `nebius/SWE-rebench-openhands-trajectories` and `nebius/SWE-agent-trajectories` (171,000 runs with a test-verified resolved label). These have no billed cost; run size is estimated as characters / 4 and called "estimated tokens" throughout.

## 4. Results

### 4.1 Cost is skewed and concentrated

Every model's mean sits well above its median (Fig. 1). The most expensive 1% of sessions carry 17% of all spend and the most expensive 10% carry 46% (Fig. 4). Within each model x trigger group, sessions above the group's p90 still carry 40% of that group's spend.

![Fig. 1](../agentlogs/figures/fig1_cost_by_model.svg)
![Fig. 4](../agentlogs/figures/fig4_spend_concentration.svg)

### 4.2 Before the run: only the floor and the coarse group hold

| Test | Result |
|---|---|
| H1 model x trigger, validation | 6/15 groups pass: **PASS** |
| H1 model x trigger, locked test (15,349 sessions, 6,509 repos) | 5/15 pass: **PASS**; best group 5.4x vs 12.8x model-only. 4 groups were wider than model-only. |
| H4 floor, locked test | held 90% [90%, 90%], 10/15 groups: **PASS** |
| H5 prompt-text clusters (92,429 sessions, 13,535 repos) | 0/10: **FAIL**. Two clusters met the width criteria but failed coverage on held-out repos. |
| H6 repository language and size (66,603 held-out sessions) | 0/37: **FAIL**, median width ratio 0.99 |
| H7 user's last 10 runs | 0/37: **FAIL**, median width ratio 0.88 |

![Fig. 2](../agentlogs/figures/fig2_hypotheses.svg)

Prompt text loses even to the coarse grouping: model x trigger ranges have a median width of 11.3x, prompt clusters 16.4x. Many prompts are short ("fix the CI", "did you do it?") and say little about the work they will cause.

Fig. 3 shows what confidence costs. A range that contains the true cost half the time runs from 46% below to 80% above the typical cost; at 90% confidence it runs from 78% below to 309% above.

![Fig. 3](../agentlogs/figures/fig3_confidence_vs_range.svg)

### 4.3 Why: the information does not exist before the run

Two measurements bound what any pre-run predictor can achieve.

- **Ceiling.** Even with the run's true call count known afterwards, the median group still spreads 3.0x (1.9x with the true token count) (Fig. 5, in-sample; on held-out development data the same ceiling was 4.3x and 5.3x).
- **Noise floor.** In `melissapan/swe-bench-lite-agent-traces-v14`, 21 harness x model setups ran the same 30 tasks three times each. Identical task, identical setup: the most and least expensive of the three runs differ by a median 1.55x (p90 2.86x). This was seen during the data survey and is reported as exploratory.

![Fig. 5](../agentlogs/figures/fig5_ceiling.svg)

### 4.4 Replication

The registered rule required two independent evaluable datasets; only one qualified, so the replication is formally **not evaluable**. One candidate set was dropped before any cost was read because 94% of its runs duplicated another. Under a secondary minimum group size added before cost was read, the Exgentic benchmark data reproduced the floor: held **90.1% [89.1%, 91.4%]**, 48 of 60 groups. Its range result (29/60 groups pass) mostly reflects baselines that mix six benchmarks; where the baseline is narrow, 1 of 11 groups passes.

### 4.5 During the run: the tail fails

| Dataset | Runs | Resolve rate, top 10% by size | Resolve rate, bottom half | Ratio [90% CI] |
|---|---|---|---|---|
| SWE-smith | 23,821 | 20.7% | 51.8% | **0.40** [0.37, 0.43] |
| SWE-rebench OpenHands | 67,074 | 22.6% | 52.9% | **0.43** [0.38, 0.47] |
| SWE-agent | 80,036 | 3.5% | 25.4% | **0.14** [0.12, 0.16] |

O1 passes in all three. The resolve rate falls almost steadily with size decile (SWE-smith: 62% in the cheapest tenth, 21% in the most expensive). A backtest that stops each run at its group's p90 saves 13.5%, 2.6% and 21.8% of estimated tokens while losing 4.9%, 5.2% and 2.1% of resolved runs: a good trade in two datasets, a bad one in the third.

### 4.6 Case study, n = 1: long conversations

The author's own agent logs (numbers only, 2,014 calls) show a pattern absent from all public datasets, which contain task runs only. In one conversation of 601 calls, 93% of 76.6M tokens were the model re-reading earlier context; context per call grew from 17k to 217k tokens (Fig. 6). Modelled as one fresh session per turn, the same work would have read about a quarter of the input (4.7x less). Some of that carried context is memory the user wants; the model cannot price that.

![Fig. 6](../agentlogs/figures/fig6_carry_vs_work.svg)

## 5. Discussion

What failed, and why it matters: H2 to H3 and H5 to H7 all tried to read the future of a run from its starting conditions. The ceiling and the noise floor say this cannot work well: the spread is created during the run, by events no input describes.

What can be built honestly from this:

1. **Before sending:** a floor and a typical value per model and trigger ("at least $0.54, usually $1.29"), with the range shown as the wide thing it is.
2. **During the run:** an alarm when the run crosses what 90% of similar runs cost. On benchmarks this is where failing runs collect. Whether stopping there is worth it depends on the setup, which the backtest makes explicit.
3. **For long conversations:** separate the work an agent does from the context it carries, and show both.

**Limits.** AgentLogs is one population (GitHub-triggered agents, mostly Claude Sonnet). The failure result uses estimated tokens on benchmark tasks, and difficulty drives both cost and failure, so it shows association, not that stopping saves money without loss. The replication did not reach its registered standard. The conversation case study is one user.

**Disclosures.** Errors and changes, all logged in `PLAN.md` with dates:
- The first H5 canary script checked the width estimate but omitted its confidence interval and appeared to pass 4/7 clusters; corrected before reporting, 0/7.
- The H2 hand-check rater's answers drifted with case order.
- Round 4 eligibility and the E3 split were amended before cost was read, from group sizes only.
- Round 4 computed its results, crashed in the summary print, and was re-run with fixed seeds (identical numbers).
- One benchmark file set held 2,255 duplicate rows; removing them changed the SWE-smith ratio from 0.38 to 0.40, verdict unchanged.
- The ceiling in Fig. 5 is in-sample and tighter than the held-out estimate.

## 6. Data and code

Repository: (to be published). Inputs are public and pinned by revision. Scripts: `run_full.py`, `full_tables.py`, `validate_h1.py`, `locked_test.py`, `h5_final.py`, `confirm_r3.py`, `replicate_r4.py`, `outcome_r5.py`, `figures.py`. Full table of 175 model x trigger groups: `agentlogs/figures/groups_full.csv`.

## Open items before publication (for Karl)

- [ ] Rewrite in your own voice; this is a reconstruction.
- [ ] Decide whether the n = 1 case study (your own logs) stays.
- [ ] Publish the lab repository (needs a GitHub remote; datasets and message-bearing files are gitignored).
- [ ] Check licences: AgentLogs CC BY 4.0, SWE-smith MIT, Nebius CC BY 4.0, melissapan CC BY 4.0; Exgentic v2 lists none, pi sessions "other" (only aggregate numbers are used).
- [ ] Port into agentrecourse.org `/journal` (site restructure not started).
