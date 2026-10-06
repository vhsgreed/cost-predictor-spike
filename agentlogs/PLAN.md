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
