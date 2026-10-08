# Agent cost prediction: lab record

Code, pre-registration and outputs behind the article draft `journal/cost-prediction-draft.md`
("What an AI coding-agent run will cost: what can be known before it starts, and what can only be seen while it runs").

- `agentlogs/PLAN.md`: every hypothesis and pass bar, with the date it was written; check commit history to see that each bar predates the data it was tested on. Also the change log of errors and amendments.
- `agentlogs/*.py`: analysis scripts. `agentlogs/*.out` / `*.json`: their committed outputs.
- `agentlogs/figures/`: figures 1 to 5 and `groups_full.csv` (175 model x trigger x prompt-length groups: n, p10, p50, p90, mean).

No raw dataset rows are committed (`agentlogs/handcheck.html` holds derived tool-call patterns for 60 AgentLogs sessions, CC BY 4.0, used for the H2 hand-check). All inputs are public and downloaded at pinned revisions (see `PLAN.md`).

## Environment

Python 3.14. `pip install -r requirements.txt`. About 60 GB of free disk for the AgentLogs pass (shards are deleted after parsing), plus about 10 GB for the external datasets. H5 also needs [Ollama](https://ollama.com) with `nomic-embed-text`.

## Reproduce

Run from `agentlogs/`, in this order:

```
python test_lab.py                   # unit tests (must pass first)
python build_dataset.py ...          # dev/val/test sets from AgentLogs shards (see PLAN.md for shard lists)
python validate_h1.py                # H1 validation
python locked_test.py                # H1 + H4 locked test (one look)
python run_full.py                   # all 276 shards -> full_rows/
python full_tables.py && python figures.py
python h5_final.py                   # H5
python explore_r3.py && python confirm_r3.py        # H6, H7
python replicate_r4.py               # R4 (needs external/ downloads, see PLAN.md round 4)
python outcome_r5.py                 # O1 (needs external/ downloads, see PLAN.md round 5)
python rescore_dev.py && python revision_v02.py && python alarm_runtime_r5.py   # v0.2 supplementary analyses
python forecast_r6.py                # R6
python posthoc_r6.py && python forecast_r7.py   # R6 per-turn correction (post-hoc), R7
```

Seeds are fixed in each script. Bootstrap CIs may differ in the last digit across platforms.

## Not included

Box A of the article (one long conversation, Fig. 6) uses the author's own agent logs. Those logs and the scripts that read them are kept on local hardware and are not published; Box A is illustrative only.

## Licence

Code: MIT. Data licences are those of the source datasets (AgentLogs CC BY 4.0; SWE-smith MIT; Nebius datasets CC BY 4.0; melissapan CC BY 4.0; Exgentic v2 no licence listed and pi sessions "other": only aggregate statistics from these two are reported).
