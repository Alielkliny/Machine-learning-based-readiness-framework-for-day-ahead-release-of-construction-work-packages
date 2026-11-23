# Release model for planned street-works packages

Decides each day which planned work packages to release for the next working day, using only records dated before the decision. An as-of evidence ledger reconstructs, at each daily cutoff, what was known about every planned package; a gradient-boosting readiness model with gates and online recalibration decides which packages to release, and the release policies are scored on crew-day losses with authority-grouped folds and held-out periods. A BIM demonstration links the same ledger logic to the elements and tasks of an open IFC model.

## Scope of this repository

This repository holds the code of the method, the comparison methods and the evaluation, with the split,
protocol and settings files that the scripts read. It does not hold the input data, run outputs, figures or tables.

## Repository layout

| Path | Contents |
|---|---|
| `src/fetch_month.sh`, `fetch_and_extract.sh`, `fetch_small_topics.sh`, `stream_zip.py`, `extract_events.py`, `extract_generic.py` | Download and extraction of the open street-works archive |
| `src/ledger.py`, `build_ledger.py`, `run_builds.sh`, `run_builds_test.sh` | As-of evidence ledger at daily cutoffs |
| `src/splits.py` | Declared splits and authority hold-out groups |
| `src/methods.py` | Release policies scored on the same snapshots |
| `src/dev_eval.py`, `dev_c2.py`, `dev_summary.py` | Development evaluation of the framework and its ablations |
| `src/test_eval.py` | Scoring of the check period and the held-out periods |
| `src/post_analysis.py`, `post_extra.py`, `review_extra.py`, `worked_example.py` | Analyses from the saved per-row results |
| `src/bim_parse.py`, `bim_parse_planned.py`, `bim_relations.py`, `bim_relations_design.py`, `bim_demo.py` | BIM demonstration on the open IFC model |

## Environment

The code was run with Python 3.13 on Windows 11. Install the packages with

```
python -m pip install -r requirements.txt
```

The versions in `requirements.txt` are those of the Python environment the code was run in.

## Input data

The input data are public and are not redistributed here. Place them as follows.

- Street Manager open data archive of the UK Department for Transport (monthly permit, activity and section 58 files), downloaded by `src/fetch_and_extract.sh` and `src/fetch_small_topics.sh` into `data/raw/streetmanager/` and extracted to `data/processed/events/`.
- Schependomlaan open BIM dataset (TU Eindhoven, CC BY 4.0): design and as-planned IFC models and the 4D event log, at the paths written in the `bim_*.py` scripts.

## Running the code

```
bash src/fetch_and_extract.sh YYYY-MM [YYYY-MM ...]   (the months listed in src/fetch_small_topics.sh)
bash src/fetch_small_topics.sh
bash src/run_builds.sh
bash src/run_builds_test.sh
python -P src/dev_eval.py
python -P src/dev_c2.py g
python -P src/dev_summary.py results/dev_eval/fold_rows.parquet results/dev_eval/summary.json
python -P src/test_eval.py SPLIT [g]
python -P src/post_analysis.py
python src/post_extra.py
python src/review_extra.py
python src/worked_example.py
python src/bim_parse.py
python src/bim_parse_planned.py
python src/bim_relations.py
python src/bim_relations_design.py
python src/bim_demo.py
```

Run every command from the repository root.
