# AI-Driven Cybersecurity Decision Intelligence

**Experimental research prototype, not submission-ready results.**

Title: *AI-Driven Cybersecurity Decision Intelligence: Cost-Sensitive Uncertainty Calibration for Adaptive Alert Escalation*.

This deliberately minimal repository contains the self-contained experiment in `experiment.py` and `requirements.txt`. The script generates **2 vector figures, 2 numeric tables, and run metadata**.

## Run
```bash
python -m pip install -r requirements.txt
python experiment.py --demo --out demo_results
python experiment.py --csv /path/to/CIC-IDS2017-attack-day.csv --out real_results
```

**DO NOT report demo results in a research paper.** Download a labeled attack-day CSV from the official CIC-IDS2017 source: https://www.unb.ca/cic/datasets/ids-2017.html. The file needs a Label column with both BENIGN and attack rows.

Method: fixed 60/20/20 train/calibration/test random stratified splits; gradient boosting; held-out isotonic and Platt calibration; compare fixed, expected-loss and capacity-limited policies. C_FN=30, C_FP=1, investigation budget 12% are **hypothetical**. Outputs: `Figure_1_calibration.pdf`, `Figure_2_capacity_cost.pdf`, `table_1_policy_results.csv`, `table_2_calibration.csv`, `run_metadata.json`.

**Limitations:** random splits may cause optimistic results from correlated traffic; test temporal/generalization splits, cost sensitivity, uncertainty intervals and novel baselines prior to submission. Top-k expected-loss escalation is a standard construction, not itself a new AI algorithm. Cite original CIC-IDS2017 dataset authors (Sharafaldin, Lashkari and Ghorbani, 2018) and preserve full data provenance. Verify manuscript claims manually.
