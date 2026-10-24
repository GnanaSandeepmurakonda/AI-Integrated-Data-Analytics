# Knowledge-based AI for Digital Transformation Readiness

Reconstructed reproducibility package for the manuscript **A knowledge-based AI decision-support model for predicting digital transformation readiness from enterprise innovation data**.

## Important provenance notice
This code was **reconstructed from the manuscript's methods description** and is **not the original experimental code**. The manuscript does not provide enough information to reconstruct the exact data-generating process, calibration procedure, or reported numerical results. The generated records are synthetic and are **not real enterprise observations**. Do not describe this repository as reproducing the paper's published tables until results have been checked and the manuscript updated accordingly.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
python train_evaluate.py
python stress_test.py
```

The scripts generate `synthetic_enterprise_data.csv`, `results/model_comparison.csv`, and `results/robustness.csv`.

## Design

- 800 synthetic enterprises with ten normalized indicators and a binary readiness label.
- Five editable knowledge rules: technology (0.55), organization (0.60), process (0.55), data (0.50), ecosystem (0.50).
- Knowledge features: five construct scores, minimum construct score, rule pass rate, and mean threshold gap.
- Raw / knowledge-based / hybrid comparisons for logistic regression, random forest, and gradient boosting.
- Stratified five-fold cross-validation and accuracy, F1, ROC-AUC, PR-AUC, Brier, and 3:1 false-negative cost.
- Three illustrative label regimes across 40 seeds.

## Limitations and outstanding work

This is a **new demonstration**, not verification of the original manuscript results. It does not yet implement the manuscript's exact five-fold isotonic calibration, bootstrap confidence intervals, permutation-importance figures, or original simulation generator (not disclosed in sufficient detail). The misspecification experiment is an illustrative rule reassignment, not necessarily the paper's original specification. The manuscript should be revised to report actual output from the available implementation, or the original experimental artifacts should be recovered.

## Data availability

All data are synthetic. `synthetic_enterprise_data.csv` can be regenerated using `generate_data.py` with seed 42.

## Additional reconstructed outputs

Run `python run_all.py` to execute all scripts. `analysis_outputs.py` additionally generates a stratified 400-resample bootstrap interval for KB logistic regression, two figures, and illustrative top-three decision levers. These outputs are reconstructed and must **not** be substituted for the manuscript's original figures or numeric claims without revision and verification.

## Manuscript-to-code verification

The current reconstructed dataset and model comparisons **do not numerically reproduce** the manuscript's reported values (e.g. LR/KB AUC 0.818). Any submission must either recover the actual original code/data or revise the manuscript's results, abstract, tables, figures, and claims to match an independently validated run of this repository. The synthetic label-generating process was inferred and is not claimed to be the original process. The current workflow does not reproduce the exact manuscript calibration pipeline.

## Repository organization

- `generate_data.py` — synthetic generator, feature schema, and five rules
- `synthetic_enterprise_data.csv` — generated demonstration cohort (not real-world evidence)
- `train_evaluate.py` — nine raw/KB/hybrid model comparisons
- `analysis_outputs.py` — supplementary plots, bootstrap CI, illustrative levers
- `stress_test.py` — three synthetic regimes, 40 seeds
- `run_all.py` — single-command execution
- `results/` — generated analysis artifacts
- `CITATION.cff`, `LICENSE` — citation metadata and open-source license

No claim of external validation or effectiveness of recommended interventions is made.
