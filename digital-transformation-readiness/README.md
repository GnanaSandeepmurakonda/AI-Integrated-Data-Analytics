# Knowledge-Based AI for Digital Transformation Readiness

Research reproducibility workspace for *A knowledge-based AI decision-support model for predicting digital transformation readiness from enterprise innovation data*.

## Status
**Work in progress — not yet validated for journal submission.** The associated manuscript describes a synthetic cohort of 800 enterprises, ten normalized indicators, five knowledge rules, raw/knowledge/hybrid models, and 40-seed robustness tests. This folder is a workspace for rebuilding and independently verifying those experiments. Do not cite it as a complete replication until scripts, generated data, metrics, tables and figures are published and checked.

## Research design
The five constructs are Technology (cloud_adoption, cyber_posture; threshold 0.55), Organization (leadership_commitment, digital_skill_share; 0.60), Process (process_digitization, agile_maturity; 0.55), Data (data_maturity, rd_intensity; 0.50), and Ecosystem (partner_ecosystem, innovation_pipeline; 0.50).

## Required next steps
1. Add an executable synthetic cohort generator with documented assumptions and seed 42.
2. Generate and commit the 800-row CSV; label it explicitly synthetic.
3. Add training/evaluation for logistic regression, random forest, gradient boosting and raw/KB/hybrid features.
4. Add out-of-fold cross-validation, calibration, bootstrap confidence intervals, permutation importance, cost metrics, and robustness regimes.
5. Generate the figures and tables from actual runs and reconcile every number with the manuscript; correct the manuscript if results differ.
6. Record Python/package versions, commands, provenance and limitations.

## Data availability
No enterprise observations were collected. Do not describe the reconstructed synthetic data as an external empirical dataset.
