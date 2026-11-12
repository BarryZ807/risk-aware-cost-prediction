# Risk-aware healthcare cost prediction: evaluation code

Evaluation code from my independent research on predicting healthcare costs when only a small share of people can be acted on.

A care team can reach only a fixed number of members. Most cost models are tuned for the average person. This work asks what a model should predict when capacity is fixed and only the top of the cost distribution matters, and how to tell whether it actually helps. This repository holds the second half: the code that decides whether a model helps.

## What the evaluation measures

- **Spending captured at a fixed head count.** Flag the top k% of people by score, then ask what share of total spending those people account for. This is the number a capacity-limited team cares about.
- **Whether a difference is real.** A paired bootstrap that resamples people, not rows, with survey weights, for model-against-baseline differences.
- **Whether the probabilities can be trusted.** Survey-weighted calibration error and decision-curve net benefit.
- **Whether the outputs agree with each other.** Audits that count quantile predictions that cross and nested risk predictions that are out of order.

## Results

The model under evaluation is a multi-task FT-Transformer in PyTorch that predicts expected cost, 7 tail quantiles and nested high-cost risk in one model. It was trained and evaluated on 72,818 person-years of US national survey data (MEPS, 2018–2023).

Share of total spending that belongs to the people flagged, model against a calibrated logistic baseline:

| People flagged | Model | Baseline | p |
| --- | --- | --- | --- |
| Top 0.5% | 6.24% | 3.11% | 0.018 |
| Top 1% | 8.84% | 6.36% | 0.009 |
| Top 2% | 14.27% | 11.84% | 0.028 |
| Top 5% | 26.62% | 24.87% | 0.072 |
| Top 10% | 38.87% | 38.16% | 0.25 |

- At the top 0.5%, the model captures 2.0 times the spending of the baseline.
- The gain is significant across the top 0.5–2%. At the 5% endpoint the difference is +1.89 points with p = 0.072, and I report it as not significant.
- 0 quantile crossings in 745,560 pairs, against 1.54% for a bank of LightGBM quantile regressors and logistic classifiers.
- Expected calibration error 0.006.

Inference is a paired, person-clustered bootstrap with survey weights and 2,000 resamples. The full study is 60 seeded, reproducible runs.

## What is here

```
evaluation/
  capture.py           share of spending captured at a fixed head count
  paired_bootstrap.py  paired, person-clustered bootstrap for model-against-baseline differences
  calibration.py       survey-weighted calibration error and decision-curve net benefit
  structure.py         audits for quantile crossings and risk-order violations
tests/                 the evaluation code, checked on synthetic data and worked examples
```

The code needs only NumPy.

## What is not here

- **The model and the training pipeline.** They are not published.
- **Data.** MEPS public-use files are available from AHRQ. They are not redistributed here.
- **Run outputs and trained weights.**
- **The manuscripts.** The lead paper is prepared for submission; companion papers are in preparation.

Three patent applications related to this work are pending (China, CNIPA; filed 2026).

## Run the tests

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

The tests use synthetic data only. They check that the capture metric matches worked examples, that the bootstrap separates a good ranking from noise and is reproducible, that calibration error is zero for perfect calibration, and that the audits catch a planted violation.

This is a published snapshot, not a development history.

© Barry (Zehui) Zhang. Shared for portfolio review. Not licensed for reuse.
