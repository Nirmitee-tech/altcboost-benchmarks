# Dataset sources

Download datasets from their providers and follow their terms. This repository distributes aggregate experimental results and code; it does not include the datasets or per-row predictions.

| Source | Kind | Provider link | Terms recorded during the study |
| --- | --- | --- | --- |
| Bank Account Fraud, Base | Synthetic applications generated from anonymized real data | [Feedzai](https://github.com/Feedzai/bank-account-fraud), [Kaggle source](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) | CC-BY-NC-SA-4.0 |
| PaySim | Synthetic mobile-money transactions | [PaySim](https://github.com/EdgarLopezPhD/PaySim), [Kaggle source](https://www.kaggle.com/datasets/ealaxi/paysim1) | CC-BY-SA-4.0 |
| Sparkov fraud-detection data | Synthetic card transactions | [Generator](https://github.com/namebrandon/Sparkov_Data_Generation), [Kaggle source](https://www.kaggle.com/datasets/kartik2112/fraud-detection) | CC0-1.0 |
| Elliptic | Real anonymized Bitcoin transactions, known-label subset | [Dataset paper](https://arxiv.org/abs/1908.02591), [Kaggle source](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) | CC-BY-NC-ND-4.0 |

Provider terms govern the source data. This table records the study's provenance and does not grant additional dataset rights.

## For whoever builds this

`experiment/multi_data.py` and `experiment/validation_data.py` contain predictor allowlists and excluded-field reasons. The study manifests record counting units, source file hashes, cohort hashes and temporal partitions. Synthetic test fixtures are generated independently and contain no original source rows.

Earlier ULB credit-card experiment code is retained in the model history and original notebook. Its source is [ULB credit-card fraud data](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud); its data are not included. IEEE-CIS access returned 403 during the study and no IEEE-CIS model result is claimed.

Decisions still needed: anyone downloading or reusing a source must check the provider's applicable terms; maintainers must document any added dataset before including a new benchmark claim.
