# ALTCBoost research benchmarks

Source code and measured fraud-detection benchmarks published by Nirmitee. This is a research prototype. Novelty, general superiority, future false-positive control and production readiness remain unproven.

**[Read the benchmark page](https://nirmitee-tech.github.io/altcboost-benchmarks/)** or inspect the [accuracy and novelty audit](docs/novelty-accuracy-audit.md).

The publication keeps two experiments separate. V8 compares six training policies on development validation cohorts. The earlier V6 study evaluates a different model on later test intervals, with thresholds selected using earlier calibration data. Those sources have since informed development, so the old test intervals are consumed. No V8 independent confirmation result is reported.

The proposed V8 adaptive policy did not beat the strongest simple training policy on any of the four development sources. Its tie check reduced objective evaluations on BAF from 296 to 236 with matching validation metrics. That single implementation result does not establish a general training speedup. The weak BAF detection result and earlier V6 Elliptic failure remain visible in the published tables.

The benchmark page reports recall and precision alongside false-positive counts and overall accuracy. At the 0.1% validation false-positive budget, the BAF candidate catches 35 of 1,129 fraud cases despite 98.808% overall accuracy. A high overall accuracy is therefore insufficient evidence of useful fraud detection.

## For whoever builds this

Skip this section unless running or reviewing the code.

Use Python 3.12 or a compatible environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make test PYTHON=python
make audit PYTHON=python
make benchmark-page PYTHON=python
```

The dependency ranges in `requirements.txt` support installation and CI; they do not guarantee bit-for-bit historical results. `experiment/requirements-tested.txt` records the original local package versions. The pinned Kaggle image is recorded in each frozen `v8-study/*/job/kernel-metadata.json`. Use the frozen notebook and image for the closest hosted reproduction. Regenerating models with different packages can change scores and tie behavior.

The clean publication copy passed 135 unit tests and 66 integration tests locally. The optional GitHub Actions template is `ci/tests.yml`; automatic test CI is not enabled in this release because the publishing CLI token lacks workflow scope. The Pages publication has its own deployment status.

`make audit` verifies source/cohort hashes and executed-code snapshots, then recomputes accuracy, recall, precision and other descriptive metrics from aggregate counts. It requires no dataset download and does not train a model. `make benchmark-page` rebuilds the static site in `docs/` from those aggregate results. The tests use synthetic fixtures; the page integration test serves a localhost HTTP endpoint.

To rerun a development benchmark after obtaining its dataset separately:

```sh
PYTHONPATH=experiment python experiment/v8_research.py paysim /path/to/downloaded/datasets /path/to/new-output
```

The development runner uses training-only preprocessing and the original temporal partitions. It fits and scores training/validation only after source discovery. It does not compute V8 calibration or final predictions. `v8_job.py` packages the controlled private Kaggle study, including the frozen plan, references and hash guards. The original hosted jobs are private; the frozen submitted source snapshots and aggregate outputs are provided here for review.

| Location | Contents |
| --- | --- |
| `experiment/altcboost*.py` | Model revisions and their dependencies. |
| `experiment/tests/` | Unit and integration suites. |
| `accuracy-novelty-audit/` | Recomputed validation metrics, units, source hashes and audit scope. |
| `v8-study/` | Six-policy development experiment, frozen jobs, aggregate outputs and verification. |
| `v7-study/` | Earlier prior-revision experiments, including negative findings. |
| `validation-study/` | Earlier V6 final-interval results, baseline selection, repeat seeds and uncertainty limitations. |
| `docs/index.html` | Public static benchmark page. |
| `docs/data/` | Aggregate CSV downloads used by the page. |

The V8 primary development selection key averages recall across empirical FPR budgets 0.05% and 0.1%, then breaks ties by lower mean FPR and higher average precision. The page's validation table reports the single 0.1% budget. V6 final results use earlier calibration thresholds targeting 0.1%, whose later observed FPR can exceed that target. Model versions, cohorts, budgets and timing claims must not be combined across those experiments.

Three benchmark sources are synthetic. Elliptic is a known-label Bitcoin AML transfer task with unknown labels and explicit neighbour-derived predictors excluded. It does not establish independent payment-fraud validation. Raw data, row identifiers, per-row prediction exports, credentials, virtual environments and trained model binaries are excluded. See [dataset sources and terms](DATASETS.md).

No paired error overlap, new threshold choice or independent uncertainty estimate can be reconstructed from confusion counts alone. Existing validation binomial intervals do not correct for adaptive model/cutoff selection or temporal dependence. Published optimizer baselines have not been run head to head, so the code and mathematical derivation do not establish a priority claim.

Decisions still needed: maintainers must freeze a candidate and comparison protocol before fresh confirmation; the application owner must choose minimum useful recall and precision based on alert capacity and missed-fraud cost. The current release records measured research results and unresolved claims.
