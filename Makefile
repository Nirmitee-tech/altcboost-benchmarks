PYTHON ?= python3

.PHONY: test unit integration audit benchmark-page
test: unit integration
unit:
	cd experiment && PYTHONPATH=.:tests $(PYTHON) -m pytest -q tests/test*unit.py
integration:
	cd experiment && PYTHONPATH=.:tests $(PYTHON) -m pytest -q tests/test*integration.py
audit:
	PYTHONPATH=experiment $(PYTHON) experiment/verify_v8.py v8-study
	PYTHONPATH=experiment $(PYTHON) experiment/accuracy_audit.py v8-study accuracy-novelty-audit
benchmark-page:
	PYTHONPATH=experiment $(PYTHON) experiment/publish_benchmark.py .
