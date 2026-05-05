.PHONY: test-fast test-integration compile reproduce-synthetic

PYTHON ?= python
PYTHONPATH ?= src

compile:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m py_compile src/lidar_snow_filter/*.py tools/*.py

test-fast:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest tests.test_bug_fixes -v

test-integration:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/test_integration.py -q
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) tests/test_reproducibility.py

reproduce-synthetic:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m lidar_snow_filter.synthetic_data_generator --num_scans 2 --seed 42 --contaminate
