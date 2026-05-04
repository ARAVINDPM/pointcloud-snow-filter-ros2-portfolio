.PHONY: test-fast test-integration compile reproduce-synthetic

PYTHON ?= python

compile:
	$(PYTHON) -m py_compile filters.py metrics.py benchmarking.py config.py synthetic_data_generator.py contaminate_with_synthetic_snow.py tools/example_workflow.py tools/test_and_visualize.py tools/evaluate_all_frames.py tools/visualize_and_animate.py

test-fast:
	$(PYTHON) -m unittest tests.test_bug_fixes -v

test-integration:
	$(PYTHON) -m pytest tests/test_integration.py -q
	$(PYTHON) tests/test_reproducibility.py

reproduce-synthetic:
	$(PYTHON) synthetic_data_generator.py --num_scans 2 --seed 42 --contaminate
