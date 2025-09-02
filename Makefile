PY := python3

.PHONY: run dev install lint format pre-commit hooks

install:
	$(PY) -m pip install -r requirements.txt
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit install; \
	fi

run:
	$(PY) run_server.py --no-static --workers 1

dev:
	$(PY) run_server.py --static --reload --debug

pre-commit:
	pre-commit run --all-files || true
