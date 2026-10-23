PYTHON ?= $(HOME)/.local/share/venv/bin/python

# at most 23 threads in total (see CLAUDE.md, Resource limits)
export OMP_NUM_THREADS := 23
export MKL_NUM_THREADS := 23
export OPENBLAS_NUM_THREADS := 23

.PHONY: test lint

test:
	$(PYTHON) -m pytest -q test

lint:
	uvx ruff check --select E,W,F,N,I --line-length 79 \
		abstraction search test writeup
