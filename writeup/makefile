# Run from anywhere: make -C writeup exp plot
PYTHON ?= $(HOME)/.local/share/venv/bin/python
ROOT := $(abspath ..)

# at most 23 threads in total (see CLAUDE.md, Resource limits)
export OMP_NUM_THREADS := 23
export MKL_NUM_THREADS := 23
export OPENBLAS_NUM_THREADS := 23

.PHONY: exp plot progress view clean

# all four configurations, one after the other (never use make -j here)
exp:
	cd $(ROOT) && $(PYTHON) -m writeup.exp.main --search=bab --abstraction=crown
	cd $(ROOT) && $(PYTHON) -m writeup.exp.main --search=bab --abstraction=neuralsat
	cd $(ROOT) && $(PYTHON) -m writeup.exp.main --search=dpll --abstraction=crown
	cd $(ROOT) && $(PYTHON) -m writeup.exp.main --search=dpll --abstraction=neuralsat

plot:
	cd $(ROOT) && $(PYTHON) -m writeup.plot.cactus

progress:
	cd $(ROOT) && $(PYTHON) -m writeup.plot.progress

# dark-themed preview in the browser (settings in ~/.config/grip/settings.py)
view:
	cd $(ROOT)/writeup && GRIPHOME=$(HOME)/.config/grip grip -b writeup.md

# delete results and figure (keeps data/threads.csv); then: make clean exp plot
clean:
	rm -f $(ROOT)/writeup/data/csv/*.csv
	rm -f $(ROOT)/writeup/data/figure/cactus.png
