# Run from anywhere: make -C writeup exp plot figures walkthrough view
#   backend/   the experiment and the plots (Python, run from the project root)
#   frontend/  the explanation website (Hugo)
PYTHON ?= $(HOME)/.local/share/venv/bin/python
ROOT := $(abspath ..)

# at most 23 threads in total (see CLAUDE.md, Resource limits)
export OMP_NUM_THREADS := 23
export MKL_NUM_THREADS := 23
export OPENBLAS_NUM_THREADS := 23

.PHONY: exp plot progress figures walkthrough view site clean

# --- backend ---------------------------------------------------------------

# every search with every abstraction, one after the other (never use make -j
# here); a finished configuration is skipped, so a rerun only adds what is new
SEARCHES := bab dpll
ABSTRACTIONS := ibp zonotope crown alphacrown neuralsat

exp:
	cd $(ROOT) && for s in $(SEARCHES); do for a in $(ABSTRACTIONS); do \
		$(PYTHON) -m writeup.backend.exp.main --search=$$s --abstraction=$$a \
		|| exit 1; done; done

# cactus figures and summary table, written into the frontend
plot:
	cd $(ROOT) && $(PYTHON) -m writeup.backend.plot.cactus

progress:
	cd $(ROOT) && $(PYTHON) -m writeup.backend.plot.progress

# the frontend's layer figures and its generated walkthrough page
figures:
	cd $(ROOT) && $(PYTHON) -m writeup.backend.plot.relaxation

walkthrough:
	cd $(ROOT) && $(PYTHON) -m writeup.backend.exp.walkthrough

# delete results and figures (keeps backend/data/threads.csv); then: make clean exp plot
clean:
	rm -f $(ROOT)/writeup/backend/data/csv/*.csv
	rm -f $(ROOT)/writeup/frontend/static/figure/cactus-*.png $(ROOT)/writeup/frontend/data/summary.json

# --- frontend --------------------------------------------------------------

# live preview in the browser, and the static build in frontend/public/
view:
	cd $(ROOT)/writeup/frontend && hugo serve --noHTTPCache --disableFastRender

site:
	cd $(ROOT)/writeup/frontend && rm -rf public && hugo
