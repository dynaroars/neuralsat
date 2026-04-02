all: clean build upload backup

clean:
	rm -rf dist

build:
	python3 -m build

upload:
	twine upload dist/*
