# Compile from this directory: make

.PHONY: all refs clean

all:
	pdflatex -interaction=nonstopmode main
	bibtex main
	pdflatex -interaction=nonstopmode main
	pdflatex -interaction=nonstopmode main

refs:
	bibtex main

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf main.toc main.synctex.gz
