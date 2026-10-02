SOURCES	= \
	main.tex

.PHONY: default all rebuild clean

default: all

all:
	pdflatex -interaction=nonstopmode main
	bibtex main
	pdflatex -interaction=nonstopmode main
	pdflatex -interaction=nonstopmode main
#	dvips -Ppdf -G0 -t letter -e 0 -o main.ps main.dvi
#	ps2pdf -dPDFSETTINGS=/prepress  main.ps

refs:
	bibtex main
clean:
	-rm -rf main.aux main.bbl main.blg main.dvi main.log main.pdf main.ps *~
sources:
	@echo  ${SOURCES}
