.PHONY: all clean arxiv

MAIN := main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib sections/*.tex
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

arxiv: $(MAIN).pdf
	tar -czf pawp_tts_arxiv.tar.gz $(MAIN).tex $(MAIN).bbl sections/*.tex figures/

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).synctex.gz pawp_tts_arxiv.tar.gz
