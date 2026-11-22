.PHONY: all clean arxiv figures

MAIN := main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib sections/*.tex figures/*.tex figures/*.pdf
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

figures: ## Regenerate all paper figures + README diagram export
	python3 figures/make_figures.py
	cd figures && pdflatex -interaction=nonstopmode architecture_standalone.tex >/dev/null
	pdftocairo -svg figures/architecture_standalone.pdf ../docs/figures/architecture_paper.svg
	pdftocairo -png -r 220 -singlefile figures/architecture_standalone.pdf ../docs/figures/architecture_paper
	python3 -c "import re,pathlib;p=pathlib.Path('../docs/figures/architecture_paper.svg');s=p.read_text();m=re.search(r'<svg[^>]*>',s);p.write_text(s[:m.end()]+'<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>'+s[m.end():])"
	rm -f figures/architecture_standalone.pdf figures/architecture_standalone.aux figures/architecture_standalone.log

arxiv: $(MAIN).pdf
	tar -czf pawp_tts_arxiv.tar.gz $(MAIN).tex $(MAIN).bbl sections/*.tex figures/

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).synctex.gz pawp_tts_arxiv.tar.gz
