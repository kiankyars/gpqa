# Build: make / make build
# arXiv tarball (with arxiv-latex-cleaner): make arxiv
# Clean: make clean

MAIN = main

build:
	latexmk -pdf -interaction=nonstopmode output/$(MAIN).tex

arxiv: build
	@mkdir -p arxiv-submit
	cp $(MAIN).tex $(MAIN).bbl references.bib PRIMEarxiv.sty arxiv-submit/
	@[ -d media ] && cp -r media arxiv-submit/ || true
	arxiv_latex_cleaner arxiv-submit --keep_bib --resize_images
	tar -czvf output/arxiv-submit.tar.gz -C arxiv-submit .
	@rm -rf arxiv-submit arxiv-submit_arXiv
	@echo "Created arxiv-submit.tar.gz — upload to arXiv"

.PHONY: build arxiv clean
