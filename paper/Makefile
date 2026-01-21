# Build: make / make build
# arXiv tarball (with arxiv-latex-cleaner): make arxiv
# Clean: make clean

MAIN = main

build:
	latexmk -pdf -interaction=nonstopmode -output-directory=../output $(MAIN).tex

arxiv: build
	@mkdir -p arxiv-submit/output
	cp ../output/summary_table.tex ../output/compliance_table.tex \
		../output/accuracy_subdomain_prompt.tex \
		../output/token_usage.pdf ../output/accuracy_with_error_bars.pdf \
		arxiv-submit/output/
	cp $(MAIN).tex ../output/$(MAIN).bbl references.bib PRIMEarxiv.sty arxiv-submit/
	sed 's|../output/|output/|g' arxiv-submit/$(MAIN).tex > arxiv-submit/$(MAIN).tex.tmp && mv arxiv-submit/$(MAIN).tex.tmp arxiv-submit/$(MAIN).tex
	@[ -d media ] && cp -r media arxiv-submit/ || true
	arxiv_latex_cleaner arxiv-submit --keep_bib --resize_images
	tar -czvf output/arxiv-submit.tar.gz -C arxiv-submit .
	@rm -rf arxiv-submit arxiv-submit_arXiv
	@echo "Created arxiv-submit.tar.gz — upload to arXiv"

.PHONY: build arxiv clean
