# Build: make / make build
# arXiv tarball (with arxiv-latex-cleaner): make arxiv
# Clean: make clean

MAIN = main

build:
	latexmk -pdf -interaction=nonstopmode $(MAIN).tex

arxiv: build
	@mkdir -p arxiv-submit
	cp $(MAIN).tex $(MAIN).bbl references.bib PRIMEarxiv.sty arxiv-submit/
	@[ -d media ] && cp -r media arxiv-submit/ || true
	arxiv_latex_cleaner arxiv-submit --keep_bib --resize_images
	tar -czvf arxiv-submit.tar.gz -C arxiv-submit .
	@rm -rf arxiv-submit
	@echo "Created arxiv-submit.tar.gz — upload to arXiv"

clean:
	latexmk -C $(MAIN).tex
	rm -rf arxiv-submit arxiv-submit.tar.gz
	@rm -f templateArxiv.pdf templateArxiv.aux templateArxiv.bbl templateArxiv.blg templateArxiv.fls templateArxiv.fdb_latexmk templateArxiv.out templateArxiv.synctex.gz templateArxiv.log

.PHONY: build arxiv clean
