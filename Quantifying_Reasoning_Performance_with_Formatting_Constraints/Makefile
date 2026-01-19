# Build: make / make build
# arXiv tarball: make arxiv

MAIN = templateArxiv

build:
	latexmk -pdf -interaction=nonstopmode $(MAIN).tex

arxiv: build
	@mkdir -p arxiv-submit
	cp $(MAIN).tex $(MAIN).bbl references.bib PRIMEarxiv.sty arxiv-submit/
	@[ -d media ] && cp -r media arxiv-submit/ || true
	tar -czvf arxiv-submit.tar.gz -C arxiv-submit .
	@rm -rf arxiv-submit
	@echo "Created arxiv-submit.tar.gz — upload to arXiv"

clean:
	latexmk -C $(MAIN).tex
	rm -rf arxiv-submit arxiv-submit.tar.gz

.PHONY: build arxiv clean
