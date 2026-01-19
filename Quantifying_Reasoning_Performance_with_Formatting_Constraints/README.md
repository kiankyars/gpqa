# Paper: Quantifying Reasoning Performance with Formatting Constraints

## Build

```bash
make          # or: latexmk -pdf templateArxiv.tex
```

## arXiv submission

```bash
make arxiv    # creates arxiv-submit.tar.gz
```

Upload `arxiv-submit.tar.gz` to [arXiv](https://arxiv.org/submit). The tarball includes `.tex`, `.bbl`, `.bib`, `PRIMEarxiv.sty`, and `media/` (if present). Including `.bbl` avoids BibTeX on arXiv’s side.

Optionally run [arxiv-latex-cleaner](https://github.com/google-research/arxiv-latex-cleaner) on the extracted folder before re-tarring if you want comments stripped and figures compressed.
