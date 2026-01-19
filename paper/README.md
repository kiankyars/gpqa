# Paper: Quantifying Reasoning Performance with Formatting Constraints

## Build

```bash
make          # or: latexmk -pdf main.tex
```

## arXiv submission

```bash
make arxiv    # build, run arxiv-latex-cleaner (strip comments, resize figures), create arxiv-submit.tar.gz
```

Upload `arxiv-submit.tar.gz` to [arXiv](https://arxiv.org/submit). The tarball includes `.tex`, `.bbl`, `.bib`, `PRIMEarxiv.sty`, and `media/` (if present), cleaned with `arxiv-latex-cleaner --keep_bib --resize_images`.
