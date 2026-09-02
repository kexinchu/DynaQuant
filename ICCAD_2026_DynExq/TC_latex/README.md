# DynaExQ — IEEE Transactions on Computers source package

Regular paper, 12 pages including references and author biographies.

## Upload to IEEE Author Portal

Use the **existing** Unsubmitted / draft manuscript (do not start a new submission):

https://ieee.submission.researchexchange.com/journal/tc-cs

1. Manuscript PDF: `main_sc.pdf`
2. Source archive: zip this directory (`TC_latex/`)
3. Cover letter: `COVER_LETTER.md` (or paste into the portal)

## Compile

```bash
latexmk -pdf main_sc.tex
```

or:

```bash
pdflatex main_sc
bibtex main_sc
pdflatex main_sc
pdflatex main_sc
```

This folder contains only the files needed to rebuild the manuscript:
`main_sc.tex`, section files, `references.bib`, `IEEEtran.cls`, `IEEEtran.bst`,
the 19 figures referenced in the paper, and author photos in `photos/`.
