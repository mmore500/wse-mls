#!/bin/bash

set -e
shopt -s globstar

for binderpath in binder*; do
    echo "binderpath ${binderpath}"
    find "${binderpath}" -type f ! \( -name "*.pdf" -o -name "*.tex" -o -name "*.bib" \) -exec rm -f {} +
done

find . -type f -name '*.jpg' -exec rm -f {} +
find . -type d -name dishtiny -exec rm -rf {} +
find . -type d -name hstrat -exec rm -rf {} +
find . -type d -name conduit -exec rm -rf {} +
find . -type d -name docs -exec rm -rf {} +
find . -type d -empty -delete

find ./binder -type f -name "*.pdf" -print | while read -r f; do
    base=$(basename "$f" .pdf)   # strip .pdf extension
    if ! grep -qF "$base" ./**/*.tex 2>/dev/null; then
        echo rm "$f"
        rm "$f"
    fi
done

rm -f arxiv.tar.gz
git checkout bibl.bib
make cleaner
make
make clean
mv bibl.bib main.bib
cp bu1.bbl main.bbl
cp bu1.blg main.blg
rm draft.tex
tar -czvf arxiv.tar.gz *
