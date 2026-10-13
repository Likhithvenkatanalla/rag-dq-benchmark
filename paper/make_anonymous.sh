#!/bin/sh
# Builds anonymous/main_anon.pdf for double-blind review: ACL "review" mode (anonymous author block,
# line numbers) and the code/data links replaced by an anonymity note. Run from paper/.
set -e
mkdir -p anonymous
python3 - <<'PY'
s = open("main.tex").read()
s = s.replace(r"\usepackage[final]{acl}", r"\usepackage[review]{acl}")
s = s.replace(r"\graphicspath{{figures/}{../figures/}}", r"\graphicspath{{../../figures/}}")
start = s.index(r"available.\footnote{")
end = s.index("}.}", start) + 3
s = s[:start] + r"available; links are omitted for anonymous review and will be added in the final version." + s[end:]
open("anonymous/main_anon.tex", "w").write(s)
PY
cp references.bib acl.sty acl_natbib.bst anonymous/
cd anonymous && TEXINPUTS=.:..: latexmk -pdf -interaction=nonstopmode -quiet main_anon.tex >/dev/null 2>&1
grep -o "Output written.*" main_anon.log
latexmk -c >/dev/null 2>&1; rm -f references.bib acl.sty acl_natbib.bst main_anon.bbl
