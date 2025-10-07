#!/usr/bin/env python3
import re, json, glob, sys
from pathlib import Path

NB_GLOB = "notebooks/*.ipynb"

swedish_chars = re.compile(r"[ÅÄÖåäö]")
# vanliga svenska ord – kompletterar tecken-sökningen
sw_words = re.compile(
    r"\b(och|är|inte|med|för|från|olika|jämför|kör|träning|kvantisering|latens|förutsägelser)\b",
    re.IGNORECASE,
)

bad = []
for p in glob.glob(NB_GLOB):
    nb = json.loads(Path(p).read_text(encoding="utf-8"))
    for i, cell in enumerate(nb.get("cells", []), 1):
        txt = "".join(cell.get("source", []))
        hits = []
        for m in swedish_chars.finditer(txt):
            s = max(0, m.start()-40); e = min(len(txt), m.end()+40)
            ctx = txt[s:e].replace("\n"," ")
            hits.append(("char", f"char={txt[m.start():m.end()]} U+{ord(txt[m.start()]):04X} …{ctx}…"))
        for m in sw_words.finditer(txt):
            s = max(0, m.start()-40); e = min(len(txt), m.end()+40)
            ctx = txt[s:e].replace("\n"," ")
            hits.append(("word", f"word={m.group(0)} …{ctx}…"))
        if hits:
            bad.append((p, i, hits))

if not bad:
    print("✔ English-only (no Å/Ä/Ö and no Swedish tokens found)")
    sys.exit(0)

print("✘ Non-English remnants found:\n")
for p, i, hits in bad:
    print(f"{p}: cell {i}")
    for kind, hit in hits:
        print(f"  [{kind}] {hit}")
    print()
sys.exit(2)
