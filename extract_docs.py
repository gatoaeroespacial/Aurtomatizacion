import docx
import os

base = r"c:\Users\juanm\Documents\Proyecto Automatización\Documentos"

for fname, outname in [("Informe avances auto.docx", "informe1.txt"),
                        ("avances con el paso a seguir.docx", "informe2.txt")]:
    path = os.path.join(base, fname)
    if os.path.exists(path):
        doc = docx.Document(path)
        text = "\n".join([p.text for p in doc.paragraphs])
        outpath = os.path.join(base, outname)
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"OK: {outname} ({len(text)} chars)")
    else:
        print(f"NOT FOUND: {path}")
