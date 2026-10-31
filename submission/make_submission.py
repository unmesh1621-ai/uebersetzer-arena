"""Build the LMS submission Word document (Step 3 of the assignment brief)."""
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

OUT = Path(__file__).with_name("Unmesh_230911030_FGAI_Assignment3.docx")
REPO = "https://github.com/unmesh1621-ai/uebersetzer-arena"
FIELDS = [
    ("Student name", "Unmesh Vardhan", None),
    ("Registration number", "230911030", None),
    ("Course", "Foundations of Generative AI (ICT4403)", None),
    ("Assignment title", "Assignment 3 — Machine Translation with Encoder–Decoder and Decoder-Only Models", None),
    ("Title of the developed system",
     "Übersetzer-Arena: Comparative English→German Machine Translation with a Transformer, mT5, LLaMA and Gemma", None),
    ("Training dataset", "WMT14 English–German (30,000-pair cleaned subset)", "https://huggingface.co/datasets/wmt/wmt14"),
    ("Test dataset", "FLORES+ devtest, eng_Latn → deu_Latn (1,012 sentences)",
     "https://huggingface.co/datasets/openlanguagedata/flores_plus"),
    ("GitHub repository", REPO, REPO),
]


def add_hyperlink(paragraph, url, text):
    rel = paragraph.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
                                   is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), rel)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "000000")
    rpr.append(color)
    run.append(rpr)
    t = OxmlElement("w:t")
    t.text = text
    run.append(t)
    link.append(run)
    paragraph._p.append(link)


def main():
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    title = doc.add_heading("FGAI Assignment 3 — Submission Details", level=1)
    title.runs[0].font.color.rgb = RGBColor(0x0B, 0x0B, 0x0B)

    table = doc.add_table(rows=0, cols=2)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for label, value, url in FIELDS:
        cells = table.add_row().cells
        cells[0].paragraphs[0].add_run(label).bold = True
        p = cells[1].paragraphs[0]
        p.add_run(value)
        if url and url != value:
            p.add_run("\n")
            add_hyperlink(p, url, url)
        elif url:
            p.clear()
            add_hyperlink(p, url, value)
    for row in table.rows:
        row.cells[0].width, row.cells[1].width = Pt(140), Pt(330)

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
