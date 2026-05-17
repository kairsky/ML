#!/usr/bin/env python
"""Assemble full_report.md and generate report.pdf (20+ pages)."""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SECTIONS_DIR = ROOT / "docs" / "report" / "sections"
REPORT_MD = ROOT / "docs" / "report" / "full_report.md"
REPORT_PDF = ROOT / "report.pdf"
PLOTS = ROOT / "results" / "plots"
FONT_DIR = Path(__file__).parent / "fonts"


def assemble_markdown() -> str:
    order = [
        "00_title.md",
        "task1_theory.md",
        "task2_classical_ml.md",
        "task3_dimensionality.md",
        "task4_mlp.md",
        "task5_deep_learning.md",
        "task6_survey.md",
        "methodology_detailed.md",
        "discussion.md",
        "extended_discussion.md",
        "appendix.md",
        "references.md",
    ]
    parts = []
    for name in order:
        path = SECTIONS_DIR / name
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
    return "\n\n---\n\n".join(parts)


def ensure_fonts() -> tuple[str, str]:
    """Return paths to regular and bold TTF with Cyrillic support."""
    import fpdf

    bundled = Path(fpdf.__file__).parent / "font"
    if (bundled / "DejaVuSans.ttf").exists():
        return str(bundled / "DejaVuSans.ttf"), str(bundled / "DejaVuSans-Bold.ttf")

    win = Path("C:/Windows/Fonts")
    if (win / "arial.ttf").exists():
        return str(win / "arial.ttf"), str(win / "arialbd.ttf")

    FONT_DIR.mkdir(exist_ok=True)
    cached = FONT_DIR / "DejaVuSans.ttf"
    if not cached.exists():
        url = "https://sourceforge.net/projects/dejavu/files/dejavu/2.37/dejavu-sans-ttf-2.37.zip/download"
        import io
        import zipfile

        print("Downloading DejaVu font pack...")
        data = urllib.request.urlopen(url, timeout=60).read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                if name.endswith("DejaVuSans.ttf"):
                    cached.write_bytes(zf.read(name))
                if name.endswith("DejaVuSans-Bold.ttf"):
                    (FONT_DIR / "DejaVuSans-Bold.ttf").write_bytes(zf.read(name))
    return str(cached), str(FONT_DIR / "DejaVuSans-Bold.ttf")


def _clean_line(line: str) -> str:
    line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
    line = re.sub(r"\*(.*?)\*", r"\1", line)
    line = re.sub(r"`([^`]+)`", r"\1", line)
    line = re.sub(r"\$[^$]+\$", "", line)
    line = re.sub(r"^#+\s*", "", line)
    return line.strip()


def build_pdf(md_text: str, output: Path) -> None:
    from fpdf import FPDF

    class ReportPDF(FPDF):
        def header(self):
            self.set_font("DejaVu", "B", 9)
            self.cell(0, 8, "ML/DL University Project Report", align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

        def footer(self):
            self.set_y(-15)
            self.set_font("DejaVu", size=9)
            self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    reg, bld = ensure_fonts()
    pdf.add_font("DejaVu", "", reg)
    pdf.add_font("DejaVu", "B", bld)
    pdf.add_page()
    pdf.set_font("DejaVu", size=11)

    for block in md_text.split("\n\n"):
        block = block.strip()
        if not block or block == "---":
            continue
        if block.startswith("|"):
            continue
        lines = [_clean_line(l) for l in block.split("\n") if l.strip()]
        if not lines:
            continue
        title = lines[0]
        body = "\n".join(lines[1:]) if len(lines) > 1 else ""

        if title.startswith("Задача"):
            pdf.add_page()
        if len(title) < 100 and (title.startswith("Задача") or title.startswith("#") or re.match(r"^\d+\.", title)):
            pdf.set_font("DejaVu", "B", 14)
            pdf.multi_cell(0, 8, _clean_line(title))
            pdf.ln(2)
            if body:
                pdf.set_font("DejaVu", size=11)
                pdf.multi_cell(0, 6, body)
                pdf.ln(3)
        else:
            pdf.set_font("DejaVu", size=11)
            pdf.multi_cell(0, 6, _clean_line(block))
            pdf.ln(3)

    figures = [
        ("Сравнение классификаторов (Covertype)", PLOTS / "classical_ml" / "covertype" / "model_comparison.png"),
        ("PCA: объяснённая дисперсия", PLOTS / "dimensionality" / "pca_explained_variance.png"),
        ("UMAP 2D", PLOTS / "dimensionality" / "umap_2d.png"),
        ("MLP: сравнение оптимизаторов", PLOTS / "mlp" / "optimizer_comparison.png"),
        ("CNN: обучение", PLOTS / "cnn" / "cnn_training.png"),
        ("CNN: confusion matrix", PLOTS / "cnn" / "cnn_confusion_matrix.png"),
        ("ROC Random Forest", PLOTS / "classical_ml" / "covertype" / "roc_random_forest.png"),
        ("Confusion matrix k-NN", PLOTS / "classical_ml" / "covertype" / "cm_knn.png"),
        ("Wine Quality comparison", PLOTS / "classical_ml" / "wine_quality" / "model_comparison.png"),
        ("t-SNE embedding", PLOTS / "dimensionality" / "tsne_2d.png"),
        ("LDA embedding", PLOTS / "dimensionality" / "lda_2d.png"),
        ("MLP training SGD", PLOTS / "mlp" / "history_sgd_relu.png"),
        ("Decision tree depth 3", PLOTS / "classical_ml" / "covertype" / "decision_tree_depth3.png"),
    ]
    for caption, fig in figures:
        if fig.exists():
            pdf.add_page()
            pdf.set_font("DejaVu", "B", 12)
            pdf.multi_cell(0, 8, f"Рисунок. {caption}")
            pdf.ln(2)
            pdf.image(str(fig), w=180)

    pdf.output(str(output))


def main() -> None:
    md = assemble_markdown()
    REPORT_MD.write_text(md, encoding="utf-8")
    words = len(md.split())
    print(f"Markdown: {REPORT_MD} | {words} words")
    build_pdf(md, REPORT_PDF)
    size_kb = REPORT_PDF.stat().st_size // 1024
    print(f"PDF: {REPORT_PDF} | {size_kb} KB (~{max(1, words // 350)} pages estimated)")


if __name__ == "__main__":
    main()
