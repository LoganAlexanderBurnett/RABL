#!/usr/bin/env python3
"""Build MACE Methodology vector diagram PDFs/previews and data figures.

Reads the retained RABL figure inputs. Never simulates, trains, recalibrates,
and never silently substitutes synthetic controls for manuscript records.
Run from any directory; figure outputs use the repository's output root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs/paper/figures"
DIAGRAMS = {
    "fig02_lstm_autoregression": "fig02_lstm_autoregression.tikz.tex",
    "fig03_mace_calibration_events": "fig03_mace_calibration_events.tikz.tex",
    "fig04_microreactor_plant": "fig04_microreactor_plant.tikz.tex",
    "figA1_recursive_forecast_algorithm": "figA1_recursive_forecast_algorithm.tikz.tex",
}


def _build_diagram(stem: str, source: str, dest: Path, *, dpi: int) -> dict:
    """Compile the exact source diagram, not an independent approximation."""
    wrapper = r"""\documentclass[border=5pt]{standalone}
\usepackage{amsmath,amssymb,graphicx,tikz}
\usetikzlibrary{arrows.meta,positioning,calc,fit,matrix}
\begin{document}
\input{SOURCE}
\end{document}
""".replace("SOURCE", (FIGURES / source).as_posix())
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mace-diagram-") as tmp:
        tmpdir = Path(tmp)
        (tmpdir / "diagram.tex").write_text(wrapper, encoding="utf-8")
        command = ["pdflatex", "-halt-on-error", "-interaction=nonstopmode",
                   "-output-directory", str(tmpdir), str(tmpdir / "diagram.tex")]
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if run.returncode:
            log = tmpdir / "diagram.log"
            detail = log.read_text(errors="replace")[-4000:] if log.exists() else run.stdout[-4000:]
            raise RuntimeError(f"Could not compile {stem}:\n{detail}")
        pdf = dest / f"{stem}.pdf"
        pdf.write_bytes((tmpdir / "diagram.pdf").read_bytes())
        try:
            import fitz
        except ImportError as e:
            raise RuntimeError("Install PyMuPDF: python -m pip install pymupdf") from e
        with fitz.open(pdf) as doc:
            page = doc[0]
            preview = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
            preview.save(dest / f"{stem}.png")
    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    return {"diagram": stem, "source": source,
            "pdf": str(pdf.relative_to(ROOT)), "pdf_sha256": sha(pdf),
            "png": str((dest / f"{stem}.png").relative_to(ROOT)),
            "png_sha256": sha(dest / f"{stem}.png")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="outputs/paper/mace_figures/methodology_diagrams")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--diagrams-only", action="store_true", help="Do not run data-figure generator")
    ap.add_argument("--illustrative-sampling", action="store_true",
                    help="Explicitly permit reproducible illustrative GP draws instead of stored controls")
    args = ap.parse_args()
    if args.dpi < 300:
        ap.error("Use >= 300 dpi for preview exports")
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    try:
        for stem, source in DIAGRAMS.items():
            entries.append(_build_diagram(stem, source, out, dpi=args.dpi))
            print(f"Built {stem}", flush=True)
    except Exception as e:
        print(f"Diagram build failed: {e}", file=sys.stderr)
        return 1

    (out / "diagram_build_manifest.json").write_text(
        json.dumps(entries, indent=2) + "\n", encoding="utf-8"
    )
    if not args.diagrams_only:
        # Stored mode for the sampling figure avoids mislabeling illustrative
        # Gaussian draws as actual training controls. Figure 6 is excluded
        # from the manuscript and its default build.
        command = [sys.executable, str(ROOT / "scripts/generate_mace_paper_figures.py"),
                   "--config", "scripts/mace_paper_figures_config.json",
                   "--figures", "sampling", "dataset",
                   "--sampling-mode", "illustrative" if args.illustrative_sampling else "stored"]
        if args.illustrative_sampling:
            print("NOTE: sampling illustration is illustrative only; not eligible as an empirical manuscript panel.")
        run = subprocess.run(command, cwd=ROOT)
        if run.returncode:
            print("The archive-dependent figures need the stored control CSVs and/or"
                  " inputs at the configured paths. Diagram PDFs are complete.",
                  file=sys.stderr)
            return run.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
