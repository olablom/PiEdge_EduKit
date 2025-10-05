#!/usr/bin/env python3
"""
i18n_notebook_fix.py - Programmatic Swedish-to-English translation for Jupyter notebooks
"""

import sys
import json
import shutil
from pathlib import Path
import nbformat as nbf

REPL = {
    "Package saknas": "Package missing",
    "installerar editable från": "installing editable from",
    "Python 3.12 krävs": "Python 3.12 required",
    "du har": "you have",
    "Package installerat": "Package installed",
    "Miljökoll + självläkning": "Environment check + self-healing",
    "Skapar syntetiskt kalibreringsset": "Creating synthetic calibration set",
    "skapats": "created",
    "Fortsätter": "Continuing",
    "Tränar en liten modell": "Training a small model",
    "Mäter hur snabb modellen är": "Measuring how fast the model is",
    "Komprimerar modellen": "Compressing the model",
    "Testar modellen och genererar kvitto": "Testing the model and generating receipt",
    "Först kontrollerar vi att miljön är korrekt": "First we check that the environment is correct",
    "Träna modell (snabb körning för demo)": "Train model (quick run for demo)",
    "Paketet importeras – kör vidare!": "Package imported – continue!",
    "data/train saknas → skapar syntetiskt kalibreringsset": "data/train missing → creating synthetic calibration set",
    "Created synthetic calibration set (32 imgs)": "Created synthetic calibration set (32 imgs)",
    "saknas": "missing",
    "installerat": "installed",
    "krävs": "required",
    "Miljökoll": "Environment check",
    "självläkning": "self-healing",
    "Träningsgrafer saknas": "Training graphs missing",
    "Benchmark-rapport saknas": "Benchmark report missing",
    "Kvantiseringsrapport saknas": "Quantization report missing",
    "Kvitto saknas": "Receipt missing",
    "Confusion matrix saknas": "Confusion matrix missing",
    "Utvärderingsrapport saknas": "Evaluation report missing",
    "Modell saknas": "Model missing",
    "Modellfiler saknas": "Model files missing",
    "Latens CSV saknas": "Latency CSV missing",
    "Träningsinfo saknas": "Training info missing",
    "ONNX-modell saknas": "ONNX model missing",
    "Artefakt-generering": "Artifact generation",
    "nödvändiga filer skapas": "necessary files are created",
    'Vad saknas för "produktion"?': 'What is missing for "production"?',
    "Tips": "Tips",
    "Om pip saknas": "If pip is missing",
    "kör": "run",
    "innan installation": "before installation",
    "Ingen PYTHONPATH krävs": "No PYTHONPATH required",
    "Installera paketet med": "Install the package with",
    "Om": "If",
    "saknas i PATH": "missing in PATH",
    "använd": "use",
    "Dessa filer skapas automatiskt": "These files are created automatically",
    "vid smoke test-körning": "during smoke test run",
}


def repl_text(txt: str) -> str:
    """Replace Swedish text with English equivalents."""
    for sv, en in REPL.items():
        txt = txt.replace(sv, en)
    return txt


def fix_nb(path: Path):
    """Fix Swedish text in a Jupyter notebook."""
    print(f"Processing: {path}")

    # Read notebook
    nb = nbf.read(path, as_version=4)
    changes_made = False

    for cell in nb.cells:
        # Cell source
        if isinstance(cell.get("source"), str):
            original = cell["source"]
            cell["source"] = repl_text(cell["source"])
            if cell["source"] != original:
                changes_made = True
        elif isinstance(cell.get("source"), list):
            original = cell["source"]
            cell["source"] = [repl_text(line) for line in cell["source"]]
            if cell["source"] != original:
                changes_made = True

        # Plain-text outputs
        if "outputs" in cell:
            for out in cell["outputs"]:
                # text/plain or stream text
                if "text" in out and isinstance(out["text"], str):
                    original = out["text"]
                    out["text"] = repl_text(out["text"])
                    if out["text"] != original:
                        changes_made = True
                elif "text" in out and isinstance(out["text"], list):
                    original = out["text"]
                    out["text"] = [repl_text(t) for t in out["text"]]
                    if out["text"] != original:
                        changes_made = True

                # Data outputs
                if "data" in out and isinstance(out["data"], dict):
                    for k, v in list(out["data"].items()):
                        if isinstance(v, str):
                            original = out["data"][k]
                            out["data"][k] = repl_text(v)
                            if out["data"][k] != original:
                                changes_made = True
                        elif isinstance(v, list):
                            original = out["data"][k]
                            out["data"][k] = [repl_text(t) for t in v]
                            if out["data"][k] != original:
                                changes_made = True

    if changes_made:
        # Create backup
        backup_path = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup_path)
        print(f"  -> Backup created: {backup_path}")

        # Write updated notebook
        nbf.write(nb, path)
        print(f"  -> Updated: {path}")
    else:
        print(f"  -> No changes needed")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/i18n_notebook_fix.py <notebook.ipynb> [...]")
        print("Example: python scripts/i18n_notebook_fix.py notebooks/00_run_everything.ipynb")
        sys.exit(1)

    for p in sys.argv[1:]:
        path = Path(p)
        if not path.exists():
            print(f"Error: {path} does not exist")
            continue
        if not path.suffix == ".ipynb":
            print(f"Warning: {path} is not a .ipynb file, skipping")
            continue

        try:
            fix_nb(path)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

    print("\nTranslation complete!")


if __name__ == "__main__":
    main()
