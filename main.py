#!/usr/bin/env python3
import os, sys, subprocess, pathlib, shutil

ROOT = pathlib.Path(__file__).resolve().parent
LABS = ROOT / "labs"
PY = sys.executable  # alltid venv:ens python om .venv är aktiv
KERNEL_NAME = "piedge-edukit-312"  # vi skapade denna tidigare

def have(cmd):
    return shutil.which(cmd) is not None

def ensure_kernel():
    # Försök använda existerande kernelspec; annars installera snabbt.
    try:
        out = subprocess.check_output([PY, "-m", "jupyter", "kernelspec", "list"], text=True)
        if KERNEL_NAME in out:
            return
    except Exception:
        pass
    print("→ Installerar Jupyter-kernel för den här miljön...")
    subprocess.check_call([PY, "-m", "pip", "install", "-q", "ipykernel", "papermill==2.6.0"])
    subprocess.check_call([PY, "-m", "ipykernel", "install", "--user",
                           "--name", KERNEL_NAME, "--display-name", "Python 3.12 (piedge)"])

def pick_notebook():
    nbs = sorted([p for p in LABS.glob("*.ipynb") if p.is_file()])
    if not nbs:
        print("Hittade inga notebooks i labs/.")
        sys.exit(1)
    print("\nVälj notebook:")
    for i, nb in enumerate(nbs, 1):
        print(f"  {i}) {nb.name}")
    while True:
        sel = input("Nummer: ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(nbs):
            return nbs[int(sel)-1]
        print("Ogiltigt val, försök igen.")

def open_in_jupyter(nb_path: pathlib.Path, lab=True):
    ensure_kernel()
    # Starta Jupyter i samma process till användaren stänger (CTRL+C)
    if lab:
        cmd = [PY, "-m", "jupyter", "lab", str(nb_path)]
    else:
        cmd = [PY, "-m", "notebook", str(nb_path)]
    print(f"\nÖppnar {'Jupyter Lab' if lab else 'Jupyter Notebook'}: {nb_path.name}\n")
    subprocess.call(cmd)

def run_with_papermill(nb_path: pathlib.Path):
    ensure_kernel()
    out = ROOT / "reports" / f"{nb_path.stem}_out.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Kör utan parametrar (notebooken funkar som är). Lägg gärna till en 'parameters'-cell i nb för -p stöd.
    cmd = [PY, "-m", "papermill", str(nb_path), str(out), "-k", KERNEL_NAME]
    print(f"\nKör headless med papermill → {out}")
    subprocess.check_call(cmd)
    print("Klart! Du kan öppna output-notebooken i Jupyter senare.")

def run_full_pipeline():
    """Snabb CLI-körning (terminal) med kvitto."""
    cmds = [
        [PY, "-m", "piedge_edukit.train", "--fakedata", "--no-pretrained", "--epochs", "1", "--batch-size", "256", "--output-dir", "./models"],
        [PY, "-m", "piedge_edukit.benchmark", "--fakedata", "--model-path", "./models/model.onnx", "--warmup", "1", "--runs", "3", "--providers", "CPUExecutionProvider"],
        [PY, "scripts/evaluate_onnx.py", "--model", "./models/model.onnx", "--fakedata", "--limit", "16"],
        [PY, "-m", "piedge_edukit.quantization", "--fakedata", "--model-path", "./models/model.onnx", "--calib-size", "32"],
        [PY, "verify.py"],
    ]
    print("\nKör hela pipelinen (terminal)…\n")
    for c in cmds:
        print(">", " ".join(c))
        subprocess.check_call(c)

def main():
    # Snabb koll: kör vi i rätt venv?
    if ".venv" not in PY.replace("\\", "/"):
        print("⚠️  Du verkar inte köra repo:ts .venv. Aktivera först:\n"
              "    Git Bash:   source .venv/Scripts/activate\n"
              "    PowerShell: .\\.venv\\Scripts\\Activate.ps1\n")
    # Meny
    print("\n=== PiEdge EduKit – terminalmeny ===")
    print("1) Öppna notebook i Jupyter Lab")
    print("2) Öppna notebook i Jupyter Notebook")
    print("3) Kör notebook headless (papermill) → reports/*_out.ipynb")
    print("4) Kör HELA pipelinen (terminal) + verify")
    print("5) Avsluta")
    choice = input("Val: ").strip()

    if choice == "1":
        nb = pick_notebook()
        open_in_jupyter(nb, lab=True)
    elif choice == "2":
        nb = pick_notebook()
        open_in_jupyter(nb, lab=False)
    elif choice == "3":
        nb = pick_notebook()
        run_with_papermill(nb)
    elif choice == "4":
        run_full_pipeline()
    else:
        print("Hejdå!")

if __name__ == "__main__":
    # Säkerställ att labs/ finns
    if not LABS.exists():
        print("Hittar inte katalogen 'labs/'. Är du i repo-roten?")
        sys.exit(1)
    # Minimal Jupyter-koll
    if not have("jupyter"):
        print("⚠️  Jupyter verkar inte finnas på PATH. Installera om du vill öppna i UI:\n"
              "    python -m pip install jupyter")
    main()
