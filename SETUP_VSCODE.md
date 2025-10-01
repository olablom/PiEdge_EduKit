# VS Code Setup - KRITISKT FÖR ATT FIXA ALLA PROBLEM

## ⚠️ GÖR DETTA NU INNAN VI COMMITTAR

Alla "reportMissingImports"-fel beror på att VS Code använder **fel Python-interpreter**.

### Steg 1: Välj rätt Python-interpreter

1. **Öppna Command Palette**: `Ctrl+Shift+P`
2. **Skriv**: `Python: Select Interpreter`
3. **Välj**: `.venv\Scripts\python.exe` (den med ".venv" i namnet)
4. **Verifiera**: Kolla bottom-right i VS Code - ska visa "Python 3.12 (.venv)"

### Steg 2: Installera dependencies i .venv

```bash
# I Git Bash terminal
source .venv/Scripts/activate
python -m pip install -r requirements-minimal.txt
```

### Steg 3: Starta om VS Code

- **Stäng** VS Code helt
- **Öppna** projektet igen
- **Vänta** 10-15 sekunder medan Pylance indexerar

### Steg 4: Verifiera att det fungerar

Öppna `labs/01_training_and_export.ipynb` och kolla att:

- ✅ Kernel visar ".venv" eller "Python 3.12"
- ✅ Inga röda "Import X could not be resolved" fel
- ✅ Problems-panelen tom eller < 10 varningar

### Varför detta är nödvändigt

VS Code's Pylance använder **system-Python** (som inte har torch, onnxruntime, etc.) istället för **repo-venv** (som har alla packages).

När du väl har gjort steg 1-3 kommer **alla** "reportMissingImports"-fel försvinna automatiskt.

### Om problem kvarstår efter Steg 1-3

Skriv till mig vilka exakta fel som finns kvar, så fixar jag dem.

