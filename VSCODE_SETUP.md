# VS Code Setup Instructions

## Quick Fix for Pylance Import Warnings

If you see "Import 'xxx' could not be resolved" warnings in VS Code:

### 1. Open the Right Folder
- **File → Open Folder…** → select the repo **root** (`piedge_edukit`)
- NOT the `notebooks/` subfolder

### 2. Select Python Interpreter
- Press **Ctrl+Shift+P** → **Python: Select Interpreter**
- Choose: `…\piedge_edukit\.venv\Scripts\python.exe`
- Confirm in bottom-right status bar shows `.venv` path

### 3. Select Jupyter Kernel
- Open any notebook → top-right kernel picker
- Choose: **Python 3.12 (piedge)**

### 4. Reload VS Code
- **Ctrl+Shift+P** → **Developer: Reload Window**
- **Ctrl+Shift+P** → **Python: Restart Language Server**

### 5. Run Helper Cell
In `00_run_everything.ipynb`, run the first Python cell to configure sys.path.

## Verification
After these steps, the Problems panel should show minimal warnings. The helper cell ensures both runtime and static analysis work correctly.

## Files Created
- `.vscode/settings.json` - VS Code workspace settings
- `.env` - PYTHONPATH configuration
- Jupyter kernel `piedge` - for notebooks
