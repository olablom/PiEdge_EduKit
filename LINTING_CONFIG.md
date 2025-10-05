# Linting Configuration

## What Was Done

The project now has **pragmatic linting** that keeps code quality high while reducing noise in notebooks:

### 1. Ruff Configuration (`.ruff.toml`)
- **Strict linting** for `src/` and `scripts/` (production code)
- **Relaxed rules** for `notebooks/*.ipynb` (educational content)
- Ignores common notebook issues: long lines, import order, unused imports, redefinitions

### 2. Pyright Configuration (`pyrightconfig.json`)
- **Excludes notebooks** from type checking
- **Focuses on** `src/` and `scripts/` directories
- **Basic type checking** mode for faster analysis

### 3. Notebook Pragmas
- Added `# pyright: reportAttributeAccessIssue=false` to suppress torchvision FakeData warnings
- This is a known issue with incomplete type stubs in some torchvision versions

## Result
- **Problems panel** should now be clean or show only relevant warnings
- **Code quality** maintained in production files (`src/`, `scripts/`)
- **Student experience** improved with fewer distracting warnings

## To Apply Changes
1. **Ctrl+Shift+P** → **Developer: Reload Window**
2. **Ctrl+Shift+P** → **Python: Restart Language Server**
3. Check Problems panel - should be much cleaner now

## Philosophy
This follows the principle: **strict where it matters, pragmatic where it doesn't**.
- Educational notebooks need flexibility for experimentation
- Production code needs strict quality control
- Students shouldn't be distracted by cosmetic warnings
